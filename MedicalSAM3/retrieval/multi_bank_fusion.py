"""Adaptive multi-bank retrieval fusion utilities."""

from __future__ import annotations

from typing import Any, Optional

import torch

from MedicalSAM3.retrieval.site_bank_resolver import SiteBankResolution


def _tensor_or_zeros(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return value.to(device=like.device, dtype=like.dtype)
    return torch.zeros(like.shape[0], device=like.device, dtype=like.dtype)


def _batch_similarity(retrieval: dict[str, Any], *, like: torch.Tensor) -> dict[str, torch.Tensor]:
    similarity_score = retrieval.get("similarity_score", {})
    stability = retrieval.get("retrieval_stability", {})
    positive = _tensor_or_zeros(
        similarity_score.get("positive_topk_mean", stability.get("positive_similarity_mean")),
        like=like,
    )
    negative = _tensor_or_zeros(
        similarity_score.get("negative_topk_mean", stability.get("negative_similarity_mean")),
        like=like,
    )
    margin = _tensor_or_zeros(similarity_score.get("margin", positive - negative), like=like)
    return {
        "positive_mean": positive,
        "negative_mean": negative,
        "margin": margin,
    }


def _bank_presence(retrieval: dict[str, Any], *, like: torch.Tensor) -> torch.Tensor:
    positive_weights = retrieval.get("positive_weights")
    negative_weights = retrieval.get("negative_weights")
    positive_sum = (
        positive_weights.to(device=like.device, dtype=like.dtype).sum(dim=1)
        if isinstance(positive_weights, torch.Tensor) and positive_weights.numel() > 0
        else torch.zeros(like.shape[0], device=like.device, dtype=like.dtype)
    )
    negative_sum = (
        negative_weights.to(device=like.device, dtype=like.dtype).sum(dim=1)
        if isinstance(negative_weights, torch.Tensor) and negative_weights.numel() > 0
        else torch.zeros(like.shape[0], device=like.device, dtype=like.dtype)
    )
    return ((positive_sum + negative_sum) > 0).to(dtype=like.dtype)


def _bank_weights(
    train_retrieval: dict[str, Any],
    site_retrieval: dict[str, Any],
    *,
    score_temperature: float = 0.125,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    reference = train_retrieval.get("positive_prototype")
    if not isinstance(reference, torch.Tensor):
        raise TypeError("train_retrieval must contain positive_prototype")
    train_similarity = _batch_similarity(train_retrieval, like=reference)
    site_similarity = _batch_similarity(site_retrieval, like=reference)
    train_score = 0.5 * (train_similarity["positive_mean"] + train_similarity["margin"])
    site_score = 0.5 * (site_similarity["positive_mean"] + site_similarity["margin"])
    stacked_scores = torch.stack([train_score, site_score], dim=1) / max(float(score_temperature), 1e-6)
    contribution = torch.softmax(stacked_scores, dim=1)

    train_present = _bank_presence(train_retrieval, like=reference)
    site_present = _bank_presence(site_retrieval, like=reference)
    only_train = (train_present > 0) & (site_present <= 0)
    only_site = (site_present > 0) & (train_present <= 0)
    neither = (train_present <= 0) & (site_present <= 0)

    contribution = contribution * torch.stack([train_present, site_present], dim=1)
    contribution_sum = contribution.sum(dim=1, keepdim=True)
    contribution = torch.where(contribution_sum > 0, contribution / contribution_sum.clamp_min(1e-6), contribution)
    contribution[only_train] = torch.tensor([1.0, 0.0], device=reference.device, dtype=reference.dtype)
    contribution[only_site] = torch.tensor([0.0, 1.0], device=reference.device, dtype=reference.dtype)
    contribution[neither] = torch.tensor([1.0, 0.0], device=reference.device, dtype=reference.dtype)
    return train_similarity, site_similarity, contribution[:, 0], contribution[:, 1]


def _combine_token_stream(
    train_features: torch.Tensor,
    site_features: torch.Tensor,
    train_weights: torch.Tensor,
    site_weights: torch.Tensor,
    train_contribution: torch.Tensor,
    site_contribution: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_train = train_weights * train_contribution.unsqueeze(1)
    scaled_site = site_weights * site_contribution.unsqueeze(1)
    combined_features = torch.cat([train_features, site_features], dim=1)
    combined_weights = torch.cat([scaled_train, scaled_site], dim=1)
    total = combined_weights.sum(dim=1, keepdim=True)
    normalized_weights = torch.where(total > 0, combined_weights / total.clamp_min(1e-6), combined_weights)
    return combined_features, normalized_weights


def _combine_prototype(
    train_prototype: torch.Tensor,
    site_prototype: torch.Tensor,
    train_contribution: torch.Tensor,
    site_contribution: torch.Tensor,
) -> torch.Tensor:
    fused = train_prototype * train_contribution.unsqueeze(1) + site_prototype * site_contribution.unsqueeze(1)
    norm = fused.norm(dim=1, keepdim=True)
    return torch.where(norm > 0, fused / norm.clamp_min(1e-6), fused)


def _combine_score_lists(train_scores: list[Any], site_scores: list[Any]) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    batch_size = max(len(train_scores), len(site_scores))
    for batch_index in range(batch_size):
        left = train_scores[batch_index] if batch_index < len(train_scores) else torch.zeros(0)
        right = site_scores[batch_index] if batch_index < len(site_scores) else torch.zeros(0)
        left_tensor = left if isinstance(left, torch.Tensor) else torch.tensor(left, dtype=torch.float32)
        right_tensor = right if isinstance(right, torch.Tensor) else torch.tensor(right, dtype=torch.float32)
        result.append(torch.cat([left_tensor, right_tensor], dim=0))
    return result


def _combine_entries(train_entries: list[list[Any]], site_entries: list[list[Any]]) -> list[list[Any]]:
    result: list[list[Any]] = []
    batch_size = max(len(train_entries), len(site_entries))
    for batch_index in range(batch_size):
        left = train_entries[batch_index] if batch_index < len(train_entries) else []
        right = site_entries[batch_index] if batch_index < len(site_entries) else []
        result.append(list(left) + list(right))
    return result


def _weighted_stability(
    train_retrieval: dict[str, Any],
    site_retrieval: dict[str, Any],
    train_contribution: torch.Tensor,
    site_contribution: torch.Tensor,
    key: str,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    train_value = _tensor_or_zeros(train_retrieval.get("retrieval_stability", {}).get(key), like=like)
    site_value = _tensor_or_zeros(site_retrieval.get("retrieval_stability", {}).get(key), like=like)
    return train_value * train_contribution + site_value * site_contribution


def annotate_single_bank_retrieval(
    retrieval: dict[str, Any],
    *,
    resolution: SiteBankResolution,
    bank_label: str,
    bank_path: str,
) -> dict[str, Any]:
    updated = dict(retrieval)
    reference = updated.get("positive_prototype")
    if not isinstance(reference, torch.Tensor):
        return updated
    device = reference.device
    dtype = reference.dtype
    batch_size = reference.shape[0]
    train_weight = torch.ones(batch_size, device=device, dtype=dtype) if bank_label == "train" else torch.zeros(batch_size, device=device, dtype=dtype)
    site_weight = torch.ones(batch_size, device=device, dtype=dtype) if bank_label != "train" else torch.zeros(batch_size, device=device, dtype=dtype)
    similarity = _batch_similarity(updated, like=reference)
    updated["multi_bank_fusion"] = {
        "mode": resolution.mode,
        "site_id": resolution.site_id,
        "parsed_site_id": resolution.site_id,
        "expected_site_bank": None if resolution.expected_site_bank is None else str(resolution.expected_site_bank),
        "fallback_reason": resolution.fallback_reason,
        "selected_bank_paths": [bank_path],
        "fallback_to_train_bank": resolution.fallback_to_train_bank,
        "warnings": list(resolution.warnings),
        "train_topk_exemplar": {
            "positive_entries": updated.get("positive_entries", []),
            "negative_entries": updated.get("negative_entries", []),
            "positive_scores": updated.get("positive_scores", []),
            "negative_scores": updated.get("negative_scores", []),
            "positive_weights": updated.get("positive_weights"),
            "negative_weights": updated.get("negative_weights"),
        } if bank_label == "train" else {},
        "site_topk_exemplar": {
            "positive_entries": updated.get("positive_entries", []),
            "negative_entries": updated.get("negative_entries", []),
            "positive_scores": updated.get("positive_scores", []),
            "negative_scores": updated.get("negative_scores", []),
            "positive_weights": updated.get("positive_weights"),
            "negative_weights": updated.get("negative_weights"),
        } if bank_label != "train" else {},
        "train_similarity": similarity if bank_label == "train" else {key: torch.zeros_like(value) for key, value in similarity.items()},
        "site_similarity": similarity if bank_label != "train" else {key: torch.zeros_like(value) for key, value in similarity.items()},
        "train_contribution": train_weight,
        "site_contribution": site_weight,
        "final_fusion_weight": site_weight,
    }
    return updated


def fuse_multi_bank_retrieval(
    *,
    train_retrieval: dict[str, Any],
    site_retrieval: dict[str, Any],
    resolution: SiteBankResolution,
    train_bank_path: str,
    site_bank_path: Optional[str],
    score_temperature: float = 0.125,
) -> dict[str, Any]:
    reference = train_retrieval.get("positive_prototype")
    if not isinstance(reference, torch.Tensor):
        raise TypeError("train_retrieval must contain positive_prototype")

    train_similarity, site_similarity, train_contribution, site_contribution = _bank_weights(
        train_retrieval,
        site_retrieval,
        score_temperature=score_temperature,
    )

    positive_features, positive_weights = _combine_token_stream(
        train_retrieval["positive_features"],
        site_retrieval["positive_features"],
        train_retrieval["positive_weights"],
        site_retrieval["positive_weights"],
        train_contribution,
        site_contribution,
    )
    negative_features, negative_weights = _combine_token_stream(
        train_retrieval["negative_features"],
        site_retrieval["negative_features"],
        train_retrieval["negative_weights"],
        site_retrieval["negative_weights"],
        train_contribution,
        site_contribution,
    )

    fused = dict(train_retrieval)
    fused["positive_features"] = positive_features
    fused["negative_features"] = negative_features
    fused["positive_weights"] = positive_weights
    fused["negative_weights"] = negative_weights
    fused["positive_prototype"] = _combine_prototype(
        train_retrieval["positive_prototype"],
        site_retrieval["positive_prototype"],
        train_contribution,
        site_contribution,
    )
    fused["negative_prototype"] = _combine_prototype(
        train_retrieval["negative_prototype"],
        site_retrieval["negative_prototype"],
        train_contribution,
        site_contribution,
    )
    fused["positive_prototype_feature"] = fused["positive_prototype"]
    fused["negative_prototype_feature"] = fused["negative_prototype"]
    fused["positive_entries"] = _combine_entries(train_retrieval.get("positive_entries", []), site_retrieval.get("positive_entries", []))
    fused["negative_entries"] = _combine_entries(train_retrieval.get("negative_entries", []), site_retrieval.get("negative_entries", []))
    fused["positive_scores"] = _combine_score_lists(train_retrieval.get("positive_scores", []), site_retrieval.get("positive_scores", []))
    fused["negative_scores"] = _combine_score_lists(train_retrieval.get("negative_scores", []), site_retrieval.get("negative_scores", []))
    fused["positive_score_tensor"] = torch.cat([train_retrieval["positive_score_tensor"], site_retrieval["positive_score_tensor"]], dim=1)
    fused["negative_score_tensor"] = torch.cat([train_retrieval["negative_score_tensor"], site_retrieval["negative_score_tensor"]], dim=1)
    fused["top_k_positive"] = int(train_retrieval.get("top_k_positive", 0)) + int(site_retrieval.get("top_k_positive", 0))
    fused["top_k_negative"] = int(train_retrieval.get("top_k_negative", 0)) + int(site_retrieval.get("top_k_negative", 0))
    fused["similarity_score"] = {
        "positive_topk_mean": train_similarity["positive_mean"] * train_contribution + site_similarity["positive_mean"] * site_contribution,
        "negative_topk_mean": train_similarity["negative_mean"] * train_contribution + site_similarity["negative_mean"] * site_contribution,
        "margin": train_similarity["margin"] * train_contribution + site_similarity["margin"] * site_contribution,
    }
    fused["retrieval_stability"] = {
        "positive_similarity_mean": fused["similarity_score"]["positive_topk_mean"],
        "negative_similarity_mean": fused["similarity_score"]["negative_topk_mean"],
        "margin": fused["similarity_score"]["margin"],
        "positive_similarity_std": _weighted_stability(train_retrieval, site_retrieval, train_contribution, site_contribution, "positive_similarity_std", like=reference),
        "negative_similarity_std": _weighted_stability(train_retrieval, site_retrieval, train_contribution, site_contribution, "negative_similarity_std", like=reference),
        "positive_weight_entropy": _weighted_stability(train_retrieval, site_retrieval, train_contribution, site_contribution, "positive_weight_entropy", like=reference),
        "negative_weight_entropy": _weighted_stability(train_retrieval, site_retrieval, train_contribution, site_contribution, "negative_weight_entropy", like=reference),
    }
    fused["multi_bank_fusion"] = {
        "mode": resolution.mode,
        "site_id": resolution.site_id,
        "parsed_site_id": resolution.site_id,
        "expected_site_bank": None if resolution.expected_site_bank is None else str(resolution.expected_site_bank),
        "fallback_reason": resolution.fallback_reason,
        "selected_bank_paths": [path for path in [train_bank_path, site_bank_path] if path],
        "fallback_to_train_bank": resolution.fallback_to_train_bank,
        "warnings": list(resolution.warnings),
        "train_topk_exemplar": {
            "positive_entries": train_retrieval.get("positive_entries", []),
            "negative_entries": train_retrieval.get("negative_entries", []),
            "positive_scores": train_retrieval.get("positive_scores", []),
            "negative_scores": train_retrieval.get("negative_scores", []),
            "positive_weights": train_retrieval.get("positive_weights"),
            "negative_weights": train_retrieval.get("negative_weights"),
        },
        "site_topk_exemplar": {
            "positive_entries": site_retrieval.get("positive_entries", []),
            "negative_entries": site_retrieval.get("negative_entries", []),
            "positive_scores": site_retrieval.get("positive_scores", []),
            "negative_scores": site_retrieval.get("negative_scores", []),
            "positive_weights": site_retrieval.get("positive_weights"),
            "negative_weights": site_retrieval.get("negative_weights"),
        },
        "train_similarity": train_similarity,
        "site_similarity": site_similarity,
        "train_contribution": train_contribution,
        "site_contribution": site_contribution,
        "final_fusion_weight": site_contribution,
    }
    return fused


__all__ = ["annotate_single_bank_retrieval", "fuse_multi_bank_retrieval"]