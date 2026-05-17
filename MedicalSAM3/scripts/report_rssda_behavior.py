"""Generate lightweight numerical reports for RSS-DA behavior and domain gap."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar_bank import PrototypeBankEntry, RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import load_retrieval_bank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    ensure_dir,
    infer_source_domain,
    read_records,
    resolve_feature_map,
)


def _resolve_hidden_dim(model: torch.nn.Module) -> int:
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def _resolve_runtime_device(requested_device: str) -> str:
    normalized = requested_device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported --device value: {requested_device}")
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


def _mask_area_ratio(mask_logits: torch.Tensor) -> float:
    prob = torch.sigmoid(mask_logits)
    return float((prob > 0.5).float().mean().item())


def _mean_confidence(outputs: dict[str, Any]) -> float:
    scores = outputs.get("scores")
    if isinstance(scores, torch.Tensor):
        return float(scores.mean().item())
    masks = outputs.get("masks")
    if isinstance(masks, torch.Tensor):
        return float(masks.mean().item())
    return 0.0


def _load_checkpoint_payload(path: Path, device: str) -> object:
    return torch.load(path, map_location=device, weights_only=False)


def _maybe_load_rssda_bundle(
    path: Path,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> bool:
    payload = _load_checkpoint_payload(path, device)
    if not isinstance(payload, dict):
        return False
    loaded = False
    adapter_state = payload.get("adapter")
    retriever_state = payload.get("retriever")
    similarity_state = payload.get("similarity_builder")
    if isinstance(adapter_state, dict):
        adapter.load_state_dict(adapter_state, strict=False)
        loaded = True
    if isinstance(retriever_state, dict):
        retriever.load_state_dict(retriever_state, strict=False)
        loaded = True
    if isinstance(similarity_state, dict):
        similarity_builder.load_state_dict(similarity_state, strict=False)
        loaded = True
    return loaded


def _apply_retrieval_mode(retrieval: dict[str, object], mode: str) -> dict[str, object]:
    if mode in {"joint", "semantic", "spatial", "positive-negative"}:
        return retrieval
    if mode != "positive-only":
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    updated = dict(retrieval)
    updated["negative_features"] = torch.zeros_like(retrieval["negative_features"])
    updated["negative_weights"] = torch.zeros_like(retrieval["negative_weights"])
    updated["negative_prototype"] = torch.zeros_like(retrieval["positive_prototype"])
    updated["negative_entries"] = [[] for _ in retrieval["positive_entries"]]
    updated["negative_scores"] = [torch.zeros_like(score) for score in retrieval["positive_scores"]]
    return updated


def _dummy_records(prefix: str, dataset_name: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "image_path": "",
            "mask_path": "",
            "dataset_name": dataset_name,
            "image_id": f"{prefix}_{index:03d}",
        }
        for index in range(count)
    ]


def _ensure_records(split_file: str | Path, dummy: bool, prefix: str, dataset_name: str, count: int) -> list[dict[str, Any]]:
    records = read_records(split_file)
    if dummy and not records:
        return _dummy_records(prefix, dataset_name, count)
    return records


def _create_dummy_bank(bank_dir: Path, hidden_dim: int, seed: int) -> RSSDABank:
    generator = torch.Generator().manual_seed(seed)
    bank = RSSDABank()
    entries = [
        ("kvasir_positive_a", "positive", "Kvasir"),
        ("kvasir_positive_b", "positive", "Kvasir"),
        ("cvc_positive_a", "positive", "CVC"),
        ("cvc_positive_b", "positive", "CVC"),
        ("negative_a", "negative", "Kvasir"),
        ("negative_b", "negative", "CVC"),
    ]
    for index, (prototype_id, polarity, source_dataset) in enumerate(entries):
        feature_dir = ensure_dir(bank_dir / ("positive_bank" if polarity == "positive" else "negative_bank"))
        feature_path = feature_dir / f"{prototype_id}.pt"
        base = torch.randn(hidden_dim, generator=generator)
        base[index % min(hidden_dim, 8)] += 3.0
        torch.save({"prototype": torch.nn.functional.normalize(base.float(), dim=0)}, feature_path)
        bank.add_entry(
            PrototypeBankEntry(
                prototype_id=prototype_id,
                feature_path=str(feature_path),
                polarity=polarity,
                source_dataset=source_dataset,
                polyp_type="polyp" if polarity == "positive" else "background",
                boundary_quality=0.8,
                confidence=0.9,
                image_id=f"{source_dataset.lower()}_{index:03d}",
                device_metadata={"runtime_device": "dummy"},
                extra_metadata={"source_group": source_dataset},
            )
        )
    bank.save(bank_dir)
    return bank


def _load_or_create_bank(
    path: str | Path,
    hidden_dim: int,
    dummy: bool,
    seed: int,
    *,
    image_size: int = 128,
    precision: str = "fp32",
    checkpoint: Optional[str] = None,
    device: str = "auto",
) -> RSSDABank:
    bank_path = Path(path)
    if bank_path.exists():
        bank_context = load_retrieval_bank(
            bank_path,
            purpose="validation",
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=dummy,
        )
        if bank_context.bank.entries:
            return bank_context.bank
    if not dummy:
        raise FileNotFoundError(f"RSS-DA bank not found or empty: {bank_path}")
    return _create_dummy_bank(bank_path, hidden_dim=hidden_dim, seed=seed)


def _entry_source_counts(bank: RSSDABank) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in bank.entries:
        polarity_counts = counts.setdefault(entry.source_dataset, {"positive": 0, "negative": 0})
        polarity_counts[entry.polarity] = polarity_counts.get(entry.polarity, 0) + 1
    return counts


def _metadata_readiness(bank: RSSDABank) -> dict[str, Any]:
    device_keys: set[str] = set()
    extra_keys: set[str] = set()
    has_hospital = False
    has_device = False
    for entry in bank.entries:
        device_keys.update(str(key) for key in entry.device_metadata.keys())
        extra_keys.update(str(key) for key in entry.extra_metadata.keys())
        has_hospital = has_hospital or ("hospital" in entry.device_metadata) or ("hospital" in entry.extra_metadata)
        has_device = has_device or ("device" in entry.device_metadata) or ("device" in entry.extra_metadata)
    return {
        "source_dataset_count": len({entry.source_dataset for entry in bank.entries}),
        "device_metadata_keys": sorted(device_keys),
        "extra_metadata_keys": sorted(extra_keys),
        "has_hospital_metadata": has_hospital,
        "has_device_metadata": has_device,
        "ready_for_hybrid_domain_score": has_hospital or has_device,
    }


def _selection_from_entries(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    dim = int(query_vector.shape[-1])
    if not indices:
        return (
            torch.zeros(1, 0, dim, device=query_vector.device),
            torch.zeros(1, 0, device=query_vector.device),
            [],
            torch.zeros(1, dim, device=query_vector.device),
            torch.zeros(0, device=query_vector.device),
        )
    selected_entries = [entries[index] for index in indices]
    features = bank.stack_features(selected_entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    raw_scores = torch.matmul(features, query_vector)
    weights = torch.softmax(raw_scores, dim=0)
    prototype = torch.nn.functional.normalize((weights.unsqueeze(-1) * features).sum(dim=0, keepdim=True), dim=-1)
    return features.unsqueeze(0), weights.unsqueeze(0), selected_entries, prototype, raw_scores


def _rank_entry_indices(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    top_k: int,
    strategy: str,
    rng: random.Random,
) -> list[int]:
    if not entries or top_k <= 0:
        return []
    features = bank.stack_features(entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    scores = torch.matmul(features, query_vector)
    count = min(top_k, len(entries))
    if strategy == "best":
        return torch.topk(scores, k=count, largest=True).indices.detach().cpu().tolist()
    if strategy == "worst":
        return torch.topk(scores, k=count, largest=False).indices.detach().cpu().tolist()
    if strategy == "random":
        shuffled = list(range(len(entries)))
        rng.shuffle(shuffled)
        return shuffled[:count]
    raise ValueError(f"Unsupported ranking strategy: {strategy}")


def _override_retrieval(
    base_retrieval: dict[str, Any],
    *,
    positive_override: Optional[tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]] = None,
    negative_override: Optional[tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]] = None,
) -> dict[str, Any]:
    retrieval = dict(base_retrieval)
    if positive_override is not None:
        retrieval["positive_features"] = positive_override[0]
        retrieval["positive_weights"] = positive_override[1]
        retrieval["positive_entries"] = [positive_override[2]]
        retrieval["positive_prototype"] = positive_override[3]
        retrieval["positive_scores"] = [positive_override[4]]
    if negative_override is not None:
        retrieval["negative_features"] = negative_override[0]
        retrieval["negative_weights"] = negative_override[1]
        retrieval["negative_entries"] = [negative_override[2]]
        retrieval["negative_prototype"] = negative_override[3]
        retrieval["negative_scores"] = [negative_override[4]]
    return retrieval


def _empty_negative_like(retrieval: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    dim = int(retrieval["positive_prototype"].shape[-1])
    device = retrieval["positive_prototype"].device
    return (
        torch.zeros(1, 0, dim, device=device),
        torch.zeros(1, 0, device=device),
        [],
        torch.zeros(1, dim, device=device),
        torch.zeros(0, device=device),
    )


def _build_variant_retrievals(
    bank: RSSDABank,
    retriever: PrototypeRetriever,
    query_feature: torch.Tensor,
    query_source: str,
    top_k_positive: int,
    rng: random.Random,
    prefer_cross_domain_positive: bool,
    retrieval_mode: str,
) -> dict[str, Optional[dict[str, Any]]]:
    base_retrieval = _apply_retrieval_mode(
        retriever(
            query_feature,
            query_source_datasets=[query_source],
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        ),
        retrieval_mode,
    )
    query_vector = base_retrieval["projected_query"][0]
    positive_entries = bank.get_entries(polarity="positive", human_verified=True)
    negative_entries = bank.get_entries(polarity="negative", human_verified=True)
    wrong_positive = _selection_from_entries(
        bank,
        query_vector,
        positive_entries,
        _rank_entry_indices(bank, query_vector, positive_entries, top_k_positive, "worst", rng),
    )
    random_positive = _selection_from_entries(
        bank,
        query_vector,
        positive_entries,
        _rank_entry_indices(bank, query_vector, positive_entries, top_k_positive, "random", rng),
    )
    negative_as_positive = _selection_from_entries(
        bank,
        query_vector,
        negative_entries,
        _rank_entry_indices(bank, query_vector, negative_entries, top_k_positive, "best", rng),
    )
    return {
        "correct_positive": base_retrieval,
        "wrong_exemplar": _override_retrieval(base_retrieval, positive_override=wrong_positive),
        "negative_exemplar": _override_retrieval(
            base_retrieval,
            positive_override=negative_as_positive,
            negative_override=_empty_negative_like(base_retrieval),
        ),
        "random_exemplar": _override_retrieval(base_retrieval, positive_override=random_positive),
        "no_retrieval": None,
    }


def _safe_entropy(values: torch.Tensor) -> float:
    flat = values.float().flatten()
    if flat.numel() == 0:
        return 0.0
    flat = flat - flat.min()
    if float(flat.sum().item()) <= 1e-6:
        return 0.0
    probs = flat / flat.sum().clamp_min(1e-6)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum()
    max_entropy = math.log(float(probs.numel())) if probs.numel() > 1 else 1.0
    return float((entropy / max(max_entropy, 1e-6)).item())


def summarize_heatmap(tensor: Optional[torch.Tensor], gt_mask: torch.Tensor, top_percent: float) -> dict[str, float]:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return {
            "max": 0.0,
            "mean": 0.0,
            "entropy": 0.0,
            "top_percent_activation": 0.0,
            "hotspot_overlap_ratio": 0.0,
        }
    values = tensor.detach().float().squeeze()
    if values.dim() != 2:
        values = values.reshape(values.shape[-2], values.shape[-1])
    flat = values.flatten()
    top_count = max(1, int(flat.numel() * top_percent))
    top_values = torch.topk(flat, k=top_count).values
    threshold = float(top_values.min().item())
    hotspot = (values >= threshold).float()
    target = gt_mask.detach().float().squeeze()
    if target.shape != values.shape:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(0).unsqueeze(0),
            size=values.shape,
            mode="nearest",
        ).squeeze(0).squeeze(0)
    hotspot_area = hotspot.sum().clamp_min(1.0)
    overlap_ratio = float(((hotspot > 0.5) * (target > 0.5)).float().sum().item() / hotspot_area.item())
    return {
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "entropy": _safe_entropy(values),
        "top_percent_activation": float(top_values.mean().item()),
        "hotspot_overlap_ratio": overlap_ratio,
    }


def _variant_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "Dice change": float(current["metrics"]["Dice"] - baseline["metrics"]["Dice"]),
        "Mask area ratio change": float(current["mask_area_ratio"] - baseline["mask_area_ratio"]),
        "Boundary F1 change": float(current["metrics"]["Boundary F1"] - baseline["metrics"]["Boundary F1"]),
        "Confidence change": float(current["mean_confidence"] - baseline["mean_confidence"]),
    }


def _sensitivity_spread(variants: dict[str, dict[str, Any]]) -> dict[str, float]:
    keys = {
        "dice_range": [variant["metrics"]["Dice"] for variant in variants.values()],
        "mask_area_ratio_range": [variant["mask_area_ratio"] for variant in variants.values()],
        "boundary_f1_range": [variant["metrics"]["Boundary F1"] for variant in variants.values()],
        "confidence_range": [variant["mean_confidence"] for variant in variants.values()],
    }
    return {name: float(max(values) - min(values)) for name, values in keys.items()}


def _to_score_list(scores: list[torch.Tensor] | list[object]) -> list[float]:
    if not scores:
        return []
    first = scores[0]
    if isinstance(first, torch.Tensor):
        return [float(value) for value in first.detach().cpu().tolist()]
    return []


def _run_variant(
    adapter: RetrievalSpatialSemanticAdapter,
    wrapper: Sam3TensorForwardWrapper,
    similarity_builder: SimilarityHeatmapBuilder,
    images: torch.Tensor,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    baseline_outputs: dict[str, Any],
    retrieval: Optional[dict[str, Any]],
    retrieval_mode: str,
    top_percent: float,
) -> dict[str, Any]:
    if retrieval is None:
        outputs = baseline_outputs
        metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
        return {
            "metrics": metrics,
            "mask_area_ratio": _mask_area_ratio(outputs["mask_logits"]),
            "mean_confidence": _mean_confidence(outputs),
            "selected_positive_ids": [],
            "selected_positive_scores": [],
            "selected_negative_ids": [],
            "selected_negative_scores": [],
            "heatmap_stats": {},
            "retrieval_summary": {},
        }

    similarity = similarity_builder(
        query_feature,
        retrieval["positive_features"],
        retrieval["negative_features"],
        retrieval["positive_weights"],
        retrieval["negative_weights"],
    )
    _, retrieval_prior, _ = adapter(
        feature_map=query_feature,
        similarity_map=similarity["fused_similarity"],
        positive_prototype=retrieval["positive_prototype"],
        negative_prototype=retrieval["negative_prototype"],
        positive_tokens=retrieval["positive_features"],
        negative_tokens=retrieval["negative_features"],
        positive_similarity=similarity["positive_similarity"],
        negative_similarity=similarity["negative_similarity"],
        positive_weights=retrieval["positive_weights"],
        negative_weights=retrieval["negative_weights"],
        positive_scores=retrieval.get("positive_score_tensor"),
        negative_scores=retrieval.get("negative_score_tensor"),
        baseline_mask_logits=baseline_outputs.get("mask_logits"),
        positive_heatmap=similarity["positive_heatmap"],
        negative_heatmap=similarity["negative_heatmap"],
        mode=retrieval_mode,
    )
    outputs = wrapper(images=images, boxes=boxes, text_prompt=text_prompt, retrieval_prior=retrieval_prior)
    metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
    return {
        "metrics": metrics,
        "mask_area_ratio": _mask_area_ratio(outputs["mask_logits"]),
        "mean_confidence": _mean_confidence(outputs),
        "selected_positive_ids": [entry.prototype_id for entry in retrieval["positive_entries"][0]],
        "selected_positive_scores": _to_score_list(retrieval["positive_scores"]),
        "selected_negative_ids": [entry.prototype_id for entry in retrieval["negative_entries"][0]],
        "selected_negative_scores": _to_score_list(retrieval["negative_scores"]),
        "heatmap_stats": {
            "positive_heatmap": summarize_heatmap(similarity["positive_heatmap"][0, 0], masks[0, 0], top_percent),
            "negative_heatmap": summarize_heatmap(similarity["negative_heatmap"][0, 0], masks[0, 0], top_percent),
            "fused_similarity": summarize_heatmap(similarity["fused_similarity"][0, 0], masks[0, 0], top_percent),
            "spatial_bias_map": summarize_heatmap(retrieval_prior.get("spatial_bias_map", None), masks[0, 0], top_percent),
        },
        "retrieval_summary": outputs.get("intermediate_features", {}).get("retrieval_prior", {}),
    }


def _accumulate_variant(
    target: dict[str, dict[str, float]],
    variant_name: str,
    payload: dict[str, float],
) -> None:
    summary = target.setdefault(variant_name, {})
    for key, value in payload.items():
        summary[key] = summary.get(key, 0.0) + float(value)


def _accumulate_heatmaps(
    target: dict[str, dict[str, dict[str, float]]],
    variant_name: str,
    heatmaps: dict[str, dict[str, float]],
) -> None:
    variant_target = target.setdefault(variant_name, {})
    for heatmap_name, stats in heatmaps.items():
        heatmap_target = variant_target.setdefault(heatmap_name, {})
        for key, value in stats.items():
            heatmap_target[key] = heatmap_target.get(key, 0.0) + float(value)


def _average_nested(values: dict[str, Any], count: int) -> dict[str, Any]:
    averaged: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            averaged[key] = _average_nested(value, count)
        else:
            averaged[key] = float(value) / max(count, 1)
    return averaged


def _report_gap(internal: dict[str, Any], external: dict[str, Any]) -> dict[str, dict[str, float]]:
    gap: dict[str, dict[str, float]] = {}
    internal_variants = internal.get("variant_metrics", {})
    external_variants = external.get("variant_metrics", {})
    for variant_name, metrics in internal_variants.items():
        if variant_name not in external_variants:
            continue
        gap[variant_name] = {
            "internal_dice": float(metrics.get("Dice", 0.0)),
            "external_dice": float(external_variants[variant_name].get("Dice", 0.0)),
            "dice_gap": float(metrics.get("Dice", 0.0) - external_variants[variant_name].get("Dice", 0.0)),
            "internal_precision": float(metrics.get("Precision", 0.0)),
            "external_precision": float(external_variants[variant_name].get("Precision", 0.0)),
            "precision_gap": float(metrics.get("Precision", 0.0) - external_variants[variant_name].get("Precision", 0.0)),
            "internal_fpr": float(metrics.get("False Positive Rate", 0.0)),
            "external_fpr": float(external_variants[variant_name].get("False Positive Rate", 0.0)),
            "fpr_gap": float(metrics.get("False Positive Rate", 0.0) - external_variants[variant_name].get("False Positive Rate", 0.0)),
        }
    return gap


def _evaluate_split(
    split_name: str,
    records: list[dict[str, Any]],
    *,
    adapter: RetrievalSpatialSemanticAdapter,
    wrapper: Sam3TensorForwardWrapper,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
    bank: RSSDABank,
    image_size: int,
    retrieval_mode: str,
    top_k_positive: int,
    top_percent: float,
    prefer_cross_domain_positive: bool,
    seed: int,
    device: str,
    max_samples: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_samples is not None:
        records = records[: max(0, max_samples)]
    loader = DataLoader(SplitSegmentationDataset(records, image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)
    rows: list[dict[str, Any]] = []
    variant_metrics_sum: dict[str, dict[str, float]] = {}
    sensitivity_sum: dict[str, dict[str, float]] = {}
    heatmap_sum: dict[str, dict[str, dict[str, float]]] = {}
    spread_sum: dict[str, float] = {}
    rng = random.Random(seed)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            record = batch["records"][0]
            source_domain = infer_source_domain(
                dataset_name=str(record.get("dataset_name", "")),
                image_id=str(record.get("image_id", "")),
                image_path=str(record.get("image_path", "")),
                mask_path=str(record.get("mask_path", "")),
            )
            baseline = wrapper(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
            query_feature = resolve_feature_map(baseline["image_embeddings"], images)
            variant_retrievals = _build_variant_retrievals(
                bank,
                retriever,
                query_feature,
                query_source=source_domain,
                top_k_positive=top_k_positive,
                rng=random.Random(seed + batch_index),
                prefer_cross_domain_positive=prefer_cross_domain_positive,
                retrieval_mode=retrieval_mode,
            )

            variants: dict[str, dict[str, Any]] = {}
            for variant_name, retrieval in variant_retrievals.items():
                variants[variant_name] = _run_variant(
                    adapter,
                    wrapper,
                    similarity_builder,
                    images,
                    masks,
                    boxes,
                    batch["text_prompt"],
                    query_feature,
                    baseline,
                    retrieval,
                    retrieval_mode,
                    top_percent,
                )

            baseline_variant = variants["no_retrieval"]
            sensitivity = {
                variant_name: _variant_delta(variant_payload, baseline_variant)
                for variant_name, variant_payload in variants.items()
                if variant_name != "no_retrieval"
            }
            spread = _sensitivity_spread(variants)
            rows.append(
                {
                    "split": split_name,
                    "image_id": str(record.get("image_id", "")),
                    "source_domain": source_domain,
                    "variants": variants,
                    "sensitivity_vs_no_retrieval": sensitivity,
                    "sensitivity_spread": spread,
                }
            )

            for variant_name, variant_payload in variants.items():
                metrics_payload = dict(variant_payload["metrics"])
                metrics_payload["mask_area_ratio"] = float(variant_payload["mask_area_ratio"])
                metrics_payload["mean_confidence"] = float(variant_payload["mean_confidence"])
                _accumulate_variant(variant_metrics_sum, variant_name, metrics_payload)
                _accumulate_heatmaps(heatmap_sum, variant_name, variant_payload["heatmap_stats"])
            for variant_name, delta_payload in sensitivity.items():
                _accumulate_variant(sensitivity_sum, variant_name, delta_payload)
            for key, value in spread.items():
                spread_sum[key] = spread_sum.get(key, 0.0) + float(value)

    count = len(rows)
    split_summary = {
        "sample_count": count,
        "variant_metrics": _average_nested(variant_metrics_sum, count),
        "sensitivity_vs_no_retrieval": _average_nested(sensitivity_sum, count),
        "heatmap_stats": _average_nested(heatmap_sum, count),
        "sensitivity_spread": _average_nested(spread_sum, count),
    }
    return rows, split_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate numerical RSS-DA behavior reports.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--internal-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt")
    parser.add_argument("--external-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/rssda_behavior_report")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--retriever-checkpoint", default=None)
    parser.add_argument("--similarity-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--top-k-positive", type=int, default=3)
    parser.add_argument("--top-k-negative", type=int, default=3)
    parser.add_argument("--negative-lambda", type=float, default=0.35)
    parser.add_argument("--retrieval-mode", choices=["joint", "semantic", "spatial", "positive-only", "positive-negative"], default="joint")
    parser.add_argument("--prefer-cross-domain-positive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hotspot-top-percent", type=float, default=0.05)
    parser.add_argument("--dummy-samples-per-split", type=int, default=3)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    internal_records = _ensure_records(args.internal_split_file, args.dummy, "internal", "Kvasir", args.dummy_samples_per_split)
    external_records = _ensure_records(args.external_split_file, args.dummy, "external", "PolypGen", args.dummy_samples_per_split)
    if not internal_records:
        raise FileNotFoundError(f"No internal validation records found in {args.internal_split_file}")
    if not external_records:
        raise FileNotFoundError(f"No external validation records found in {args.external_split_file}")

    device = _resolve_runtime_device(args.device)
    base_model = build_official_sam3_image_model(
        args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
        allow_dummy_fallback=args.dummy,
    )
    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        apply_lora_to_model(base_model, LoRAConfig(stage=args.lora_stage, min_replaced_modules=0))
        load_lora_weights(base_model, args.lora_checkpoint, strict=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    hidden_dim = _resolve_hidden_dim(base_model)
    bank = _load_or_create_bank(
        args.memory_bank,
        hidden_dim=hidden_dim,
        dummy=args.dummy,
        seed=args.seed,
        image_size=args.image_size,
        precision=args.precision,
        checkpoint=args.checkpoint,
        device=device,
    )
    retriever = PrototypeRetriever(bank=bank, feature_dim=hidden_dim, top_k_positive=args.top_k_positive, top_k_negative=args.top_k_negative).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=args.negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(dim=hidden_dim).to(device)
    if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
        loaded_bundle = _maybe_load_rssda_bundle(Path(args.adapter_checkpoint), device, adapter, retriever, similarity_builder)
        if not loaded_bundle:
            adapter.load_state_dict(_load_checkpoint_payload(Path(args.adapter_checkpoint), device), strict=False)
    if args.retriever_checkpoint and Path(args.retriever_checkpoint).exists():
        retriever.load_state_dict(torch.load(args.retriever_checkpoint, map_location=device, weights_only=False), strict=False)
    if args.similarity_checkpoint and Path(args.similarity_checkpoint).exists():
        similarity_builder.load_state_dict(torch.load(args.similarity_checkpoint, map_location=device, weights_only=False), strict=False)
    adapter.eval()
    retriever.eval()
    similarity_builder.eval()

    internal_rows, internal_summary = _evaluate_split(
        "internal",
        internal_records,
        adapter=adapter,
        wrapper=wrapper,
        retriever=retriever,
        similarity_builder=similarity_builder,
        bank=bank,
        image_size=args.image_size,
        retrieval_mode=args.retrieval_mode,
        top_k_positive=args.top_k_positive,
        top_percent=args.hotspot_top_percent,
        prefer_cross_domain_positive=args.prefer_cross_domain_positive,
        seed=args.seed,
        device=device,
        max_samples=args.max_samples_per_split,
    )
    external_rows, external_summary = _evaluate_split(
        "external",
        external_records,
        adapter=adapter,
        wrapper=wrapper,
        retriever=retriever,
        similarity_builder=similarity_builder,
        bank=bank,
        image_size=args.image_size,
        retrieval_mode=args.retrieval_mode,
        top_k_positive=args.top_k_positive,
        top_percent=args.hotspot_top_percent,
        prefer_cross_domain_positive=args.prefer_cross_domain_positive,
        seed=args.seed + 1000,
        device=device,
        max_samples=args.max_samples_per_split,
    )

    report = {
        "config": vars(args),
        "bank_source_counts": _entry_source_counts(bank),
        "metadata_readiness": _metadata_readiness(bank),
        "internal": internal_summary,
        "external": external_summary,
        "gap_report": _report_gap(internal_summary, external_summary),
    }
    (output_dir / "per_image_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in internal_rows + external_rows),
        encoding="utf-8",
    )
    (output_dir / "summary_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())