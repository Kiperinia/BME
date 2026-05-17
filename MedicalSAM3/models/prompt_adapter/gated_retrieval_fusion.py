"""Lightweight gated retrieval fusion for retrieval-conditioned segmentation."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from MedicalSAM3.retrieval.region_gate import build_retrieval_region_mask
from MedicalSAM3.retrieval.region_uncertainty import build_region_uncertainty_maps


def _ensure_token_weights(tokens: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    batch_size, token_count = tokens.shape[:2]
    if token_count == 0:
        return tokens.new_zeros(batch_size, 0)
    if weights is None or weights.numel() == 0:
        return tokens.new_full((batch_size, token_count), 1.0 / float(token_count))
    return weights.to(tokens.device)


def _align_token_tensor(
    values: Optional[torch.Tensor],
    *,
    batch_size: int,
    token_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if values is None or values.numel() == 0 or token_count == 0:
        return torch.zeros(batch_size, token_count, device=device, dtype=dtype)
    tensor = values.to(device=device, dtype=dtype)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    output = torch.zeros(batch_size, token_count, device=device, dtype=dtype)
    batch_limit = min(batch_size, tensor.shape[0])
    token_limit = min(token_count, tensor.shape[1])
    output[:batch_limit, :token_limit] = tensor[:batch_limit, :token_limit]
    return output


class GatedRetrievalFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        positive_weight: float = 1.0,
        negative_weight: float = 0.25,
        similarity_threshold: float = 0.5,
        confidence_scale: float = 8.0,
        similarity_weighting: str = "hard",
        similarity_temperature: float | None = None,
        retrieval_policy: str = "uncertainty-aware",
        uncertainty_threshold: float = 0.35,
        uncertainty_scale: float = 10.0,
        policy_activation_threshold: float = 0.05,
        residual_strength: float = 0.5,
    ) -> None:
        super().__init__()
        if retrieval_policy not in {"always-on", "similarity-threshold", "uncertainty-aware", "region-aware", "residual"}:
            raise ValueError(f"Unsupported retrieval policy: {retrieval_policy}")
        if similarity_weighting not in {"hard", "soft"}:
            raise ValueError(f"Unsupported similarity weighting: {similarity_weighting}")
        hidden_dim = max(dim // 2, 16)
        self.query_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.prototype_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 3, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.35))
        self.negative_lambda = nn.Parameter(torch.tensor(0.35))
        self.mask_logit_scale = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))
        self.register_buffer("negative_weight", torch.tensor(float(negative_weight)))
        self.register_buffer("similarity_threshold", torch.tensor(float(similarity_threshold)))
        self.register_buffer("confidence_scale", torch.tensor(float(confidence_scale)))
        default_similarity_temperature = 1.0 / max(float(confidence_scale), 1e-3)
        self.register_buffer(
            "similarity_temperature",
            torch.tensor(float(default_similarity_temperature if similarity_temperature is None else similarity_temperature)),
        )
        self.register_buffer("uncertainty_threshold", torch.tensor(float(uncertainty_threshold)))
        self.register_buffer("uncertainty_scale", torch.tensor(float(uncertainty_scale)))
        self.register_buffer("policy_activation_threshold", torch.tensor(float(policy_activation_threshold)))
        self.register_buffer("residual_strength", torch.tensor(float(residual_strength)))
        self.register_buffer("high_confidence_threshold", torch.tensor(0.85))
        self.similarity_weighting = similarity_weighting
        self.retrieval_policy = retrieval_policy

    def _context_map(
        self,
        prototype_tokens: Optional[torch.Tensor],
        similarity_tokens: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        spatial_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = 1
        if prototype_tokens is not None:
            batch_size = int(prototype_tokens.shape[0])
        elif similarity_tokens is not None:
            batch_size = int(similarity_tokens.shape[0])
        height, width = spatial_shape
        if prototype_tokens is None or prototype_tokens.numel() == 0:
            empty_context = torch.zeros(batch_size, 0, height, width, device=device, dtype=dtype)
            empty_weight = torch.zeros(batch_size, 0, device=device, dtype=dtype)
            empty_summary = torch.zeros(batch_size, 0, device=device, dtype=dtype)
            return empty_context, empty_weight, empty_summary

        projected_tokens = self.prototype_proj(prototype_tokens)
        normalized_weights = _ensure_token_weights(projected_tokens, weights).to(dtype=projected_tokens.dtype)
        if similarity_tokens is None or similarity_tokens.numel() == 0:
            similarity_tokens = torch.zeros(
                projected_tokens.shape[0],
                projected_tokens.shape[1],
                height,
                width,
                device=device,
                dtype=dtype,
            )
        token_attention = torch.sigmoid(similarity_tokens.to(dtype=projected_tokens.dtype))
        weighted_attention = token_attention * normalized_weights.unsqueeze(-1).unsqueeze(-1)
        context = torch.einsum("bkc,bkhw->bchw", projected_tokens, weighted_attention)
        return context, normalized_weights, weighted_attention.flatten(2).mean(dim=-1)

    def _summarize_similarity(
        self,
        scores: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        similarity_tokens: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if scores is not None and scores.numel() > 0:
            score_tensor = scores.to(device=device, dtype=dtype)
            if score_tensor.dim() == 1:
                score_tensor = score_tensor.unsqueeze(0)
        elif similarity_tokens is not None and similarity_tokens.numel() > 0:
            score_tensor = similarity_tokens.to(device=device, dtype=dtype).flatten(2).mean(dim=-1)
        else:
            return torch.zeros(batch_size, device=device, dtype=dtype)

        aligned_weights = _align_token_tensor(
            weights,
            batch_size=batch_size,
            token_count=score_tensor.shape[1],
            device=device,
            dtype=dtype,
        )
        if aligned_weights.numel() > 0 and torch.any(aligned_weights > 0):
            denom = aligned_weights.sum(dim=1).clamp_min(1e-6)
            return (score_tensor * aligned_weights).sum(dim=1) / denom

        valid_mask = (score_tensor.abs() > 0).to(dtype=dtype)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return (score_tensor * valid_mask).sum(dim=1) / denom

    def build_calibration(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        positive_scores: Optional[torch.Tensor],
        negative_scores: Optional[torch.Tensor],
        positive_weights: Optional[torch.Tensor],
        negative_weights: Optional[torch.Tensor],
        positive_similarity: Optional[torch.Tensor],
        negative_similarity: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        positive_score = self._summarize_similarity(
            positive_scores,
            positive_weights,
            positive_similarity,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        negative_score = self._summarize_similarity(
            negative_scores,
            negative_weights,
            negative_similarity,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        similarity_threshold = self.similarity_threshold.to(device=device, dtype=dtype)
        confidence_scale = self.confidence_scale.to(device=device, dtype=dtype).clamp_min(1e-3)
        similarity_temperature = self.similarity_temperature.to(device=device, dtype=dtype).clamp_min(1e-3)
        if self.retrieval_policy == "always-on":
            ones = torch.ones(batch_size, device=device, dtype=dtype)
            return {
                "positive_score": positive_score,
                "negative_score": negative_score,
                "positive_confidence": ones,
                "negative_confidence": ones,
                "positive_scale": self.positive_weight.to(device=device, dtype=dtype).expand(batch_size),
                "negative_scale": self.negative_weight.to(device=device, dtype=dtype).expand(batch_size),
                "retrieval_confidence": ones,
                "positive_active": ones,
                "negative_active": ones,
                "retrieval_active": ones,
                "positive_similarity_weight": ones,
                "negative_similarity_weight": ones,
                "retrieval_similarity_weight": ones,
                "similarity_temperature": similarity_temperature.expand(batch_size),
            }
        positive_confidence = torch.sigmoid((positive_score - similarity_threshold) * confidence_scale)
        negative_confidence = torch.sigmoid((negative_score - similarity_threshold) * confidence_scale)
        positive_similarity_weight = torch.sigmoid((positive_score - similarity_threshold) / similarity_temperature)
        negative_similarity_weight = torch.sigmoid((negative_score - similarity_threshold) / similarity_temperature)
        if self.similarity_weighting == "hard":
            positive_active = (positive_score >= similarity_threshold).to(dtype=dtype)
            negative_active = (negative_score >= similarity_threshold).to(dtype=dtype)
        else:
            positive_active = positive_similarity_weight
            negative_active = negative_similarity_weight
        positive_scale = self.positive_weight.to(device=device, dtype=dtype) * positive_active
        negative_scale = self.negative_weight.to(device=device, dtype=dtype) * negative_active
        retrieval_confidence = torch.maximum(positive_confidence, negative_confidence)
        retrieval_active = torch.maximum(positive_active, negative_active)
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "positive_confidence": positive_confidence,
            "negative_confidence": negative_confidence,
            "positive_scale": positive_scale,
            "negative_scale": negative_scale,
            "retrieval_confidence": retrieval_confidence,
            "positive_active": positive_active,
            "negative_active": negative_active,
            "retrieval_active": retrieval_active,
            "positive_similarity_weight": positive_similarity_weight,
            "negative_similarity_weight": negative_similarity_weight,
            "retrieval_similarity_weight": torch.maximum(positive_similarity_weight, negative_similarity_weight),
            "similarity_temperature": similarity_temperature.expand(batch_size),
        }

    def build_policy_state(
        self,
        *,
        feature_map: torch.Tensor,
        calibration: dict[str, torch.Tensor],
        baseline_mask_logits: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = feature_map.shape
        device = feature_map.device
        dtype = feature_map.dtype
        retrieval_active = calibration.get("retrieval_active", torch.ones(batch_size, device=device, dtype=dtype)).to(device=device, dtype=dtype)
        retrieval_similarity_weight = calibration.get("retrieval_similarity_weight", retrieval_active).to(device=device, dtype=dtype)

        if baseline_mask_logits is None or baseline_mask_logits.numel() == 0:
            confidence_map = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
            uncertainty_map = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
            entropy_map = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
            boundary_uncertainty_map = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
            low_confidence_lesion_map = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
            retrieval_region_mask = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
            high_confidence_preserve_mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
            region_activation_ratio = retrieval_region_mask.flatten(1).mean(dim=1)
            region_type_statistics = {
                "boundary": torch.zeros(batch_size, device=device, dtype=dtype),
                "low_confidence_lesion": torch.zeros(batch_size, device=device, dtype=dtype),
                "high_confidence_foreground": torch.zeros(batch_size, device=device, dtype=dtype),
                "high_confidence_background": torch.zeros(batch_size, device=device, dtype=dtype),
            }
            uncertainty_signal = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
            used_baseline_uncertainty = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            resized_logits = F.interpolate(
                baseline_mask_logits.to(device=device, dtype=dtype),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            region_maps = build_region_uncertainty_maps(
                resized_logits,
                confidence_threshold=float(self.high_confidence_threshold.item()),
            )
            probability = region_maps["probability_map"]
            confidence_map = region_maps["confidence_map"]
            uncertainty_map = region_maps["uncertainty_map"]
            entropy_map = region_maps["entropy_map"]
            boundary_uncertainty_map = region_maps["boundary_uncertainty_map"]
            low_confidence_lesion_map = region_maps["low_confidence_lesion_map"]
            gate_maps = build_retrieval_region_mask(
                probability_map=probability,
                confidence_map=confidence_map,
                entropy_map=entropy_map,
                boundary_uncertainty_map=boundary_uncertainty_map,
                low_confidence_lesion_map=low_confidence_lesion_map,
                high_confidence_threshold=float(self.high_confidence_threshold.item()),
            )
            retrieval_region_mask = gate_maps["retrieval_region_mask"]
            high_confidence_preserve_mask = gate_maps["high_confidence_preserve_mask"]
            region_activation_ratio = gate_maps["activation_ratio"]
            region_type_statistics = gate_maps["region_type_statistics"]
            uncertainty_signal = 0.5 * (uncertainty_map + entropy_map)
            used_baseline_uncertainty = torch.ones(batch_size, device=device, dtype=dtype)

        if self.retrieval_policy in {"always-on", "similarity-threshold"}:
            uncertainty_gate_map = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        else:
            uncertainty_gate_map = torch.sigmoid(
                (uncertainty_signal - self.uncertainty_threshold.to(device=device, dtype=dtype))
                * self.uncertainty_scale.to(device=device, dtype=dtype).clamp_min(1e-3)
            )

        similarity_gate_map = retrieval_similarity_weight.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
        if self.retrieval_policy == "always-on":
            inference_gate_map = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
            policy_gate_map = inference_gate_map
        elif self.retrieval_policy == "similarity-threshold":
            inference_gate_map = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
            policy_gate_map = similarity_gate_map
        elif self.retrieval_policy in {"region-aware", "residual"}:
            inference_gate_map = retrieval_region_mask
            policy_gate_map = similarity_gate_map * retrieval_region_mask
        else:
            inference_gate_map = uncertainty_gate_map
            policy_gate_map = similarity_gate_map * uncertainty_gate_map

        policy_strength = policy_gate_map.flatten(1).mean(dim=1)
        activation_threshold = self.policy_activation_threshold.to(device=device, dtype=dtype)
        activation_ratio = (policy_gate_map > activation_threshold).to(dtype=dtype).flatten(1).mean(dim=1)
        suppression_ratio = 1.0 - activation_ratio
        similarity_activation_ratio = (similarity_gate_map > activation_threshold).to(dtype=dtype).flatten(1).mean(dim=1)
        residual_scale = self.residual_strength.to(device=device, dtype=dtype).expand(batch_size) if self.retrieval_policy == "residual" else torch.ones(batch_size, device=device, dtype=dtype)

        return {
            "segmentation_confidence_map": confidence_map,
            "segmentation_uncertainty_map": uncertainty_map,
            "segmentation_entropy_map": entropy_map,
            "boundary_uncertainty_map": boundary_uncertainty_map,
            "low_confidence_lesion_map": low_confidence_lesion_map,
            "retrieval_region_mask": retrieval_region_mask,
            "high_confidence_preserve_mask": high_confidence_preserve_mask,
            "similarity_gate_map": similarity_gate_map,
            "uncertainty_gate_map": uncertainty_gate_map,
            "inference_gate_map": inference_gate_map,
            "policy_gate_map": policy_gate_map,
            "segmentation_confidence": confidence_map.flatten(1).mean(dim=1),
            "segmentation_uncertainty": uncertainty_map.flatten(1).mean(dim=1),
            "segmentation_entropy": entropy_map.flatten(1).mean(dim=1),
            "positive_similarity_weight": calibration.get("positive_similarity_weight", retrieval_active),
            "negative_similarity_weight": calibration.get("negative_similarity_weight", retrieval_active),
            "retrieval_similarity_weight": retrieval_similarity_weight,
            "uncertainty_gate": uncertainty_gate_map.flatten(1).mean(dim=1),
            "policy_strength": policy_strength,
            "retrieval_activation_ratio": torch.maximum(activation_ratio, region_activation_ratio)
            if self.retrieval_policy in {"region-aware", "residual"}
            else activation_ratio,
            "retrieval_suppression_ratio": suppression_ratio,
            "similarity_activation_ratio": similarity_activation_ratio,
            "residual_strength": residual_scale,
            "used_baseline_uncertainty": used_baseline_uncertainty,
            "region_type_statistics": region_type_statistics,
        }

    def forward(
        self,
        feature_map: torch.Tensor,
        positive_tokens: torch.Tensor,
        negative_tokens: Optional[torch.Tensor] = None,
        positive_similarity: Optional[torch.Tensor] = None,
        negative_similarity: Optional[torch.Tensor] = None,
        positive_weights: Optional[torch.Tensor] = None,
        negative_weights: Optional[torch.Tensor] = None,
        spatial_prior: Optional[torch.Tensor] = None,
        positive_scores: Optional[torch.Tensor] = None,
        negative_scores: Optional[torch.Tensor] = None,
        baseline_mask_logits: Optional[torch.Tensor] = None,
        calibration: Optional[dict[str, torch.Tensor]] = None,
        policy_state: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        batch_size, _, height, width = feature_map.shape
        query_feature = self.query_proj(feature_map)
        positive_context, positive_weights_norm, positive_token_response = self._context_map(
            positive_tokens,
            positive_similarity,
            positive_weights,
            spatial_shape=(height, width),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        negative_context, negative_weights_norm, negative_token_response = self._context_map(
            negative_tokens,
            negative_similarity,
            negative_weights,
            spatial_shape=(height, width),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        if positive_context.shape[1] == 0:
            positive_context = torch.zeros_like(query_feature)
        if negative_context.shape[1] == 0:
            negative_context = torch.zeros_like(query_feature)

        if spatial_prior is None:
            spatial_prior = torch.ones(batch_size, 1, height, width, device=feature_map.device, dtype=feature_map.dtype)
        else:
            spatial_prior = spatial_prior.to(feature_map.device, dtype=feature_map.dtype)

        if calibration is None:
            calibration = self.build_calibration(
                batch_size=batch_size,
                device=feature_map.device,
                dtype=feature_map.dtype,
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                positive_similarity=positive_similarity,
                negative_similarity=negative_similarity,
            )

        if policy_state is None:
            policy_state = self.build_policy_state(
                feature_map=feature_map,
                calibration=calibration,
                baseline_mask_logits=baseline_mask_logits,
            )

        negative_scale = self.negative_lambda.clamp(min=0.0, max=2.0)
        positive_scale_map = calibration["positive_scale"].view(batch_size, 1, 1, 1)
        negative_scale_map = calibration["negative_scale"].view(batch_size, 1, 1, 1)
        policy_gate_map = policy_state["policy_gate_map"]
        inference_gate_map = policy_state["inference_gate_map"]
        positive_context = positive_context * spatial_prior * positive_scale_map
        negative_context = negative_context * (1.0 - spatial_prior) * negative_scale_map
        combined_context = positive_context - negative_scale * negative_context
        gate_input = torch.cat([query_feature, positive_context, negative_context], dim=1)
        base_gate = self.gate(gate_input)
        gate = base_gate * inference_gate_map
        residual = self.delta_proj(torch.cat([query_feature, combined_context, query_feature * spatial_prior], dim=1))
        fused_delta = gate * residual
        retrieval_region_mask = policy_state.get("retrieval_region_mask", inference_gate_map)
        high_confidence_preserve_mask = policy_state.get(
            "high_confidence_preserve_mask",
            torch.zeros_like(retrieval_region_mask),
        )
        localized_delta = fused_delta * retrieval_region_mask * (1.0 - high_confidence_preserve_mask)
        high_confidence_modification = (localized_delta.abs().mean(dim=1, keepdim=True) > 1e-6).to(dtype=feature_map.dtype)
        protected_denom = high_confidence_preserve_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        high_confidence_region_modification_ratio = (
            (high_confidence_modification * high_confidence_preserve_mask).sum(dim=(1, 2, 3)) / protected_denom
        )
        alpha = self.alpha.tanh()
        if self.retrieval_policy == "residual":
            effective_alpha = alpha * policy_state["residual_strength"] * policy_state["policy_strength"]
        else:
            effective_alpha = alpha * policy_state["residual_strength"]
        mask_logit_scale = self.mask_logit_scale.clamp(min=0.0, max=8.0)
        adapted_feature = feature_map + effective_alpha.view(batch_size, 1, 1, 1) * localized_delta

        positive_prototype = torch.einsum("bkc,bk->bc", self.prototype_proj(positive_tokens), positive_weights_norm) if positive_tokens.numel() > 0 else torch.zeros(batch_size, feature_map.shape[1], device=feature_map.device, dtype=feature_map.dtype)
        negative_prototype = torch.einsum("bkc,bk->bc", self.prototype_proj(negative_tokens), negative_weights_norm) if negative_tokens is not None and negative_tokens.numel() > 0 else torch.zeros_like(positive_prototype)
        calibrated_positive_prototype = positive_prototype * calibration["positive_scale"].unsqueeze(-1)
        calibrated_negative_prototype = negative_prototype * calibration["negative_scale"].unsqueeze(-1)

        prior = {
            "semantic_prototype": calibrated_positive_prototype - negative_scale * calibrated_negative_prototype,
            "semantic_prototype_map": combined_context,
            "spatial_bias_map": spatial_prior,
            "decoder_feature_bias_map": effective_alpha.view(batch_size, 1, 1, 1) * localized_delta,
            "mask_logit_bias_map": mask_logit_scale * effective_alpha.view(batch_size, 1, 1, 1) * localized_delta.mean(dim=1, keepdim=True),
            "positive_context_map": positive_context,
            "negative_context_map": negative_context,
            "fusion_gate_map": gate,
            "base_fusion_gate_map": base_gate,
            "fusion_alpha": effective_alpha,
            "negative_lambda": negative_scale.view(1),
            "mask_logit_scale": mask_logit_scale.view(1),
            "retrieval_confidence_gate": calibration["retrieval_confidence"].view(batch_size, 1, 1, 1),
            "positive_similarity_score": calibration["positive_score"],
            "negative_similarity_score": calibration["negative_score"],
            "positive_confidence_gate": calibration["positive_confidence"],
            "negative_confidence_gate": calibration["negative_confidence"],
            "positive_calibrated_weight": calibration["positive_scale"],
            "negative_calibrated_weight": calibration["negative_scale"],
            "segmentation_confidence_map": policy_state["segmentation_confidence_map"],
            "segmentation_uncertainty_map": policy_state["segmentation_uncertainty_map"],
            "segmentation_entropy_map": policy_state["segmentation_entropy_map"],
            "boundary_uncertainty_map": policy_state["boundary_uncertainty_map"],
            "low_confidence_lesion_map": policy_state["low_confidence_lesion_map"],
            "retrieval_region_mask": retrieval_region_mask,
            "high_confidence_preserve_mask": high_confidence_preserve_mask,
            "similarity_gate_map": policy_state["similarity_gate_map"],
            "uncertainty_gate_map": policy_state["uncertainty_gate_map"],
            "inference_gate_map": inference_gate_map,
            "policy_gate_map": policy_gate_map,
            "segmentation_confidence": policy_state["segmentation_confidence"],
            "segmentation_uncertainty": policy_state["segmentation_uncertainty"],
            "segmentation_entropy": policy_state["segmentation_entropy"],
            "positive_similarity_weight": policy_state["positive_similarity_weight"],
            "negative_similarity_weight": policy_state["negative_similarity_weight"],
            "retrieval_similarity_weight": policy_state["retrieval_similarity_weight"],
            "uncertainty_gate": policy_state["uncertainty_gate"],
            "retrieval_activation_ratio": policy_state["retrieval_activation_ratio"],
            "retrieval_suppression_ratio": policy_state["retrieval_suppression_ratio"],
            "similarity_activation_ratio": policy_state["similarity_activation_ratio"],
            "residual_strength": policy_state["residual_strength"],
            "high_confidence_region_modification_ratio": high_confidence_region_modification_ratio,
            "region_type_statistics": policy_state["region_type_statistics"],
            "similarity_temperature": calibration["similarity_temperature"],
            "encoder_memory_bias": effective_alpha.view(batch_size, 1, 1, 1) * localized_delta,
        }
        aux = {
            "query_feature": query_feature,
            "positive_context_map": positive_context,
            "negative_context_map": negative_context,
            "fusion_gate_map": gate,
            "base_gate_map": base_gate,
            "policy_gate_map": policy_gate_map.detach(),
            "fused_delta": localized_delta,
            "alpha": effective_alpha.detach(),
            "raw_alpha": alpha.detach(),
            "negative_lambda": negative_scale.detach(),
            "mask_logit_scale": mask_logit_scale.detach(),
            "positive_similarity_score": calibration["positive_score"].detach(),
            "negative_similarity_score": calibration["negative_score"].detach(),
            "positive_confidence_gate": calibration["positive_confidence"].detach(),
            "negative_confidence_gate": calibration["negative_confidence"].detach(),
            "retrieval_confidence_gate": calibration["retrieval_confidence"].detach(),
            "positive_calibrated_weight": calibration["positive_scale"].detach(),
            "negative_calibrated_weight": calibration["negative_scale"].detach(),
            "segmentation_confidence_map": policy_state["segmentation_confidence_map"].detach(),
            "segmentation_uncertainty_map": policy_state["segmentation_uncertainty_map"].detach(),
            "segmentation_entropy_map": policy_state["segmentation_entropy_map"].detach(),
            "boundary_uncertainty_map": policy_state["boundary_uncertainty_map"].detach(),
            "low_confidence_lesion_map": policy_state["low_confidence_lesion_map"].detach(),
            "retrieval_region_mask": retrieval_region_mask.detach(),
            "high_confidence_preserve_mask": high_confidence_preserve_mask.detach(),
            "similarity_gate_map": policy_state["similarity_gate_map"].detach(),
            "uncertainty_gate_map": policy_state["uncertainty_gate_map"].detach(),
            "inference_gate_map": inference_gate_map.detach(),
            "segmentation_confidence": policy_state["segmentation_confidence"].detach(),
            "segmentation_uncertainty": policy_state["segmentation_uncertainty"].detach(),
            "segmentation_entropy": policy_state["segmentation_entropy"].detach(),
            "positive_similarity_weight": policy_state["positive_similarity_weight"].detach(),
            "negative_similarity_weight": policy_state["negative_similarity_weight"].detach(),
            "retrieval_similarity_weight": policy_state["retrieval_similarity_weight"].detach(),
            "uncertainty_gate": policy_state["uncertainty_gate"].detach(),
            "retrieval_activation_ratio": policy_state["retrieval_activation_ratio"].detach(),
            "retrieval_suppression_ratio": policy_state["retrieval_suppression_ratio"].detach(),
            "similarity_activation_ratio": policy_state["similarity_activation_ratio"].detach(),
            "residual_strength": policy_state["residual_strength"].detach(),
            "high_confidence_region_modification_ratio": high_confidence_region_modification_ratio.detach(),
            "region_type_statistics": {key: value.detach() for key, value in policy_state["region_type_statistics"].items()},
            "used_baseline_uncertainty": policy_state["used_baseline_uncertainty"].detach(),
            "similarity_temperature": calibration["similarity_temperature"].detach(),
            "positive_token_response": positive_token_response.detach(),
            "negative_token_response": negative_token_response.detach(),
        }
        return adapted_feature, prior, aux