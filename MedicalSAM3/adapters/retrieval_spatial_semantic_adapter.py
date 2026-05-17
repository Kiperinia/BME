"""Spatial-semantic retrieval adapter for RSS-DA."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from MedicalSAM3.models.prompt_adapter import GatedRetrievalFusion


class RetrievalSpatialSemanticAdapter(nn.Module):
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
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(2, dim // 4 if dim >= 16 else 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4 if dim >= 16 else 4, 1, kernel_size=1),
        )
        self.spatial_scale = nn.Parameter(torch.tensor(0.4))
        self.gated_fusion = GatedRetrievalFusion(
            dim=dim,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            similarity_threshold=similarity_threshold,
            confidence_scale=confidence_scale,
            similarity_weighting=similarity_weighting,
            similarity_temperature=similarity_temperature,
            retrieval_policy=retrieval_policy,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_scale=uncertainty_scale,
            policy_activation_threshold=policy_activation_threshold,
            residual_strength=residual_strength,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        similarity_map: torch.Tensor,
        positive_prototype: torch.Tensor,
        negative_prototype: Optional[torch.Tensor] = None,
        positive_tokens: Optional[torch.Tensor] = None,
        negative_tokens: Optional[torch.Tensor] = None,
        positive_similarity: Optional[torch.Tensor] = None,
        negative_similarity: Optional[torch.Tensor] = None,
        positive_weights: Optional[torch.Tensor] = None,
        negative_weights: Optional[torch.Tensor] = None,
        positive_scores: Optional[torch.Tensor] = None,
        negative_scores: Optional[torch.Tensor] = None,
        baseline_mask_logits: Optional[torch.Tensor] = None,
        positive_heatmap: Optional[torch.Tensor] = None,
        negative_heatmap: Optional[torch.Tensor] = None,
        mode: str = "joint",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        if similarity_map.dim() != 4:
            raise ValueError("similarity_map must have shape [B, 1, H, W]")
        if negative_prototype is None:
            negative_prototype = torch.zeros_like(positive_prototype)

        pos_heat = positive_heatmap if positive_heatmap is not None else similarity_map
        neg_heat = negative_heatmap if negative_heatmap is not None else torch.zeros_like(similarity_map)
        spatial_bias = torch.sigmoid(self.spatial_fusion(torch.cat([pos_heat, neg_heat], dim=1)))
        use_semantic = mode in {"joint", "semantic", "positive-only", "negative-only", "positive-negative"}
        use_spatial = mode in {"joint", "spatial", "positive-only", "negative-only", "positive-negative"}

        if positive_tokens is None:
            positive_tokens = positive_prototype.unsqueeze(1)
        if negative_tokens is None:
            negative_tokens = negative_prototype.unsqueeze(1)

        calibration = self.gated_fusion.build_calibration(
            batch_size=feature_map.shape[0],
            device=feature_map.device,
            dtype=feature_map.dtype,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
            positive_similarity=positive_similarity,
            negative_similarity=negative_similarity,
        )
        policy_state = self.gated_fusion.build_policy_state(
            feature_map=feature_map,
            calibration=calibration,
            baseline_mask_logits=baseline_mask_logits,
        )
        confidence_gate_map = policy_state["inference_gate_map"]
        pos_heat = pos_heat * calibration["positive_scale"].view(feature_map.shape[0], 1, 1, 1)
        neg_heat = neg_heat * calibration["negative_scale"].view(feature_map.shape[0], 1, 1, 1)

        spatial_bias = torch.sigmoid(self.spatial_fusion(torch.cat([pos_heat, neg_heat], dim=1))) * confidence_gate_map

        adapted_feature, retrieval_prior, fusion_aux = self.gated_fusion(
            feature_map=feature_map,
            positive_tokens=positive_tokens,
            negative_tokens=negative_tokens,
            positive_similarity=positive_similarity,
            negative_similarity=negative_similarity,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            baseline_mask_logits=baseline_mask_logits,
            spatial_prior=spatial_bias,
            calibration=calibration,
            policy_state=policy_state,
        )
        if use_spatial and not use_semantic:
            adapted_feature = feature_map + self.spatial_scale * spatial_bias * fusion_aux["query_feature"]
            retrieval_prior["decoder_feature_bias_map"] = adapted_feature - feature_map
            retrieval_prior["encoder_memory_bias"] = adapted_feature - feature_map
        elif not use_spatial and use_semantic:
            adapted_feature = feature_map + retrieval_prior["semantic_prototype_map"]
            retrieval_prior["decoder_feature_bias_map"] = adapted_feature - feature_map
            retrieval_prior["encoder_memory_bias"] = adapted_feature - feature_map
        elif not use_spatial and not use_semantic:
            adapted_feature = feature_map
            retrieval_prior["decoder_feature_bias_map"] = torch.zeros_like(feature_map)
            retrieval_prior["encoder_memory_bias"] = torch.zeros_like(feature_map)
            retrieval_prior["mask_logit_bias_map"] = torch.zeros(feature_map.shape[0], 1, feature_map.shape[2], feature_map.shape[3], device=feature_map.device, dtype=feature_map.dtype)

        if not use_spatial:
            retrieval_prior["spatial_bias_map"] = torch.zeros_like(spatial_bias)
        if not use_semantic:
            retrieval_prior["semantic_prototype"] = torch.zeros_like(retrieval_prior["semantic_prototype"])
            retrieval_prior["semantic_prototype_map"] = torch.zeros_like(retrieval_prior["semantic_prototype_map"])
            retrieval_prior["positive_context_map"] = torch.zeros_like(retrieval_prior["positive_context_map"])
            retrieval_prior["negative_context_map"] = torch.zeros_like(retrieval_prior["negative_context_map"])
            retrieval_prior["decoder_feature_bias_map"] = torch.zeros_like(retrieval_prior["decoder_feature_bias_map"])
            retrieval_prior["encoder_memory_bias"] = torch.zeros_like(retrieval_prior["encoder_memory_bias"])
            retrieval_prior["mask_logit_bias_map"] = torch.zeros_like(retrieval_prior["mask_logit_bias_map"])

        aux = {
            "spatial_bias": spatial_bias,
            "semantic_prototype": retrieval_prior["semantic_prototype"],
            "semantic_prototype_map": retrieval_prior["semantic_prototype_map"],
            "negative_prompt_mask_logits": neg_heat,
            "query_feature": fusion_aux["query_feature"],
            "positive_context_map": fusion_aux["positive_context_map"],
            "negative_context_map": fusion_aux["negative_context_map"],
            "fusion_gate_map": fusion_aux["fusion_gate_map"],
            "policy_gate_map": fusion_aux["policy_gate_map"],
            "retrieval_confidence_gate": fusion_aux["retrieval_confidence_gate"],
            "positive_confidence_gate": fusion_aux["positive_confidence_gate"],
            "negative_confidence_gate": fusion_aux["negative_confidence_gate"],
            "positive_similarity_score": fusion_aux["positive_similarity_score"],
            "negative_similarity_score": fusion_aux["negative_similarity_score"],
            "positive_calibrated_weight": fusion_aux["positive_calibrated_weight"],
            "negative_calibrated_weight": fusion_aux["negative_calibrated_weight"],
            "positive_similarity_weight": fusion_aux["positive_similarity_weight"],
            "negative_similarity_weight": fusion_aux["negative_similarity_weight"],
            "retrieval_similarity_weight": fusion_aux["retrieval_similarity_weight"],
            "similarity_gate_map": fusion_aux["similarity_gate_map"],
            "segmentation_confidence_map": fusion_aux["segmentation_confidence_map"],
            "segmentation_uncertainty_map": fusion_aux["segmentation_uncertainty_map"],
            "segmentation_entropy_map": fusion_aux["segmentation_entropy_map"],
            "boundary_uncertainty_map": fusion_aux["boundary_uncertainty_map"],
            "low_confidence_lesion_map": fusion_aux["low_confidence_lesion_map"],
            "retrieval_region_mask": fusion_aux["retrieval_region_mask"],
            "high_confidence_preserve_mask": fusion_aux["high_confidence_preserve_mask"],
            "uncertainty_gate_map": fusion_aux["uncertainty_gate_map"],
            "inference_gate_map": fusion_aux["inference_gate_map"],
            "segmentation_confidence": fusion_aux["segmentation_confidence"],
            "segmentation_uncertainty": fusion_aux["segmentation_uncertainty"],
            "segmentation_entropy": fusion_aux["segmentation_entropy"],
            "uncertainty_gate": fusion_aux["uncertainty_gate"],
            "retrieval_activation_ratio": fusion_aux["retrieval_activation_ratio"],
            "retrieval_suppression_ratio": fusion_aux["retrieval_suppression_ratio"],
            "similarity_activation_ratio": fusion_aux["similarity_activation_ratio"],
            "similarity_temperature": fusion_aux["similarity_temperature"],
            "residual_strength": fusion_aux["residual_strength"],
            "high_confidence_region_modification_ratio": fusion_aux["high_confidence_region_modification_ratio"],
            "region_type_statistics": fusion_aux["region_type_statistics"],
            "used_baseline_uncertainty": fusion_aux["used_baseline_uncertainty"],
            "fused_delta": fusion_aux["fused_delta"],
            "fusion_alpha": fusion_aux["alpha"],
            "negative_lambda": fusion_aux["negative_lambda"],
            "mask_logit_scale": fusion_aux["mask_logit_scale"],
            "positive_token_response": fusion_aux["positive_token_response"],
            "negative_token_response": fusion_aux["negative_token_response"],
            "mode": mode,
        }
        return adapted_feature, retrieval_prior, aux