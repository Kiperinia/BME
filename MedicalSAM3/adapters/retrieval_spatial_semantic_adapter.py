"""MedEx-SAM3 的空间-语义检索适配器（RSS-DA）。

将检索得到的正/负原型与相似度先验，通过空间偏置与门控融合机制
注入到分割特征中，实现基于检索的域自适应增强。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from MedicalSAM3.models.prompt_adapter import GatedRetrievalFusion


class RetrievalSpatialSemanticAdapter(nn.Module):
    """空间-语义检索适配器，融合空间先验与语义原型增强分割特征。

    参数：
        - dim: 特征维度
        - positive_weight: 正例原型融合权重
        - negative_weight: 负例原型融合权重
        - similarity_threshold: 相似度激活阈值
        - confidence_scale: 置信度缩放系数
        - similarity_weighting: 相似度加权方式（hard/soft）
        - similarity_temperature: 软相似度温度系数
        - retrieval_policy: 检索策略名称
        - uncertainty_threshold: 不确定性激活阈值
        - uncertainty_scale: 不确定性缩放系数
        - policy_activation_threshold: 策略激活比例阈值
        - residual_strength: 残差融合强度
    """

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
        """初始化空间偏置卷积分支与门控检索融合模块。

        参数：
            - dim: 特征维度
            - positive_weight: 正例原型融合权重
            - negative_weight: 负例原型融合权重
            - similarity_threshold: 相似度激活阈值
            - confidence_scale: 置信度缩放系数
            - similarity_weighting: 相似度加权方式（hard/soft）
            - similarity_temperature: 软相似度温度系数
            - retrieval_policy: 检索策略名称
            - uncertainty_threshold: 不确定性激活阈值
            - uncertainty_scale: 不确定性缩放系数
            - policy_activation_threshold: 策略激活比例阈值
            - residual_strength: 残差融合强度

        返回：
            - 无返回值，完成空间偏置与门控融合模块的构建
        """
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
        """前向计算：按模式融合空间偏置与语义原型，输出增强特征与检索先验。

        参数：
            - feature_map: 形如 [B, C, H, W] 的输入特征图
            - similarity_map: 形如 [B, 1, H, W] 的相似度图
            - positive_prototype: 正例原型向量
            - negative_prototype: 负例原型向量，可选
            - positive_tokens: 正例提示 token，可选
            - negative_tokens: 负例提示 token，可选
            - positive_similarity: 正例相似度标量，可选
            - negative_similarity: 负例相似度标量，可选
            - positive_weights: 正例检索权重，可选
            - negative_weights: 负例检索权重，可选
            - positive_scores: 正例检索分数，可选
            - negative_scores: 负例检索分数，可选
            - baseline_mask_logits: 基线掩码 logits，用于不确定性估计，可选
            - positive_heatmap: 正例热力图，可选
            - negative_heatmap: 负例热力图，可选
            - mode: 融合模式，如 joint/spatial/semantic 等

        返回：
            - 三元组：(增强后的特征图, 检索先验字典, 辅助信息字典)
        """
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
