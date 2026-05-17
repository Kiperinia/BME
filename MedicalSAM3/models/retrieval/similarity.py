"""Spatial similarity maps and positive-negative fusion for RSS-DA."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity_map(feature_map: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    if prototypes.dim() == 2:
        prototypes = prototypes.unsqueeze(1)
    if prototypes.dim() != 3:
        raise ValueError("prototypes must have shape [B, K, C] or [B, C]")
    normalized_feature = F.normalize(feature_map, dim=1)
    normalized_proto = F.normalize(prototypes, dim=-1)
    similarity = torch.einsum("bchw,bkc->bkhw", normalized_feature, normalized_proto)
    return similarity


class SimilarityHeatmapBuilder(nn.Module):
    def __init__(self, lambda_negative: float = 0.35, temperature: float = 1.0) -> None:
        super().__init__()
        self.fusion_proj = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(max(temperature, 1e-6)))))
        with torch.no_grad():
            self.fusion_proj.weight.zero_()
            self.fusion_proj.bias.zero_()
            self.fusion_proj.weight[0, 0, 0, 0] = 1.0
            self.fusion_proj.weight[0, 1, 0, 0] = -float(lambda_negative)

    @staticmethod
    def _fuse(similarity: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        if similarity.numel() == 0:
            batch_size = similarity.shape[0]
            spatial_shape = similarity.shape[-2:]
            return torch.zeros(batch_size, 1, *spatial_shape, device=similarity.device)
        if weights is None or weights.numel() == 0:
            fused = similarity.mean(dim=1, keepdim=True)
        else:
            fused = (similarity * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        return fused

    def forward(
        self,
        feature_map: torch.Tensor,
        positive_prototypes: torch.Tensor,
        negative_prototypes: Optional[torch.Tensor] = None,
        positive_weights: Optional[torch.Tensor] = None,
        negative_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        positive_similarity = cosine_similarity_map(feature_map, positive_prototypes)
        positive_heatmap = self._fuse(positive_similarity, positive_weights)
        if negative_prototypes is None or negative_prototypes.numel() == 0:
            negative_similarity = torch.zeros_like(positive_similarity[:, :1])
            negative_heatmap = torch.zeros_like(positive_heatmap)
        else:
            negative_similarity = cosine_similarity_map(feature_map, negative_prototypes)
            negative_heatmap = self._fuse(negative_similarity, negative_weights)
        fused_similarity = self.fusion_proj(torch.cat([positive_heatmap, negative_heatmap], dim=1))
        spatial_prior = torch.sigmoid(fused_similarity / self.log_temperature.exp().clamp_min(1e-6))
        fusion_weight = self.fusion_proj.weight.detach().clone()
        return {
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "positive_heatmap": positive_heatmap,
            "negative_heatmap": negative_heatmap,
            "fused_similarity": fused_similarity,
            "spatial_prior": spatial_prior,
            "fusion_weight": fusion_weight,
            "temperature": self.log_temperature.detach().exp().clone(),
            "positive_fusion_weight": fusion_weight[:, :1].detach().clone(),
            "negative_fusion_weight": fusion_weight[:, 1:].detach().clone(),
        }