"""Exemplar encoder for positive, negative and boundary embeddings."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


def _masked_average_pool(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = feature_map * mask
    denom = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
    return masked.sum(dim=(-2, -1)) / denom


def _boundary_band(mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    return (dilated - eroded).clamp(0, 1)


class ExemplarEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def _encode_feature_map(self, crop_image: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            features = self.backbone(crop_image)
            if isinstance(features, dict):
                for value in features.values():
                    if isinstance(value, torch.Tensor) and value.dim() == 4:
                        return value
            if isinstance(features, torch.Tensor) and features.dim() == 4:
                return features
        return self.proj(self.stem(crop_image))

    def forward(
        self,
        crop_image: torch.Tensor,
        crop_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        features = self._encode_feature_map(crop_image)
        global_embedding = _l2_normalize(features.mean(dim=(-2, -1)))

        if crop_mask is not None:
            if crop_mask.shape[-2:] != features.shape[-2:]:
                crop_mask = F.interpolate(crop_mask.float(), size=features.shape[-2:], mode="nearest")
            foreground_mask = crop_mask.float().clamp(0, 1)
            background_mask = (1.0 - foreground_mask).clamp(0, 1)
            boundary_mask = _boundary_band(foreground_mask)
            foreground_embedding = _l2_normalize(_masked_average_pool(features, foreground_mask))
            boundary_embedding = _l2_normalize(_masked_average_pool(features, boundary_mask))
            context_embedding = _l2_normalize(_masked_average_pool(features, background_mask))
        else:
            foreground_embedding = global_embedding
            boundary_embedding = global_embedding
            context_embedding = global_embedding

        return {
            "global_embedding": global_embedding,
            "foreground_embedding": foreground_embedding,
            "boundary_embedding": boundary_embedding,
            "context_embedding": context_embedding,
        }
