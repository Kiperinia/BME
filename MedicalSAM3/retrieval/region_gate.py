"""Region-aware gating for localized retrieval correction."""

from __future__ import annotations

import torch


def build_retrieval_region_mask(
    *,
    probability_map: torch.Tensor,
    confidence_map: torch.Tensor,
    entropy_map: torch.Tensor,
    boundary_uncertainty_map: torch.Tensor,
    low_confidence_lesion_map: torch.Tensor,
    high_confidence_threshold: float = 0.85,
) -> dict[str, torch.Tensor]:
    dtype = probability_map.dtype
    uncertain_focus = torch.maximum(entropy_map, boundary_uncertainty_map)
    lesion_focus = torch.maximum(uncertain_focus, low_confidence_lesion_map)
    high_confidence_mask = (confidence_map >= high_confidence_threshold).to(dtype=dtype)
    high_confidence_foreground = ((probability_map >= 0.5).to(dtype=dtype) * high_confidence_mask)
    high_confidence_background = ((probability_map < 0.5).to(dtype=dtype) * high_confidence_mask)
    high_confidence_preserve_mask = torch.clamp(high_confidence_foreground + high_confidence_background, 0.0, 1.0)
    retrieval_region_mask = torch.clamp(lesion_focus * (1.0 - high_confidence_preserve_mask), 0.0, 1.0)
    activation_ratio = retrieval_region_mask.flatten(1).mean(dim=1)
    region_type_statistics = {
        "boundary": boundary_uncertainty_map.flatten(1).mean(dim=1),
        "low_confidence_lesion": low_confidence_lesion_map.flatten(1).mean(dim=1),
        "high_confidence_foreground": high_confidence_foreground.flatten(1).mean(dim=1),
        "high_confidence_background": high_confidence_background.flatten(1).mean(dim=1),
    }
    return {
        "retrieval_region_mask": retrieval_region_mask,
        "high_confidence_preserve_mask": high_confidence_preserve_mask,
        "high_confidence_foreground_mask": high_confidence_foreground,
        "high_confidence_background_mask": high_confidence_background,
        "activation_ratio": activation_ratio,
        "region_type_statistics": region_type_statistics,
    }


__all__ = ["build_retrieval_region_mask"]