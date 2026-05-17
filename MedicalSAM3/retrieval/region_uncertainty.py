"""Region-level uncertainty maps for localized retrieval correction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def entropy_from_logits(mask_logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(mask_logits)
    safe_probability = probability.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(
        safe_probability * safe_probability.log()
        + (1.0 - safe_probability) * (1.0 - safe_probability).log()
    ) / math.log(2.0)
    return entropy


def confidence_from_logits(mask_logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(mask_logits)
    return (probability - 0.5).abs() * 2.0


def boundary_uncertainty_from_logits(mask_logits: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    probability = torch.sigmoid(mask_logits)
    pooled = F.avg_pool2d(probability, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    boundary_residual = (probability - pooled).abs()
    return torch.clamp(0.5 * boundary_residual + 0.5 * entropy_from_logits(mask_logits), 0.0, 1.0)


def low_confidence_lesion_from_logits(mask_logits: torch.Tensor, confidence_threshold: float = 0.85) -> torch.Tensor:
    probability = torch.sigmoid(mask_logits)
    confidence = confidence_from_logits(mask_logits)
    lesion_region = (probability >= 0.5).to(dtype=mask_logits.dtype)
    low_confidence = (confidence < confidence_threshold).to(dtype=mask_logits.dtype)
    return lesion_region * low_confidence * (1.0 - confidence)


def build_region_uncertainty_maps(mask_logits: torch.Tensor, confidence_threshold: float = 0.85) -> dict[str, torch.Tensor]:
    confidence_map = confidence_from_logits(mask_logits)
    entropy_map = entropy_from_logits(mask_logits)
    boundary_uncertainty_map = boundary_uncertainty_from_logits(mask_logits)
    low_confidence_lesion_map = low_confidence_lesion_from_logits(mask_logits, confidence_threshold=confidence_threshold)
    return {
        "probability_map": torch.sigmoid(mask_logits),
        "confidence_map": confidence_map,
        "uncertainty_map": 1.0 - confidence_map,
        "entropy_map": entropy_map,
        "boundary_uncertainty_map": boundary_uncertainty_map,
        "low_confidence_lesion_map": low_confidence_lesion_map,
        "low_confidence_region_map": low_confidence_lesion_map,
    }


__all__ = [
    "build_region_uncertainty_maps",
    "boundary_uncertainty_from_logits",
    "confidence_from_logits",
    "entropy_from_logits",
    "low_confidence_lesion_from_logits",
]