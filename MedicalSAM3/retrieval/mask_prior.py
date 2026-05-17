"""Retrieved mask prior aggregation for localized retrieval guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


def _load_mask(mask_path: str | None, spatial_size: tuple[int, int]) -> torch.Tensor | None:
    if not mask_path:
        return None
    target = Path(mask_path)
    if not target.exists():
        return None
    image = Image.open(target).convert("L").resize((spatial_size[1], spatial_size[0]), resample=Image.NEAREST)
    array = np.asarray(image).astype("float32")
    threshold = 0.0 if array.max() <= 1.0 else 127.0
    return torch.from_numpy((array > threshold).astype("float32")).unsqueeze(0)


def _weighted_mask_prior(entries: list[list[Any]], weights: torch.Tensor | None, spatial_size: tuple[int, int]) -> torch.Tensor | None:
    if weights is None or not isinstance(weights, torch.Tensor) or not entries:
        return None
    priors = []
    for batch_index, batch_entries in enumerate(entries):
        masks = []
        mask_weights = []
        for entry_index, entry in enumerate(batch_entries):
            if batch_index >= weights.shape[0] or entry_index >= weights.shape[1]:
                continue
            weight = float(weights[batch_index, entry_index].detach().cpu().item())
            if weight <= 0.0:
                continue
            mask = _load_mask(getattr(entry, "mask_path", None), spatial_size)
            if mask is None:
                continue
            masks.append(mask)
            mask_weights.append(weight)
        if not masks:
            priors.append(torch.zeros(1, spatial_size[0], spatial_size[1], dtype=torch.float32))
            continue
        stacked = torch.stack(masks, dim=0)
        weight_tensor = torch.tensor(mask_weights, dtype=torch.float32).view(-1, 1, 1, 1)
        prior = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6)
        priors.append(prior)
    return torch.stack(priors, dim=0)


def attach_retrieved_mask_priors(retrieval: dict[str, Any], spatial_size: tuple[int, int]) -> dict[str, Any]:
    updated = dict(retrieval)
    positive_prior = _weighted_mask_prior(
        retrieval.get("positive_entries", []),
        retrieval.get("positive_weights"),
        spatial_size,
    )
    negative_prior = _weighted_mask_prior(
        retrieval.get("negative_entries", []),
        retrieval.get("negative_weights"),
        spatial_size,
    )
    if positive_prior is not None:
        updated["positive_mask_prior"] = positive_prior.to(device=retrieval["positive_features"].device, dtype=retrieval["positive_features"].dtype)
    if negative_prior is not None:
        updated["negative_mask_prior"] = negative_prior.to(device=retrieval["negative_features"].device, dtype=retrieval["negative_features"].dtype)
    updated["mask_prior_available"] = bool(positive_prior is not None or negative_prior is not None)
    return updated


__all__ = ["attach_retrieved_mask_priors"]