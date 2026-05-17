"""Prototype extraction for RSS-DA memory banks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .bank import PrototypeBankEntry


def masked_average_pool(feature_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError("mask must have shape [B, 1, H, W]")
    resized_mask = F.interpolate(mask.float(), size=feature_map.shape[-2:], mode="nearest")
    weights = resized_mask.flatten(2)
    features = feature_map.flatten(2)
    denom = weights.sum(dim=-1).clamp_min(eps)
    pooled = (features * weights).sum(dim=-1) / denom
    return F.normalize(pooled, dim=1)


def _resolve_feature_map(feature: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    if feature.dim() == 4:
        return feature
    if feature.dim() == 3:
        batch_size, tokens, channels = feature.shape
        side = int(tokens ** 0.5)
        if side * side == tokens:
            return feature.transpose(1, 2).reshape(batch_size, channels, side, side)
        reduced = feature.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        return reduced.repeat(1, 1, max(images.shape[-2] // 16, 1), max(images.shape[-1] // 16, 1))
    raise ValueError("Unsupported feature tensor shape")


class PrototypeExtractor:
    def __init__(self, wrapper: Optional[torch.nn.Module] = None) -> None:
        self.wrapper = wrapper

    def extract_from_feature_map(self, feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_average_pool(feature_map, mask)

    def extract_from_outputs(self, outputs: dict[str, object], images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feature = outputs.get("image_embeddings")
        if not isinstance(feature, torch.Tensor):
            raise ValueError("SAM3 outputs did not contain image_embeddings for prototype extraction")
        feature_map = _resolve_feature_map(feature, images)
        return self.extract_from_feature_map(feature_map, mask)

    def extract_from_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        text_prompt: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if self.wrapper is None:
            raise RuntimeError("PrototypeExtractor requires a SAM3 wrapper to extract from raw images")
        outputs = self.wrapper(images=images, boxes=boxes, text_prompt=text_prompt)
        return self.extract_from_outputs(outputs, images, masks), outputs

    def save_prototype(
        self,
        root: str | Path,
        prototype: torch.Tensor,
        entry: PrototypeBankEntry,
    ) -> PrototypeBankEntry:
        destination = Path(root)
        bank_dir = destination / ("positive_bank" if entry.polarity == "positive" else "negative_bank")
        bank_dir.mkdir(parents=True, exist_ok=True)
        feature_path = bank_dir / f"{entry.prototype_id}.pt"
        metadata_path = bank_dir / f"{entry.prototype_id}.json"
        stored_entry = replace(entry, feature_path=str(feature_path))
        torch.save({"prototype": prototype.detach().cpu()}, feature_path)
        metadata_path.write_text(json.dumps(stored_entry.__dict__, indent=2), encoding="utf-8")
        return stored_entry