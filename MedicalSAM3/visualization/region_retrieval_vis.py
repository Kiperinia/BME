"""Region-aware retrieval visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


def _to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float()
    if image.dim() == 4:
        image = image[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    array = image.permute(1, 2, 0).numpy()
    return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)


def _to_gray(mask_tensor: torch.Tensor, *, sigmoid: bool = False) -> np.ndarray:
    mask = mask_tensor.detach().cpu().float()
    if sigmoid:
        mask = torch.sigmoid(mask)
    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    array = mask.numpy()
    if array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _resize_map(value: Any, size: tuple[int, int], *, mode: str = "bilinear") -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    tensor = value.detach().float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[-2:] == size:
        return tensor
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    if align_corners is None:
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)


def _heatmap_rgb(values: torch.Tensor) -> np.ndarray:
    tensor = values.detach().cpu().float()
    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]
    array = tensor.numpy()
    array = array - array.min()
    denom = max(float(array.max()), 1e-6)
    norm = array / denom
    rgb = np.stack([norm, np.sqrt(norm), 1.0 - norm], axis=-1)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _delta_rgb(delta_logits: torch.Tensor) -> np.ndarray:
    tensor = delta_logits.detach().cpu().float()
    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]
    array = tensor.numpy()
    scale = max(float(np.abs(array).max()), 1e-6)
    normalized = np.clip(array / scale, -1.0, 1.0)
    red = np.clip(-normalized, 0.0, 1.0)
    green = np.clip(normalized, 0.0, 1.0)
    blue = 1.0 - np.abs(normalized)
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _change_map(baseline_logits: torch.Tensor, corrected_logits: torch.Tensor, gt_mask: torch.Tensor | None) -> np.ndarray:
    baseline = (torch.sigmoid(baseline_logits.detach().float()) > 0.5).float()
    corrected = (torch.sigmoid(corrected_logits.detach().float()) > 0.5).float()
    if gt_mask is None or gt_mask.numel() == 0:
        delta = corrected - baseline
        return _delta_rgb(delta)
    target = (gt_mask.detach().float() > 0.5).float()
    improved = (corrected == target) & (baseline != target)
    worsened = (corrected != target) & (baseline == target)
    unchanged_error = (corrected != target) & (baseline != target)
    if improved.dim() == 4:
        improved = improved[0, 0]
        worsened = worsened[0, 0]
        unchanged_error = unchanged_error[0, 0]
    image = np.zeros((*improved.shape, 3), dtype=np.uint8)
    image[..., :] = 36
    image[unchanged_error.cpu().numpy(), :] = np.array([96, 96, 96], dtype=np.uint8)
    image[worsened.cpu().numpy(), :] = np.array([220, 68, 48], dtype=np.uint8)
    image[improved.cpu().numpy(), :] = np.array([48, 180, 84], dtype=np.uint8)
    return image


def _caption(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    extra_height = 34 if subtitle else 22
    canvas = Image.new("RGB", (image.width, image.height + extra_height), color=(18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, image.height + 4), title[:32], fill=(255, 255, 255))
    if subtitle:
        draw.text((6, image.height + 18), subtitle[:32], fill=(180, 180, 180))
    return canvas


def _tile_from_path(path: Optional[str], tile_size: int, title: str) -> Image.Image:
    if path and Path(path).is_file():
        image = Image.open(path).convert("RGB").resize((tile_size, tile_size))
    else:
        image = Image.new("RGB", (tile_size, tile_size), color=(54, 60, 68))
        draw = ImageDraw.Draw(image)
        draw.rectangle([8, 8, tile_size - 8, tile_size - 8], outline=(110, 120, 132), width=2)
    return _caption(image, title)


def _first_entry_path(payload: dict[str, Any], polarity: str = "positive") -> Optional[str]:
    entries = payload.get(f"{polarity}_entries", [])
    if not isinstance(entries, list) or not entries:
        return None
    batch_entries = entries[0]
    if not isinstance(batch_entries, list):
        return None
    for entry in batch_entries:
        crop_path = getattr(entry, "crop_path", None)
        if crop_path:
            return str(crop_path)
    return None


def save_region_retrieval_panel(
    *,
    query_image: torch.Tensor,
    baseline_mask_logits: torch.Tensor,
    corrected_mask_logits: torch.Tensor,
    adapter_aux: dict[str, Any],
    retrieval: dict[str, Any],
    gt_mask: torch.Tensor | None,
    output_path: str | Path,
    tile_size: int = 160,
) -> Path:
    size = tuple(int(item) for item in corrected_mask_logits.shape[-2:])
    entropy_map = _resize_map(adapter_aux.get("segmentation_entropy_map"), size, mode="bilinear")
    if entropy_map is None:
        entropy_map = torch.zeros_like(corrected_mask_logits)
    retrieval_region_mask = _resize_map(adapter_aux.get("retrieval_region_mask"), size, mode="bilinear")
    if retrieval_region_mask is None:
        retrieval_region_mask = torch.zeros_like(corrected_mask_logits)
    positive_mask_prior = _resize_map(retrieval.get("positive_mask_prior"), size, mode="bilinear")
    if positive_mask_prior is None:
        positive_mask_prior = torch.zeros_like(corrected_mask_logits)
    multi_bank_fusion = retrieval.get("multi_bank_fusion", {}) if isinstance(retrieval.get("multi_bank_fusion"), dict) else {}
    train_entry_path = _first_entry_path(multi_bank_fusion.get("train_topk_exemplar", {}), polarity="positive")
    site_entry_path = _first_entry_path(multi_bank_fusion.get("site_topk_exemplar", {}), polarity="positive")

    query_tile = _caption(Image.fromarray(_to_rgb(query_image)).resize((tile_size, tile_size)), "query")
    baseline_tile = _caption(Image.fromarray(_to_gray(baseline_mask_logits, sigmoid=True)).convert("RGB").resize((tile_size, tile_size)), "baseline mask")
    corrected_tile = _caption(Image.fromarray(_to_gray(corrected_mask_logits, sigmoid=True)).convert("RGB").resize((tile_size, tile_size)), "retrieval mask")
    entropy_tile = _caption(Image.fromarray(_heatmap_rgb(entropy_map)).resize((tile_size, tile_size)), "uncertainty")
    region_tile = _caption(Image.fromarray(_heatmap_rgb(retrieval_region_mask)).resize((tile_size, tile_size)), "retrieval region")
    train_tile = _tile_from_path(train_entry_path, tile_size, "train exemplar")
    site_tile = _tile_from_path(site_entry_path, tile_size, "site exemplar")
    mask_prior_tile = _caption(Image.fromarray(_heatmap_rgb(positive_mask_prior)).resize((tile_size, tile_size)), "mask prior")
    delta_tile = _caption(Image.fromarray(_delta_rgb(corrected_mask_logits - baseline_mask_logits)).resize((tile_size, tile_size)), "correction heatmap")
    gt_for_change = _resize_map(gt_mask, size, mode="nearest") if isinstance(gt_mask, torch.Tensor) and gt_mask.numel() > 0 else None
    change_tile = _caption(Image.fromarray(_change_map(baseline_mask_logits, corrected_mask_logits, gt_for_change)).resize((tile_size, tile_size)), "change map", "green=better red=worse")

    tiles = [
        query_tile,
        baseline_tile,
        corrected_tile,
        entropy_tile,
        region_tile,
        train_tile,
        site_tile,
        mask_prior_tile,
        delta_tile,
        change_tile,
    ]
    columns = 5
    rows = 2
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + 34)), color=(10, 10, 10))
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        canvas.paste(tile, (col * tile_size, row * (tile_size + 34)))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


__all__ = ["save_region_retrieval_panel"]
