"""Visualization helpers for retrieval influence diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch


def _to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float()
    if image.dim() == 4:
        image = image[0]
    if image.dim() != 3:
        raise ValueError("Expected image tensor with shape [C, H, W] or [B, C, H, W]")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    array = image.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return array


def _to_uint8_gray(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().float()
    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    array = mask.numpy()
    if array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _normalize_map(values: torch.Tensor) -> np.ndarray:
    heatmap = values.detach().cpu().float()
    if heatmap.dim() == 4:
        heatmap = heatmap[0, 0]
    elif heatmap.dim() == 3:
        heatmap = heatmap[0]
    array = heatmap.numpy()
    array = array - array.min()
    denom = max(float(array.max()), 1e-6)
    return array / denom


def _caption_tile(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    caption_height = 34 if subtitle else 22
    canvas = Image.new("RGB", (image.width, image.height + caption_height), color=(18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, image.height + 4), title[:28], fill=(255, 255, 255))
    if subtitle:
        draw.text((6, image.height + 18), subtitle[:28], fill=(180, 180, 180))
    return canvas


def _load_tile_from_path(path: Optional[str], tile_size: int, title: str, subtitle: str) -> Image.Image:
    if path and Path(path).is_file():
        image = Image.open(path).convert("RGB").resize((tile_size, tile_size))
    else:
        image = Image.new("RGB", (tile_size, tile_size), color=(54, 60, 68))
        draw = ImageDraw.Draw(image)
        draw.rectangle([8, 8, tile_size - 8, tile_size - 8], outline=(110, 120, 132), width=2)
        draw.line([16, 16, tile_size - 16, tile_size - 16], fill=(110, 120, 132), width=2)
        draw.line([tile_size - 16, 16, 16, tile_size - 16], fill=(110, 120, 132), width=2)
    return _caption_tile(image, title, subtitle)


def save_retrieved_prototype_panel(
    query_image: torch.Tensor,
    positive_entries: list[dict[str, Any]],
    negative_entries: list[dict[str, Any]],
    output_path: str | Path,
    tile_size: int = 128,
) -> Path:
    query_tile = _caption_tile(Image.fromarray(_to_uint8_rgb(query_image)).resize((tile_size, tile_size)), "query")
    tiles = [query_tile]
    for entry in positive_entries:
        tiles.append(
            _load_tile_from_path(
                entry.get("crop_path"),
                tile_size,
                f"pos:{entry.get('prototype_id', 'unknown')}",
                f"sim={entry.get('similarity_score', 0.0):.3f}",
            )
        )
    for entry in negative_entries:
        tiles.append(
            _load_tile_from_path(
                entry.get("crop_path"),
                tile_size,
                f"neg:{entry.get('prototype_id', 'unknown')}",
                f"sim={entry.get('similarity_score', 0.0):.3f}",
            )
        )
    columns = max(1, min(4, len(tiles)))
    rows = int(np.ceil(len(tiles) / columns))
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + 34)), color=(12, 12, 12))
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        canvas.paste(tile, (col * tile_size, row * (tile_size + 34)))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def save_similarity_heatmap_overlay(
    query_image: torch.Tensor,
    heatmap: torch.Tensor,
    output_path: str | Path,
    title: str = "heatmap",
    alpha: float = 0.45,
) -> Path:
    rgb = _to_uint8_rgb(query_image)
    norm = _normalize_map(heatmap)
    if norm.shape[0] != rgb.shape[0] or norm.shape[1] != rgb.shape[1]:
        norm = np.asarray(
            Image.fromarray((norm * 255.0).clip(0.0, 255.0).astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]))
        ).astype(np.float32) / 255.0
    colored = np.stack([norm, np.square(norm), 1.0 - norm], axis=-1)
    overlay = (1.0 - alpha) * (rgb.astype(np.float32) / 255.0) + alpha * colored
    image = _caption_tile(Image.fromarray((overlay * 255.0).clip(0.0, 255.0).astype(np.uint8)), title)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def save_mask_difference_visualization(
    query_image: torch.Tensor,
    reference_logits: torch.Tensor,
    compared_logits: torch.Tensor,
    output_path: str | Path,
    reference_label: str,
    compared_label: str,
) -> Path:
    base = _to_uint8_rgb(query_image)
    ref_mask = (torch.sigmoid(reference_logits.detach().float()) > 0.5).float()
    cmp_mask = (torch.sigmoid(compared_logits.detach().float()) > 0.5).float()
    if ref_mask.shape[-2:] != (base.shape[0], base.shape[1]):
        ref_mask = torch.nn.functional.interpolate(ref_mask.float(), size=(base.shape[0], base.shape[1]), mode="nearest")
    if cmp_mask.shape[-2:] != (base.shape[0], base.shape[1]):
        cmp_mask = torch.nn.functional.interpolate(cmp_mask.float(), size=(base.shape[0], base.shape[1]), mode="nearest")
    ref_array = _to_uint8_gray(ref_mask)
    cmp_array = _to_uint8_gray(cmp_mask)
    xor = ((ref_mask > 0.5) ^ (cmp_mask > 0.5)).detach().cpu().numpy().astype(np.uint8)
    if xor.ndim == 4:
        xor = xor[0, 0]
    elif xor.ndim == 3:
        xor = xor[0]
    overlay = base.copy()
    overlay[xor > 0, 0] = 255
    overlay[xor > 0, 1] = 210
    overlay[xor > 0, 2] = 0
    tiles = [
        _caption_tile(Image.fromarray(base), "query"),
        _caption_tile(Image.fromarray(ref_array).convert("RGB"), reference_label),
        _caption_tile(Image.fromarray(cmp_array).convert("RGB"), compared_label),
        _caption_tile(Image.fromarray(overlay), "difference"),
    ]
    tile_width = tiles[0].width
    tile_height = tiles[0].height
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 2), color=(14, 14, 14))
    for index, tile in enumerate(tiles):
        row = index // 2
        col = index % 2
        canvas.paste(tile, (col * tile_width, row * tile_height))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def save_false_positive_overlay(
    query_image: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    output_path: str | Path,
    title: str,
) -> Path:
    base = _to_uint8_rgb(query_image)
    pred = (torch.sigmoid(pred_logits.detach().float()) > 0.5).float()
    target = (gt_mask.detach().float() > 0.5).float()
    if pred.shape[-2:] != (base.shape[0], base.shape[1]):
        pred = torch.nn.functional.interpolate(pred.float(), size=(base.shape[0], base.shape[1]), mode="nearest")
    if target.shape[-2:] != (base.shape[0], base.shape[1]):
        target = torch.nn.functional.interpolate(target.float(), size=(base.shape[0], base.shape[1]), mode="nearest")
    if pred.dim() == 4:
        pred = pred[0, 0]
    elif pred.dim() == 3:
        pred = pred[0]
    if target.dim() == 4:
        target = target[0, 0]
    elif target.dim() == 3:
        target = target[0]
    pred_array = pred.cpu().numpy() > 0.5
    target_array = target.cpu().numpy() > 0.5
    fp = pred_array & ~target_array
    tp = pred_array & target_array
    overlay = base.copy()
    overlay[tp, 1] = 255
    overlay[fp, 0] = 255
    overlay[fp, 1] = 64
    image = _caption_tile(Image.fromarray(overlay), title, "green=tp red=fp")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination