"""Shared helpers for MedEx-SAM3 data, metrics and lightweight training."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from MedicalSAM3.adapters.boundary_adapter import BoundaryAwareAdapter
from MedicalSAM3.adapters.medical_adapter import MedicalImageAdapter, MultiScaleMedicalAdapter
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: Optional[str | Path]) -> dict[str, Any]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:
        config: dict[str, Any] = {}
        for line in target.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip()
            lowered = value.lower()
            if lowered in {"true", "false"}:
                parsed: Any = lowered == "true"
            else:
                try:
                    parsed = json.loads(value)
                except Exception:
                    try:
                        parsed = int(value)
                    except ValueError:
                        try:
                            parsed = float(value)
                        except ValueError:
                            parsed = value
            config[key.strip()] = parsed
        return config


def dump_config(path: str | Path, config: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except Exception:
        destination.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return destination


def read_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            records.append(json.loads(stripped))
        else:
            image_path, mask_path, dataset_name, image_id = stripped.split("\t")
            records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "dataset_name": dataset_name,
                    "image_id": image_id,
                }
            )
    return records


def write_records(path: str | Path, records: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )
    return destination


def synthetic_polyp_sample(image_size: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.new("RGB", (image_size, image_size), color=(30 + index * 3 % 80, 20, 20))
    mask = Image.new("L", (image_size, image_size), color=0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    radius = image_size // 6 + (index % 5) * max(image_size // 40, 1)
    cx = image_size // 2 + (index % 3 - 1) * image_size // 10
    cy = image_size // 2 + (index % 4 - 2) * image_size // 12
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw_image.ellipse(bbox, fill=(170, 80 + index * 7 % 60, 90))
    draw_mask.ellipse(bbox, fill=255)
    image_tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
    mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype("float32")).unsqueeze(0)
    return image_tensor, mask_tensor


def load_record_tensors(record: dict[str, Any], image_size: int, fallback_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    image_path = Path(record.get("image_path", ""))
    mask_path = Path(record.get("mask_path", ""))
    if image_path.exists() and mask_path.exists():
        if image_path.stem.endswith("_0000"):
            channel_paths = [image_path.with_name(image_path.name.replace("_0000", f"_000{i}")) for i in range(3)]
            if all(channel_path.exists() for channel_path in channel_paths):
                channels = [
                    np.asarray(Image.open(channel_path).convert("L").resize((image_size, image_size))).astype("float32")
                    for channel_path in channel_paths
                ]
                stacked = np.stack(channels, axis=-1) / 255.0
                image_tensor = torch.from_numpy(stacked).permute(2, 0, 1)
            else:
                image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
                image_tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
        else:
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
            image_tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
        mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
        mask_tensor = torch.from_numpy((np.asarray(mask) > 127).astype("float32")).unsqueeze(0)
        return image_tensor, mask_tensor
    return synthetic_polyp_sample(image_size, fallback_index)


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        _, height, width = mask.shape
        return torch.tensor([0.0, 0.0, float(width), float(height)])
    y1 = coords[:, -2].min().float()
    x1 = coords[:, -1].min().float()
    y2 = coords[:, -2].max().float() + 1.0
    x2 = coords[:, -1].max().float() + 1.0
    return torch.tensor([x1, y1, x2, y2])


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    return (dilated - eroded).clamp(0, 1)


def _surface_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> tuple[float, float]:
    pred_points = torch.nonzero(boundary_band(pred_mask) > 0.5, as_tuple=False).float()
    gt_points = torch.nonzero(boundary_band(gt_mask) > 0.5, as_tuple=False).float()
    if pred_points.numel() == 0 or gt_points.numel() == 0:
        return 0.0, 0.0
    distances = torch.cdist(pred_points[:, -2:], gt_points[:, -2:])
    symmetric = torch.cat([distances.min(dim=1).values, distances.min(dim=0).values])
    hd95 = float(torch.quantile(symmetric, 0.95).item())
    assd = float(symmetric.mean().item())
    return hd95, assd


def compute_segmentation_metrics(mask_logits: torch.Tensor, gt_mask: torch.Tensor) -> dict[str, float]:
    if gt_mask.shape != mask_logits.shape:
        gt_mask = F.interpolate(gt_mask.float(), size=mask_logits.shape[-2:], mode="nearest")
    prob = torch.sigmoid(mask_logits)
    pred = (prob > 0.5).float()
    target = gt_mask.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = float(((2.0 * intersection + 1e-6) / (union + 1e-6)).item())
    iou = float(((intersection + 1e-6) / ((pred + target).clamp(0, 1).sum() + 1e-6)).item())
    precision = float(((intersection + 1e-6) / (pred.sum() + 1e-6)).item())
    recall = float(((intersection + 1e-6) / (target.sum() + 1e-6)).item())
    pred_boundary = boundary_band(pred)
    gt_boundary = boundary_band(target)
    boundary_intersection = (pred_boundary * gt_boundary).sum()
    boundary_union = pred_boundary.sum() + gt_boundary.sum()
    boundary_f1 = float(((2.0 * boundary_intersection + 1e-6) / (boundary_union + 1e-6)).item())
    hd95, assd = _surface_metrics(pred, target)
    fp = ((pred == 1) & (target == 0)).float().sum()
    fn = ((pred == 0) & (target == 1)).float().sum()
    tn = ((pred == 0) & (target == 0)).float().sum()
    tp = ((pred == 1) & (target == 1)).float().sum()
    fpr = float((fp / (fp + tn + 1e-6)).item())
    fnr = float((fn / (fn + tp + 1e-6)).item())
    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Boundary F1": boundary_f1,
        "HD95": hd95,
        "ASSD": assd,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
    }


class SplitSegmentationDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, mask = load_record_tensors(record, self.image_size, fallback_index=index)
        return {
            "image": image,
            "mask": mask,
            "box": mask_to_box(mask),
            "text_prompt": ["polyp"],
            "record": record,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "boxes": torch.stack([item["box"] for item in batch], dim=0),
        "text_prompt": [item["text_prompt"][0] for item in batch],
        "records": [item["record"] for item in batch],
    }


def resolve_feature_map(feature: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    if feature is None:
        return fallback.mean(dim=1, keepdim=True).repeat(1, 128, 1, 1)
    if feature.dim() == 4:
        return feature
    if feature.dim() == 3:
        batch_size, tokens, channels = feature.shape
        side = int(math.sqrt(tokens))
        if side * side == tokens:
            return feature.transpose(1, 2).reshape(batch_size, channels, side, side)
        feature = feature.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        return feature.repeat(1, 1, fallback.shape[-2] // 4, fallback.shape[-1] // 4)
    raise ValueError("Unsupported feature shape")


class MedExSam3SegmentationModel(nn.Module):
    def __init__(
        self,
        wrapper: Sam3TensorForwardWrapper,
        enable_medical_adapter: bool = False,
        enable_msfa_adapter: bool = False,
        enable_boundary_adapter: bool = False,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.medical_adapter = MedicalImageAdapter(embed_dim, max(embed_dim // 4, 8)) if enable_medical_adapter else None
        self.msfa_adapter = MultiScaleMedicalAdapter(embed_dim) if enable_msfa_adapter else None
        self.boundary_adapter = BoundaryAwareAdapter(embed_dim) if enable_boundary_adapter else None
        self.refine_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        text_prompt: Optional[list[str]] = None,
        exemplar_prompt_tokens: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        outputs = self.wrapper(
            images=images,
            text_prompt=text_prompt,
            boxes=boxes,
            exemplar_prompt_tokens=exemplar_prompt_tokens,
        )
        feature_map = resolve_feature_map(outputs.get("image_embeddings"), images)
        if self.medical_adapter is not None:
            feature_map = self.medical_adapter(feature_map)
        if self.msfa_adapter is not None:
            feature_map = self.msfa_adapter(feature_map)
        aux = {}
        if self.boundary_adapter is not None:
            feature_map, aux = self.boundary_adapter(feature_map, coarse_mask_logits=outputs["mask_logits"], gt_mask=gt_mask)
        delta = self.refine_head(feature_map)
        delta = F.interpolate(delta, size=outputs["mask_logits"].shape[-2:], mode="bilinear", align_corners=False)
        outputs["mask_logits"] = outputs["mask_logits"] + 0.1 * delta
        outputs["masks"] = torch.sigmoid(outputs["mask_logits"])
        outputs["query_embedding"] = F.normalize(F.adaptive_avg_pool2d(feature_map, 1).flatten(1), dim=1)
        outputs["adapter_aux"] = aux
        outputs["image_embeddings"] = feature_map
        return outputs
