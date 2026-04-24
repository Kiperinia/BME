from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return ((2 * intersection + 1e-6) / (union + 1e-6)).mean()


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) == 9).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0).float()
    return (dilated - eroded).clamp(0, 1)


def boundary_f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_b = boundary_band(pred)
    target_b = boundary_band(target)
    tp = (pred_b * target_b).sum(dim=(-2, -1))
    precision = (tp + 1e-6) / (pred_b.sum(dim=(-2, -1)) + 1e-6)
    recall = (tp + 1e-6) / (target_b.sum(dim=(-2, -1)) + 1e-6)
    return (2 * precision * recall / (precision + recall + 1e-6)).mean()


@dataclass
class SampleMeta:
    low_contrast: bool
    blurry_boundary: bool
    small_polyp: bool


def stratified_report(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    metas: list[SampleMeta],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    groups = {
        "all": [True] * len(metas),
        "low_contrast": [meta.low_contrast for meta in metas],
        "blurry_boundary": [meta.blurry_boundary for meta in metas],
        "small_polyp": [meta.small_polyp for meta in metas],
    }

    for name, flags in groups.items():
        idx = [i for i, flag in enumerate(flags) if flag]
        subset_pred = pred_masks[idx]
        subset_gt = gt_masks[idx]
        report[name] = {
            "dice": round(dice_score(subset_pred, subset_gt).item(), 4),
            "boundary_f1": round(boundary_f1(subset_pred, subset_gt).item(), 4),
            "count": float(len(idx)),
        }
    return report


def demo() -> None:
    torch.manual_seed(3)

    gt_masks = torch.zeros(4, 1, 64, 64)
    pred_masks = torch.zeros(4, 1, 64, 64)
    metas = [
        SampleMeta(low_contrast=True, blurry_boundary=True, small_polyp=False),
        SampleMeta(low_contrast=False, blurry_boundary=True, small_polyp=True),
        SampleMeta(low_contrast=True, blurry_boundary=False, small_polyp=True),
        SampleMeta(low_contrast=False, blurry_boundary=False, small_polyp=False),
    ]

    for i in range(4):
        gt_masks[i, :, 16 + i:40 + i, 20:44] = 1.0
        pred_masks[i, :, 18 + i:38 + i, 22:46] = 1.0

    report = stratified_report(pred_masks, gt_masks, metas)
    for name, metrics in report.items():
        print(name, metrics)


if __name__ == "__main__":
    demo()