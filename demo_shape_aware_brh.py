from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_error_targets import build_error_targets
from demo_polyp_shape_prior import build_polyp_shape_prior


class ShapeAwareBoundaryRefinementHead(nn.Module):
    """
    BRH 的演示版本：
    1. 预测边界修正量 delta
    2. 预测误差置信度 error confidence
    3. 用息肉形状先验调节精修强度
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.error_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, coarse_logits: torch.Tensor, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.shape[-2:] != coarse_logits.shape[-2:]:
            image = F.interpolate(image, size=coarse_logits.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([coarse_logits, image], dim=1)
        feat = self.trunk(x)
        delta = self.delta_head(feat)
        error_confidence = self.error_head(feat).sigmoid()
        shape_prior = build_polyp_shape_prior(image, coarse_logits)

        refinement_gate = error_confidence * shape_prior
        refined_logits = coarse_logits + delta * refinement_gate
        return {
            "refined_logits": refined_logits,
            "delta": delta,
            "error_confidence": error_confidence,
            "shape_prior": shape_prior,
            "refinement_gate": refinement_gate,
        }


@dataclass
class DemoLosses:
    seg: torch.Tensor
    boundary: torch.Tensor
    error: torch.Tensor
    smooth: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.seg + 0.5 * self.boundary + 0.5 * self.error + 0.1 * self.smooth


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.sigmoid()
    intersection = (prob * target).sum(dim=(-2, -1))
    union = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


def build_losses(outputs: dict[str, torch.Tensor], coarse_logits: torch.Tensor, gt_mask: torch.Tensor) -> DemoLosses:
    targets = build_error_targets(coarse_logits, gt_mask)
    seg = dice_loss(outputs["refined_logits"], gt_mask) + F.binary_cross_entropy_with_logits(
        outputs["refined_logits"], gt_mask
    )
    boundary = F.l1_loss(outputs["refinement_gate"], targets["gt_boundary"])
    error = F.binary_cross_entropy(outputs["error_confidence"], targets["error_region"].clamp(0, 1))
    smooth = (outputs["delta"][:, :, :, 1:] - outputs["delta"][:, :, :, :-1]).abs().mean()
    return DemoLosses(seg=seg, boundary=boundary, error=error, smooth=smooth)


def demo_train_step() -> None:
    torch.manual_seed(2)
    model = ShapeAwareBoundaryRefinementHead()

    image = torch.rand(2, 3, 64, 64) * 0.15 + 0.45
    gt_mask = torch.zeros(2, 1, 64, 64)
    gt_mask[0, :, 17:45, 18:42] = 1.0
    gt_mask[1, :, 22:50, 24:47] = 1.0

    coarse_logits = torch.full((2, 1, 64, 64), -1.8)
    coarse_logits[0, :, 15:42, 22:46] = 1.6
    coarse_logits[1, :, 24:52, 22:44] = 1.0

    outputs = model(coarse_logits, image)
    losses = build_losses(outputs, coarse_logits, gt_mask)

    print("seg loss     =", round(losses.seg.item(), 4))
    print("boundary loss=", round(losses.boundary.item(), 4))
    print("error loss   =", round(losses.error.item(), 4))
    print("smooth loss  =", round(losses.smooth.item(), 4))
    print("total loss   =", round(losses.total.item(), 4))


if __name__ == "__main__":
    demo_train_step()