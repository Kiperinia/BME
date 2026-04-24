from __future__ import annotations

import torch
import torch.nn.functional as F


def boundary_from_mask(mask: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    """从二值 mask 构造一个软边界带。"""
    if mask.dim() != 4:
        raise ValueError("mask must be shaped as (B, 1, H, W)")

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    boundary = (grad_x.abs() + grad_y.abs()).clamp(0, 1)
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=dilation)
    return boundary


def build_error_targets(
    coarse_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    dilation: int = 2,
) -> dict[str, torch.Tensor]:
    """
    构造监督信号，教 BRH 学会“哪里该修、该修多强”。

    返回:
        coarse_prob: coarse_logits 的 sigmoid 概率图
        coarse_boundary: 从粗分割结果得到的边界带
        gt_boundary: 从 GT mask 得到的边界带
        error_region: 限制在边界附近的误差区域图
        signed_error: 带符号误差图，表示应该往哪一侧修正
    """
    coarse_prob = coarse_logits.sigmoid()
    coarse_boundary = boundary_from_mask((coarse_prob > 0.5).float(), dilation=dilation)
    gt_boundary = boundary_from_mask(gt_mask, dilation=dilation)

    disagreement = (coarse_prob - gt_mask).abs()
    boundary_union = torch.maximum(coarse_boundary, gt_boundary)
    error_region = disagreement * boundary_union
    signed_error = (gt_mask - coarse_prob) * boundary_union

    return {
        "coarse_prob": coarse_prob,
        "coarse_boundary": coarse_boundary,
        "gt_boundary": gt_boundary,
        "error_region": error_region,
        "signed_error": signed_error,
    }


def demo() -> None:
    torch.manual_seed(0)

    gt_mask = torch.zeros(1, 1, 64, 64)
    gt_mask[:, :, 18:46, 20:44] = 1.0

    coarse_logits = torch.full((1, 1, 64, 64), -2.0)
    coarse_logits[:, :, 16:42, 24:48] = 2.0
    coarse_logits[:, :, 38:50, 34:48] = 1.0

    targets = build_error_targets(coarse_logits, gt_mask)
    for name, value in targets.items():
        print(f"{name:16s} shape={tuple(value.shape)} mean={value.mean().item():.4f} max={value.max().item():.4f}")


if __name__ == "__main__":
    demo()