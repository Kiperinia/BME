from __future__ import annotations

import torch
import torch.nn.functional as F


def local_contrast_map(image: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """估计内镜低对比场景下每个位置的模糊程度。"""
    gray = image.mean(dim=1, keepdim=True)
    avg = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    contrast = (gray - avg).abs()
    normalized = 1.0 - contrast / (contrast.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return normalized.clamp(0, 1)


def smoothness_prior(prob_mask: torch.Tensor) -> torch.Tensor:
    """偏向局部平滑、闭合的轮廓，而不是带毛刺的边界。"""
    avg = F.avg_pool2d(prob_mask, kernel_size=5, stride=1, padding=2)
    prior = 1.0 - (prob_mask - avg).abs()
    return prior.clamp(0, 1)


def compactness_proxy(prob_mask: torch.Tensor) -> torch.Tensor:
    """在推理时不依赖 GT 的轻量“息肉样紧致区域”代理先验。"""
    pooled = F.avg_pool2d(prob_mask, kernel_size=11, stride=1, padding=5)
    center_bias = pooled / (prob_mask.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return center_bias.clamp(0, 1)


def build_polyp_shape_prior(image: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
    """
    将低对比模糊性、平滑性和紧致性合成为一个先验门控图。
    值越大，表示该位置越值得做更强的息肉边界精修。
    """
    coarse_prob = coarse_logits.sigmoid()
    ambiguity = local_contrast_map(image)
    smooth = smoothness_prior(coarse_prob)
    compact = compactness_proxy(coarse_prob)
    shape_prior = 0.45 * ambiguity + 0.35 * smooth + 0.20 * compact
    return shape_prior.clamp(0, 1)


def demo() -> None:
    torch.manual_seed(1)

    image = torch.rand(1, 3, 64, 64) * 0.2 + 0.4
    image[:, :, 18:46, 20:44] += 0.04

    coarse_logits = torch.full((1, 1, 64, 64), -1.4)
    coarse_logits[:, :, 17:44, 23:46] = 1.2
    coarse_logits[:, :, 14:17, 35:41] = 0.6

    prior = build_polyp_shape_prior(image, coarse_logits)
    print("shape_prior mean=", round(prior.mean().item(), 4))
    print("shape_prior max= ", round(prior.max().item(), 4))
    print("shape_prior min= ", round(prior.min().item(), 4))


if __name__ == "__main__":
    demo()