"""区域感知门控机制，用于局部检索校正。"""

from __future__ import annotations

import torch


def build_retrieval_region_mask(
    *,
    probability_map: torch.Tensor,
    confidence_map: torch.Tensor,
    entropy_map: torch.Tensor,
    boundary_uncertainty_map: torch.Tensor,
    low_confidence_lesion_map: torch.Tensor,
    high_confidence_threshold: float = 0.85,
) -> dict[str, torch.Tensor]:
    """构建检索区域掩码，基于不确定性、边界和置信度信息确定需要检索修正的区域。

    参数：
        - probability_map: 分割概率图。
        - confidence_map: 置信度图。
        - entropy_map: 熵图，表示预测不确定性。
        - boundary_uncertainty_map: 边界不确定性图。
        - low_confidence_lesion_map: 低置信度病变区域图。
        - high_confidence_threshold: 高置信度阈值，默认为 0.85。

    返回：
        - 包含检索区域掩码、高置信度保留掩码、激活比率和区域类型统计的字典。
    """
    dtype = probability_map.dtype
    uncertain_focus = torch.maximum(entropy_map, boundary_uncertainty_map)
    lesion_focus = torch.maximum(uncertain_focus, low_confidence_lesion_map)
    high_confidence_mask = (confidence_map >= high_confidence_threshold).to(dtype=dtype)
    high_confidence_foreground = ((probability_map >= 0.5).to(dtype=dtype) * high_confidence_mask)
    high_confidence_background = ((probability_map < 0.5).to(dtype=dtype) * high_confidence_mask)
    high_confidence_preserve_mask = torch.clamp(high_confidence_foreground + high_confidence_background, 0.0, 1.0)
    retrieval_region_mask = torch.clamp(lesion_focus * (1.0 - high_confidence_preserve_mask), 0.0, 1.0)
    activation_ratio = retrieval_region_mask.flatten(1).mean(dim=1)
    region_type_statistics = {
        "boundary": boundary_uncertainty_map.flatten(1).mean(dim=1),
        "low_confidence_lesion": low_confidence_lesion_map.flatten(1).mean(dim=1),
        "high_confidence_foreground": high_confidence_foreground.flatten(1).mean(dim=1),
        "high_confidence_background": high_confidence_background.flatten(1).mean(dim=1),
    }
    return {
        "retrieval_region_mask": retrieval_region_mask,
        "high_confidence_preserve_mask": high_confidence_preserve_mask,
        "high_confidence_foreground_mask": high_confidence_foreground,
        "high_confidence_background_mask": high_confidence_background,
        "activation_ratio": activation_ratio,
        "region_type_statistics": region_type_statistics,
    }


__all__ = ["build_retrieval_region_mask"]
