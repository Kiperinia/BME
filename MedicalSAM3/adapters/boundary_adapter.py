"""MedEx-SAM3 的边界感知特征精炼模块。

通过掩码形态学或对比度先验提取边界图，再用门控机制将边界信息
注入图像特征，增强分割模型对组织边界的刻画能力。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .medical_adapter import BottleneckAdapter


def _boundary_from_mask(mask: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    """从二值掩码通过腐蚀与膨胀的差集提取边界带。

    参数：
        - mask: 输入二值掩码张量
        - dilation: 边界带膨胀的程度

    返回：
        - 边界带掩码张量，取值范围为 [0, 1]
    """
    mask = mask.float()
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    band = (dilated - eroded).clamp(0, 1)
    if dilation > 1:
        band = F.max_pool2d(band, kernel_size=dilation * 2 + 1, stride=1, padding=dilation)
    return band


def _contrast_prior(feature_map: torch.Tensor) -> torch.Tensor:
    """从特征图通道均值计算局部对比度先验作为边界近似。

    参数：
        - feature_map: 输入特征图张量

    返回：
        - 归一化到 [0, 1] 的对比度先验图
    """
    contrast_source = feature_map.mean(dim=1, keepdim=True)
    local_mean = F.avg_pool2d(contrast_source, kernel_size=7, stride=1, padding=3)
    local_std = (contrast_source - local_mean).abs()
    normalized = local_std / (local_std.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return normalized.clamp(0, 1)


class BoundaryAwareAdapter(nn.Module):
    """边界感知适配器，融合边界特征与图像特征以精炼分割边界。

    参数：
        - channels: 输入特征通道数
        - bottleneck_dim: 内部瓶颈适配器的瓶颈维度
    """

    def __init__(self, channels: int, bottleneck_dim: Optional[int] = None) -> None:
        """初始化边界编码器、边界头、门控头与瓶颈适配器。

        参数：
            - channels: 输入特征通道数
            - bottleneck_dim: 内部瓶颈适配器的瓶颈维度，为空时自动取 channels//4

        返回：
            - 无返回值，完成各子层的构建
        """
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(channels // 4, 8)
        self.boundary_encoder = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.adapter = BottleneckAdapter(channels, bottleneck_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        coarse_mask_logits: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """前向计算：根据边界图门控增强图像特征，并输出辅助监督信息。

        参数：
            - image_features: 形如 [B, C, H, W] 的图像特征
            - coarse_mask_logits: 粗预测掩码 logits，用于推断边界
            - gt_mask: 真值掩码，用于生成边界监督目标

        返回：
            - 二元组：(边界增强后的特征, 辅助信息字典，可能含边界图与边界损失)
        """
        if image_features.dim() != 4:
            raise ValueError("BoundaryAwareAdapter expects [B, C, H, W] image features")

        feature_size = image_features.shape[-2:]
        boundary_target = None
        boundary_map = None
        if gt_mask is not None:
            if gt_mask.shape[-2:] != feature_size:
                gt_mask = F.interpolate(gt_mask.float(), size=feature_size, mode="nearest")
            boundary_target = _boundary_from_mask(gt_mask)
        if coarse_mask_logits is not None:
            coarse = coarse_mask_logits.sigmoid()
            if coarse.shape[-2:] != feature_size:
                coarse = F.interpolate(coarse, size=feature_size, mode="bilinear", align_corners=False)
            boundary_map = _boundary_from_mask((coarse > 0.5).float())

        if boundary_map is None:
            boundary_map = _contrast_prior(image_features)
        boundary_features = self.boundary_encoder(boundary_map)
        boundary_logits = self.boundary_head(boundary_features)
        boundary_gate = self.gate_head(torch.cat([image_features, boundary_features], dim=1))
        enhanced = image_features + boundary_gate * self.adapter(image_features)

        aux: dict[str, torch.Tensor] = {
            "boundary_map": boundary_map,
            "boundary_gate": boundary_gate,
        }
        if boundary_target is not None:
            with torch.autocast(device_type=image_features.device.type, enabled=False):
                aux["boundary_loss"] = F.binary_cross_entropy_with_logits(
                    boundary_logits.float(),
                    boundary_target.float(),
                )
        return enhanced, aux
