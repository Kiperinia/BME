"""示例编码器，生成正例、负例与边界嵌入。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """对输入张量在最后一维做 L2 归一化。

    参数：
        - x: 待归一化的张量。

    返回：
        - L2 归一化后的张量。
    """
    return F.normalize(x, dim=-1, eps=1e-6)


def _masked_average_pool(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """在掩码区域内对特征图做平均池化。

    参数：
        - feature_map: 特征图张量，形状为 (B, C, H, W)。
        - mask: 掩码张量，形状与特征图空间维度一致。

    返回：
        - 掩码区域平均池化后的嵌入张量，形状为 (B, C)。
    """
    masked = feature_map * mask
    denom = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
    return masked.sum(dim=(-2, -1)) / denom


def _boundary_band(mask: torch.Tensor) -> torch.Tensor:
    """通过形态学膨胀与腐蚀之差提取边界带。

    参数：
        - mask: 二值掩码张量，形状为 (B, 1, H, W)。

    返回：
        - 边界带掩码张量，取值范围 [0, 1]。
    """
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    return (dilated - eroded).clamp(0, 1)


class ExemplarEncoder(nn.Module):
    """示例编码器模块。

    将裁剪后的图像（及可选掩码）编码为全局、前景、边界、背景四类嵌入，
    可选地复用外部骨干网络提取特征。

    参数：
        - embed_dim: 输出嵌入维度。
        - backbone: 可选的外部骨干网络；为 None 时使用内置 stem。
    """
    def __init__(self, embed_dim: int = 128, backbone: Optional[nn.Module] = None) -> None:
        """初始化示例编码器。

        参数：
            - embed_dim: 输出嵌入维度。
            - backbone: 可选的外部骨干网络。

        返回：
            - 无返回值，仅完成模块初始化。
        """
        super().__init__()
        self.backbone = backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def _encode_feature_map(self, crop_image: torch.Tensor) -> torch.Tensor:
        """提取裁剪图像的特征图。

        优先使用外部骨干网络；若骨干不可用或返回不可用特征，则回退到内置 stem。

        参数：
            - crop_image: 裁剪后的图像张量，形状为 (B, 3, H, W)。

        返回：
            - 特征图张量，形状为 (B, C, H', W')。
        """
        if self.backbone is not None:
            features = self.backbone(crop_image)
            if isinstance(features, dict):
                for value in features.values():
                    if isinstance(value, torch.Tensor) and value.dim() == 4:
                        return value
            if isinstance(features, torch.Tensor) and features.dim() == 4:
                return features
        return self.proj(self.stem(crop_image))

    def forward(
        self,
        crop_image: torch.Tensor,
        crop_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """将裁剪图像编码为多类型嵌入。

        参数：
            - crop_image: 裁剪后的图像张量，形状为 (B, 3, H, W)。
            - crop_mask: 可选的裁剪掩码张量；为 None 时各类嵌入退化为全局嵌入。

        返回：
            - 包含 global_embedding、foreground_embedding、boundary_embedding、
              context_embedding 四类嵌入的字典。
        """
        features = self._encode_feature_map(crop_image)
        global_embedding = _l2_normalize(features.mean(dim=(-2, -1)))

        if crop_mask is not None:
            if crop_mask.shape[-2:] != features.shape[-2:]:
                crop_mask = F.interpolate(crop_mask.float(), size=features.shape[-2:], mode="nearest")
            foreground_mask = crop_mask.float().clamp(0, 1)
            background_mask = (1.0 - foreground_mask).clamp(0, 1)
            boundary_mask = _boundary_band(foreground_mask)
            foreground_embedding = _l2_normalize(_masked_average_pool(features, foreground_mask))
            boundary_embedding = _l2_normalize(_masked_average_pool(features, boundary_mask))
            context_embedding = _l2_normalize(_masked_average_pool(features, background_mask))
        else:
            foreground_embedding = global_embedding
            boundary_embedding = global_embedding
            context_embedding = global_embedding

        return {
            "global_embedding": global_embedding,
            "foreground_embedding": foreground_embedding,
            "boundary_embedding": boundary_embedding,
            "context_embedding": context_embedding,
        }
