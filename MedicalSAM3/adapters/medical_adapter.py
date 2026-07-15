"""MedEx-SAM3 的医学图像特征适配器。

提供基于瓶颈结构的特征适配模块，支持序列与图像两种输入形态，
并可选叠加深度可分离卷积以增强纹理表达。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """瓶颈结构特征适配器，通过降维-升维残差路径调整特征。

    参数：
        - dim: 输入特征维度
        - bottleneck_dim: 瓶颈维度
        - dropout: dropout 概率
        - scale_init: 残差路径初始缩放值
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
        scale_init: float = 1e-3,
    ) -> None:
        """初始化瓶颈适配器的各子层与可学习缩放因子。

        参数：
            - dim: 输入特征维度
            - bottleneck_dim: 瓶颈维度
            - dropout: dropout 概率
            - scale_init: 残差路径初始缩放值

        返回：
            - 无返回值，完成各子层的构建
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """对序列形态特征执行瓶颈残差前向计算。

        参数：
            - x: 形如 [B, N, C] 的序列特征张量

        返回：
            - 残差融合后的序列特征张量
        """
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + self.scale * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算，自动适配序列或图像输入形态。

        参数：
            - x: 输入张量，支持 [B, N, C] 或 [B, C, H, W]

        返回：
            - 适配后的特征张量，形态与输入一致
        """
        if x.dim() == 3:
            return self._forward_sequence(x)
        if x.dim() == 4:
            x_perm = x.permute(0, 2, 3, 1)
            out = self._forward_sequence(x_perm)
            return out.permute(0, 3, 1, 2)
        raise ValueError("BottleneckAdapter only supports [B, N, C] or [B, C, H, W]")


class MedicalImageAdapter(nn.Module):
    """医学图像特征适配器，组合瓶颈适配与可选的深度可分离卷积分支。

    参数：
        - dim: 输入特征维度
        - bottleneck_dim: 瓶颈维度
        - dropout: dropout 概率
        - scale_init: 残差路径初始缩放值
        - use_depthwise_conv: 是否启用深度可分离卷积纹理分支
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
        scale_init: float = 1e-3,
        use_depthwise_conv: bool = True,
    ) -> None:
        """初始化医学图像适配器的瓶颈子模块与卷积分支。

        参数：
            - dim: 输入特征维度
            - bottleneck_dim: 瓶颈维度
            - dropout: dropout 概率
            - scale_init: 残差路径初始缩放值
            - use_depthwise_conv: 是否启用深度可分离卷积纹理分支

        返回：
            - 无返回值，完成各子层的构建
        """
        super().__init__()
        self.adapter = BottleneckAdapter(dim, bottleneck_dim, dropout=dropout, scale_init=scale_init)
        self.use_depthwise_conv = use_depthwise_conv
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) if use_depthwise_conv else None
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1) if use_depthwise_conv else None
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算：瓶颈适配输出叠加可选的卷积纹理特征。

        参数：
            - x: 输入张量，支持 [B, N, C] 或 [B, C, H, W]

        返回：
            - 适配后的特征张量
        """
        out = self.adapter(x)
        if x.dim() == 4 and self.use_depthwise_conv and self.depthwise is not None and self.pointwise is not None:
            texture = self.pointwise(self.activation(self.depthwise(x)))
            out = out + texture
        return out
