"""Medical feature adapters for MedEx-SAM3."""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
        scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + self.scale * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self._forward_sequence(x)
        if x.dim() == 4:
            x_perm = x.permute(0, 2, 3, 1)
            out = self._forward_sequence(x_perm)
            return out.permute(0, 3, 1, 2)
        raise ValueError("BottleneckAdapter only supports [B, N, C] or [B, C, H, W]")


class MedicalImageAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
        scale_init: float = 1e-3,
        use_depthwise_conv: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = BottleneckAdapter(dim, bottleneck_dim, dropout=dropout, scale_init=scale_init)
        self.use_depthwise_conv = use_depthwise_conv
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) if use_depthwise_conv else None
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1) if use_depthwise_conv else None
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.adapter(x)
        if x.dim() == 4 and self.use_depthwise_conv and self.depthwise is not None and self.pointwise is not None:
            texture = self.pointwise(self.activation(self.depthwise(x)))
            out = out + texture
        return out


class MultiScaleMedicalAdapter(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 6, 12, 18)) -> None:
        super().__init__()
        branch_channels = max(channels // max(len(dilations), 1), 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.GELU(),
                )
                for dilation in dilations
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )
        fused_channels = branch_channels * (len(dilations) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("MultiScaleMedicalAdapter expects [B, C, H, W]")
        residual = x
        features = [branch(x) for branch in self.branches]
        pooled = self.global_pool(x).expand(-1, -1, x.shape[-2], x.shape[-1])
        features.append(pooled)
        return residual + self.fuse(torch.cat(features, dim=1))
