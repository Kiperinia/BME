"""Multi-Scale Feature Adapter 扩展模块。

为轻量特征支路提供多尺度上下文聚合和通道重标定能力。
"""

import torch
import torch.nn as nn


class MultiScaleFeatureAdapter(nn.Module):
    """
    Multi-Scale Feature Adapter (MSFA)

    对编码器输出的特征图进行多尺度空洞卷积 + 通道注意力融合。
    类似 ASPP 但更轻量化，适合嵌入到 SAM 架构中。
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 dilations: tuple = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(dilations),
                          kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels // len(dilations)),
                nn.GELU(),
            ))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // len(dilations),
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // len(dilations)),
            nn.GELU(),
        )

        fused_ch = out_channels // len(dilations) * (len(dilations) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """融合多尺度分支特征，并通过残差形式返回增强结果。"""

        residual = x
        branch_outs = [branch(x) for branch in self.branches]

        gp = self.global_pool(x)
        gp = gp.expand(-1, -1, x.shape[2], x.shape[3])
        branch_outs.append(gp)

        fused = self.fuse(torch.cat(branch_outs, dim=1))
        attn = self.channel_attn(fused).unsqueeze(-1).unsqueeze(-1)
        fused = fused * attn

        return fused + residual