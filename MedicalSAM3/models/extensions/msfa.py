"""Multi-Scale Feature Adapter 扩展模块。

为轻量特征支路提供多尺度上下文聚合和通道重标定能力。
"""

import torch
import torch.nn as nn


class MultiScaleFeatureAdapter(nn.Module):
    """多尺度特征适配器，聚合多膨胀率分支与全局上下文并做通道重标定。

    参数：
        - in_channels: 输入特征通道数。
        - out_channels: 输出特征通道数。
        - dilations: 各膨胀分支的膨胀率元组。

    返回：
        - 构建可用的多尺度特征适配模块实例。
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 dilations: tuple = (1, 6, 12, 18)):
        """初始化多尺度膨胀分支、全局池化分支、融合层与通道注意力。

        参数：
            - in_channels: 输入特征通道数。
            - out_channels: 输出特征通道数。
            - dilations: 各膨胀分支的膨胀率元组。

        返回：
            - 无返回值，完成各子模块的构建。
        """
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
        """对输入特征做多尺度聚合、通道注意力加权并叠加残差。

        参数：
            - x: 形状为 [B, C, H, W] 的输入特征张量。

        返回：
            - 与输入同形状的融合后特征张量。
        """

        residual = x
        branch_outs = [branch(x) for branch in self.branches]

        gp = self.global_pool(x)
        gp = gp.expand(-1, -1, x.shape[2], x.shape[3])
        branch_outs.append(gp)

        fused = self.fuse(torch.cat(branch_outs, dim=1))
        attn = self.channel_attn(fused).unsqueeze(-1).unsqueeze(-1)
        fused = fused * attn

        return fused + residual
