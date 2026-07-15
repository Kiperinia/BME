"""
创新训练策略 — 难例挖掘损失 (Online Hard Example Mining Loss)

设计动机:
  医学图像分割中，大部分像素属于 "简单" 背景区域。
  OHEM 策略筛选出模型预测最困难的像素，仅对这些区域计算损失，
  迫使网络更关注难以分辨的边界区域和小目标。

实现:
  1. 计算所有像素的 loss map
  2. 按 loss 值排序，取 Top-K% 最困难的像素
  3. 仅用这些像素的 loss 做反传
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):
    """难例挖掘损失，结合 Focal Loss 和 Dice Loss 并聚焦于困难像素。

    通过筛选损失最高的 Top-K 像素进行反传，迫使网络关注难以分辨的边界和小目标。

    参数：
        - hard_ratio: 难例像素保留比例
        - min_kept: 最少保留的像素数
        - focal_alpha: Focal Loss 的 alpha 平衡因子
        - focal_gamma: Focal Loss 的 gamma 聚焦参数
        - dice_weight: Dice Loss 的权重系数
        - focal_weight: Focal Loss 的权重系数
    """

    def __init__(
        self,
        hard_ratio: float = 0.3,
        min_kept: int = 1000,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
    ):
        """初始化难例挖掘损失模块。

        参数：
            - hard_ratio: 难例保留比例，默认 0.3
            - min_kept: 最少保留像素数，默认 1000
            - focal_alpha: Focal Loss alpha，默认 0.25
            - focal_gamma: Focal Loss gamma，默认 2.0
            - dice_weight: Dice 损失权重，默认 0.5
            - focal_weight: Focal 损失权重，默认 0.5
        """
        super().__init__()
        self.hard_ratio = hard_ratio
        self.min_kept = min_kept
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def _pixel_focal_loss(self, pred: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """计算逐像素的 Focal Loss，不进行归约。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 与输入形状相同的逐像素损失图
        """
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        p_t = torch.exp(-bce)
        return self.focal_alpha * (1 - p_t) ** self.focal_gamma * bce

    def _hard_mining(self, loss_map: torch.Tensor) -> torch.Tensor:
        """对损失图进行难例挖掘，取 Top-K 最高损失像素的均值。

        参数：
            - loss_map: 逐像素损失图

        返回：
            - 难例像素的平均损失值
        """
        B = loss_map.shape[0]
        flat = loss_map.flatten(1)  # (B, N)
        n_pixels = flat.shape[1]
        n_keep = max(int(n_pixels * self.hard_ratio), self.min_kept)
        n_keep = min(n_keep, n_pixels)

        # 对每个样本取 Top-K
        topk_loss, _ = flat.topk(n_keep, dim=1)
        return topk_loss.mean()

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Dice Loss。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 标量 Dice 损失值
        """
        pred_sig = pred.sigmoid().flatten(1)
        target_flat = target.float().flatten(1)
        inter = (pred_sig * target_flat).sum(dim=1)
        dice = (2.0 * inter + 1.0) / (pred_sig.sum(1) + target_flat.sum(1) + 1.0)
        return (1.0 - dice).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 OHEM 损失，组合难例挖掘后的 Focal Loss 和全局 Dice Loss。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 加权组合后的总损失值
        """
        # 逐像素 Focal loss
        focal_map = self._pixel_focal_loss(pred, target)
        # 难例挖掘
        hard_focal = self._hard_mining(focal_map)

        # Dice loss (全局)
        dice_loss = self._dice_loss(pred, target)

        return self.dice_weight * dice_loss + self.focal_weight * hard_focal
