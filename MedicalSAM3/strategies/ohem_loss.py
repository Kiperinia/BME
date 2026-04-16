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
    """
    Online Hard Example Mining (OHEM) Loss

    组合 Dice 和 Focal 损失，并在像素级别做难例挖掘。
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
        """
        Args:
            hard_ratio: 保留最困难像素的比例
            min_kept: 至少保留的像素数
            focal_alpha: Focal Loss α
            focal_gamma: Focal Loss γ
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
        """逐像素 Focal Loss (不 reduce)"""
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        p_t = torch.exp(-bce)
        return self.focal_alpha * (1 - p_t) ** self.focal_gamma * bce

    def _hard_mining(self, loss_map: torch.Tensor) -> torch.Tensor:
        """
        从 loss_map 中选择最困难的像素
        Args:
            loss_map: (B, 1, H, W) 逐像素损失
        Returns:
            标量损失 (仅含困难像素)
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
        pred_sig = pred.sigmoid().flatten(1)
        target_flat = target.float().flatten(1)
        inter = (pred_sig * target_flat).sum(dim=1)
        dice = (2.0 * inter + 1.0) / (pred_sig.sum(1) + target_flat.sum(1) + 1.0)
        return (1.0 - dice).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary mask
        """
        # 逐像素 Focal loss
        focal_map = self._pixel_focal_loss(pred, target)
        # 难例挖掘
        hard_focal = self._hard_mining(focal_map)

        # Dice loss (全局)
        dice_loss = self._dice_loss(pred, target)

        return self.dice_weight * dice_loss + self.focal_weight * hard_focal
