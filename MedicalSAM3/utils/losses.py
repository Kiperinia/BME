"""
损失函数
提供 Dice Loss、Focal Loss 及其组合，用于医学图像分割训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - Dice Coefficient
    适用于前景/背景严重不平衡的医学分割任务。
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: logits (B, 1, H, W)
            target: binary mask (B, 1, H, W)
        """
        pred = pred.sigmoid()
        pred = pred.flatten(1)
        target = target.float().flatten(1)
        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)
    对难分类样本赋予更大权重。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: logits (B, 1, H, W)
            target: binary mask (B, 1, H, W)
        """
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """
    边界损失：通过 Sobel 算子提取 mask 边界区域，加权 BCE 损失。
    促进模型对分割边界的精度提升。
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """提取二值 mask 的边界区域"""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        gx = F.conv2d(mask.float(), self.sobel_x, padding=1)
        gy = F.conv2d(mask.float(), self.sobel_y, padding=1)
        boundary = (gx.abs() + gy.abs()).clamp(0, 1)
        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        boundary = self._get_boundary(target)
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        # 边界区域加权
        weighted = bce * (1.0 + self.weight * boundary)
        return weighted.mean()


class CombinedSegLoss(nn.Module):
    """
    组合损失: w_dice * DiceLoss + w_focal * FocalLoss + w_bce * BCELoss
    """
    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 bce_weight: float = 1.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice_loss(pred, target)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal_loss(pred, target)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(
                pred, target.float()
            )
        return loss
