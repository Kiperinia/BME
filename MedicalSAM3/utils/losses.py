"""
损失函数
提供 Dice Loss、Focal Loss 及其组合，用于医学图像分割训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss，用于医学图像分割的区域相似性度量。

    参数：
        - smooth: 平滑因子，防止除零
    """
    def __init__(self, smooth: float = 1.0):
        """初始化 Dice Loss。

        参数：
            - smooth: 平滑因子，默认 1.0
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Dice Loss。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 标量损失值
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
    """Focal Loss，聚焦于难分类样本的损失函数。

    参数：
        - alpha: 正负样本平衡因子
        - gamma: 难易样本聚焦参数
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """初始化 Focal Loss。

        参数：
            - alpha: 平衡因子，默认 0.25
            - gamma: 聚焦参数，默认 2.0
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Focal Loss。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 标量损失值
        """
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """边界损失，通过 Sobel 算子对边界区域赋予更高权重。

    参数：
        - weight: 边界区域的额外权重系数
    """
    def __init__(self, weight: float = 1.0):
        """初始化边界损失。

        参数：
            - weight: 边界权重系数，默认 1.0
        """
        super().__init__()
        self.weight = weight
        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """使用 Sobel 算子提取掩码的边界。

        参数：
            - mask: 二值掩码张量

        返回：
            - 边界概率图 (0~1)
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        gx = F.conv2d(mask.float(), self.sobel_x, padding=1)
        gy = F.conv2d(mask.float(), self.sobel_y, padding=1)
        boundary = (gx.abs() + gy.abs()).clamp(0, 1)
        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边界加权损失。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 标量损失值
        """
        boundary = self._get_boundary(target)
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        # 边界区域加权
        weighted = bce * (1.0 + self.weight * boundary)
        return weighted.mean()


class CombinedSegLoss(nn.Module):
    """组合分割损失，加权融合 Dice Loss、Focal Loss 和 BCE Loss。

    参数：
        - dice_weight: Dice Loss 权重
        - focal_weight: Focal Loss 权重
        - bce_weight: BCE Loss 权重
    """
    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 bce_weight: float = 1.0):
        """初始化组合分割损失。

        参数：
            - dice_weight: Dice 损失权重，默认 1.0
            - focal_weight: Focal 损失权重，默认 1.0
            - bce_weight: BCE 损失权重，默认 1.0
        """
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算组合损失。

        按配置的权重加权求和 Dice Loss、Focal Loss 和 BCE Loss。

        参数：
            - pred: 预测 logits 张量
            - target: 二值目标掩码张量

        返回：
            - 加权组合后的总损失值
        """
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
