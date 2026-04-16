"""
评估指标
提供 Dice Coefficient、IoU、Precision、Recall 等医学图像分割评估指标。
"""

import torch
import numpy as np
from typing import Dict


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> torch.Tensor:
    """
    计算 Dice 系数
    Args:
        pred: 预测 mask (B, H, W) 或 (B, 1, H, W)，值为 0/1
        target: 真实 mask，同 shape
        smooth: 平滑因子，避免除零
    Returns:
        per-sample Dice 系数 (B,)
    """
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    return (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """
    计算 Intersection over Union (IoU / Jaccard Index)
    """
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred: torch.Tensor, target: torch.Tensor,
                    smooth: float = 1e-6) -> torch.Tensor:
    """计算精确率"""
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred: torch.Tensor, target: torch.Tensor,
                 smooth: float = 1e-6) -> torch.Tensor:
    """计算召回率"""
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    return (tp + smooth) / (tp + fn + smooth)


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
    """
    计算所有指标，返回字典
    Args:
        pred: 模型输出的 logits 或概率图 (B, 1, H, W) 或 (B, H, W)
        target: 二值 mask
        threshold: 二值化阈值
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # 二值化
    pred_binary = (pred.sigmoid() > threshold).float() if pred.min() < 0 else (pred > threshold).float()
    target_binary = target.float()

    dice = dice_coefficient(pred_binary, target_binary).mean().item()
    iou = iou_score(pred_binary, target_binary).mean().item()
    prec = precision_score(pred_binary, target_binary).mean().item()
    rec = recall_score(pred_binary, target_binary).mean().item()

    return {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
    }
