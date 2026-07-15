"""
评估指标
提供 Dice Coefficient、IoU、Precision、Recall 等医学图像分割评估指标。
"""

import torch
import numpy as np
from typing import Dict


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> torch.Tensor:
    """计算 Dice 系数，衡量预测与目标的重叠程度。

    参数：
        - pred: 预测掩码张量
        - target: 目标掩码张量
        - smooth: 平滑因子，默认 1e-6

    返回：
        - 每个样本的 Dice 系数张量
    """
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    return (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """计算 IoU（交并比）分数。

    参数：
        - pred: 预测掩码张量
        - target: 目标掩码张量
        - smooth: 平滑因子，默认 1e-6

    返回：
        - 每个样本的 IoU 张量
    """
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred: torch.Tensor, target: torch.Tensor,
                    smooth: float = 1e-6) -> torch.Tensor:
    """计算精确率（Precision）。

    参数：
        - pred: 预测掩码张量
        - target: 目标掩码张量
        - smooth: 平滑因子，默认 1e-6

    返回：
        - 每个样本的精确率张量
    """
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred: torch.Tensor, target: torch.Tensor,
                 smooth: float = 1e-6) -> torch.Tensor:
    """计算召回率（Recall）。

    参数：
        - pred: 预测掩码张量
        - target: 目标掩码张量
        - smooth: 平滑因子，默认 1e-6

    返回：
        - 每个样本的召回率张量
    """
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    return (tp + smooth) / (tp + fn + smooth)


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
    """计算 Dice、IoU、精确率和召回率所有指标。

    参数：
        - pred: 预测掩码张量（logits 或二值）
        - target: 目标掩码张量
        - threshold: 二值化阈值，默认 0.5

    返回：
        - 包含 dice、iou、precision、recall 的字典
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
