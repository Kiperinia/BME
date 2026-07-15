"""MedEx-SAM3 的示例感知损失函数集合。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算两个（可软化的）掩码之间的 Soft Dice 系数。

    参数：
        - mask_a: 第一个掩码张量，形状为 (B, H, W) 或 (B, N)。
        - mask_b: 第二个掩码张量，形状与 mask_a 相同。
        - eps: 用于数值稳定的小常数，防止除零。

    返回：
        - 每个样本的 Dice 系数张量，形状为 (B,)。
    """
    a = mask_a.flatten(1)
    b = mask_b.flatten(1)
    inter = (a * b).sum(dim=1)
    union = a.sum(dim=1) + b.sum(dim=1)
    return (2.0 * inter + eps) / (union + eps)


def _boundary_band(mask: torch.Tensor) -> torch.Tensor:
    """提取二值/软掩码的边界带区域。

    通过最大池化实现膨胀，通过对负掩码做最大池化实现腐蚀，
    二者之差即为边界带。

    参数：
        - mask: 输入掩码张量，形状为 (B, H, W)。

    返回：
        - 边界带掩码张量，取值范围 [0, 1]，形状与输入相同。
    """
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


class ExemplarInfoNCELoss(nn.Module):
    """基于 InfoNCE 的示例对比损失。

    将锚点嵌入拉近正例嵌入、推远若干负例嵌入，用于学习具有区分性的示例表示。

    参数：
        - temperature: InfoNCE 中的温度系数，控制分布的尖锐程度。
    """
    def __init__(self, temperature: float = 0.07) -> None:
        """初始化示例对比损失模块。

        参数：
            - temperature: InfoNCE 中的温度系数。

        返回：
            - 无返回值，仅完成模块初始化。
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算示例对比损失。

        参数：
            - anchor_embedding: 锚点嵌入，形状为 (B, C)。
            - positive_embedding: 正例嵌入，形状为 (B, C)。
            - negative_embeddings: 负例嵌入，形状为 (B, N, C) 或 (N, C)。

        返回：
            - 对比损失标量张量。
        """
        anchor = F.normalize(anchor_embedding, dim=-1)
        positive = F.normalize(positive_embedding, dim=-1)
        negatives = F.normalize(negative_embeddings, dim=-1)
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(0).expand(anchor.shape[0], -1, -1)

        pos_logits = (anchor * positive).sum(dim=-1, keepdim=True) / self.temperature
        neg_logits = torch.einsum("bc,bnc->bn", anchor, negatives) / self.temperature
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class NegativeSuppressionLoss(nn.Module):
    """负例提示抑制损失。

    鼓励负例提示对应的掩码预测概率趋近于 0，从而抑制假阳性响应。
    """
    def forward(self, negative_prompt_mask_logits: torch.Tensor) -> torch.Tensor:
        """计算负例提示掩码的抑制损失。

        参数：
            - negative_prompt_mask_logits: 负例提示对应的掩码 logits 张量。

        返回：
            - 负例掩码 sigmoid 均值损失标量张量。
        """
        return torch.sigmoid(negative_prompt_mask_logits).mean()


class CrossDomainConsistencyLoss(nn.Module):
    """跨域一致性损失。

    通过余弦相似度约束锚点嵌入与原型嵌入的方向一致性，提升跨域泛化能力。
    """
    def forward(self, anchor_embedding: torch.Tensor, prototype_embedding: torch.Tensor) -> torch.Tensor:
        """计算跨域一致性损失。

        参数：
            - anchor_embedding: 锚点嵌入，形状为 (B, C)。
            - prototype_embedding: 原型嵌入，形状为 (B, C)。

        返回：
            - 一致性损失标量张量（1 - 余弦相似度均值）。
        """
        anchor = F.normalize(anchor_embedding, dim=-1)
        prototype = F.normalize(prototype_embedding, dim=-1)
        return 1.0 - (anchor * prototype).sum(dim=-1).mean()


class ExemplarConsistencyLoss(nn.Module):
    """示例一致性损失。

    基于 Soft Dice 衡量两份掩码预测之间的一致性，鼓励同一目标在不同视角/扰动下输出一致结果。
    """
    def forward(self, mask_logits_a: torch.Tensor, mask_logits_b: torch.Tensor) -> torch.Tensor:
        """计算示例一致性损失。

        参数：
            - mask_logits_a: 第一份掩码 logits 张量。
            - mask_logits_b: 第二份掩码 logits 张量。

        返回：
            - 一致性损失标量张量（1 - Soft Dice 均值）。
        """
        return 1.0 - _soft_dice(torch.sigmoid(mask_logits_a), torch.sigmoid(mask_logits_b)).mean()


class PrototypeVarianceLoss(nn.Module):
    """原型方差损失。

    约束样本嵌入到原型的距离均值不超过给定裕度（margin），从而控制原型表示的紧致性。

    参数：
        - margin: 允许的距离裕度，超过该裕度才会产生损失。
    """
    def __init__(self, margin: float = 0.1) -> None:
        """初始化原型方差损失模块。

        参数：
            - margin: 距离裕度阈值。

        返回：
            - 无返回值，仅完成模块初始化。
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """计算原型方差损失。

        参数：
            - embeddings: 样本嵌入张量，形状为 (N, C)。
            - prototype: 原型嵌入，形状为 (C,) 或 (M, C)。

        返回：
            - 方差损失标量张量（超过裕度的部分经 ReLU 截断）。
        """
        if prototype.dim() == 1:
            distances = (embeddings - prototype.unsqueeze(0)).pow(2).sum(dim=-1)
        else:
            distances = torch.cdist(embeddings, prototype).min(dim=1).values.pow(2)
        return torch.relu(distances.mean() - self.margin)


class BoundaryBandDiceLoss(nn.Module):
    """边界带 Dice 损失。

    仅在预测与真值的边界带区域上计算 Soft Dice，强化模型对边界的刻画能力。
    """
    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """计算边界带 Dice 损失。

        参数：
            - pred_logits: 预测掩码 logits 张量。
            - target_mask: 真值掩码张量，形状需与 pred_logits 匹配（否则就近插值对齐）。

        返回：
            - 边界带 Dice 损失标量张量（1 - 边界 Soft Dice 均值）。
        """
        if target_mask.shape != pred_logits.shape:
            target_mask = F.interpolate(target_mask.float(), size=pred_logits.shape[-2:], mode="nearest")
        pred_band = _boundary_band(torch.sigmoid(pred_logits))
        target_band = _boundary_band(target_mask.float())
        return 1.0 - _soft_dice(pred_band, target_band).mean()


class SoftHausdorffLoss(nn.Module):
    """软 Hausdorff 损失。

    对预测与真值掩码分别做均值平滑后再计算绝对误差均值，
    作为 Hausdorff 距离的可微近似，用于关注边界整体偏移。
    """
    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """计算软 Hausdorff 损失。

        参数：
            - pred_logits: 预测掩码 logits 张量。
            - target_mask: 真值掩码张量，形状需与预测匹配（否则就近插值对齐）。

        返回：
            - 平滑后掩码的绝对误差均值标量张量。
        """
        pred = torch.sigmoid(pred_logits)
        if target_mask.shape != pred.shape:
            target_mask = F.interpolate(target_mask.float(), size=pred.shape[-2:], mode="nearest")
        pred_smooth = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        target_smooth = F.avg_pool2d(target_mask.float(), kernel_size=5, stride=1, padding=2)
        return (pred_smooth - target_smooth).abs().mean()


class MedExLossComposer(nn.Module):
    """MedEx-SAM3 多项损失组合器。

    将 BCE、Dice、边界、对比、负例抑制、一致性等多项损失按权重加权合成总损失，
    并返回各项明细便于日志记录与监控。

    参数：
        - w_bce: BCE 损失权重。
        - w_dice: Dice 损失权重。
        - w_boundary: 边界带 Dice 损失权重。
        - w_contrast: 示例对比损失权重。
        - w_neg: 负例抑制损失权重。
        - w_consistency: 示例一致性损失权重。
    """
    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_boundary: float = 0.3,
        w_contrast: float = 0.1,
        w_neg: float = 0.1,
        w_consistency: float = 0.05,
    ) -> None:
        """初始化损失组合器。

        参数：
            - w_bce: BCE 损失权重。
            - w_dice: Dice 损失权重。
            - w_boundary: 边界带 Dice 损失权重。
            - w_contrast: 示例对比损失权重。
            - w_neg: 负例抑制损失权重。
            - w_consistency: 示例一致性损失权重。

        返回：
            - 无返回值，仅完成模块初始化。
        """
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.w_contrast = w_contrast
        self.w_neg = w_neg
        self.w_consistency = w_consistency
        self.boundary = BoundaryBandDiceLoss()
        self.contrast = ExemplarInfoNCELoss()
        self.neg = NegativeSuppressionLoss()
        self.consistency = ExemplarConsistencyLoss()

    def forward(
        self,
        mask_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        anchor_embedding: Optional[torch.Tensor] = None,
        positive_embedding: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        negative_prompt_mask_logits: Optional[torch.Tensor] = None,
        consistency_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算加权合成的总损失及各项明细。

        参数：
            - mask_logits: 预测掩码 logits 张量。
            - gt_mask: 真值掩码张量，形状需与 mask_logits 匹配（否则就近插值对齐）。
            - anchor_embedding: 锚点嵌入，用于对比损失；为 None 则不计算对比项。
            - positive_embedding: 正例嵌入，用于对比损失。
            - negative_embeddings: 负例嵌入，用于对比损失。
            - negative_prompt_mask_logits: 负例提示掩码 logits，用于负例抑制损失。
            - consistency_pair: 一致性损失所需的掩码对 (logits_a, logits_b)。

        返回：
            - 总损失标量张量，以及包含各项损失明细的字典。
        """
        if gt_mask.shape != mask_logits.shape:
            gt_mask = F.interpolate(gt_mask.float(), size=mask_logits.shape[-2:], mode="nearest")
        bce = F.binary_cross_entropy_with_logits(mask_logits, gt_mask.float())
        dice = 1.0 - _soft_dice(torch.sigmoid(mask_logits), gt_mask.float()).mean()
        boundary = self.boundary(mask_logits, gt_mask)
        total = self.w_bce * bce + self.w_dice * dice + self.w_boundary * boundary
        aux = {
            "bce": bce,
            "dice": dice,
            "boundary": boundary,
        }
        if anchor_embedding is not None and positive_embedding is not None and negative_embeddings is not None:
            contrast = self.contrast(anchor_embedding, positive_embedding, negative_embeddings)
            aux["contrast"] = contrast
            total = total + self.w_contrast * contrast
        if negative_prompt_mask_logits is not None:
            neg = self.neg(negative_prompt_mask_logits)
            aux["negative"] = neg
            total = total + self.w_neg * neg
        if consistency_pair is not None:
            consistency = self.consistency(*consistency_pair)
            aux["consistency"] = consistency
            total = total + self.w_consistency * consistency
        aux["total"] = total
        return total, aux
