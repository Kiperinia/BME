"""Exemplar-aware losses for MedEx-SAM3."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = mask_a.flatten(1)
    b = mask_b.flatten(1)
    inter = (a * b).sum(dim=1)
    union = a.sum(dim=1) + b.sum(dim=1)
    return (2.0 * inter + eps) / (union + eps)


def _boundary_band(mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    return (dilated - eroded).clamp(0, 1)


class ExemplarInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
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
    def forward(self, negative_prompt_mask_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(negative_prompt_mask_logits).mean()


class ExemplarConsistencyLoss(nn.Module):
    def forward(self, mask_logits_a: torch.Tensor, mask_logits_b: torch.Tensor) -> torch.Tensor:
        return 1.0 - _soft_dice(torch.sigmoid(mask_logits_a), torch.sigmoid(mask_logits_b)).mean()


class PrototypeVarianceLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        if prototype.dim() == 1:
            distances = (embeddings - prototype.unsqueeze(0)).pow(2).sum(dim=-1)
        else:
            distances = torch.cdist(embeddings, prototype).min(dim=1).values.pow(2)
        return torch.relu(distances.mean() - self.margin)


class BoundaryBandDiceLoss(nn.Module):
    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        if target_mask.shape != pred_logits.shape:
            target_mask = F.interpolate(target_mask.float(), size=pred_logits.shape[-2:], mode="nearest")
        pred_band = _boundary_band(torch.sigmoid(pred_logits))
        target_band = _boundary_band(target_mask.float())
        return 1.0 - _soft_dice(pred_band, target_band).mean()


class SoftHausdorffLoss(nn.Module):
    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        if target_mask.shape != pred.shape:
            target_mask = F.interpolate(target_mask.float(), size=pred.shape[-2:], mode="nearest")
        pred_smooth = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        target_smooth = F.avg_pool2d(target_mask.float(), kernel_size=5, stride=1, padding=2)
        return (pred_smooth - target_smooth).abs().mean()


class MedExLossComposer(nn.Module):
    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_boundary: float = 0.3,
        w_contrast: float = 0.1,
        w_neg: float = 0.1,
        w_consistency: float = 0.05,
    ) -> None:
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
