"""Quality and failure heuristics for segmentation outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _boundary_band(mask: torch.Tensor) -> torch.Tensor:
    """计算掩码的边界带状区域（膨胀减腐蚀）。

    参数：
        - mask: 二值掩码张量，形状 (B, 1, H, W)

    返回：
        - 形状相同的边界区域张量
    """
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    return (dilated - eroded).clamp(0, 1)


class QualityEvaluator:
    """分割质量与失败类型评估器。

    支持有真实掩码和无真实掩码两种评估模式，输出多项质量指标和失败类型。
    """
    def evaluate(
        self,
        mask_logits: torch.Tensor,
        mask: torch.Tensor,
        score: torch.Tensor | float,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, float | str]:
        """对分割结果进行质量评估和失败分类。

        无真实掩码时基于置信度和不确定性推断；有真实掩码时计算 Dice、边界质量等指标。

        参数：
            - mask_logits: 模型输出的 logits 张量
            - mask: 二值预测掩码
            - score: 置信度分数（张量或标量）
            - gt_mask: 可选；真实掩码张量

        返回：
            - 包含 mask_quality、boundary_quality、uncertainty、failure_type 等的字典
        """
        prob = torch.sigmoid(mask_logits)
        pred = (prob > 0.5).float()
        uncertainty = float((4.0 * prob * (1.0 - prob)).mean().item())
        score_value = float(score.mean().item()) if isinstance(score, torch.Tensor) else float(score)

        if gt_mask is None:
            fp_risk = float((pred.mean() * (1.0 - score_value)).item())
            fn_risk = float(((1.0 - pred.mean()) * (1.0 - score_value)).item())
            return {
                "mask_quality": score_value,
                "boundary_quality": score_value,
                "uncertainty": uncertainty,
                "fp_risk": fp_risk,
                "fn_risk": fn_risk,
                "failure_type": "uncertain" if uncertainty > 0.25 else "false_positive",
            }

        if gt_mask.shape != pred.shape:
            gt_mask = F.interpolate(gt_mask.float(), size=pred.shape[-2:], mode="nearest")
        intersection = (pred * gt_mask).sum()
        union = pred.sum() + gt_mask.sum()
        dice = float(((2.0 * intersection + 1e-6) / (union + 1e-6)).item())
        pred_boundary = _boundary_band(pred)
        gt_boundary = _boundary_band(gt_mask.float())
        boundary_intersection = (pred_boundary * gt_boundary).sum()
        boundary_union = pred_boundary.sum() + gt_boundary.sum()
        boundary_quality = float(((2.0 * boundary_intersection + 1e-6) / (boundary_union + 1e-6)).item())

        false_positive = float((((pred == 1) & (gt_mask == 0)).float().mean()).item())
        false_negative = float((((pred == 0) & (gt_mask == 1)).float().mean()).item())
        if uncertainty > 0.25:
            failure_type = "uncertain"
        elif false_positive > 0.1 and false_negative < 0.05:
            failure_type = "over_segmentation"
        elif false_negative > 0.1 and false_positive < 0.05:
            failure_type = "under_segmentation"
        elif boundary_quality < 0.5:
            failure_type = "boundary_leak"
        elif pred.sum() == 0 and gt_mask.sum() > 0:
            failure_type = "false_negative"
        elif pred.sum() > 0 and gt_mask.sum() == 0:
            failure_type = "false_positive"
        else:
            failure_type = "uncertain"

        return {
            "mask_quality": dice,
            "boundary_quality": boundary_quality,
            "uncertainty": uncertainty,
            "fp_risk": false_positive,
            "fn_risk": false_negative,
            "failure_type": failure_type,
        }
