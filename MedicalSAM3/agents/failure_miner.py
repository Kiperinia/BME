"""Mine exemplar candidates from failed segmentation cases."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .quality_evaluator import QualityEvaluator


def _bbox_from_mask(mask: torch.Tensor) -> list[int]:
    """从二值掩码张量中计算边界框坐标。

    参数：
        - mask: 形状为 (H, W) 的二值掩码张量

    返回：
        - [x1, y1, x2, y2] 边界框坐标列表；若掩码为空则返回整图范围
    """
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return [0, 0, mask.shape[-1], mask.shape[-2]]
    y1, x1 = coords[:, -2].min().item(), coords[:, -1].min().item()
    y2, x2 = coords[:, -2].max().item() + 1, coords[:, -1].max().item() + 1
    return [x1, y1, x2, y2]


class FailureMiner:
    """从失败分割案例中挖掘范例候选。

    自动保存裁剪后的失败区域图像，并由质量评估器分析失败类型。
    """
    def __init__(self, output_dir: str | Path) -> None:
        """初始化失败挖掘器。

        参数：
            - output_dir: 输出目录路径，用于保存裁剪图像

        返回：
            - None
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = QualityEvaluator()

    def mine(
        self,
        image: torch.Tensor,
        mask_logits: torch.Tensor,
        score: torch.Tensor,
        image_id: str,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> list[dict[str, object]]:
        """对单张图像的分割结果进行失败挖掘。

        根据预测掩码与真实掩码计算质量指标，裁剪 ROI 区域并保存，返回结构化范例候选列表。

        参数：
            - image: 输入图像张量，形状 (1, C, H, W)
            - mask_logits: 模型输出的 logits 掩码
            - score: 置信度分数
            - image_id: 图像标识符
            - gt_mask: 可选的真实掩码

        返回：
            - 范例候选字典列表，包含 image_id、裁剪路径、边界框、类型等信息
        """
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float()
        quality = self.evaluator.evaluate(mask_logits=mask_logits, mask=pred_mask, score=score, gt_mask=gt_mask)
        if pred_mask.shape[-2:] != image.shape[-2:]:
            pred_mask = F.interpolate(pred_mask, size=image.shape[-2:], mode="nearest")

        bbox = _bbox_from_mask(pred_mask[0, 0])
        x1, y1, x2, y2 = bbox
        crop = image[0, :, y1:y2, x1:x2].detach().cpu().permute(1, 2, 0).numpy()
        crop = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
        crop_path = self.output_dir / f"{image_id}_{quality['failure_type']}.png"
        Image.fromarray(crop).save(crop_path)

        if gt_mask is not None and gt_mask.sum() > 0:
            exemplar_type = "boundary" if quality["failure_type"] == "boundary_leak" else "positive"
        else:
            exemplar_type = "negative"

        return [
            {
                "image_id": image_id,
                "crop_path": str(crop_path),
                "bbox": bbox,
                "type": exemplar_type,
                "failure_type": quality["failure_type"],
                "requires_human_review": gt_mask is None,
            }
        ]
