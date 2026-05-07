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
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return [0, 0, mask.shape[-1], mask.shape[-2]]
    y1, x1 = coords[:, -2].min().item(), coords[:, -1].min().item()
    y2, x2 = coords[:, -2].max().item() + 1, coords[:, -1].max().item() + 1
    return [x1, y1, x2, y2]


class FailureMiner:
    def __init__(self, output_dir: str | Path) -> None:
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
