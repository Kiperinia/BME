"""
Medical SAM3 — SAM3 包装器

将 SAM3 模型封装为统一的 forward 接口，供 medsam3_base 使用。
依赖: sam3 包必须可用 (Linux / macOS 环境)。
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor

logger = logging.getLogger(__name__)


def _normalize_bbox_xywh(bbox_xywh: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Normalize an XYWH bbox to [0, 1] coordinates."""
    normalized_bbox = bbox_xywh.clone()
    normalized_bbox[..., 0] /= img_w
    normalized_bbox[..., 1] /= img_h
    normalized_bbox[..., 2] /= img_w
    normalized_bbox[..., 3] /= img_h
    return normalized_bbox


class MedSAM3Wrapper(nn.Module):
    """
    基于 SAM3 的 Medical SAM3 包装器。
    加载 MedSAM3 checkpoint 并提供统一的 forward 接口。
    参考: https://github.com/AIM-Research-Lab/Medical-SAM3
    """

    def __init__(self, sam3_model: Any, confidence_threshold: float = 0.1):
        super().__init__()
        self.sam_model = sam3_model
        model_device = getattr(next(sam3_model.parameters(), None), "device", torch.device("cpu"))
        self.processor = Sam3Processor(
            sam3_model,
            device=str(model_device),
            confidence_threshold=confidence_threshold,
        )

    def load_custom_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载自定义 checkpoint，兼容 SAM3 / MedSAM3 格式。

        SAM3 格式: key 带 'detector.' 前缀。
        MedSAM3 格式: key 无 'detector.' 前缀。
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        sample_key = next(iter(state_dict), "")
        if "detector." in sample_key:
            state_dict = {
                k.replace("detector.", ""): v
                for k, v in state_dict.items()
                if "detector" in k
            }

        missing, unexpected = self.sam_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"Checkpoint missing keys: {len(missing)}")
        if unexpected:
            logger.info(f"Checkpoint unexpected keys: {len(unexpected)}")

    def forward(
        self,
        images: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images:       (B, 3, H, W) 且值范围在 [0, 1] 的 RGB 图像
            bboxes:       (B, 4) bounding boxes [x1, y1, x2, y2]
            points:       (B, N, 2) point prompts
            point_labels: (B, N) point labels (1=foreground, 0=background)
        Returns:
            dict: masks (B, 1, H, W), iou_predictions (B, 1)
        """
        batch_masks = []
        batch_scores = []

        for i in range(images.shape[0]):
            img = images[i]  # (3, H, W)
            img_h, img_w = img.shape[1], img.shape[2]

            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)
            inference_state = self.processor.set_image(pil_img)

            result: Dict[str, Any] = {"masks": None, "scores": None}

            if bboxes is not None:
                self.processor.reset_all_prompts(inference_state)
                x1, y1, x2, y2 = bboxes[i].cpu().float().tolist()
                w, h = x2 - x1, y2 - y1
                box_xywh = torch.tensor([x1, y1, w, h], dtype=torch.float32).view(1, 4)
                box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                norm_box = _normalize_bbox_xywh(box_cxcywh, img_w, img_h).flatten().tolist()
                result = self.processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=True
                )

            if result["masks"] is not None and len(result["masks"]) > 0:
                best_idx = torch.argmax(result["scores"]).item()
                mask = result["masks"][best_idx].float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                score = result["scores"][best_idx].float().unsqueeze(0)
                batch_masks.append(mask)
                batch_scores.append(score)
            else:
                batch_masks.append(torch.zeros(1, img_h, img_w))
                batch_scores.append(torch.tensor([0.0]))

        return {
            "masks": torch.stack(batch_masks).to(images.device),
            "iou_predictions": torch.stack(batch_scores).to(images.device),
        }
