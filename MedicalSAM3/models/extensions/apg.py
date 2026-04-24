from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePromptGenerator(nn.Module):
    """
    Adaptive Prompt Generator (APG)

    从图像特征自动推断候选 bounding box 和关键点，减少对外部 prompt 的依赖。
    """

    def __init__(self, in_channels: int = 256, num_queries: int = 8,
                 embed_dim: int = 256):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.rpn_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cls_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bbox_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.point_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor,
                gt_bbox: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        B, C, H, W = features.shape
        rpn_feat = self.rpn_conv(features)

        cls_map = self.cls_head(rpn_feat).sigmoid()
        bbox_map = self.bbox_head(rpn_feat).sigmoid()

        flat_cls = cls_map.flatten(2)
        soft_weights = F.softmax(flat_cls / self.temperature.clamp(min=0.01), dim=-1)

        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, H, device=features.device),
            torch.linspace(0, 1, W, device=features.device),
            indexing="ij",
        )
        coords = torch.stack([gx, gy], dim=-1).flatten(0, 1)
        centroid = (soft_weights @ coords.unsqueeze(0).expand(B, -1, -1)).squeeze(1)

        bbox_params = (
            soft_weights.unsqueeze(-1)
            * bbox_map.flatten(2).permute(0, 2, 1).unsqueeze(1)
        ).sum(2).squeeze(1)
        cx, cy = centroid[:, 0], centroid[:, 1]
        bw = bbox_params[:, 2].clamp(0.05, 0.95)
        bh = bbox_params[:, 3].clamp(0.05, 0.95)
        pred_bbox = torch.stack([
            (cx - bw / 2).clamp(0, 1),
            (cy - bh / 2).clamp(0, 1),
            (cx + bw / 2).clamp(0, 1),
            (cy + bh / 2).clamp(0, 1),
        ], dim=1)

        point_map = self.point_head(rpn_feat)
        flat_pts = point_map.flatten(2)
        topk_vals, topk_idx = flat_pts.squeeze(1).topk(self.num_queries, dim=-1)

        topk_y = topk_idx // W
        topk_x = topk_idx % W
        pred_points = torch.stack([
            topk_x.float() / W,
            topk_y.float() / H,
        ], dim=-1)
        point_scores = topk_vals.sigmoid()

        result = {
            "pred_bbox": pred_bbox,
            "pred_points": pred_points,
            "point_scores": point_scores,
            "cls_map": cls_map,
        }

        if gt_bbox is not None:
            gt_norm = gt_bbox.clone()
            bbox_loss = F.l1_loss(pred_bbox, gt_norm)
            result["bbox_loss"] = bbox_loss

        return result