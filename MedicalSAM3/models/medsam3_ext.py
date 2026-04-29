"""Medical SAM3 可运行扩展版。

当前扩展版围绕已有高层包装器接口构建：
1. 轻量图像特征支路负责 MSFA / APG / TGA。
2. 基础模型负责给出初始 coarse mask。
3. BRH 使用误差置信度和息肉形状先验联合控制边界精修强度。
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .extensions import (
    AdaptivePromptGenerator,
    BoundaryRefinementHead,
    MultiScaleFeatureAdapter,
    TextGuidedAttention,
)


# ═══════════════════════════════════════════════════════
#  扩展模型: 整合所有创新模块
# ═══════════════════════════════════════════════════════

class MedSAM3Extended(nn.Module):
    """
    扩展版 Medical SAM3。

    该实现兼容当前的高层基础模型接口：
    base_model(images, bboxes=..., points=..., point_labels=...) -> dict
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_msfa: bool = True,
        use_apg: bool = True,
        use_brh: bool = True,
        use_tga: bool = True,
        image_size: int = 1024,
        feature_channels: int = 256,
    ):
        super().__init__()
        self.base = base_model
        self.image_size = image_size
        self.use_msfa = use_msfa
        self.use_apg = use_apg
        self.use_brh = use_brh
        self.use_tga = use_tga
        self.feature_channels = feature_channels

        self.image_feature_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
        )
        self.fallback_text_embedding = nn.Embedding(256, feature_channels)

        if use_msfa:
            self.msfa = MultiScaleFeatureAdapter(in_channels=feature_channels, out_channels=feature_channels)
        if use_apg:
            self.apg = AdaptivePromptGenerator(in_channels=feature_channels)
        if use_brh:
            self.brh = BoundaryRefinementHead()
        if use_tga:
            self.tga = TextGuidedAttention(embed_dim=feature_channels)

    @staticmethod
    def _ensure_mask_4d(mask: torch.Tensor) -> torch.Tensor:
        while mask.dim() > 4:
            mask = mask.squeeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _to_logits(mask: torch.Tensor) -> torch.Tensor:
        if mask.min() < 0 or mask.max() > 1:
            return mask
        return torch.logit(mask.clamp(1e-4, 1 - 1e-4))

    @staticmethod
    def _normalize_xyxy_boxes(bboxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        norm = bboxes.clone().float()
        norm[..., 0] /= width
        norm[..., 2] /= width
        norm[..., 1] /= height
        norm[..., 3] /= height
        return norm

    @staticmethod
    def _denormalize_xyxy_boxes(bboxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        boxes = bboxes.clone().float()
        boxes[..., 0] *= width
        boxes[..., 2] *= width
        boxes[..., 1] *= height
        boxes[..., 3] *= height
        return boxes

    def _encode_text_prompt(self, text_prompt: Optional[List[str]], device: torch.device) -> Optional[torch.Tensor]:
        if text_prompt is None:
            return None
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]

        if hasattr(self.base, "tokenizer") and hasattr(self.base, "text_encoder"):
            token_ids = self.base.tokenizer.batch_encode(text_prompt).to(device)
            return self.base.text_encoder(token_ids)

        max_len = max(max(len(item.encode("utf-8")), 1) for item in text_prompt)
        byte_ids = torch.zeros(len(text_prompt), max_len, dtype=torch.long, device=device)
        for row, prompt in enumerate(text_prompt):
            encoded = list(prompt.encode("utf-8"))
            if encoded:
                byte_ids[row, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
        return self.fallback_text_embedding(byte_ids).mean(dim=1)

    def forward(
        self,
        images: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        text_prompt: Optional[List[str]] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        扩展前向流程:
        1. 轻量特征支路生成图像特征。
        2. 可选执行 MSFA / TGA / APG。
        3. 基础模型给出初始 coarse mask。
        4. BRH 结合误差置信度和形状先验做边界精修。
        """
        image_features = self.image_feature_stem(images)
        if self.use_msfa:
            image_features = self.msfa(image_features)

        text_embed = self._encode_text_prompt(text_prompt, images.device)
        if self.use_tga and text_embed is not None:
            image_features = self.tga(image_features, text_embed)

        results: Dict[str, torch.Tensor] = {
            "image_features": image_features,
        }

        working_bboxes = bboxes
        if self.use_apg:
            gt_bbox_norm = None
            if bboxes is not None:
                gt_bbox_norm = self._normalize_xyxy_boxes(bboxes, images.shape[-1], images.shape[-2])
            apg_out = self.apg(image_features, gt_bbox=gt_bbox_norm)
            results["apg_output"] = apg_out
            if working_bboxes is None:
                working_bboxes = self._denormalize_xyxy_boxes(
                    apg_out["pred_bbox"], images.shape[-1], images.shape[-2]
                )

        base_outputs = self.base(
            images,
            bboxes=working_bboxes,
            points=points,
            point_labels=point_labels,
        )
        coarse_masks = self._ensure_mask_4d(base_outputs["masks"].float())
        coarse_logits = self._to_logits(coarse_masks)

        results["coarse_masks"] = coarse_masks
        results["coarse_logits"] = coarse_logits
        results["used_bboxes"] = working_bboxes if working_bboxes is not None else torch.empty(0, device=images.device)

        if self.use_brh:
            brh_outputs = self.brh(coarse_logits, images, gt_mask=gt_masks, return_aux=True)
            results.update(brh_outputs)
            results["masks"] = brh_outputs["masks"]
            results["mask_logits"] = brh_outputs["mask_logits"]
        else:
            results["masks"] = coarse_logits.sigmoid()
            results["mask_logits"] = coarse_logits

        results["iou_predictions"] = base_outputs["iou_predictions"].float()

        return results

    def load_extension_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, List[str]]:
        """加载 train_ext.py 保存的整套扩展模型 checkpoint。"""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        return {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    def load_brh_checkpoint(self, checkpoint_path: str, strict: bool = True) -> Dict[str, List[str]]:
        """只加载 BRH 权重，适用于 brh_best.pt / brh_last.pt 或整套 checkpoint 中的 brh.* 子集。"""
        if not hasattr(self, "brh"):
            raise ValueError("Current model does not have BRH enabled.")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

        if any(key.startswith("brh.") for key in state_dict):
            brh_state = {
                key.removeprefix("brh."): value
                for key, value in state_dict.items()
                if key.startswith("brh.")
            }
        else:
            brh_state = state_dict

        missing, unexpected = self.brh.load_state_dict(brh_state, strict=strict)
        return {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }


def build_medsam3_extended(
    base_model: nn.Module,
    use_msfa: bool = True,
    use_apg: bool = True,
    use_brh: bool = True,
    use_tga: bool = True,
    image_size: int = 1024,
    feature_channels: int = 256,
) -> MedSAM3Extended:
    """构建扩展版模型"""
    model = MedSAM3Extended(
        base_model=base_model,
        use_msfa=use_msfa,
        use_apg=use_apg,
        use_brh=use_brh,
        use_tga=use_tga,
        image_size=image_size,
        feature_channels=feature_channels,
    )
    base_param = next(base_model.parameters(), None)
    if base_param is not None:
        model = model.to(base_param.device)
    return model


def load_medsam3_extended_checkpoint(
    model: MedSAM3Extended,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict[str, List[str]]:
    return model.load_extension_checkpoint(checkpoint_path, strict=strict)


def load_brh_checkpoint(
    model: MedSAM3Extended,
    checkpoint_path: str,
    strict: bool = True,
) -> Dict[str, List[str]]:
    return model.load_brh_checkpoint(checkpoint_path, strict=strict)
