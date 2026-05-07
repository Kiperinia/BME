"""Tensor-native forward wrapper around official or dummy SAM3 image models."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_model import build_official_sam3_image_model
from .feature_hooks import FeatureHookManager, register_feature_hooks

try:
    from sam3.model.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
    from sam3.model.data_misc import FindStage
    from sam3.model.sam3_image_processor import Sam3Processor

    HAS_OFFICIAL_SAM3_RUNTIME = True
except Exception:
    HAS_OFFICIAL_SAM3_RUNTIME = False


def _to_mask_logits(masks: torch.Tensor) -> torch.Tensor:
    if masks.min() < 0 or masks.max() > 1:
        return masks
    return torch.logit(masks.clamp(1e-4, 1 - 1e-4))


def _mean_tensor_from_feature_map(features: dict[str, object], key_hint: str) -> Optional[torch.Tensor]:
    for name, value in features.items():
        if key_hint not in name.lower():
            continue
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            return value[0]
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, torch.Tensor):
                    return nested
    return None


def _is_official_sam3_model(model: nn.Module) -> bool:
    return HAS_OFFICIAL_SAM3_RUNTIME and all(
        hasattr(model, name)
        for name in ["backbone", "_encode_prompt", "_run_encoder", "_run_decoder", "_run_segmentation_heads"]
    )


def _ensure_text_prompt(text_prompt: Optional[list[str]], batch_size: int) -> list[str]:
    if text_prompt is None:
        return ["visual"] * batch_size
    if len(text_prompt) == batch_size:
        return text_prompt
    if len(text_prompt) == 1 and batch_size > 1:
        return text_prompt * batch_size
    raise ValueError("text_prompt length must be 1 or match batch size")


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_official_resolution(model: nn.Module) -> int:
    default_resolution = 1008
    try:
        patch_proj = model.backbone.vision_backbone.trunk.patch_embed.proj
        patch_stride = int(patch_proj.stride[0])
        attn_sizes = {
            tuple(block.attn.input_size)
            for block in model.backbone.vision_backbone.trunk.blocks
            if getattr(block.attn, "input_size", None) is not None
        }
        if not attn_sizes:
            return default_resolution
        max_side_tokens = max(size[0] for size in attn_sizes)
        return int(max_side_tokens * patch_stride)
    except Exception:
        return default_resolution


def _normalize_xyxy_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.float()
    if boxes.numel() == 0:
        return boxes
    if boxes.max() > 1.0 or boxes.min() < 0.0:
        scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
        boxes = boxes / scale
    return boxes.clamp(0.0, 1.0)


def _normalize_xy_points(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    points = points.float()
    if points.numel() == 0:
        return points
    if points.max() > 1.0 or points.min() < 0.0:
        scale = torch.tensor([width, height], device=points.device, dtype=points.dtype)
        points = points / scale
    return points.clamp(0.0, 1.0)


class Sam3TensorForwardWrapper(nn.Module):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        dtype: str = "fp32",
        use_hooks: bool = True,
        hook_keywords: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.model = model or build_official_sam3_image_model(
            checkpoint_path=None,
            device=device,
            dtype=dtype,
            compile_model=False,
        )
        self.use_hooks = use_hooks
        self.hook_keywords = hook_keywords or [
            "image_encoder",
            "prompt",
            "decoder",
            "mask",
            "detector",
            "backbone",
            "geometry_encoder",
            "transformer",
            "segmentation_head",
        ]
        self.hooks: Optional[FeatureHookManager] = None
        self.processor_transform = None
        self.official_resolution = None
        if _is_official_sam3_model(self.model):
            self.official_resolution = _infer_official_resolution(self.model)
            self.processor_transform = Sam3Processor(
                self.model,
                resolution=self.official_resolution,
                device=str(_model_device(self.model)),
            ).transform
        if use_hooks:
            self.hooks = register_feature_hooks(self.model, keywords=self.hook_keywords, max_hooks=24)

    def _preprocess_official_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.processor_transform is None or self.official_resolution is None:
            raise RuntimeError("Official SAM3 preprocessing is unavailable")
        processed = [self.processor_transform(image.cpu()).to(images.device) for image in images]
        return torch.stack(processed, dim=0)

    def _build_find_stage(self, batch_size: int, device: torch.device) -> FindStage:
        return FindStage(
            img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            input_boxes=torch.zeros(0, batch_size, 4, device=device, dtype=torch.float32),
            input_boxes_mask=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
            input_boxes_label=torch.zeros(0, batch_size, device=device, dtype=torch.long),
            input_points=torch.zeros(0, batch_size, 2, device=device, dtype=torch.float32),
            input_points_mask=torch.zeros(batch_size, 0, device=device, dtype=torch.bool),
            object_ids=[[0] for _ in range(batch_size)],
        )

    def _call_official_model(
        self,
        images: torch.Tensor,
        text_prompt: Optional[list[str]],
        boxes: Optional[torch.Tensor],
        points: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        exemplar_prompt_tokens: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if not _is_official_sam3_model(self.model):
            raise TypeError("Current model is not an official SAM3 image model")

        batch_size, _, orig_height, orig_width = images.shape
        model_device = _model_device(self.model)
        runtime_images = self._preprocess_official_images(images).to(model_device)
        prompts = _ensure_text_prompt(text_prompt, batch_size)

        backbone_out = {"img_batch_all_stages": runtime_images}
        backbone_out.update(self.model.backbone.forward_image(runtime_images))
        backbone_out.update(self.model.backbone.forward_text(prompts, device=model_device))

        find_input = self._build_find_stage(batch_size=batch_size, device=model_device)
        geometric_prompt = self.model._get_dummy_prompt(num_prompts=batch_size)

        if boxes is not None:
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(1)
            if boxes.dim() != 3 or boxes.shape[0] != batch_size or boxes.shape[-1] != 4:
                raise ValueError("boxes must have shape [B, 4] or [B, N, 4]")
            boxes = boxes.to(model_device)
            normalized_boxes = _normalize_xyxy_boxes(boxes, height=orig_height, width=orig_width)
            box_embeddings = box_xyxy_to_cxcywh(normalized_boxes).permute(1, 0, 2)
            box_labels = torch.ones(box_embeddings.shape[:2], device=model_device, dtype=torch.long)
            box_mask = torch.zeros(batch_size, box_embeddings.shape[0], device=model_device, dtype=torch.bool)
            geometric_prompt.append_boxes(box_embeddings, box_labels, mask=box_mask)

        if points is not None:
            if points.dim() == 2:
                points = points.unsqueeze(1)
            if points.dim() != 3 or points.shape[0] != batch_size or points.shape[-1] != 2:
                raise ValueError("points must have shape [B, N, 2] or [B, 2]")
            points = points.to(model_device)
            normalized_points = _normalize_xy_points(points, height=orig_height, width=orig_width)
            point_embeddings = normalized_points.permute(1, 0, 2)
            if point_labels is None:
                point_labels = torch.ones(batch_size, points.shape[1], device=model_device, dtype=torch.long)
            else:
                point_labels = point_labels.to(model_device).long()
                if point_labels.dim() == 1:
                    point_labels = point_labels.unsqueeze(1)
            if point_labels.shape != points.shape[:2]:
                raise ValueError("point_labels must match points shape [B, N]")
            point_mask = torch.zeros(batch_size, point_embeddings.shape[0], device=model_device, dtype=torch.bool)
            geometric_prompt.append_points(point_embeddings, point_labels.transpose(0, 1), mask=point_mask)

        visual_prompt_embed = None
        visual_prompt_mask = None
        if exemplar_prompt_tokens is not None:
            exemplar_prompt_tokens = exemplar_prompt_tokens.to(model_device)
            if exemplar_prompt_tokens.dim() == 2:
                exemplar_prompt_tokens = exemplar_prompt_tokens.unsqueeze(1)
            if exemplar_prompt_tokens.dim() != 3 or exemplar_prompt_tokens.shape[0] != batch_size:
                raise ValueError("exemplar_prompt_tokens must have shape [B, C] or [B, N, C]")
            hidden_dim = int(getattr(self.model, "hidden_dim", exemplar_prompt_tokens.shape[-1]))
            if exemplar_prompt_tokens.shape[-1] != hidden_dim:
                raise ValueError(
                    f"Official SAM3 expects exemplar prompt dim {hidden_dim}, got {exemplar_prompt_tokens.shape[-1]}"
                )
            visual_prompt_embed = exemplar_prompt_tokens.permute(1, 0, 2)
            visual_prompt_mask = torch.zeros(batch_size, visual_prompt_embed.shape[0], device=model_device, dtype=torch.bool)

        prompt, prompt_mask, backbone_out = self.model._encode_prompt(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            visual_prompt_embed=visual_prompt_embed,
            visual_prompt_mask=visual_prompt_mask,
            encode_text=True,
        )
        backbone_out, encoder_out, _ = self.model._run_encoder(
            backbone_out=backbone_out,
            find_input=find_input,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        out: dict[str, Any] = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        out, hs = self.model._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )
        self.model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )
        out = self.model._postprocess_out(out, multimask_output=False)

        pred_logits = out["pred_logits"]
        pred_masks = out["pred_masks"]
        pred_boxes = out.get("pred_boxes_xyxy")
        if pred_boxes is None:
            pred_boxes = box_cxcywh_to_xyxy(out["pred_boxes"])

        scores = pred_logits.sigmoid()
        if "presence_logit_dec" in out:
            presence_score = out["presence_logit_dec"].sigmoid()
            if presence_score.dim() == 1:
                scores = scores * presence_score[:, None, None]
            elif presence_score.dim() == 2:
                scores = scores * presence_score.unsqueeze(-1)

        best_idx = scores.squeeze(-1).argmax(dim=1)
        batch_idx = torch.arange(batch_size, device=model_device)
        mask_logits = pred_masks[batch_idx, best_idx].unsqueeze(1)
        selected_scores = scores.squeeze(-1)[batch_idx, best_idx].unsqueeze(1)
        selected_boxes = pred_boxes[batch_idx, best_idx]

        scale = torch.tensor([orig_width, orig_height, orig_width, orig_height], device=model_device, dtype=selected_boxes.dtype)
        selected_boxes = selected_boxes * scale.unsqueeze(0)
        mask_logits = F.interpolate(mask_logits, size=(orig_height, orig_width), mode="bilinear", align_corners=False)

        return {
            "masks": torch.sigmoid(mask_logits),
            "mask_logits": mask_logits,
            "boxes": selected_boxes,
            "scores": selected_scores,
            "image_embeddings": backbone_out.get("backbone_fpn", [None])[-1],
            "prompt_embeddings": encoder_out.get("prompt_after_enc").transpose(0, 1) if encoder_out.get("prompt_after_enc") is not None else None,
            "exemplar_embeddings": exemplar_prompt_tokens,
            "detector_queries": out.get("queries"),
            "intermediate_features": {
                "backbone_out": backbone_out,
                "encoder_out": encoder_out,
                "raw_output": out,
                "queries": out.get("queries"),
                "prompt_before_enc": encoder_out.get("prompt_before_enc"),
                "prompt_after_enc": encoder_out.get("prompt_after_enc"),
            },
        }

    def _call_model(
        self,
        images: torch.Tensor,
        text_prompt: Optional[list[str]],
        boxes: Optional[torch.Tensor],
        points: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        exemplar_prompt_tokens: Optional[torch.Tensor],
    ) -> Any:
        if _is_official_sam3_model(self.model):
            return self._call_official_model(
                images=images,
                text_prompt=text_prompt,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                exemplar_prompt_tokens=exemplar_prompt_tokens,
            )

        if hasattr(self.model, "tensor_forward"):
            return self.model.tensor_forward(
                images=images,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                text_prompt=text_prompt,
                exemplar_prompt_tokens=exemplar_prompt_tokens,
            )

        try:
            return self.model(
                images=images,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                text_prompt=text_prompt,
                exemplar_prompt_tokens=exemplar_prompt_tokens,
            )
        except TypeError:
            return self.model(
                images,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
            )

    def forward(
        self,
        images: torch.Tensor,
        text_prompt: Optional[list[str]] = None,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        exemplar_prompt_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if images.min() < 0 or images.max() > 1:
            raise ValueError("images must be normalized to [0, 1]")

        if self.hooks is not None:
            self.hooks.clear()

        outputs = self._call_model(
            images=images,
            text_prompt=text_prompt,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            exemplar_prompt_tokens=exemplar_prompt_tokens,
        )

        if isinstance(outputs, torch.Tensor):
            outputs = {"mask_logits": outputs}
        elif not isinstance(outputs, dict):
            raise TypeError("SAM3 model output must be a Tensor or dict")

        if "mask_logits" not in outputs and "masks" in outputs:
            outputs["mask_logits"] = _to_mask_logits(outputs["masks"])
        if "masks" not in outputs and "mask_logits" in outputs:
            outputs["masks"] = torch.sigmoid(outputs["mask_logits"])
        if "scores" not in outputs:
            outputs["scores"] = outputs["masks"].flatten(1).mean(dim=1, keepdim=True)
        if "boxes" not in outputs:
            outputs["boxes"] = boxes

        intermediate = {}
        if self.hooks is not None:
            intermediate.update(self.hooks.features)
        intermediate.update(outputs.get("intermediate_features") or {})

        outputs.setdefault("image_embeddings", _mean_tensor_from_feature_map(intermediate, "image"))
        outputs.setdefault("prompt_embeddings", _mean_tensor_from_feature_map(intermediate, "prompt"))
        outputs.setdefault("exemplar_embeddings", None)
        outputs.setdefault("detector_queries", _mean_tensor_from_feature_map(intermediate, "detector"))
        outputs["intermediate_features"] = intermediate
        return outputs
