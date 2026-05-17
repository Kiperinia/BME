"""Build and inspect official SAM3 image models with MedEx-SAM3 fallbacks."""

from __future__ import annotations

import ctypes
import inspect
import json
import logging
from pathlib import Path
import site
import warnings
from datetime import datetime, timezone
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_CHECKPOINTS = ("sam3.pt", "MedSAM3.pt")
_DEFAULT_BUILD_REPORT = (
    Path(__file__).resolve().parents[1] / "outputs" / "medex_sam3" / "preflight" / "model_build_report.json"
)
_CUDA_RUNTIME_PRIMED = False


def _prime_cuda_runtime_libraries(device: str) -> None:
    global _CUDA_RUNTIME_PRIMED
    if _CUDA_RUNTIME_PRIMED or str(device) != "cuda":
        return

    cuda_version = str(getattr(torch.version, "cuda", "") or "")
    if not cuda_version:
        return

    major_minor = cuda_version.split(".")
    if len(major_minor) < 2:
        return

    major, minor = major_minor[0], major_minor[1]
    site_packages = [Path(path) for path in site.getsitepackages()]
    versioned_dir = f"cu{major}"
    versioned_builtins = f"libnvrtc-builtins.so.{major}.{minor}"
    versioned_nvrtc = f"libnvrtc.so.{major}"
    fallback_builtins = f"libnvrtc-builtins.so.{major}.{minor}"
    fallback_nvrtc = f"libnvrtc.so.{major}"

    candidate_files: list[Path] = []
    for package_dir in site_packages:
        candidate_files.extend(
            [
                package_dir / "nvidia" / versioned_dir / "lib" / versioned_builtins,
                package_dir / "nvidia" / versioned_dir / "lib" / versioned_nvrtc,
                package_dir / "nvidia" / "cuda_nvrtc" / "lib" / fallback_builtins,
                package_dir / "nvidia" / "cuda_nvrtc" / "lib" / fallback_nvrtc,
            ]
        )

    loaded_any = False
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        loaded_any = True

    if loaded_any:
        _CUDA_RUNTIME_PRIMED = True


def _resolve_dtype(dtype: str) -> torch.dtype:
    normalized = dtype.lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[normalized]


def _find_default_local_checkpoint() -> Optional[str]:
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoint"
    for file_name in _DEFAULT_LOCAL_CHECKPOINTS:
        candidate = checkpoint_dir / file_name
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if checkpoint_path is None:
        return _find_default_local_checkpoint()

    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return str(candidate)

    repo_relative_candidate = Path(__file__).resolve().parents[2] / candidate
    if repo_relative_candidate.exists():
        return str(repo_relative_candidate)

    raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")


def _dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return mapping.get(dtype, str(dtype))


def _resolve_runtime_dtype(device: str, dtype: torch.dtype) -> tuple[torch.dtype, Optional[str]]:
    if str(device).startswith("cpu") and dtype != torch.float32:
        return torch.float32, "CPU execution falls back to fp32 for stability."
    return dtype, None


def _default_hidden_dim(model: nn.Module) -> Optional[int]:
    hidden_dim = getattr(model, "hidden_dim", None)
    if hidden_dim is not None:
        return int(hidden_dim)
    embed_dim = getattr(model, "embed_dim", None)
    if embed_dim is not None:
        return int(embed_dim)
    return None


def _annotate_model(model: nn.Module, *, used_official_sam3: bool, used_dummy_fallback: bool) -> nn.Module:
    hidden_dim = _default_hidden_dim(model)
    model._medex_used_official_sam3 = bool(used_official_sam3)
    model._medex_used_dummy_fallback = bool(used_dummy_fallback)
    model._medex_hidden_dim = hidden_dim
    return model


def _build_model_report(
    *,
    requested_checkpoint: Optional[str],
    resolved_checkpoint: Optional[str],
    device: str,
    requested_dtype: str,
    effective_dtype: torch.dtype,
    model: Optional[nn.Module],
    used_official_sam3: bool,
    used_dummy_fallback: bool,
    error: Optional[str],
    warning: Optional[str],
) -> dict[str, Any]:
    hidden_dim = None if model is None else getattr(model, "hidden_dim", None)
    embed_dim = None if model is None else getattr(model, "embed_dim", None)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_checkpoint": requested_checkpoint,
        "resolved_checkpoint": resolved_checkpoint,
        "device": device,
        "dtype": _dtype_name(effective_dtype),
        "requested_dtype": requested_dtype,
        "model_class": None if model is None else model.__class__.__name__,
        "used_official_sam3": bool(used_official_sam3),
        "used_dummy_fallback": bool(used_dummy_fallback),
        "has_backbone": bool(model is not None and hasattr(model, "backbone")),
        "has_encode_prompt": bool(model is not None and hasattr(model, "_encode_prompt")),
        "has_run_encoder": bool(model is not None and hasattr(model, "_run_encoder")),
        "has_run_decoder": bool(model is not None and hasattr(model, "_run_decoder")),
        "has_run_segmentation_heads": bool(model is not None and hasattr(model, "_run_segmentation_heads")),
        "hidden_dim": None if hidden_dim is None else int(hidden_dim),
        "embed_dim": None if embed_dim is None else int(embed_dim),
        "error": error,
    }
    if warning is not None:
        report["warning"] = warning
    return report


def _write_model_report(report: dict[str, Any], report_path: Optional[str]) -> Path:
    destination = Path(report_path) if report_path is not None else _DEFAULT_BUILD_REPORT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


class DummySelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None:
            context = x
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        return self.out_proj(torch.matmul(attn, v))


class DummyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DummyTransformerBlock(nn.Module):
    def __init__(self, dim: int, with_cross_attn: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = DummySelfAttention(dim)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = DummySelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DummyMLP(dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        if self.with_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context=context)
        x = x + self.mlp(self.norm2(x))
        return x


class DummyEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, with_cross_attn: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DummyTransformerBlock(dim, with_cross_attn=with_cross_attn) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, context=context)
        return x


class DummyPromptEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.box_proj = nn.Linear(4, dim)
        self.point_proj = nn.Linear(3, dim)
        self.text_proj = nn.Linear(32, dim)
        self.exemplar_projection = nn.Linear(dim, dim)

    def encode(
        self,
        batch_size: int,
        device: torch.device,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        text_prompt: Optional[list[str]] = None,
        exemplar_prompt_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prompts = []

        if boxes is not None:
            if boxes.dim() == 2:
                valid_mask = torch.isfinite(boxes).all(dim=-1, keepdim=True)
                box_embeddings = self.box_proj(torch.nan_to_num(boxes.float(), nan=0.0))
                box_embeddings = box_embeddings * valid_mask.to(box_embeddings.dtype)
                prompts.append(box_embeddings)
            elif boxes.dim() == 3:
                valid_mask = torch.isfinite(boxes).all(dim=-1, keepdim=True)
                box_embeddings = self.box_proj(torch.nan_to_num(boxes.float(), nan=0.0))
                box_embeddings = box_embeddings * valid_mask.to(box_embeddings.dtype)
                prompts.append(box_embeddings.mean(dim=1))
            else:
                raise ValueError("DummyPromptEncoder boxes must have shape [B, 4] or [B, N, 4]")

        if points is not None:
            if point_labels is None:
                point_labels = torch.ones(points.shape[:2], device=device, dtype=points.dtype)
            point_features = torch.cat([points.float(), point_labels.unsqueeze(-1).float()], dim=-1)
            prompts.append(self.point_proj(point_features).mean(dim=1))

        if text_prompt is not None:
            text_tensor = torch.zeros(batch_size, 32, device=device)
            for index, prompt in enumerate(text_prompt):
                values = list(prompt.encode("utf-8")[:32])
                if values:
                    text_tensor[index, : len(values)] = torch.tensor(values, device=device)
            prompts.append(self.text_proj(text_tensor / 255.0))

        exemplar_embeddings = None
        if exemplar_prompt_tokens is not None:
            if exemplar_prompt_tokens.dim() == 2:
                exemplar_embeddings = exemplar_prompt_tokens
            else:
                exemplar_embeddings = exemplar_prompt_tokens.mean(dim=1)
            prompts.append(self.exemplar_projection(exemplar_embeddings))

        if not prompts:
            prompt_embeddings = torch.zeros(batch_size, self.box_proj.out_features, device=device)
        else:
            prompt_embeddings = torch.stack(prompts, dim=0).mean(dim=0)
        return prompt_embeddings, exemplar_embeddings


class DummyMaskDecoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.transformer = DummyEncoder(dim=dim, depth=2, with_cross_attn=True)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.mask_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        self.score_head = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, features_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.transformer(tokens, context=tokens)
        token = self.out_proj(self.v_proj(tokens[:, :1]))
        feature_map = features_2d + token.transpose(1, 2).unsqueeze(-1)
        logits = self.mask_head(feature_map)
        scores = self.score_head(tokens[:, 0]).sigmoid()
        return logits, scores


class DummyOfficialSam3ImageModel(nn.Module):
    """A tensor-native stand-in for the official SAM3 image model."""

    def __init__(self, embed_dim: int = 128, image_stride: int = 4, depth: int = 6) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_stride = image_stride
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.image_encoder = DummyEncoder(dim=embed_dim, depth=depth)
        self.detector_encoder = DummyEncoder(dim=embed_dim, depth=2)
        self.detector_decoder = DummyEncoder(dim=embed_dim, depth=2, with_cross_attn=True)
        self.prompt_encoder = DummyPromptEncoder(embed_dim)
        self.mask_decoder = DummyMaskDecoder(embed_dim)
        self.text_encoder = nn.Linear(32, embed_dim)

    def _image_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.stem(images)
        batch_size, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        encoded = self.image_encoder(tokens)
        feature_map = encoded.transpose(1, 2).reshape(batch_size, channels, height, width)
        return encoded, feature_map

    def tensor_forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        text_prompt: Optional[list[str]] = None,
        exemplar_prompt_tokens: Optional[torch.Tensor] = None,
        retrieval_prior: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        image_tokens, image_features = self._image_tokens(images)
        query_tokens = self.detector_encoder(image_tokens[:, :1])
        prompt_embeddings, exemplar_embeddings = self.prompt_encoder.encode(
            batch_size=images.shape[0],
            device=images.device,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            text_prompt=text_prompt,
            exemplar_prompt_tokens=exemplar_prompt_tokens,
        )
        retrieval_summary: dict[str, Any] = {}
        if isinstance(retrieval_prior, dict):
            decoder_bias = retrieval_prior.get("decoder_feature_bias_map")
            if isinstance(decoder_bias, torch.Tensor) and decoder_bias.dim() == 4:
                resized_bias = F.interpolate(
                    decoder_bias.to(image_features.device, dtype=image_features.dtype),
                    size=image_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                image_features = image_features + resized_bias
                retrieval_summary["used_decoder_feature_bias_map"] = True
            spatial_bias = retrieval_prior.get("spatial_bias_map")
            if isinstance(spatial_bias, torch.Tensor) and spatial_bias.dim() == 4:
                resized_spatial = F.interpolate(
                    spatial_bias.to(image_features.device, dtype=image_features.dtype),
                    size=image_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                image_features = image_features + image_features * resized_spatial
                retrieval_summary["used_spatial_bias_map"] = True
            semantic_prototype = retrieval_prior.get("semantic_prototype")
            if isinstance(semantic_prototype, torch.Tensor) and semantic_prototype.dim() == 2:
                prompt_embeddings = prompt_embeddings + semantic_prototype.to(prompt_embeddings.device, dtype=prompt_embeddings.dtype)
                retrieval_summary["used_semantic_prototype"] = True
        detector_queries = self.detector_decoder(
            query_tokens + prompt_embeddings.unsqueeze(1),
            context=image_tokens,
        )
        mask_logits, scores = self.mask_decoder(detector_queries, image_features)
        if isinstance(retrieval_prior, dict):
            logit_bias = retrieval_prior.get("mask_logit_bias_map")
            if isinstance(logit_bias, torch.Tensor) and logit_bias.dim() == 4:
                resized_logit_bias = F.interpolate(
                    logit_bias.to(mask_logits.device, dtype=mask_logits.dtype),
                    size=mask_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                mask_logits = mask_logits + resized_logit_bias
                retrieval_summary["used_mask_logit_bias_map"] = True
        masks = torch.sigmoid(mask_logits)
        return {
            "masks": masks,
            "mask_logits": mask_logits,
            "boxes": boxes,
            "scores": scores,
            "image_embeddings": image_features,
            "prompt_embeddings": prompt_embeddings,
            "exemplar_embeddings": exemplar_embeddings,
            "detector_queries": detector_queries,
            "intermediate_features": {
                "image_tokens": image_tokens,
                "image_features": image_features,
                "retrieval_prior": retrieval_summary,
            },
        }

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.tensor_forward(*args, **kwargs)


def _move_model(model: nn.Module, device: str, dtype: torch.dtype) -> nn.Module:
    runtime_dtype, runtime_warning = _resolve_runtime_dtype(device, dtype)
    if runtime_warning is not None:
        warnings.warn(runtime_warning, stacklevel=2)
    return model.to(device=device, dtype=runtime_dtype)


def _reset_official_runtime_caches(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "compilable_cord_cache"):
            module.compilable_cord_cache = None
        if hasattr(module, "compilable_stored_size"):
            module.compilable_stored_size = None
        if hasattr(module, "coord_cache") and isinstance(module.coord_cache, dict):
            module.coord_cache = {}


def _move_official_model(model: nn.Module, device: str) -> nn.Module:
    model = model.to(device=device)
    _reset_official_runtime_caches(model)
    return model


def _build_from_official_builder(
    checkpoint_path: Optional[str],
    device: str,
    dtype: torch.dtype,
) -> nn.Module:
    from sam3.model_builder import build_sam3_image_model as official_builder

    signature = inspect.signature(official_builder)
    kwargs: dict[str, Any] = {}
    alias_groups = {
        "checkpoint_path": ["checkpoint_path", "checkpoint", "ckpt_path", "model_path"],
        "device": ["device"],
        "dtype": ["dtype"],
        "load_from_HF": ["load_from_HF", "load_from_hf"],
    }

    if checkpoint_path is not None:
        for alias in alias_groups["checkpoint_path"]:
            if alias in signature.parameters:
                kwargs[alias] = checkpoint_path
                break

    for alias in alias_groups["device"]:
        if alias in signature.parameters:
            kwargs[alias] = device
            break

    for alias in alias_groups["dtype"]:
        if alias in signature.parameters:
            kwargs[alias] = dtype
            break

    if checkpoint_path is None:
        for alias in alias_groups["load_from_HF"]:
            if alias in signature.parameters:
                kwargs[alias] = True
                break

    model = official_builder(**kwargs)
    return _move_official_model(model, device=device)


def build_official_sam3_image_model(
    checkpoint_path: Optional[str],
    device: str,
    dtype: str = "fp16",
    compile_model: bool = False,
    allow_dummy_fallback: bool = False,
    report_path: Optional[str] = None,
) -> nn.Module:
    """Build the official SAM3 image model with a tensor-native fallback for smoke tests."""

    target_dtype = _resolve_dtype(dtype)
    effective_dtype, runtime_warning = _resolve_runtime_dtype(device, target_dtype)
    resolved_checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    model: Optional[nn.Module] = None
    used_official_sam3 = False
    used_dummy_fallback = False
    error_message: Optional[str] = None
    if checkpoint_path is None and resolved_checkpoint_path is not None:
        logger.info("Using local SAM3 checkpoint: %s", resolved_checkpoint_path)

    try:
        _prime_cuda_runtime_libraries(device)
        model = _build_from_official_builder(
            checkpoint_path=resolved_checkpoint_path,
            device=device,
            dtype=effective_dtype,
        )
        used_official_sam3 = True
        logger.info("Built official SAM3 image model.")
    except Exception as exc:
        error_message = str(exc)
        build_report = _build_model_report(
            requested_checkpoint=checkpoint_path,
            resolved_checkpoint=resolved_checkpoint_path,
            device=device,
            requested_dtype=dtype,
            effective_dtype=effective_dtype,
            model=None,
            used_official_sam3=False,
            used_dummy_fallback=False,
            error=error_message,
            warning=runtime_warning,
        )
        if checkpoint_path is not None or not allow_dummy_fallback:
            _write_model_report(build_report, report_path)
            raise RuntimeError("Failed to build official SAM3 image model.") from exc

        warnings.warn(
            f"Falling back to DummyOfficialSam3ImageModel because official builder failed: {exc}",
            stacklevel=2,
        )
        model = _move_model(DummyOfficialSam3ImageModel(), device=device, dtype=effective_dtype)
        used_dummy_fallback = True

    model = _annotate_model(
        model,
        used_official_sam3=used_official_sam3,
        used_dummy_fallback=used_dummy_fallback,
    )
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            model = _annotate_model(
                model,
                used_official_sam3=used_official_sam3,
                used_dummy_fallback=used_dummy_fallback,
            )
        except Exception as exc:
            warnings.warn(f"torch.compile skipped: {exc}", stacklevel=2)

    build_report = _build_model_report(
        requested_checkpoint=checkpoint_path,
        resolved_checkpoint=resolved_checkpoint_path,
        device=device,
        requested_dtype=dtype,
        effective_dtype=effective_dtype,
        model=model,
        used_official_sam3=used_official_sam3,
        used_dummy_fallback=used_dummy_fallback,
        error=error_message,
        warning=runtime_warning,
    )
    _write_model_report(build_report, report_path)
    return model


def freeze_model(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def unfreeze_by_keywords(model: nn.Module, keywords: list[str]) -> nn.Module:
    lowered = [keyword.lower() for keyword in keywords]
    for name, parameter in model.named_parameters():
        if any(keyword in name.lower() for keyword in lowered):
            parameter.requires_grad = True
    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    ratio = float(trainable) / float(total) if total else 0.0
    return trainable, total, ratio


def print_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
    trainable, total, ratio = count_trainable_parameters(model)
    message = (
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({ratio * 100:.2f}%)"
    )
    logger.info(message)
    print(message)
    return trainable, total, ratio
