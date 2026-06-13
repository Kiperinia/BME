from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for the fallback path.
    torch = None  # type: ignore[assignment]

try:
    from monai.transforms import (
        AdjustContrast,
        Compose,
        EnsureChannelFirst,
        NormalizeIntensity,
        Resize,
        ScaleIntensity,
    )
    from monai.networks.nets import SwinUNETR
except Exception:  # MONAI is optional; the tool has an OpenCV fallback.
    AdjustContrast = None  # type: ignore[assignment]
    Compose = None  # type: ignore[assignment]
    EnsureChannelFirst = None  # type: ignore[assignment]
    NormalizeIntensity = None  # type: ignore[assignment]
    Resize = None  # type: ignore[assignment]
    ScaleIntensity = None  # type: ignore[assignment]
    SwinUNETR = None  # type: ignore[assignment]


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class ImageQualityThresholds:
    min_width: int = 64
    min_height: int = 64
    max_width: int = 8192
    max_height: int = 8192
    dark_mean_threshold: float = 45.0
    overexposed_pixel_threshold: int = 245
    overexposed_ratio_threshold: float = 0.18
    blur_laplacian_var_threshold: float = 80.0
    low_contrast_std_threshold: float = 32.0
    severe_dark_threshold: float = 25.0
    severe_blur_threshold: float = 25.0
    severe_contrast_threshold: float = 15.0


@dataclass(slots=True)
class PreprocessConfig:
    target_size: tuple[int, int] = (512, 512)
    thresholds: ImageQualityThresholds = field(default_factory=ImageQualityThresholds)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    denoise_kernel_size: int = 3
    always_call_monai_tool: bool = True
    call_monai_on_warnings: bool = True
    embedding_dim: int = 128
    monai_encoder_input_size: tuple[int, int] = (256, 256)
    monai_encoder_depth: int = 32
    monai_encoder_weights_path: str | None = None
    monai_encoder_device: str = "cpu"
    allow_untrained_monai_encoder: bool = False
    use_2d_polyp_model: bool = True
    polyp_model_repo_id: str = "andreribeiro87/unet3plus-efficientnet-kvasir-seg"
    polyp_model_threshold: float = 0.5


def _normalize_uint8(array: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def _bbox_from_heatmap(heatmap: np.ndarray, threshold_ratio: float = 0.72) -> dict[str, Any]:
    if heatmap.size == 0:
        return {"bbox": None, "confidence": 0.0, "source": "empty_heatmap"}

    heatmap_u8 = _normalize_uint8(heatmap)
    high_percentile = int(np.percentile(heatmap_u8, 92))
    threshold = max(12, int(float(heatmap_u8.max()) * threshold_ratio), high_percentile)
    binary = (heatmap_u8 >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = heatmap.shape[:2]
        return {
            "bbox": [0, 0, int(w), int(h)],
            "confidence": 0.1,
            "source": "fallback_full_image",
        }

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    pad_x = max(4, int(w * 0.18))
    pad_y = max(4, int(h * 0.18))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(heatmap.shape[1], x + w + pad_x)
    y2 = min(heatmap.shape[0], y + h + pad_y)
    area_ratio = float(cv2.contourArea(contour) / max(1, heatmap.shape[0] * heatmap.shape[1]))
    confidence = min(1.0, max(0.15, area_ratio * 8.0))
    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "confidence": round(confidence, 4),
        "source": "suspicious_heatmap",
        "heatmap_threshold": int(threshold),
        "area_ratio": round(area_ratio, 6),
    }


class MonaiTransformerTool:
    """
    MONAI-based preprocessing and transformer feature tool.

    MONAI transform wrapper with a 2D polyp segmentation model mounted inside.
    SwinViT support is kept for explicit experiments, but the default clinical
    preprocessing path uses MONAI transforms plus `PolypSegmentation2DTool`.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (512, 512),
        embedding_dim: int = 128,
        encoder_input_size: tuple[int, int] = (256, 256),
        encoder_depth: int = 32,
        encoder_weights_path: str | None = None,
        device: str = "cpu",
        allow_untrained_encoder: bool = False,
        use_2d_polyp_model: bool = True,
        polyp_model_repo_id: str = "andreribeiro87/unet3plus-efficientnet-kvasir-seg",
        polyp_model_threshold: float = 0.5,
        polyp_2d_tool: Any | None = None,
    ):
        self.target_size = target_size
        self.encoder_input_size = encoder_input_size
        self.encoder_depth = encoder_depth
        self.embedding_dim = embedding_dim
        self.device = device
        self.encoder_weights_path = encoder_weights_path
        self.allow_untrained_encoder = allow_untrained_encoder
        self.encoder_in_channels = 3
        self.encoder_spatial_dims = 2
        self.encoder_feature_size = 24
        self.encoder_depths = (1, 1, 1, 1)
        self.encoder_num_heads = (3, 6, 12, 24)
        self.monai_available = Compose is not None
        self.encoder_status = "unavailable"
        self.encoder_weight_report: dict[str, Any] = {}
        self.pipeline = self._build_monai_pipeline() if self.monai_available else None
        self.encoder = self._build_encoder() if self.monai_available and encoder_weights_path else None
        if self.encoder is not None and self.encoder_status == "unavailable":
            self.encoder_status = "untrained_encoder"
        self.polyp_2d_tool = polyp_2d_tool
        if self.polyp_2d_tool is None and use_2d_polyp_model:
            self.polyp_2d_tool = PolypSegmentation2DTool(
                repo_id=polyp_model_repo_id,
                device=device,
                threshold=polyp_model_threshold,
            )

    def _build_monai_pipeline(self) -> Any:
        return Compose(
            [
                EnsureChannelFirst(channel_dim=-1),
                Resize(spatial_size=self.encoder_input_size),
                ScaleIntensity(minv=0.0, maxv=1.0),
                AdjustContrast(gamma=0.9),
                NormalizeIntensity(nonzero=False, channel_wise=True),
            ]
        )

    def _build_encoder(self) -> Any:
        if SwinUNETR is None or torch is None:
            return None
        encoder_state = None
        if self.encoder_weights_path:
            encoder_state = self._load_encoder_state_dict(Path(self.encoder_weights_path))
            self._infer_encoder_shape(encoder_state)
        elif not self.allow_untrained_encoder:
            self.encoder_status = "not_loaded_no_weights"
            self.encoder_weight_report = {
                "note": "No trained SwinUNETR weights supplied; using MONAI transforms plus deterministic image saliency instead of random encoder features.",
            }
            return None

        encoder = SwinUNETR(
            in_channels=self.encoder_in_channels,
            out_channels=1,
            feature_size=self.encoder_feature_size,
            depths=self.encoder_depths,
            num_heads=self.encoder_num_heads,
            spatial_dims=self.encoder_spatial_dims,
            use_checkpoint=False,
        )

        if encoder_state is not None:
            missing, unexpected = encoder.swinViT.load_state_dict(encoder_state, strict=False)
            self.encoder_status = "trained_weights_loaded"
            self.encoder_weight_report = {
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "loaded_into": "SwinUNETR.swinViT",
                "spatial_dims": self.encoder_spatial_dims,
                "in_channels": self.encoder_in_channels,
                "feature_size": self.encoder_feature_size,
                "depths": list(self.encoder_depths),
            }
        else:
            self.encoder_weight_report = {
                "missing_keys": 0,
                "unexpected_keys": 0,
                "note": "No trained SwinUNETR weights supplied; random encoder features are enabled only because allow_untrained_encoder=True.",
            }

        encoder.to(self.device)
        encoder.eval()
        return encoder

    def _load_encoder_state_dict(self, path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "encoder", "swinViT"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")

        converted: dict[str, Any] = {}
        for raw_key, value in payload.items():
            key = str(raw_key)
            if key.startswith("module."):
                key = key.removeprefix("module.")
            if key.startswith("swinViT."):
                key = key.removeprefix("swinViT.")
            if key.startswith("encoder."):
                key = key.removeprefix("encoder.")
            if any(part in key for part in ("convTrans3d", "rotation_head", "contrastive_head")):
                continue
            if key.startswith("norm."):
                continue
            key = key.replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2.")
            converted[key] = value
        return converted

    def _infer_encoder_shape(self, state_dict: dict[str, Any]) -> None:
        patch_weight = state_dict.get("patch_embed.proj.weight")
        if patch_weight is not None and hasattr(patch_weight, "shape"):
            shape = tuple(int(v) for v in patch_weight.shape)
            self.encoder_feature_size = shape[0]
            self.encoder_in_channels = shape[1]
            self.encoder_spatial_dims = max(2, len(shape) - 2)

        depths: list[int] = []
        for layer_idx in range(1, 5):
            block_ids: set[int] = set()
            prefix = f"layers{layer_idx}.0.blocks."
            for key in state_dict:
                if key.startswith(prefix) and key.endswith(".norm1.weight"):
                    parts = key.split(".")
                    if len(parts) > 3:
                        block_ids.add(int(parts[3]))
            depths.append(len(block_ids) or 1)
        self.encoder_depths = tuple(depths)  # type: ignore[assignment]

    def preprocess_with_monai(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            resized = cv2.resize(image_rgb, self.encoder_input_size[::-1], interpolation=cv2.INTER_AREA)
            resized = resized.astype(np.float32) / 255.0
            return np.moveaxis(resized, -1, 0)

        transformed = self.pipeline(image_rgb.astype(np.float32))
        if torch is not None and isinstance(transformed, torch.Tensor):
            transformed = transformed.detach().cpu().numpy()

        transformed = np.asarray(transformed, dtype=np.float32)
        if transformed.ndim == 3 and transformed.shape[-1] in (1, 3):
            transformed = np.moveaxis(transformed, -1, 0)
        return transformed

    def extract_transformer_features(self, image_rgb: np.ndarray, quality_report: dict[str, Any] | None = None) -> dict[str, Any]:
        transformed_chw = self.preprocess_with_monai(image_rgb)
        if self.encoder is None:
            classical_result = self._extract_classical_features(transformed_chw, quality_report or {})
            polyp_result = self._call_polyp_2d_tool(image_rgb)
            return self._merge_monai_and_polyp_results(classical_result, polyp_result)
        if self.encoder is None or torch is None:
            return self._extract_classical_features(transformed_chw, quality_report or {})
        swin_result = self._extract_swin_unetr_features(transformed_chw, quality_report or {})
        polyp_result = self._call_polyp_2d_tool(image_rgb)
        return self._merge_monai_and_polyp_results(swin_result, polyp_result)

    def __call__(self, image_rgb: np.ndarray, quality_report: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.extract_transformer_features(image_rgb, quality_report)

    def _call_polyp_2d_tool(self, image_rgb: np.ndarray) -> dict[str, Any] | None:
        if self.polyp_2d_tool is None:
            return None
        return self.polyp_2d_tool(image_rgb)

    @staticmethod
    def _merge_monai_and_polyp_results(
        monai_result: dict[str, Any],
        polyp_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        polyp_hint = (polyp_result or {}).get("candidate_region_hint") or {}
        monai_hint = monai_result.get("candidate_region_hint") or {}
        if polyp_hint.get("bbox") is not None:
            selected_hint = {**polyp_hint, "selected_source": "2d_polyp_model"}
        elif monai_hint.get("bbox") is not None:
            selected_hint = {**monai_hint, "selected_source": "monai_saliency"}
        else:
            selected_hint = {"bbox": None, "confidence": 0.0, "selected_source": "none"}

        return {
            **monai_result,
            "transformer_backend": "monai_transforms_with_2d_polyp_model",
            "polyp_2d_tool_result": polyp_result,
            "candidate_region_hint": selected_hint,
        }

    def _extract_swin_unetr_features(self, image_chw: np.ndarray, quality_report: dict[str, Any]) -> dict[str, Any]:
        tensor = self._prepare_encoder_tensor(image_chw)
        with torch.no_grad():
            feature_maps = self.encoder.swinViT(tensor)

        high_level = feature_maps[-1].detach().float()
        mid_level = feature_maps[-2].detach().float()
        heatmap_tensor = mid_level.abs().mean(dim=1, keepdim=True)
        if heatmap_tensor.ndim == 5:
            heatmap_tensor = heatmap_tensor.mean(dim=2)
        heatmap_tensor = torch.nn.functional.interpolate(
            heatmap_tensor,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        heatmap = _normalize_uint8(heatmap_tensor.squeeze().cpu().numpy())

        if high_level.ndim == 5:
            pooled = torch.nn.functional.adaptive_avg_pool3d(high_level, output_size=1).flatten()
        else:
            pooled = torch.nn.functional.adaptive_avg_pool2d(high_level, output_size=1).flatten()
        embedding = pooled.cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        if embedding.size < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - embedding.size))
        semantic_embedding = [round(float(v), 6) for v in embedding[: self.embedding_dim]]

        image_hwc = np.moveaxis(image_chw, 0, -1)
        image_u8 = _normalize_uint8(image_hwc)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
        candidate_region_hint = _bbox_from_heatmap(heatmap)
        quality_score = self._quality_score_from_report(quality_report, gray)

        return {
            "tool_name": "MonaiTransformerTool",
            "monai_available": True,
            "transformer_backend": "monai_swin_unetr_encoder",
            "encoder_status": self.encoder_status,
            "encoder_weight_report": getattr(self, "encoder_weight_report", {}),
            "image_quality_score": round(quality_score, 4),
            "suspicious_heatmap": heatmap,
            "semantic_embedding": semantic_embedding,
            "candidate_region_hint": candidate_region_hint,
            "feature_shapes": [list(feature.shape) for feature in feature_maps],
        }

    def _prepare_encoder_tensor(self, image_chw: np.ndarray) -> Any:
        if torch is None:
            raise RuntimeError("torch is required for MONAI encoder inference")

        image_hwc = np.moveaxis(image_chw, 0, -1)
        image_hwc = np.clip(image_hwc, 0.0, 1.0).astype(np.float32)
        if self.encoder_in_channels == 1:
            gray = cv2.cvtColor(_normalize_uint8(image_hwc), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            if self.encoder_spatial_dims == 3:
                volume = np.repeat(gray[None, :, :], self.encoder_depth, axis=0)
                tensor_np = volume[None, :, :, :]
            else:
                tensor_np = gray[None, :, :]
        else:
            tensor_np = image_chw[: self.encoder_in_channels]
            if tensor_np.shape[0] < self.encoder_in_channels:
                pad = np.repeat(tensor_np[-1:], self.encoder_in_channels - tensor_np.shape[0], axis=0)
                tensor_np = np.concatenate([tensor_np, pad], axis=0)
            if self.encoder_spatial_dims == 3:
                tensor_np = np.repeat(tensor_np[:, None, :, :], self.encoder_depth, axis=1)

        return torch.from_numpy(tensor_np).float().unsqueeze(0).to(self.device)

    def _extract_classical_features(self, image_chw: np.ndarray, quality_report: dict[str, Any]) -> dict[str, Any]:
        image_hwc = np.moveaxis(image_chw, 0, -1) if image_chw.ndim == 3 and image_chw.shape[0] in (1, 3) else image_chw
        image_01 = np.clip(image_hwc, 0.0, 1.0)
        image_u8 = _normalize_uint8(image_01)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY) if image_u8.ndim == 3 else image_u8

        edges = cv2.Laplacian(gray, cv2.CV_32F)
        local_contrast = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), sigmaX=7))
        heatmap = _normalize_uint8(np.abs(edges) * 0.55 + local_contrast.astype(np.float32) * 0.45)
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
        heatmap = cv2.resize(heatmap, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)

        semantic_embedding = self._build_classical_embedding(image_u8, gray)
        candidate_region_hint = _bbox_from_heatmap(heatmap)

        quality_score = self._quality_score_from_report(quality_report, gray)
        return {
            "tool_name": "MonaiTransformerTool",
            "monai_available": self.monai_available,
            "transformer_backend": "monai_transforms_classical_saliency" if self.monai_available else "opencv_classical_fallback",
            "encoder_status": self.encoder_status,
            "encoder_weight_report": getattr(self, "encoder_weight_report", {}),
            "image_quality_score": round(quality_score, 4),
            "suspicious_heatmap": heatmap,
            "semantic_embedding": semantic_embedding,
            "candidate_region_hint": candidate_region_hint,
        }

    def _build_classical_embedding(self, image_u8: np.ndarray, gray: np.ndarray) -> list[float]:
        features: list[float] = []
        for channel in cv2.split(image_u8):
            hist = cv2.calcHist([channel], [0], None, [8], [0, 256]).flatten()
            hist = hist / max(float(hist.sum()), 1.0)
            features.extend(float(v) for v in hist)

        features.extend(
            [
                float(gray.mean() / 255.0),
                float(gray.std() / 128.0),
                float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0),
                float(np.percentile(gray, 90) / 255.0),
                float(np.percentile(gray, 10) / 255.0),
                float((gray > 245).mean()),
                float((gray < 20).mean()),
                float(cv2.Canny(gray, 80, 160).mean() / 255.0),
            ]
        )
        if len(features) < self.embedding_dim:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        return [round(float(v), 6) for v in features[: self.embedding_dim]]

    @staticmethod
    def _quality_score_from_report(quality_report: dict[str, Any], gray: np.ndarray) -> float:
        if quality_report:
            brightness = float(quality_report.get("brightness_score", gray.mean()))
            blur = float(quality_report.get("blur_score", cv2.Laplacian(gray, cv2.CV_64F).var()))
            contrast = float(quality_report.get("contrast_score", gray.std()))
        else:
            brightness = float(gray.mean())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast = float(gray.std())

        brightness_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
        blur_score = min(blur / 250.0, 1.0)
        contrast_score = min(contrast / 64.0, 1.0)
        return 0.35 * brightness_score + 0.35 * contrast_score + 0.30 * blur_score


class PolypSegmentation2DTool:
    """2D polyp segmentation tool using a Kvasir-SEG pretrained UNet3+ model."""

    def __init__(
        self,
        repo_id: str = "andreribeiro87/unet3plus-efficientnet-kvasir-seg",
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.repo_id = repo_id
        self.device = device
        self.threshold = threshold
        self.model = None
        self.config: dict[str, Any] = {}
        self.model_status = "not_loaded"
        self.snapshot_dir: Path | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
        except Exception as exc:
            self.model_status = f"dependency_missing: {exc}"
            return

        snapshot_dir = Path(snapshot_download(self.repo_id))
        self.snapshot_dir = snapshot_dir
        self.config = json.loads((snapshot_dir / "config.json").read_text(encoding="utf-8"))

        if str(snapshot_dir) not in sys.path:
            sys.path.insert(0, str(snapshot_dir))

        module_path = snapshot_dir / "modeling_unet3plus.py"
        spec = importlib.util.spec_from_file_location("hf_polyp_modeling_unet3plus", module_path)
        if spec is None or spec.loader is None:
            self.model_status = "modeling_module_load_failed"
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.UNet3PlusConfig(
            backbone=self.config.get("backbone", "efficientnet"),
            in_channels=int(self.config.get("in_channels", 3)),
            out_channels=int(self.config.get("out_channels", 1)),
            inter_ch=int(self.config.get("inter_ch", 64)),
            img_size=int(self.config.get("img_size", 256)),
        )
        model = module.UNet3PlusForSegmentation(config)
        state_dict = load_file(str(snapshot_dir / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_status = "loaded"
        self.load_report = {
            "repo_id": self.repo_id,
            "snapshot_dir": str(snapshot_dir),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "dataset": self.config.get("dataset"),
            "test_metrics": self.config.get("test_metrics", {}),
        }

    def __call__(self, image_rgb: np.ndarray) -> dict[str, Any]:
        if self.model is None or torch is None:
            return {
                "tool_name": "PolypSegmentation2DTool",
                "model_status": self.model_status,
                "polyp_probability_map": None,
                "candidate_region_hint": {"bbox": None, "confidence": 0.0, "source": "2d_model_unavailable"},
            }

        img_size = int(self.config.get("img_size", 256))
        resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(pixel_values=tensor)
            logits = output["logits"] if isinstance(output, dict) else output
            probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)

        probability_full = cv2.resize(probability, image_rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        heatmap = _normalize_uint8(probability_full)
        candidate = self._candidate_from_probability(probability_full)
        return {
            "tool_name": "PolypSegmentation2DTool",
            "model_status": self.model_status,
            "model_report": getattr(self, "load_report", {}),
            "polyp_probability_map": probability_full,
            "polyp_heatmap": heatmap,
            "candidate_region_hint": candidate,
        }

    def _candidate_from_probability(self, probability: np.ndarray) -> dict[str, Any]:
        mask = (probability >= self.threshold).astype(np.uint8) * 255
        if mask.sum() == 0:
            adaptive_threshold = max(0.15, float(np.percentile(probability, 98)))
            mask = (probability >= adaptive_threshold).astype(np.uint8) * 255
        else:
            adaptive_threshold = self.threshold

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"bbox": None, "confidence": 0.0, "source": "2d_polyp_model"}

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        pad_x = max(6, int(w * 0.12))
        pad_y = max(6, int(h * 0.12))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(probability.shape[1], x + w + pad_x)
        y2 = min(probability.shape[0], y + h + pad_y)
        region = probability[y1:y2, x1:x2]
        confidence = float(region.mean()) if region.size else float(probability.max())
        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(confidence, 4),
            "source": "2d_polyp_model",
            "threshold": round(float(adaptive_threshold), 4),
            "max_probability": round(float(probability.max()), 4),
            "mean_probability": round(float(probability.mean()), 4),
        }

    def _extract_classical_features(self, image_chw: np.ndarray, quality_report: dict[str, Any]) -> dict[str, Any]:
        image_hwc = np.moveaxis(image_chw, 0, -1) if image_chw.ndim == 3 and image_chw.shape[0] in (1, 3) else image_chw
        image_01 = np.clip(image_hwc, 0.0, 1.0)
        image_u8 = _normalize_uint8(image_01)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY) if image_u8.ndim == 3 else image_u8

        edges = cv2.Laplacian(gray, cv2.CV_32F)
        local_contrast = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), sigmaX=7))
        heatmap = _normalize_uint8(np.abs(edges) * 0.55 + local_contrast.astype(np.float32) * 0.45)
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
        heatmap = cv2.resize(heatmap, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)

        semantic_embedding = self._build_classical_embedding(image_u8, gray)
        candidate_region_hint = _bbox_from_heatmap(heatmap)

        quality_score = self._quality_score_from_report(quality_report, gray)
        return {
            "tool_name": "MonaiTransformerTool",
            "monai_available": self.monai_available,
            "transformer_backend": "monai_transforms_classical_saliency" if self.monai_available else "opencv_classical_fallback",
            "encoder_status": self.encoder_status,
            "encoder_weight_report": getattr(self, "encoder_weight_report", {}),
            "image_quality_score": round(quality_score, 4),
            "suspicious_heatmap": heatmap,
            "semantic_embedding": semantic_embedding,
            "candidate_region_hint": candidate_region_hint,
        }

    def _build_classical_embedding(self, image_u8: np.ndarray, gray: np.ndarray) -> list[float]:
        features: list[float] = []
        for channel in cv2.split(image_u8):
            hist = cv2.calcHist([channel], [0], None, [8], [0, 256]).flatten()
            hist = hist / max(float(hist.sum()), 1.0)
            features.extend(float(v) for v in hist)

        features.extend(
            [
                float(gray.mean() / 255.0),
                float(gray.std() / 128.0),
                float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0),
                float(np.percentile(gray, 90) / 255.0),
                float(np.percentile(gray, 10) / 255.0),
                float((gray > 245).mean()),
                float((gray < 20).mean()),
                float(cv2.Canny(gray, 80, 160).mean() / 255.0),
            ]
        )
        if len(features) < self.embedding_dim:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        return [round(float(v), 6) for v in features[: self.embedding_dim]]

    @staticmethod
    def _quality_score_from_report(quality_report: dict[str, Any], gray: np.ndarray) -> float:
        if quality_report:
            brightness = float(quality_report.get("brightness_score", gray.mean()))
            blur = float(quality_report.get("blur_score", cv2.Laplacian(gray, cv2.CV_64F).var()))
            contrast = float(quality_report.get("contrast_score", gray.std()))
        else:
            brightness = float(gray.mean())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast = float(gray.std())

        brightness_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
        blur_score = min(blur / 250.0, 1.0)
        contrast_score = min(contrast / 64.0, 1.0)
        return 0.35 * brightness_score + 0.35 * contrast_score + 0.30 * blur_score


class PreprocessAgent:
    """Quality-control and preprocessing agent for endoscopic images."""

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        monai_tool: MonaiTransformerTool | None = None,
        polyp_2d_tool: PolypSegmentation2DTool | None = None,
    ):
        self.config = config or PreprocessConfig()
        self.monai_tool = monai_tool or MonaiTransformerTool(
            target_size=self.config.target_size,
            embedding_dim=self.config.embedding_dim,
            encoder_input_size=self.config.monai_encoder_input_size,
            encoder_depth=self.config.monai_encoder_depth,
            encoder_weights_path=self.config.monai_encoder_weights_path,
            device=self.config.monai_encoder_device,
            allow_untrained_encoder=self.config.allow_untrained_monai_encoder,
            use_2d_polyp_model=self.config.use_2d_polyp_model,
            polyp_model_repo_id=self.config.polyp_model_repo_id,
            polyp_model_threshold=self.config.polyp_model_threshold,
            polyp_2d_tool=polyp_2d_tool,
        )

    def check_image_quality(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        report: dict[str, Any] = {
            "is_dark": False,
            "is_overexposed": False,
            "is_blurry": False,
            "is_low_contrast": False,
            "brightness_score": 0.0,
            "blur_score": 0.0,
            "contrast_score": 0.0,
            "overexposed_ratio": 0.0,
            "image_shape": None,
            "warnings": [],
            "errors": [],
        }

        if not path.exists():
            report["errors"].append(f"file_not_found: {path}")
            return report
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            report["warnings"].append(f"unsupported_extension: {path.suffix}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            report["errors"].append("image_decode_failed")
            return report

        h, w = image.shape[:2]
        report["image_shape"] = [int(h), int(w), int(image.shape[2])]
        t = self.config.thresholds
        if w < t.min_width or h < t.min_height or w > t.max_width or h > t.max_height:
            report["warnings"].append("abnormal_image_size")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        overexposed_ratio = float((gray >= t.overexposed_pixel_threshold).mean())

        report["brightness_score"] = round(brightness, 4)
        report["blur_score"] = round(blur, 4)
        report["contrast_score"] = round(contrast, 4)
        report["overexposed_ratio"] = round(overexposed_ratio, 6)

        report["is_dark"] = brightness < t.dark_mean_threshold
        report["is_overexposed"] = overexposed_ratio > t.overexposed_ratio_threshold
        report["is_blurry"] = blur < t.blur_laplacian_var_threshold
        report["is_low_contrast"] = contrast < t.low_contrast_std_threshold

        if report["is_dark"]:
            report["warnings"].append("image_too_dark")
        if report["is_overexposed"]:
            report["warnings"].append("image_overexposed")
        if report["is_blurry"]:
            report["warnings"].append("image_blurry")
        if report["is_low_contrast"]:
            report["warnings"].append("image_low_contrast")

        return report

    def basic_preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, self.config.target_size[::-1], interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        kernel = max(1, int(self.config.denoise_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        denoised = cv2.medianBlur(enhanced_rgb, kernel)
        return denoised.astype(np.float32) / 255.0

    def call_monai_tool(self, image: np.ndarray, quality_report: dict[str, Any] | None = None) -> dict[str, Any]:
        if image.dtype != np.uint8:
            image_rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image_rgb = image
        return self.monai_tool(image_rgb, quality_report)

    def run(self, image_path: str | Path, target_size: tuple[int, int] | None = None) -> dict[str, Any]:
        if target_size is not None and target_size != self.config.target_size:
            self.config = PreprocessConfig(
                target_size=target_size,
                thresholds=self.config.thresholds,
                clahe_clip_limit=self.config.clahe_clip_limit,
                clahe_tile_grid_size=self.config.clahe_tile_grid_size,
                denoise_kernel_size=self.config.denoise_kernel_size,
                always_call_monai_tool=self.config.always_call_monai_tool,
                call_monai_on_warnings=self.config.call_monai_on_warnings,
                embedding_dim=self.config.embedding_dim,
                monai_encoder_input_size=self.config.monai_encoder_input_size,
                monai_encoder_depth=self.config.monai_encoder_depth,
                monai_encoder_weights_path=self.config.monai_encoder_weights_path,
                monai_encoder_device=self.config.monai_encoder_device,
                allow_untrained_monai_encoder=self.config.allow_untrained_monai_encoder,
                use_2d_polyp_model=self.config.use_2d_polyp_model,
                polyp_model_repo_id=self.config.polyp_model_repo_id,
                polyp_model_threshold=self.config.polyp_model_threshold,
            )
            self.monai_tool = MonaiTransformerTool(
                target_size=target_size,
                embedding_dim=self.config.embedding_dim,
                encoder_input_size=self.config.monai_encoder_input_size,
                encoder_depth=self.config.monai_encoder_depth,
                encoder_weights_path=self.config.monai_encoder_weights_path,
                device=self.config.monai_encoder_device,
                allow_untrained_encoder=self.config.allow_untrained_monai_encoder,
                use_2d_polyp_model=self.config.use_2d_polyp_model,
                polyp_model_repo_id=self.config.polyp_model_repo_id,
                polyp_model_threshold=self.config.polyp_model_threshold,
            )

        quality_report = self.check_image_quality(image_path)
        if quality_report["errors"]:
            return {
                "status": "reject",
                "quality_report": quality_report,
                "preprocessed_image": None,
                "monai_tool_result": None,
                "candidate_region_hint": {"bbox": None, "confidence": 0.0, "selected_source": "none"},
                "next_step_suggestion": "reject_input",
            }

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            quality_report["errors"].append("image_decode_failed_after_quality_check")
            return {
                "status": "reject",
                "quality_report": quality_report,
                "preprocessed_image": None,
                "monai_tool_result": None,
                "candidate_region_hint": {"bbox": None, "confidence": 0.0, "selected_source": "none"},
                "next_step_suggestion": "reject_input",
            }

        preprocessed_image = self.basic_preprocess(image_bgr)
        status = self._decide_status(quality_report)
        should_call_monai = self._should_call_monai_tool(quality_report)
        monai_result = self.call_monai_tool(preprocessed_image, quality_report) if should_call_monai else None
        polyp_2d_result = (monai_result or {}).get("polyp_2d_tool_result")
        candidate_region_hint = (monai_result or {}).get(
            "candidate_region_hint",
            {"bbox": None, "confidence": 0.0, "selected_source": "none"},
        )

        return {
            "status": status,
            "quality_report": quality_report,
            "preprocessed_image": {
                "array": preprocessed_image,
                "shape": list(preprocessed_image.shape),
                "dtype": str(preprocessed_image.dtype),
                "color_space": "RGB",
                "intensity_range": [0.0, 1.0],
            },
            "monai_tool_result": monai_result,
            "candidate_region_hint": candidate_region_hint,
            "next_step_suggestion": self._next_step(status, quality_report, monai_result, polyp_2d_result),
        }

    def _decide_status(self, quality_report: dict[str, Any]) -> str:
        if quality_report["errors"]:
            return "reject"

        t = self.config.thresholds
        severe_flags = [
            quality_report["brightness_score"] < t.severe_dark_threshold,
            quality_report["blur_score"] < t.severe_blur_threshold,
            quality_report["contrast_score"] < t.severe_contrast_threshold,
            "abnormal_image_size" in quality_report["warnings"],
        ]
        if sum(bool(flag) for flag in severe_flags) >= 2:
            return "reject"
        if quality_report["warnings"]:
            return "warning"
        return "success"

    def _should_call_monai_tool(self, quality_report: dict[str, Any]) -> bool:
        if self.config.always_call_monai_tool:
            return True
        if not self.config.call_monai_on_warnings:
            return False
        trigger_flags = (
            "image_low_contrast",
            "image_blurry",
            "image_too_dark",
            "image_overexposed",
        )
        return any(flag in quality_report["warnings"] for flag in trigger_flags)

    @staticmethod
    def _next_step(
        status: str,
        quality_report: dict[str, Any],
        monai_result: dict[str, Any] | None,
        polyp_2d_result: dict[str, Any] | None = None,
    ) -> str:
        if status == "reject":
            return "reject_input"
        if status == "warning":
            quality_score = 1.0
            if monai_result is not None:
                quality_score = float(monai_result.get("image_quality_score", 0.0))
            if quality_score < 0.35 or len(quality_report["warnings"]) >= 3:
                return "require_manual_review"
        if polyp_2d_result is not None:
            hint = polyp_2d_result.get("candidate_region_hint", {})
            if hint.get("bbox") is None and status == "warning":
                return "require_manual_review"
        return "send_to_bbox_agent"

    @staticmethod
    def _merge_candidate_region_hints(
        polyp_2d_result: dict[str, Any] | None,
        monai_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        polyp_hint = (polyp_2d_result or {}).get("candidate_region_hint") or {}
        if polyp_hint.get("bbox") is not None:
            return {
                **polyp_hint,
                "selected_source": "2d_polyp_model",
            }

        monai_hint = (monai_result or {}).get("candidate_region_hint") or {}
        if monai_hint.get("bbox") is not None:
            return {
                **monai_hint,
                "selected_source": "monai_swinvit_or_saliency",
            }

        return {
            "bbox": None,
            "confidence": 0.0,
            "selected_source": "none",
        }


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)) if value.size else 0.0,
            "max": float(np.max(value)) if value.size else 0.0,
            "mean": float(np.mean(value)) if value.size else 0.0,
        }
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _parse_target_size(raw: str) -> tuple[int, int]:
    normalized = raw.lower().replace("x", ",")
    parts = [int(part.strip()) for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target size must look like 512x512 or 512,512")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the endoscopy PreprocessAgent.")
    parser.add_argument("image_path", type=Path, help="Path to a jpg/png endoscopy image.")
    parser.add_argument("--target-size", type=_parse_target_size, default=(512, 512))
    parser.add_argument("--encoder-input-size", type=_parse_target_size, default=(256, 256))
    parser.add_argument("--encoder-depth", type=int, default=32)
    parser.add_argument("--encoder-weights", type=str, default=None, help="Optional trained SwinUNETR state_dict path.")
    parser.add_argument("--device", default="cpu", help="Torch device for MONAI encoder, for example cpu or cuda.")
    parser.add_argument("--disable-2d-polyp-model", action="store_true")
    parser.add_argument("--polyp-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-untrained-encoder",
        action="store_true",
        help="Run a randomly initialized SwinUNETR encoder. Useful only for plumbing tests.",
    )
    args = parser.parse_args()

    agent = PreprocessAgent(
        PreprocessConfig(
            target_size=args.target_size,
            monai_encoder_input_size=args.encoder_input_size,
            monai_encoder_depth=args.encoder_depth,
            monai_encoder_weights_path=args.encoder_weights,
            monai_encoder_device=args.device,
            allow_untrained_monai_encoder=args.allow_untrained_encoder,
            use_2d_polyp_model=not args.disable_2d_polyp_model,
            polyp_model_threshold=args.polyp_threshold,
        )
    )
    result = agent.run(args.image_path)
    print(json.dumps(to_serializable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
