"""BBox provider that maps YOLO detections into SAM3 input coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
import torch

from MedicalSAM3.scripts.common import full_image_box, mask_to_box, removed_box_sentinel
from .detector import BBoxDetection, UltralyticsYoloDetector


FallbackMode = Literal["none", "full", "mask", "error"]


def _record_key(record: dict[str, Any]) -> str:
    image_path = str(record.get("image_path", ""))
    image_id = str(record.get("image_id", ""))
    return image_id or Path(image_path).stem or image_path


def _image_size(path: str | Path) -> tuple[int, int] | None:
    target = Path(path)
    if not target.is_file():
        return None
    with Image.open(target) as image:
        return image.size


def _scale_xyxy_to_square(
    xyxy: list[float],
    *,
    original_size: tuple[int, int],
    image_size: int,
) -> torch.Tensor:
    width, height = original_size
    if width <= 0 or height <= 0:
        return full_image_box(image_size)
    sx = float(image_size) / float(width)
    sy = float(image_size) / float(height)
    x1, y1, x2, y2 = xyxy
    box = torch.tensor([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=torch.float32)
    clamp = torch.tensor([float(image_size), float(image_size), float(image_size), float(image_size)])
    box = torch.maximum(box, torch.zeros_like(box))
    box = torch.minimum(box, clamp)
    if box[2] <= box[0] or box[3] <= box[1]:
        return full_image_box(image_size)
    return box


class YoloBoxProvider:
    """Provides bbox prompts from YOLO detections, with optional JSON cache."""

    def __init__(
        self,
        *,
        weights: str | Path = "yolo/models/yolov8_polyp.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
        imgsz: int | None = None,
        cache_path: str | Path | None = None,
        fallback: FallbackMode = "none",
    ) -> None:
        self.detector = UltralyticsYoloDetector(weights=weights, conf=conf, iou=iou, device=device, imgsz=imgsz)
        self.cache_path = Path(cache_path) if cache_path else None
        self.fallback = fallback
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.cache = payload

    def get_box(
        self,
        record: dict[str, Any],
        image_size: int,
        *,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        fallback_index: int = 0,
    ) -> torch.Tensor:
        del image, fallback_index
        image_path = str(record.get("image_path", ""))
        key = _record_key(record)
        detection = self._load_or_predict(key, image_path)
        if detection is not None:
            original_size = _image_size(image_path) or (image_size, image_size)
            return _scale_xyxy_to_square(detection.xyxy, original_size=original_size, image_size=image_size)

        if self.fallback == "mask" and mask is not None:
            return mask_to_box(mask)
        if self.fallback == "error":
            raise RuntimeError(f"YOLO produced no bbox for record {key}")
        if self.fallback == "none":
            return removed_box_sentinel()
        return full_image_box(image_size)

    def _load_or_predict(self, key: str, image_path: str) -> BBoxDetection | None:
        cached = self.cache.get(key) or self.cache.get(Path(image_path).stem)
        if isinstance(cached, dict):
            xyxy = cached.get("xyxy") or cached.get("bbox")
            if isinstance(xyxy, list) and len(xyxy) == 4:
                return BBoxDetection(
                    xyxy=[float(value) for value in xyxy],
                    confidence=float(cached.get("confidence", cached.get("conf", 0.0))),
                    class_id=int(cached.get("class_id", 0)),
                    class_name=str(cached.get("class_name", "polyp")),
                )
        if not image_path or not Path(image_path).is_file():
            return None
        detection = self.detector.predict_one(image_path)
        if detection is not None:
            self.cache[key] = detection.to_dict()
            self._flush_cache()
        return detection

    def _flush_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")


class NoBoxProvider:
    """Always disables geometric bbox prompting."""

    def get_box(
        self,
        record: dict[str, Any],
        image_size: int,
        *,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        fallback_index: int = 0,
    ) -> torch.Tensor:
        del record, image_size, image, mask, fallback_index
        return removed_box_sentinel()


def create_box_provider(
    *,
    source: str = "mask",
    yolo_weights: str | Path = "yolo/models/yolov8_polyp.pt",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    yolo_device: str | None = None,
    yolo_imgsz: int | None = None,
    yolo_cache: str | Path | None = None,
    yolo_fallback: FallbackMode = "none",
) -> YoloBoxProvider | None:
    if source == "mask":
        return None
    if source == "none":
        return NoBoxProvider()
    if source != "yolo":
        raise ValueError(f"Unsupported bbox source: {source}")
    return YoloBoxProvider(
        weights=yolo_weights,
        conf=yolo_conf,
        iou=yolo_iou,
        device=yolo_device,
        imgsz=yolo_imgsz,
        cache_path=yolo_cache,
        fallback=yolo_fallback,
    )
