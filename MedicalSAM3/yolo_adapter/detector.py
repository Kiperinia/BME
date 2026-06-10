"""Ultralytics YOLO detector wrapper used to generate SAM3 bbox prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BBoxDetection:
    xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str = "polyp"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UltralyticsYoloDetector:
    """Lazy YOLOv8 detector that returns the highest-confidence bbox."""

    def __init__(
        self,
        weights: str | Path = "yolo/models/yolov8_polyp.pt",
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
        imgsz: int | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("ultralytics is required for YOLO bbox inference") from exc
        self._model = YOLO(str(self.weights))
        return self._model

    def predict_one(self, source: str | Path) -> BBoxDetection | None:
        image = self._load_image_array(source)
        if image is None:
            return None
        return self._predict_source(image)

    def predict_one_array(self, image: Any) -> BBoxDetection | None:
        return self._predict_source(image)

    def _predict_source(self, source: Any) -> BBoxDetection | None:
        model = self._load_model()
        kwargs: dict[str, Any] = {
            "source": source,
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
            "save": False,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz

        results = model.predict(**kwargs)
        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        best_index = 0
        confidences = getattr(boxes, "conf", None)
        if confidences is not None and len(confidences) > 0:
            best_index = int(confidences.argmax().item())
            confidence = float(confidences[best_index].item())
        else:
            confidence = 0.0

        xyxy_tensor = boxes.xyxy[best_index].detach().cpu().float()
        class_tensor = getattr(boxes, "cls", None)
        class_id = int(class_tensor[best_index].item()) if class_tensor is not None and len(class_tensor) > 0 else 0
        names = getattr(result, "names", {}) or {}
        class_name = str(names.get(class_id, "polyp")) if isinstance(names, dict) else "polyp"
        return BBoxDetection(
            xyxy=[float(value) for value in xyxy_tensor.tolist()],
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
        )

    @staticmethod
    def _load_image_array(source: str | Path) -> np.ndarray | None:
        target = Path(source)
        if not target.is_file():
            return None
        try:
            if target.stem.endswith("_0000"):
                channel_paths = [target.with_name(target.name.replace("_0000", f"_000{i}")) for i in range(3)]
                if all(channel_path.exists() for channel_path in channel_paths):
                    channels = [np.asarray(Image.open(channel_path).convert("L")) for channel_path in channel_paths]
                    return np.stack(channels, axis=-1)
            return np.asarray(Image.open(target).convert("RGB"))
        except Exception:
            return None
