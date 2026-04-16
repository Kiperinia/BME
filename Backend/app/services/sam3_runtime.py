from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import Settings, get_settings

try:
    import torch
except Exception:
    torch = None


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessResult:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale_x: float
    scale_y: float
    normalized_image: np.ndarray
    tensor: Any | None


class SAM3Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = settings.model_device
        self.input_size = settings.model_input_size
        self.keep_aspect_ratio = settings.model_keep_aspect_ratio
        self.mask_threshold = settings.model_mask_threshold
        self.polygon_epsilon_ratio = settings.model_polygon_epsilon_ratio
        self.min_contour_area = settings.model_min_contour_area
        self.model = self._load_model()

        if self.settings.model_warmup_enabled:
            self._warmup()

    @property
    def is_mock_mode(self) -> bool:
        return self.settings.model_load_mode == "mock"

    def preprocess(self, image: np.ndarray) -> PreprocessResult:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("invalid or corrupted image")

        if image.ndim == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError("unsupported image layout")

        original_height, original_width = rgb_image.shape[:2]
        if original_width <= 0 or original_height <= 0:
            raise ValueError("invalid or corrupted image")

        if self.keep_aspect_ratio:
            resize_ratio = min(self.input_size / original_width, self.input_size / original_height)
            resized_width = max(1, int(round(original_width * resize_ratio)))
            resized_height = max(1, int(round(original_height * resize_ratio)))
            resized_image = cv2.resize(
                rgb_image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            pad_left = (self.input_size - resized_width) // 2
            pad_top = (self.input_size - resized_height) // 2
            canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized_image
        else:
            resized_width = self.input_size
            resized_height = self.input_size
            pad_left = 0
            pad_top = 0
            canvas = cv2.resize(
                rgb_image,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )

        normalized_image = canvas.astype(np.float32) / 255.0
        tensor = None
        if torch is not None:
            tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0).contiguous()

        return PreprocessResult(
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            pad_left=pad_left,
            pad_top=pad_top,
            scale_x=resized_width / original_width,
            scale_y=resized_height / original_height,
            normalized_image=normalized_image,
            tensor=tensor,
        )

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        preprocess_result = self.preprocess(image=image)
        if self.is_mock_mode:
            return self.mock_predict(
                image_width=preprocess_result.original_width,
                image_height=preprocess_result.original_height,
            )

        if torch is None or preprocess_result.tensor is None:
            raise RuntimeError("PyTorch runtime is unavailable")

        model_input = None
        outputs = None
        mask_tensor = None
        try:
            model_input = preprocess_result.tensor.to(
                self.device,
                non_blocking=self.device.startswith("cuda"),
            )
            prompt_bbox = self._build_prompt_bbox(preprocess_result=preprocess_result).to(model_input.device)

            with torch.inference_mode():
                outputs = self.model(model_input, bboxes=prompt_bbox)

            mask_tensor = outputs["masks"].detach().float().cpu()
            binary_mask = (np.squeeze(mask_tensor[0].numpy()) > self.mask_threshold).astype(np.uint8)
            return self.postprocess(binary_mask=binary_mask, preprocess_result=preprocess_result)
        finally:
            del outputs
            del mask_tensor
            del model_input
            if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def postprocess(self, binary_mask: np.ndarray, preprocess_result: PreprocessResult) -> dict[str, Any]:
        mask = (binary_mask > 0).astype(np.uint8)
        if mask.ndim != 2:
            raise ValueError("invalid mask shape")

        mask = mask[
            preprocess_result.pad_top : preprocess_result.pad_top + preprocess_result.resized_height,
            preprocess_result.pad_left : preprocess_result.pad_left + preprocess_result.resized_width,
        ]
        if mask.size == 0:
            return self._empty_result()

        if (
            preprocess_result.resized_width != preprocess_result.original_width
            or preprocess_result.resized_height != preprocess_result.original_height
        ):
            mask = cv2.resize(
                mask,
                (preprocess_result.original_width, preprocess_result.original_height),
                interpolation=cv2.INTER_NEAREST,
            )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [
            contour for contour in contours if cv2.contourArea(contour) >= self.min_contour_area
        ]
        if not valid_contours:
            return self._empty_result()

        largest_contour = max(valid_contours, key=cv2.contourArea)
        epsilon = max(1.0, self.polygon_epsilon_ratio * cv2.arcLength(largest_contour, True))
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        if polygon.shape[0] < 3:
            polygon = largest_contour

        points = polygon.reshape(-1, 2).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, preprocess_result.original_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, preprocess_result.original_height - 1)

        x, y, width, height = cv2.boundingRect(largest_contour)
        bounding_box = [
            int(x),
            int(y),
            int(x + max(width - 1, 0)),
            int(y + max(height - 1, 0)),
        ]
        mask_coordinates = [[int(px), int(py)] for px, py in points.tolist()]
        return {
            "mask_coordinates": mask_coordinates,
            "bounding_box": bounding_box,
        }

    def mock_predict(self, image_width: int, image_height: int) -> dict[str, Any]:
        if self.settings.model_mock_delay_ms > 0:
            time.sleep(self.settings.model_mock_delay_ms / 1000)

        template = np.array(
            [
                [0.18, 0.22],
                [0.41, 0.21],
                [0.52, 0.34],
                [0.49, 0.52],
                [0.31, 0.61],
                [0.16, 0.47],
            ],
            dtype=np.float32,
        )
        polygon = np.empty_like(template, dtype=np.int32)
        polygon[:, 0] = np.clip(np.rint(template[:, 0] * image_width), 0, max(image_width - 1, 0)).astype(np.int32)
        polygon[:, 1] = np.clip(np.rint(template[:, 1] * image_height), 0, max(image_height - 1, 0)).astype(np.int32)

        x_min = int(polygon[:, 0].min())
        y_min = int(polygon[:, 1].min())
        x_max = int(polygon[:, 0].max())
        y_max = int(polygon[:, 1].max())
        return {
            "mask_coordinates": [[int(x), int(y)] for x, y in polygon.tolist()],
            "bounding_box": [x_min, y_min, x_max, y_max],
        }

    def predict_bytes(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        del filename
        image = self._decode_image(image_bytes=image_bytes)
        return self.predict(image=image)

    def predict_path(self, image_path: str) -> dict[str, Any]:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("invalid or corrupted image")
        return self.predict(image=image)

    def _load_model(self) -> Any | None:
        if self.is_mock_mode:
            logger.info("Loading SAM3Engine in mock mode")
            return None

        if torch is None:
            raise RuntimeError("PyTorch runtime is unavailable")

        checkpoint_path = Path(self.settings.model_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

        workspace_dir = str(Path(__file__).resolve().parents[3])
        if workspace_dir not in sys.path:
            sys.path.insert(0, workspace_dir)

        try:
            from MedicalSAM3.models.medsam3_base import build_medsam3
        except Exception as exc:
            raise RuntimeError("failed to import MedicalSAM3 backend") from exc

        try:
            model = build_medsam3(
                checkpoint_path=str(checkpoint_path),
                image_size=self.input_size,
                device=self.device,
                load_from_hf=False,
            )
        except Exception as exc:
            raise RuntimeError("failed to initialize SAM3 model") from exc

        model.eval()
        logger.info("SAM3Engine loaded on %s", self.device)
        return model

    def _warmup(self) -> None:
        warmup_image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        if self.is_mock_mode:
            self.mock_predict(image_width=self.input_size, image_height=self.input_size)
            logger.info("SAM3Engine mock warm-up finished")
            return

        self.predict(image=warmup_image)
        logger.info("SAM3Engine warm-up finished")

    def _build_prompt_bbox(self, preprocess_result: PreprocessResult) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch runtime is unavailable")

        x_min = float(preprocess_result.pad_left)
        y_min = float(preprocess_result.pad_top)
        x_max = float(preprocess_result.pad_left + preprocess_result.resized_width - 1)
        y_max = float(preprocess_result.pad_top + preprocess_result.resized_height - 1)
        return torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("empty image payload")

        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError("invalid or corrupted image")
        return image

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "mask_coordinates": [],
            "bounding_box": [0, 0, 0, 0],
        }


class SAM3RuntimeSingleton:
    _instance: "SAM3RuntimeSingleton | None" = None
    _lock = threading.Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = SAM3Engine(settings=settings)

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> "SAM3RuntimeSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings=settings or get_settings())
        return cls._instance

    def run_inference(self, image_path: str) -> dict[str, Any]:
        result = self.engine.predict_path(image_path=image_path)
        polygon = [{"x": x, "y": y} for x, y in result["mask_coordinates"]]
        lesions = []
        if polygon:
            lesions.append(
                {
                    "label": "sam3_segmentation",
                    "confidence": 1.0 if self.engine.is_mock_mode else 0.0,
                    "location": None,
                    "area_mm2": None,
                    "mask_coordinates": polygon,
                }
            )

        return {
            "image_path": image_path,
            "mask_coordinates": [polygon] if polygon else [],
            "bounding_box": result["bounding_box"],
            "lesions": lesions,
        }
