from __future__ import annotations

import base64
import json
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
MODEL_RUNTIME_PRECISION = "fp32"


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
        return self.predict_with_context(image=image, retrieval_context=None)

    def predict_with_context(self, image: np.ndarray, retrieval_context: dict[str, Any] | None) -> dict[str, Any]:
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
        retrieval_package = None
        retrieval_response = None
        try:
            model_input = preprocess_result.tensor.to(
                self.device,
                non_blocking=self.device.startswith("cuda"),
            )
            prompt_bbox = self._build_prompt_bbox(preprocess_result=preprocess_result).to(model_input.device)
            retrieval_package, retrieval_response = self._build_retrieval_artifacts(
                preprocess_result=preprocess_result,
                retrieval_context=retrieval_context,
            )
            exemplar_prompt_tokens = None if retrieval_package is None else retrieval_package.prompt_tokens.to(model_input.device)
            retrieval_prior = None
            if retrieval_package is not None:
                retrieval_prior = {
                    key: value.to(model_input.device) if hasattr(value, "to") else value
                    for key, value in retrieval_package.retrieval_prior.items()
                }

            with torch.inference_mode():
                outputs = self.model(
                    model_input,
                    boxes=prompt_bbox,
                    exemplar_prompt_tokens=exemplar_prompt_tokens,
                    retrieval_prior=retrieval_prior,
                )

            mask_tensor = outputs["masks"].detach().float().cpu()
            binary_mask = (np.squeeze(mask_tensor[0].numpy()) > self.mask_threshold).astype(np.uint8)
            return self.postprocess(
                binary_mask=binary_mask,
                preprocess_result=preprocess_result,
                retrieval_response=retrieval_response,
            )
        finally:
            del outputs
            del mask_tensor
            del model_input
            if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def postprocess(
        self,
        binary_mask: np.ndarray,
        preprocess_result: PreprocessResult,
        retrieval_response: Any | None = None,
    ) -> dict[str, Any]:
        mask = (binary_mask > 0).astype(np.uint8)
        if mask.ndim != 2:
            raise ValueError("invalid mask shape")

        mask = mask[
            preprocess_result.pad_top : preprocess_result.pad_top + preprocess_result.resized_height,
            preprocess_result.pad_left : preprocess_result.pad_left + preprocess_result.resized_width,
        ]
        if mask.size == 0:
            return self._empty_result(retrieval_response=retrieval_response)

        if (
            preprocess_result.resized_width != preprocess_result.original_width
            or preprocess_result.resized_height != preprocess_result.original_height
        ):
            mask = cv2.resize(
                mask,
                (preprocess_result.original_width, preprocess_result.original_height),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_area_pixels = int(mask.sum())
        mask_data_url = self._encode_mask_data_url(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [
            contour for contour in contours if cv2.contourArea(contour) >= self.min_contour_area
        ]
        if not valid_contours:
            return self._empty_result(retrieval_response=retrieval_response)

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
        result = {
            "mask_data_url": mask_data_url,
            "mask_coordinates": mask_coordinates,
            "bounding_box": bounding_box,
            "mask_area_pixels": mask_area_pixels,
        }
        if retrieval_response is not None:
            result.update(
                {
                    "retrieval_applied": True,
                    "retrieval_confidence": retrieval_response.confidence,
                    "retrieval_uncertainty": retrieval_response.uncertainty,
                    "retrieval_candidate_count": retrieval_response.candidateCount,
                    "retrieval_bank_id": retrieval_response.bankId,
                    "retrieval_prior_keys": retrieval_response.priorKeys,
                }
            )
        return result

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
        mask = np.zeros((max(image_height, 1), max(image_width, 1)), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 1)
        return {
            "mask_data_url": self._encode_mask_data_url(mask),
            "mask_coordinates": [[int(x), int(y)] for x, y in polygon.tolist()],
            "bounding_box": [x_min, y_min, x_max, y_max],
            "mask_area_pixels": int(mask.sum()),
        }

    def predict_bytes(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        *,
        content_type: str | None = None,
        retrieval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image = self._decode_image(image_bytes=image_bytes)
        normalized_context = dict(retrieval_context or {})
        if filename:
            normalized_context.setdefault("filename", filename)
        if content_type:
            normalized_context.setdefault("content_type", content_type)
        normalized_context.setdefault("image_bytes", image_bytes)
        return self.predict_with_context(image=image, retrieval_context=normalized_context)

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
            from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
            from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model
            from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
        except Exception as exc:
            raise RuntimeError("failed to import MedicalSAM3 backend") from exc

        try:
            base_model = build_official_sam3_image_model(
                checkpoint_path=str(checkpoint_path),
                device=self.device,
                dtype=MODEL_RUNTIME_PRECISION,
                compile_model=False,
                allow_dummy_fallback=False,
            )
        except Exception as exc:
            raise RuntimeError("failed to initialize SAM3 model") from exc

        if self.settings.model_lora_enabled and self.settings.model_lora_path:
            lora_path = Path(self.settings.model_lora_path)
            if not lora_path.exists():
                raise FileNotFoundError(f"SAM3 LoRA checkpoint not found: {lora_path}")
            lora_config = LoRAConfig(stage=self.settings.model_lora_stage, min_replaced_modules=1)
            replaced_modules = apply_lora_to_model(base_model, lora_config)
            missing_keys, unexpected_keys = load_lora_weights(base_model, lora_path, strict=False)
            matched_lora_keys = self._count_loaded_lora_keys(base_model=base_model, lora_path=lora_path)
            if matched_lora_keys <= 0:
                raise RuntimeError(f"LoRA checkpoint did not match any injected modules: {lora_path}")
            if unexpected_keys:
                logger.warning(
                    "Loaded SAM3 LoRA checkpoint with %s unexpected keys: %s",
                    len(unexpected_keys),
                    unexpected_keys[:8],
                )
            logger.info(
                "Loaded SAM3 LoRA weights from %s with stage=%s replaced_modules=%s matched_lora_keys=%s missing_keys=%s",
                lora_path,
                self.settings.model_lora_stage,
                len(replaced_modules),
                matched_lora_keys,
                len(missing_keys),
            )

        model = Sam3TensorForwardWrapper(
            model=base_model,
            device=self.device,
            dtype=MODEL_RUNTIME_PRECISION,
            use_hooks=False,
        )
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
    def _empty_result(retrieval_response: Any | None = None) -> dict[str, Any]:
        payload = {
            "mask_data_url": "",
            "mask_coordinates": [],
            "bounding_box": [0, 0, 0, 0],
            "mask_area_pixels": 0,
        }
        if retrieval_response is not None:
            payload.update(
                {
                    "retrieval_applied": True,
                    "retrieval_confidence": retrieval_response.confidence,
                    "retrieval_uncertainty": retrieval_response.uncertainty,
                    "retrieval_candidate_count": retrieval_response.candidateCount,
                    "retrieval_bank_id": retrieval_response.bankId,
                    "retrieval_prior_keys": retrieval_response.priorKeys,
                }
            )
        return payload

    @staticmethod
    def _encode_mask_data_url(mask: np.ndarray) -> str:
        if mask.ndim != 2:
            raise ValueError("mask image must be 2D")

        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        foreground = mask.astype(bool)
        rgba[foreground] = np.array([16, 185, 129, 160], dtype=np.uint8)
        success, encoded = cv2.imencode(".png", rgba)
        if not success:
            raise RuntimeError("failed to encode mask png")
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    @staticmethod
    def _count_loaded_lora_keys(base_model: Any, lora_path: Path) -> int:
        if torch is None:
            return 0

        payload = torch.load(lora_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return 0

        model_keys = set(base_model.state_dict().keys())
        prefixes = ("wrapper.model.", "model.", "module.")
        matched_keys = 0

        for key, value in payload.items():
            if not isinstance(key, str) or not hasattr(value, "shape"):
                continue

            candidates = [key]
            pending = [key]
            seen = {key}
            while pending:
                current = pending.pop()
                for prefix in prefixes:
                    if not current.startswith(prefix):
                        continue
                    stripped = current[len(prefix) :]
                    if stripped in seen:
                        continue
                    seen.add(stripped)
                    candidates.append(stripped)
                    pending.append(stripped)

            if any(candidate in model_keys for candidate in candidates):
                matched_keys += 1

        return matched_keys

    def _build_retrieval_artifacts(
        self,
        *,
        preprocess_result: PreprocessResult,
        retrieval_context: dict[str, Any] | None,
    ) -> tuple[Any | None, Any | None]:
        if not retrieval_context:
            return None, None

        bank_id = str(retrieval_context.get("bank_id", "default-bank") or "default-bank")
        try:
            image_bytes = retrieval_context.get("image_bytes")
            if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
                return None, None

            workspace_dir = str(Path(__file__).resolve().parents[3])
            if workspace_dir not in sys.path:
                sys.path.insert(0, workspace_dir)

            from app.schemas.workspace import (
                ExpertConfigurationSchema,
                ExemplarRetrievalRequestSchema,
                ParisDetailSchema,
                WorkspaceImageSchema,
                WorkspacePatientSchema,
                WorkspaceSegmentationSchema,
            )
            from app.services.exemplar_bank_service import ExemplarBankService

            patient_payload = self._parse_json_payload(retrieval_context.get("patient_payload"))
            expert_payload = self._parse_json_payload(retrieval_context.get("expert_config_payload"))
            content_type = str(retrieval_context.get("content_type", "") or "image/png")
            filename = str(retrieval_context.get("filename", "") or "segment-frame.png")
            image_data_url = self._encode_input_image_data_url(bytes(image_bytes), content_type=content_type)

            request = ExemplarRetrievalRequestSchema(
                patient=WorkspacePatientSchema(**patient_payload) if patient_payload else WorkspacePatientSchema(),
                image=WorkspaceImageSchema(
                    filename=filename,
                    contentType=content_type,
                    dataUrl=image_data_url,
                    width=preprocess_result.original_width,
                    height=preprocess_result.original_height,
                ),
                segmentation=WorkspaceSegmentationSchema(
                    maskDataUrl="",
                    maskCoordinates=[],
                    boundingBox=(0, 0, preprocess_result.original_width - 1, preprocess_result.original_height - 1),
                    maskAreaPixels=preprocess_result.original_width * preprocess_result.original_height,
                    maskAreaRatio=1.0,
                    pointCount=4,
                ),
                expertConfig=ExpertConfigurationSchema(**expert_payload) if expert_payload else ExpertConfigurationSchema(parisDetail=ParisDetailSchema()),
                topK=int(retrieval_context.get("top_k", 6) or 6),
                bankId=bank_id,
            )
            service = ExemplarBankService()
            response, package = service.build_retrieval_artifacts(request)
            if package is None or response.candidateCount <= 0:
                return None, None
            return package, response
        except Exception as exc:
            logger.warning("failed to build retrieval prior for SAM3 inference: %s", exc)
            return None, None

    @staticmethod
    def _parse_json_payload(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str) and payload.strip():
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _encode_input_image_data_url(image_bytes: bytes, *, content_type: str) -> str:
        payload = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{content_type};base64,{payload}"


class SAM3RuntimeSingleton:
    _instance: "SAM3RuntimeSingleton | None" = None
    _lock = threading.Lock()
    _last_reload_error: str | None = None
    _preload_thread: threading.Thread | None = None
    _preload_error: str | None = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = SAM3Engine(settings=settings)

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> "SAM3RuntimeSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings=settings or get_settings())
                    cls._last_reload_error = None
        return cls._instance

    @classmethod
    def ensure_preload_started(cls, settings: Settings | None = None) -> dict[str, Any]:
        instance = cls.peek_instance()
        if instance is not None:
            return cls.get_preload_status()

        with cls._lock:
            if cls._instance is not None:
                return cls.get_preload_status()

            if cls._preload_thread is not None and cls._preload_thread.is_alive():
                return cls.get_preload_status()

            runtime_settings = settings or get_settings()
            cls._preload_error = None
            thread = threading.Thread(
                target=cls._preload_worker,
                args=(runtime_settings,),
                name="sam3-preload",
                daemon=True,
            )
            cls._preload_thread = thread
            thread.start()

        return cls.get_preload_status()

    @classmethod
    def _preload_worker(cls, settings: Settings) -> None:
        try:
            cls.get_instance(settings=settings)
        except Exception as exc:  # pragma: no cover - runtime dependent
            cls._preload_error = str(exc)
            logger.exception("SAM3 preload failed: %s", exc)

    @classmethod
    def get_preload_status(cls) -> dict[str, Any]:
        instance = cls._instance
        preload_thread = cls._preload_thread
        in_progress = preload_thread is not None and preload_thread.is_alive()

        return {
            "started": instance is not None or preload_thread is not None,
            "ready": instance is not None,
            "in_progress": in_progress,
            "load_mode": instance.engine.settings.model_load_mode if instance is not None else "",
            "device": instance.engine.device if instance is not None else "",
            "warmup_enabled": instance.engine.settings.model_warmup_enabled if instance is not None else False,
            "last_error": cls._preload_error or "",
        }

    @classmethod
    def peek_instance(cls) -> "SAM3RuntimeSingleton | None":
        return cls._instance

    @classmethod
    def get_last_reload_error(cls) -> str | None:
        return cls._last_reload_error

    @classmethod
    def reload_instance(cls, settings: Settings | None = None) -> "SAM3RuntimeSingleton":
        with cls._lock:
            previous_instance = cls._instance
            try:
                next_instance = cls(settings=settings or get_settings())
            except Exception as exc:
                cls._last_reload_error = str(exc)
                cls._instance = previous_instance
                raise

            cls._instance = next_instance
            cls._last_reload_error = None
            return next_instance

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
