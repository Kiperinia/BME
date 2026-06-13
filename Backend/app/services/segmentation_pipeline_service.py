from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.sam3_runtime import SAM3Engine


logger = logging.getLogger(__name__)
WORKSPACE_DIR = Path(__file__).resolve().parents[3]


class SegmentationPipelineService:
    def __init__(self, settings: "Settings", engine: "SAM3Engine"):
        self.settings = settings
        self.engine = engine

    def segment_image_bytes(
        self,
        *,
        image_bytes: bytes,
        filename: str | None,
        content_type: str | None,
        retrieval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pipeline_warnings: list[str] = []
        temp_path = self._write_temp_image(image_bytes=image_bytes, filename=filename)
        try:
            preprocess_result = self._run_preprocess(temp_path=temp_path, warnings=pipeline_warnings)
            yolo_result = self._run_yolo(temp_path=temp_path, warnings=pipeline_warnings)
            candidate = self._select_candidate(preprocess_result=preprocess_result, yolo_result=yolo_result)

            context = dict(retrieval_context or {})
            context.update(
                {
                    "filename": filename,
                    "content_type": content_type,
                    "image_bytes": image_bytes,
                    "pipeline_candidate_box": candidate["box"],
                    "pipeline_candidate_source": candidate["source"],
                    "pipeline_candidate_confidence": candidate["confidence"],
                    "preprocess_status": preprocess_result["status"],
                    "quality_warnings": preprocess_result["quality_warnings"],
                    "pipeline_warnings": pipeline_warnings,
                }
            )

            result = self.engine.predict_bytes(
                image_bytes,
                filename,
                content_type=content_type,
                retrieval_context=context,
            )
            result.update(
                {
                    "preprocess_status": preprocess_result["status"],
                    "quality_warnings": preprocess_result["quality_warnings"],
                    "candidate_box": candidate["box"],
                    "candidate_source": candidate["source"],
                    "candidate_confidence": candidate["confidence"],
                    "pipeline_warnings": pipeline_warnings,
                }
            )
            return result
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("failed to remove temporary pipeline image %s: %s", temp_path, exc)

    def _write_temp_image(self, *, image_bytes: bytes, filename: str | None) -> Path:
        suffix = Path(filename or "upload.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            suffix = ".png"
        upload_dir = Path(self.settings.upload_dir) / "segmentation-pipeline"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(image_bytes)
        return path

    def _run_preprocess(self, *, temp_path: Path, warnings: list[str]) -> dict[str, Any]:
        if not bool(getattr(self.settings, "preprocess_enabled", True)):
            return {
                "status": "disabled",
                "quality_warnings": [],
                "candidate_box": None,
                "candidate_confidence": None,
                "candidate_source": "none",
            }

        agent_root = str((WORKSPACE_DIR / "agent").resolve())
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)

        try:
            from preprocess_agent.preprocess_agent import PreprocessAgent
        except Exception as exc:
            warnings.append(f"PreprocessAgent unavailable: {type(exc).__name__}: {exc}")
            return {
                "status": "unavailable",
                "quality_warnings": [],
                "candidate_box": None,
                "candidate_confidence": None,
                "candidate_source": "none",
            }

        try:
            payload = PreprocessAgent().run(temp_path)
        except Exception as exc:
            warnings.append(f"PreprocessAgent failed: {type(exc).__name__}: {exc}")
            return {
                "status": "failed",
                "quality_warnings": [],
                "candidate_box": None,
                "candidate_confidence": None,
                "candidate_source": "none",
            }

        quality_report = payload.get("quality_report") or {}
        hint = payload.get("candidate_region_hint") or {}
        return {
            "status": str(payload.get("status") or "unknown"),
            "quality_warnings": list(quality_report.get("warnings") or []),
            "candidate_box": self._normalize_box(hint.get("bbox")),
            "candidate_confidence": self._normalize_confidence(hint.get("confidence")),
            "candidate_source": str(hint.get("selected_source") or "preprocess"),
        }

    def _run_yolo(self, *, temp_path: Path, warnings: list[str]) -> dict[str, Any]:
        if not bool(getattr(self.settings, "yolo_detection_enabled", True)):
            return {"box": None, "confidence": None, "source": "none"}

        weights_path = Path(str(getattr(self.settings, "yolo_weights_path", "")))
        if not weights_path.is_absolute():
            weights_path = WORKSPACE_DIR / weights_path
        weights_path = weights_path.resolve()
        if not weights_path.exists():
            warnings.append(f"YOLO weights not found: {weights_path}")
            return {"box": None, "confidence": None, "source": "none"}

        try:
            from ultralytics import YOLO
        except Exception as exc:
            warnings.append(f"YOLO unavailable: {type(exc).__name__}: {exc}")
            return {"box": None, "confidence": None, "source": "none"}

        try:
            model = YOLO(str(weights_path))
            results = model.predict(
                source=str(temp_path),
                save=False,
                conf=float(getattr(self.settings, "yolo_confidence_threshold", 0.25)),
                verbose=False,
            )
        except Exception as exc:
            warnings.append(f"YOLO inference failed: {type(exc).__name__}: {exc}")
            return {"box": None, "confidence": None, "source": "none"}

        best_box = None
        best_confidence = None
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy_values = getattr(boxes, "xyxy", [])
            conf_values = getattr(boxes, "conf", [])
            for index, xyxy in enumerate(xyxy_values):
                confidence = self._tensor_item(conf_values[index]) if index < len(conf_values) else None
                if confidence is None:
                    continue
                if best_confidence is None or confidence > best_confidence:
                    best_confidence = confidence
                    best_box = self._normalize_box(xyxy)

        return {"box": best_box, "confidence": best_confidence, "source": "yolo" if best_box else "none"}

    @staticmethod
    def _select_candidate(*, preprocess_result: dict[str, Any], yolo_result: dict[str, Any]) -> dict[str, Any]:
        if yolo_result.get("box"):
            return {
                "box": yolo_result["box"],
                "source": "yolo",
                "confidence": yolo_result.get("confidence"),
            }
        if preprocess_result.get("candidate_box"):
            return {
                "box": preprocess_result["candidate_box"],
                "source": preprocess_result.get("candidate_source") or "preprocess",
                "confidence": preprocess_result.get("candidate_confidence"),
            }
        return {"box": None, "source": "none", "confidence": None}

    @staticmethod
    def _normalize_box(value: Any) -> list[int] | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            return [max(0, int(round(float(item)))) for item in value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_confidence(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(SegmentationPipelineService._tensor_item(value))))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_item(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
