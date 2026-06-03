from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import base64

from app.services.segmentation_pipeline_service import SegmentationPipelineService


class FakeEngine:
    def predict_bytes(self, image_bytes, filename, *, content_type=None, retrieval_context=None):
        assert retrieval_context is not None
        assert retrieval_context["filename"] == filename
        assert retrieval_context["image_bytes"] == image_bytes
        assert "pipeline_candidate_box" in retrieval_context
        return {
            "mask_data_url": "",
            "mask_coordinates": [[1, 1], [8, 1], [8, 8], [1, 8]],
            "bounding_box": [1, 1, 8, 8],
            "mask_area_pixels": 49,
        }


def _png_bytes() -> bytes:
    payload = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNsaGgAAwAD"
        "AZGfQPYAAAAASUVORK5CYII="
    )
    return base64.b64decode(payload)


def test_pipeline_falls_back_when_yolo_weights_are_missing(tmp_path: Path):
    settings = SimpleNamespace(
        upload_dir=str(tmp_path),
        preprocess_enabled=False,
        yolo_detection_enabled=True,
        yolo_weights_path=str(tmp_path / "missing-yolo.pt"),
        yolo_confidence_threshold=0.25,
    )
    service = SegmentationPipelineService(settings=settings, engine=FakeEngine())

    result = service.segment_image_bytes(
        image_bytes=_png_bytes(),
        filename="case.png",
        content_type="image/png",
        retrieval_context={"bank_id": "default-bank"},
    )

    assert result["mask_area_pixels"] == 49
    assert result["preprocess_status"] == "disabled"
    assert result["quality_warnings"] == []
    assert result["candidate_source"] in {"preprocess", "none"}
    assert result["pipeline_warnings"]
    assert any("YOLO" in warning for warning in result["pipeline_warnings"])
