"""YOLO bbox adapters for SAM3 prompt generation."""

from .bbox_provider import YoloBoxProvider, create_box_provider
from .detector import BBoxDetection, UltralyticsYoloDetector

__all__ = [
    "BBoxDetection",
    "UltralyticsYoloDetector",
    "YoloBoxProvider",
    "create_box_provider",
]
