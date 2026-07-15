"""YOLO 边界框适配器，用于生成 SAM3 提示。"""

from .bbox_provider import YoloBoxProvider, create_box_provider
from .detector import BBoxDetection, UltralyticsYoloDetector

__all__ = [
    "BBoxDetection",
    "UltralyticsYoloDetector",
    "YoloBoxProvider",
    "create_box_provider",
]
