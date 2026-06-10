"""CLI helpers for enabling YOLO bbox prompts in SAM3 scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .bbox_provider import create_box_provider


def add_yolo_bbox_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bbox-source", choices=["none", "mask", "yolo"], default="mask")
    parser.add_argument("--yolo-weights", default="yolo/models/yolov8_polyp.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-cache", default=None)
    parser.add_argument("--yolo-fallback", choices=["none", "full", "mask", "error"], default="none")


def build_box_provider_from_args(args: Any, *, default_cache_name: str | None = None) -> Any | None:
    cache_path = getattr(args, "yolo_cache", None)
    if cache_path is None and default_cache_name:
        cache_path = Path("MedicalSAM3/outputs/medex_sam3/yolo_bbox_cache") / default_cache_name
    return create_box_provider(
        source=getattr(args, "bbox_source", "mask"),
        yolo_weights=getattr(args, "yolo_weights", "yolo/models/yolov8_polyp.pt"),
        yolo_conf=float(getattr(args, "yolo_conf", 0.25)),
        yolo_iou=float(getattr(args, "yolo_iou", 0.7)),
        yolo_device=getattr(args, "yolo_device", None),
        yolo_imgsz=getattr(args, "yolo_imgsz", None),
        yolo_cache=cache_path,
        yolo_fallback=getattr(args, "yolo_fallback", "full"),
    )
