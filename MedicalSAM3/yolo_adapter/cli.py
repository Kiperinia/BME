"""CLI 辅助工具，用于在 SAM3 脚本中启用 YOLO 边界框提示。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .bbox_provider import create_box_provider


def add_yolo_bbox_args(parser: argparse.ArgumentParser) -> None:
    """向 ArgumentParser 添加 YOLO 边界框相关的命令行参数。

    参数：
        - parser: 要添加参数的命令行解析器。
    """
    parser.add_argument("--bbox-source", choices=["none", "mask", "yolo"], default="mask")
    parser.add_argument("--yolo-weights", default="yolo/models/yolov8_polyp.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-cache", default=None)
    parser.add_argument("--yolo-fallback", choices=["none", "full", "mask", "error"], default="none")


def build_box_provider_from_args(args: Any, *, default_cache_name: str | None = None) -> Any | None:
    """根据解析后的命令行参数构建边界框提供器。

    参数：
        - args: 解析后的命令行参数对象。
        - default_cache_name: 默认缓存文件名（可选）。

    返回：
        - 边界框提供器实例；若无可用提供器则返回 None。
    """
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
