"""边界框提供器，将 YOLO 检测结果映射为 SAM3 输入坐标。"""

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
    """从记录字典中提取唯一键值。

    参数：
        - record: 包含 image_path 和/或 image_id 的记录字典。

    返回：
        - 唯一字符串键值。
    """
    image_path = str(record.get("image_path", ""))
    image_id = str(record.get("image_id", ""))
    return image_id or Path(image_path).stem or image_path


def _image_size(path: str | Path) -> tuple[int, int] | None:
    """获取图像尺寸 (宽, 高)。

    参数：
        - path: 图像文件路径。

    返回：
        - 若文件存在则返回 (宽, 高) 元组，否则返回 None。
    """
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
    """将原始坐标中的 xyxy 缩放到方形图像坐标系。

    参数：
        - xyxy: 原始坐标系中的 [x1, y1, x2, y2] 列表。
        - original_size: 原始图像尺寸 (宽, 高)。
        - image_size: 目标方形图像边长。

    返回：
        - 缩放后的 [x1, y1, x2, y2] 张量，值域被裁剪到 [0, image_size]。
    """
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
    """基于 YOLO 检测结果的边界框提供器，负责为 SAM3 提供输入坐标。

    支持从 JSON 缓存读取检测结果，或在运行时调用 YOLO 模型进行预测。
    当检测缺失时，可根据 fallback 策略返回全图框、掩码框或哨兵值。

    参数：
        - weights: YOLO 模型权重路径。
        - conf: 检测置信度阈值。
        - iou: NMS 的 IoU 阈值。
        - device: 推理设备。
        - imgsz: 推理图像尺寸。
        - cache_path: JSON 缓存文件路径（可选）。
        - fallback: 检测缺失时的回退策略。
    """

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
        """初始化 YoloBoxProvider 实例。"""
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
        """获取记录对应的边界框张量。

        参数：
            - record: 数据记录字典。
            - image_size: 目标方形图像尺寸。
            - image: 图像张量（当前未使用）。
            - mask: 掩码张量，用于 mask 回退模式。
            - fallback_index: 回退索引（当前未使用）。

        返回：
            - 形状为 [4] 的边界框张量 [x1, y1, x2, y2]。
        """
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
        """从缓存加载或调用 YOLO 模型预测单个图像的边界框。

        参数：
            - key: 缓存键值。
            - image_path: 图像文件路径。

        返回：
            - 检测结果对象；若无法检测则返回 None。
        """
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
        """将当前缓存写入磁盘文件。"""
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")


class NoBoxProvider:
    """始终返回空哨兵值的边界框提供器。"""

    def get_box(
        self,
        record: dict[str, Any],
        image_size: int,
        *,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        fallback_index: int = 0,
    ) -> torch.Tensor:
        """返回已移除框的哨兵值。

        参数：
            - record: 数据记录（未使用）。
            - image_size: 图像尺寸（未使用）。
            - image: 图像张量（未使用）。
            - mask: 掩码张量（未使用）。
            - fallback_index: 回退索引（未使用）。

        返回：
            - 哨兵值张量，表示无检测框。
        """
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
    """根据 source 参数创建相应的边界框提供器。

    参数：
        - source: 边界框来源，"mask" 返回 None，"none" 返回 NoBoxProvider，"yolo" 返回 YoloBoxProvider。
        - yolo_weights: YOLO 模型权重路径。
        - yolo_conf: 检测置信度阈值。
        - yolo_iou: NMS 的 IoU 阈值。
        - yolo_device: 推理设备。
        - yolo_imgsz: 推理图像尺寸。
        - yolo_cache: 缓存路径。
        - yolo_fallback: 检测缺失时的回退策略。

    返回：
        - 边界框提供器实例；若 source 为 "mask" 则返回 None。
    """
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
