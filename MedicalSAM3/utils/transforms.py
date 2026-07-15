"""
数据增强与预处理
提供训练/验证阶段的图像与 mask 增强管线。
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False


def get_train_transforms(image_size: int = 1024) -> Any:
    """获取训练阶段的数据增强管线。

    包含翻转、旋转、仿射变换、模糊、亮度对比度调整、噪声和随机遮挡。

    参数：
        - image_size: 目标图像尺寸，默认 1024

    返回：
        - albumentations 组合变换或 ResizeNormalize 回退
    """
    if not HAS_ALBUM:
        return ResizeNormalize(image_size)

    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                 rotate=(-15, 15), p=0.5,
                 border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0, p=0.2,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 1024) -> Any:
    """获取验证阶段的变换管线（仅缩放和归一化）。

    参数：
        - image_size: 目标图像尺寸，默认 1024

    返回：
        - albumentations 组合变换或 ResizeNormalize 回退
    """
    if not HAS_ALBUM:
        return ResizeNormalize(image_size)

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class ResizeNormalize:
    """当 albumentations 不可用时的简单缩放和归一化回退实现。

    参数：
        - image_size: 目标图像尺寸
    """

    def __init__(self, image_size: int = 1024):
        """初始化回退变换。

        参数：
            - image_size: 目标图像尺寸，默认 1024
        """
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image: np.ndarray = None,
                 mask: np.ndarray = None, **kwargs) -> Dict[str, np.ndarray]:
        """执行缩放和归一化变换。

        参数：
            - image: 输入图像数组，可选
            - mask: 输入掩码数组，可选
            - **kwargs: 额外参数

        返回：
            - 包含变换后 image 和 mask 的字典
        """
        result = {}
        if image is not None:
            img = cv2.resize(image, (self.image_size, self.image_size))
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            result["image"] = img
        if mask is not None:
            msk = cv2.resize(mask, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_NEAREST)
            result["mask"] = msk
        return result


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """从二值掩码中提取边界框（非零区域的最小外接矩形）。

    参数：
        - mask: 二值掩码数组

    返回：
        - [xmin, ymin, xmax, ymax] 边界框数组
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return np.array([0, 0, 1, 1], dtype=np.float32)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def jitter_bbox(bbox: np.ndarray, jitter_ratio: float = 0.05,
                img_h: int = 1024, img_w: int = 1024) -> np.ndarray:
    """对边界框添加随机扰动，增强训练鲁棒性。

    参数：
        - bbox: 原始边界框 [xmin, ymin, xmax, ymax]
        - jitter_ratio: 扰动比例，默认 0.05
        - img_h: 图像高度，默认 1024
        - img_w: 图像宽度，默认 1024

    返回：
        - 扰动后的边界框数组
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    dx = w * jitter_ratio * (np.random.random() * 2 - 1)
    dy = h * jitter_ratio * (np.random.random() * 2 - 1)
    x_min = max(0, x_min + dx)
    y_min = max(0, y_min + dy)
    x_max = min(img_w, x_max + dx)
    y_max = min(img_h, y_max + dy)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
