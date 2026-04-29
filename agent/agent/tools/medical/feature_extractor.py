"""
feature_extractor.py — 从 mask + image 提取定量特征

输入：
  - image: np.ndarray (H, W, 3) BGR 格式内镜图像
  - mask:  np.ndarray (H, W)   二值掩码，前景=255

输出：
  - LesionFeatures dataclass，包含几何、颜色、纹理三大类特征

所有计算均为纯 OpenCV / NumPy，不依赖额外深度学习模型。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class SurfacePattern(str, Enum):
    """表面纹理模式（规则引擎判定）"""
    SMOOTH = "smooth"
    IRREGULAR = "irregular"
    GRANULAR = "granular"
    VILLOUS = "villous"
    UNKNOWN = "unknown"


class ColorTone(str, Enum):
    """病变颜色基调"""
    RED = "red"
    PALE = "pale"
    BROWN = "brown"
    MIXED = "mixed"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class GeometricFeatures:
    """几何特征"""
    area_px: int = 0                    # 掩码面积（像素）
    area_mm2: float | None = None       # 估算面积 mm²（需传入 pixel_size_mm）
    perimeter_px: float = 0.0           # 周长（像素）
    circularity: float = 0.0            # 4π·area / perimeter²  (0~1)
    aspect_ratio: float = 0.0           # 最小外接矩形宽高比
    convexity: float = 0.0              # 凸包面积 / 掩码面积
    solidity: float = 0.0               # 掩码面积 / 凸包面积
    equivalent_diameter_px: float = 0.0 # 等效圆直径
    major_axis_length: float = 0.0      # 拟合椭圆长轴
    minor_axis_length: float = 0.0      # 拟合椭圆短轴
    orientation_deg: float = 0.0        # 拟合椭圆角度（度）
    num_contours: int = 0               # 有效轮廓数（多连通域时 >1）
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)


@dataclass(slots=True)
class ColorFeatures:
    """颜色特征（在病变 ROI 内统计）"""
    mean_bgr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std_bgr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_hsv: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std_hsv: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dominant_color: ColorTone = ColorTone.UNKNOWN
    redness_ratio: float = 0.0          # R 通道均值 / (R+G+B)
    whiteness_ratio: float = 0.0        # (R+G+B)/3 / 255
    color_variance: float = 0.0         # ROI 内颜色方差（标量）
    border_contrast: float = 0.0        # 病变边缘与周围正常黏膜的对比度


@dataclass(slots=True)
class TextureFeatures:
    """纹理特征（基于灰度共生矩阵 GLCM 简化版）"""
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    entropy: float = 0.0
    energy: float = 0.0
    homogeneity: float = 0.0
    contrast: float = 0.0
    surface_pattern: SurfacePattern = SurfacePattern.UNKNOWN
    vessel_density: float = 0.0         # 血管纹理密度（Frangi 简化近似）


@dataclass(slots=True)
class LesionFeatures:
    """病变完整特征集"""
    geometric: GeometricFeatures = field(default_factory=GeometricFeatures)
    color: ColorFeatures = field(default_factory=ColorFeatures)
    texture: TextureFeatures = field(default_factory=TextureFeatures)
    raw_mask: np.ndarray | None = field(default=None, repr=False)
    roi_image: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典"""
        return {
            "geometric": {
                "area_px": self.geometric.area_px,
                "area_mm2": self.geometric.area_mm2,
                "perimeter_px": round(self.geometric.perimeter_px, 2),
                "circularity": round(self.geometric.circularity, 4),
                "aspect_ratio": round(self.geometric.aspect_ratio, 4),
                "convexity": round(self.geometric.convexity, 4),
                "solidity": round(self.geometric.solidity, 4),
                "equivalent_diameter_px": round(self.geometric.equivalent_diameter_px, 2),
                "major_axis_length": round(self.geometric.major_axis_length, 2),
                "minor_axis_length": round(self.geometric.minor_axis_length, 2),
                "orientation_deg": round(self.geometric.orientation_deg, 2),
                "num_contours": self.geometric.num_contours,
                "bbox": list(self.geometric.bbox),
            },
            "color": {
                "mean_bgr": [round(v, 2) for v in self.color.mean_bgr],
                "std_bgr": [round(v, 2) for v in self.color.std_bgr],
                "mean_hsv": [round(v, 2) for v in self.color.mean_hsv],
                "std_hsv": [round(v, 2) for v in self.color.std_hsv],
                "dominant_color": self.color.dominant_color.value,
                "redness_ratio": round(self.color.redness_ratio, 4),
                "whiteness_ratio": round(self.color.whiteness_ratio, 4),
                "color_variance": round(self.color.color_variance, 2),
                "border_contrast": round(self.color.border_contrast, 4),
            },
            "texture": {
                "mean_intensity": round(self.texture.mean_intensity, 2),
                "std_intensity": round(self.texture.std_intensity, 2),
                "entropy": round(self.texture.entropy, 4),
                "energy": round(self.texture.energy, 4),
                "homogeneity": round(self.texture.homogeneity, 4),
                "contrast": round(self.texture.contrast, 4),
                "surface_pattern": self.texture.surface_pattern.value,
                "vessel_density": round(self.texture.vessel_density, 4),
            },
        }


# ---------------------------------------------------------------------------
# 特征提取器
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    从内镜图像 + SAM3 掩码中提取完整的定量特征集。

    Usage::

        extractor = FeatureExtractor()
        features = extractor.extract(image_bgr, mask_binary)
        print(features.to_dict())
    """

    def __init__(
        self,
        pixel_size_mm: float | None = None,
        glcm_distance: int = 1,
        glcm_angles: int = 4,
        min_contour_area: int = 50,
    ):
        """
        Args:
            pixel_size_mm: 每像素对应的物理尺寸 (mm)，用于面积换算。
                           内镜通常 0.1~0.3 mm/px，不传则不计算 area_mm2。
            glcm_distance: GLCM 像素间距。
            glcm_angles:  GLCM 计算方向数（0°, 45°, 90°, 135°）。
            min_contour_area: 最小有效轮廓面积（像素）。
        """
        self.pixel_size_mm = pixel_size_mm
        self.glcm_distance = glcm_distance
        self.glcm_angles = glcm_angles
        self.min_contour_area = min_contour_area

    # ---- 公共接口 ----

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> LesionFeatures:
        """
        主入口：从 BGR 图像 + 二值掩码提取全部特征。

        Args:
            image: (H, W, 3) uint8 BGR 内镜图像
            mask:  (H, W) uint8 二值掩码，前景=255

        Returns:
            LesionFeatures
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be (H, W, 3) BGR")
        if mask.ndim != 2:
            raise ValueError("mask must be (H, W) 2D")
        if image.shape[:2] != mask.shape:
            raise ValueError("image and mask must have same spatial dimensions")

        binary = (mask > 127).astype(np.uint8)

        geo = self._extract_geometry(binary)
        color = self._extract_color(image, binary, geo)
        texture = self._extract_texture(image, binary, geo)

        return LesionFeatures(
            geometric=geo,
            color=color,
            texture=texture,
            raw_mask=binary,
            roi_image=self._crop_roi(image, binary),
        )

    def extract_from_polygon(
        self,
        image: np.ndarray,
        polygon: list[tuple[int, int]],
    ) -> LesionFeatures:
        """
        从多边形坐标直接提取特征（无需预先生成 mask）。

        Args:
            image: (H, W, 3) uint8 BGR
            polygon: [(x1,y1), (x2,y2), ...] 多边形顶点

        Returns:
            LesionFeatures
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return self.extract(image, mask)

    # ---- 几何特征 ----

    def _extract_geometry(self, binary: np.ndarray) -> GeometricFeatures:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

        if not valid:
            return GeometricFeatures()

        # 取最大轮廓
        largest = max(valid, key=cv2.contourArea)
        area = int(cv2.contourArea(largest))
        perimeter = cv2.arcLength(largest, True)

        # 圆度
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

        # 最小外接矩形
        rect = cv2.minAreaRect(largest)
        (w, h) = rect[1]
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0.0

        # 凸包
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        convexity = hull_area / area if area > 0 else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0

        # 等效直径
        equiv_diam = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0

        # 拟合椭圆
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            major = max(ellipse[1])
            minor = min(ellipse[1])
            orientation = ellipse[2]
        else:
            major, minor, orientation = 0.0, 0.0, 0.0

        # 边界框
        x, y, bw, bh = cv2.boundingRect(largest)
        bbox = (x, y, x + bw, y + bh)

        # 面积换算
        area_mm2 = None
        if self.pixel_size_mm is not None and self.pixel_size_mm > 0:
            area_mm2 = round(area * (self.pixel_size_mm ** 2), 2)

        return GeometricFeatures(
            area_px=area,
            area_mm2=area_mm2,
            perimeter_px=round(perimeter, 2),
            circularity=round(circularity, 4),
            aspect_ratio=round(aspect_ratio, 4),
            convexity=round(convexity, 4),
            solidity=round(solidity, 4),
            equivalent_diameter_px=round(equiv_diam, 2),
            major_axis_length=round(major, 2),
            minor_axis_length=round(minor, 2),
            orientation_deg=round(orientation, 2),
            num_contours=len(valid),
            bbox=bbox,
        )

    # ---- 颜色特征 ----

    def _extract_color(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        geo: GeometricFeatures,
    ) -> ColorFeatures:
        roi = self._crop_roi(image, binary)
        if roi.size == 0:
            return ColorFeatures()

        # BGR 统计
        mean_bgr = cv2.mean(roi)[:3]
        std_bgr = tuple(float(v) for v in cv2.meanStdDev(roi)[1].flatten())

        # HSV 统计
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_roi)[:3]
        std_hsv = tuple(float(v) for v in cv2.meanStdDev(hsv_roi)[1].flatten())

        # 颜色基调判定
        dominant = self._classify_color_tone(mean_bgr, mean_hsv)

        # 红色比例 (BGR → R is channel 2)
        b, g, r = mean_bgr
        total = b + g + r
        redness = r / total if total > 0 else 0.0

        # 亮度
        whiteness = total / (3.0 * 255.0)

        # 颜色方差（标量，取三通道方差均值）
        color_var = float(np.mean([v * v for v in std_bgr]))

        # 边缘对比度：病变边缘 vs 周围 5px 环带
        border_contrast = self._compute_border_contrast(image, binary, geo)

        return ColorFeatures(
            mean_bgr=tuple(round(float(v), 2) for v in mean_bgr),
            std_bgr=tuple(round(v, 2) for v in std_bgr),
            mean_hsv=tuple(round(float(v), 2) for v in mean_hsv),
            std_hsv=tuple(round(v, 2) for v in std_hsv),
            dominant_color=dominant,
            redness_ratio=round(redness, 4),
            whiteness_ratio=round(whiteness, 4),
            color_variance=round(color_var, 2),
            border_contrast=round(border_contrast, 4),
        )

    @staticmethod
    def _classify_color_tone(
        mean_bgr: tuple[float, ...],
        mean_hsv: tuple[float, ...],
    ) -> ColorTone:
        """基于 BGR 均值和 HSV 色调判定颜色基调"""
        b, g, r = mean_bgr
        h, s, v = mean_hsv

        # 低饱和度 → 偏白/正常
        if s < 30:
            return ColorTone.NORMAL if v > 100 else ColorTone.PALE

        # 高红色分量
        if r > 120 and r > g * 1.2 and r > b * 1.3:
            return ColorTone.RED

        # 棕色调（中等 R+G，低 B）
        if r > 80 and g > 60 and b < 80 and s > 30:
            return ColorTone.BROWN

        # 混合色
        if s > 40:
            return ColorTone.MIXED

        return ColorTone.NORMAL

    def _compute_border_contrast(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        geo: GeometricFeatures,
    ) -> float:
        """计算病变边缘与周围黏膜的对比度"""
        x1, y1, x2, y2 = geo.bbox
        h, w = binary.shape

        # 扩展 5px 环带
        pad = 5
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        ring_mask = np.zeros_like(binary)
        ring_mask[ry1:ry2, rx1:rx2] = 255
        ring_mask = ring_mask & (~binary.astype(bool)).astype(np.uint8)

        if np.sum(ring_mask) < 10:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lesion_mean = float(cv2.mean(gray, mask=binary)[0])
        ring_mean = float(cv2.mean(gray, mask=ring_mask)[0])

        if ring_mean == 0:
            return 0.0

        return abs(lesion_mean - ring_mean) / 255.0

    # ---- 纹理特征 ----

    def _extract_texture(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        geo: GeometricFeatures,
    ) -> TextureFeatures:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi_gray = self._crop_roi(gray, binary)

        if roi_gray.size == 0:
            return TextureFeatures()

        # 基本统计
        mean_int = float(np.mean(roi_gray))
        std_int = float(np.std(roi_gray))

        # GLCM 简化特征
        glcm_feats = self._compute_glcm_features(roi_gray)

        # 表面纹理模式判定
        surface = self._classify_surface_pattern(std_int, glcm_feats)

        # 血管密度（Frangi 简化：用 Top-Hat + 阈值近似）
        vessel_density = self._estimate_vessel_density(roi_gray)

        return TextureFeatures(
            mean_intensity=round(mean_int, 2),
            std_intensity=round(std_int, 2),
            entropy=round(glcm_feats["entropy"], 4),
            energy=round(glcm_feats["energy"], 4),
            homogeneity=round(glcm_feats["homogeneity"], 4),
            contrast=round(glcm_feats["contrast"], 4),
            surface_pattern=surface,
            vessel_density=round(vessel_density, 4),
        )

    def _compute_glcm_features(self, roi_gray: np.ndarray) -> dict[str, float]:
        """
        简化版 GLCM 特征提取。
        使用 NumPy 直接计算，避免依赖 skimage。
        """
        # 量化到 8 级以加速
        quantized = (roi_gray / 32).astype(np.uint8)
        quantized = np.clip(quantized, 0, 7)

        glcm = np.zeros((8, 8), dtype=np.float64)

        angles = [
            (0, self.glcm_distance),
            (self.glcm_distance, self.glcm_distance),
            (self.glcm_distance, 0),
            (self.glcm_distance, -self.glcm_distance),
        ][: self.glcm_angles]

        rows, cols = quantized.shape
        for dy, dx in angles:
            y_start = max(0, -dy)
            y_end = min(rows, rows - dy)
            x_start = max(0, -dx)
            x_end = min(cols, cols - dx)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    i = quantized[y, x]
                    j = quantized[y + dy, x + dx]
                    glcm[i, j] += 1.0

        # 归一化
        total = glcm.sum()
        if total == 0:
            return {"entropy": 0.0, "energy": 0.0, "homogeneity": 0.0, "contrast": 0.0}
        glcm /= total

        # 防止 log(0)
        glcm_safe = glcm.copy()
        glcm_safe[glcm_safe == 0] = 1e-10

        entropy = -float(np.sum(glcm * np.log2(glcm_safe)))
        energy = float(np.sum(glcm ** 2))

        # Homogeneity (Inverse Difference Moment)
        i_idx, j_idx = np.indices(glcm.shape)
        homogeneity = float(np.sum(glcm / (1.0 + np.abs(i_idx - j_idx).astype(np.float64))))

        # Contrast
        contrast = float(np.sum(glcm * (np.abs(i_idx - j_idx).astype(np.float64)) ** 2))

        return {
            "entropy": entropy,
            "energy": energy,
            "homogeneity": homogeneity,
            "contrast": contrast,
        }

    @staticmethod
    def _classify_surface_pattern(
        std_intensity: float,
        glcm_feats: dict[str, float],
    ) -> SurfacePattern:
        """基于纹理统计判定表面模式"""
        entropy = glcm_feats["entropy"]
        homogeneity = glcm_feats["homogeneity"]
        contrast = glcm_feats["contrast"]

        # 高均匀性 + 低对比度 → 光滑
        if homogeneity > 0.6 and contrast < 0.3:
            return SurfacePattern.SMOOTH

        # 高熵 + 高对比度 → 不规则
        if entropy > 3.0 and contrast > 0.5:
            return SurfacePattern.IRREGULAR

        # 中等熵 + 中等对比度 → 颗粒状
        if 1.5 < entropy < 3.0 and 0.2 < contrast < 0.5:
            return SurfacePattern.GRANULAR

        # 高对比度 + 低均匀性 → 绒毛状
        if contrast > 0.4 and homogeneity < 0.4:
            return SurfacePattern.VILLOUS

        return SurfacePattern.UNKNOWN

    @staticmethod
    def _estimate_vessel_density(roi_gray: np.ndarray) -> float:
        """
        简化版血管密度估计。
        使用形态学 Top-Hat 提取亮/暗线状结构，再统计占比。
        """
        if roi_gray.size == 0:
            return 0.0

        kernel_size = max(3, min(roi_gray.shape) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 暗线结构（血管多为暗色）
        tophat_dark = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, kernel)

        # 阈值化
        _, vessel_mask = cv2.threshold(tophat_dark, 10, 255, cv2.THRESH_BINARY)

        density = np.sum(vessel_mask > 0) / vessel_mask.size
        return float(density)

    # ---- 工具方法 ----

    @staticmethod
    def _crop_roi(image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """裁剪出掩码包围盒内的 ROI"""
        coords = np.where(binary > 0)
        if len(coords[0]) == 0:
            return np.array([], dtype=image.dtype)

        y_min, y_max = int(coords[0].min()), int(coords[0].max()) + 1
        x_min, x_max = int(coords[1].min()), int(coords[1].max()) + 1
        return image[y_min:y_max, x_min:x_max].copy()
