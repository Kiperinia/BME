"""
morphology_classifier.py — 形态分类（规则 + LLM 混合）

对内镜下病变进行形态学分类：
  - 有蒂 / 亚蒂 / 无蒂
  - 大小分级（微小 / 小 / 中 / 大 / 巨大）
  - 表面形态描述

策略：
  1. 规则引擎基于 FeatureExtractor 的定量特征做初步判定
  2. 当规则置信度 < threshold 时，调用 LLM 做二次确认
  3. 两者结果冲突时，以 LLM 为准（可配置）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .feature_extractor import (
    ColorTone,
    FeatureExtractor,
    LesionFeatures,
    SurfacePattern,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts" / "medical"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class PedicleType(str, Enum):
    """brief:
        Represent PedicleType state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    PEDUNCULATED = "pedunculated"    # 有蒂
    SESSILE = "sessile"              # 无蒂
    SUBPEDUNCULATED = "subpedunculated"  # 亚蒂
    FLAT = "flat"                    # 扁平
    UNCERTAIN = "uncertain"


class SizeGrade(str, Enum):
    """brief:
        Represent SizeGrade state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    TINY = "tiny"          # ≤ 3 mm
    SMALL = "small"        # 3–5 mm
    MEDIUM = "medium"      # 5–10 mm
    LARGE = "large"        # 10–20 mm
    GIANT = "giant"        # > 20 mm


@dataclass(slots=True)
class MorphologyResult:
    """brief:
        Represent MorphologyResult state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    pedicle_type: PedicleType = PedicleType.UNCERTAIN
    size_grade: SizeGrade = SizeGrade.SMALL
    estimated_size_mm: float = 0.0
    surface_pattern: str = "unknown"
    dominant_color: str = "unknown"
    shape_description: str = ""
    confidence: float = 0.0           # 规则引擎置信度 0~1
    used_llm: bool = False            # 是否调用了 LLM
    llm_reasoning: str = ""           # LLM 推理过程

    def to_dict(self) -> dict[str, Any]:
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "pedicle_type": self.pedicle_type.value,
            "size_grade": self.size_grade.value,
            "estimated_size_mm": round(self.estimated_size_mm, 1),
            "surface_pattern": self.surface_pattern,
            "dominant_color": self.dominant_color,
            "shape_description": self.shape_description,
            "confidence": round(self.confidence, 4),
            "used_llm": self.used_llm,
            "llm_reasoning": self.llm_reasoning,
        }


# ---------------------------------------------------------------------------
# LLM 接口协议
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """brief:
        Represent LLMClient state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """brief:
            Handle chat.

        parameter:
            - messages: Input value for messages.
            - temperature: Input value for temperature.
            - max_tokens: Input value for max_tokens.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        ...


# ---------------------------------------------------------------------------
# 形态分类器
# ---------------------------------------------------------------------------

class MorphologyClassifier:
    """brief:
        Represent MorphologyClassifier state and behavior.

    parameter:
        - pixel_size_mm: Input value for pixel_size_mm.
        - llm_client: Input value for llm_client.
        - llm_confidence_threshold: Input value for llm_confidence_threshold.
        - llm_overrides_rules: Input value for llm_overrides_rules.
        - prompt_path: Input value for prompt_path.

    retrival:
        - Provides instances used by the surrounding workflow.
    """

    # 大小分级阈值 (mm)
    SIZE_THRESHOLDS = {
        SizeGrade.TINY: 3.0,
        SizeGrade.SMALL: 5.0,
        SizeGrade.MEDIUM: 10.0,
        SizeGrade.LARGE: 20.0,
    }

    def __init__(
        self,
        pixel_size_mm: float | None = None,
        llm_client: LLMClient | None = None,
        llm_confidence_threshold: float = 0.6,
        llm_overrides_rules: bool = True,
        prompt_path: Path | None = None,
    ):
        """brief:
            Initialize this object.

        parameter:
            - pixel_size_mm: Input value for pixel_size_mm.
            - llm_client: Input value for llm_client.
            - llm_confidence_threshold: Input value for llm_confidence_threshold.
            - llm_overrides_rules: Input value for llm_overrides_rules.
            - prompt_path: Input value for prompt_path.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.pixel_size_mm = pixel_size_mm
        self.llm_client = llm_client
        self.llm_threshold = llm_confidence_threshold
        self.llm_overrides = llm_overrides_rules
        self._prompt_template = self._load_prompt(prompt_path)

        self._extractor = FeatureExtractor(pixel_size_mm=pixel_size_mm)

    # ---- 公共接口 ----

    def classify(self, features: LesionFeatures) -> MorphologyResult:
        """brief:
            Handle classify.

        parameter:
            - features: Input value for features.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        result = self._rule_based_classify(features)

        # 置信度不足且 LLM 可用时，调用 LLM 增强
        if (
            self.llm_client is not None
            and result.confidence < self.llm_threshold
        ):
            llm_result = self._llm_classify(features)
            if llm_result and self.llm_overrides:
                result = llm_result
            elif llm_result:
                # 不覆盖但记录 LLM 意见
                result.llm_reasoning = llm_result.llm_reasoning
                result.used_llm = True

        return result

    def classify_from_image(
        self,
        image: "np.ndarray",
        mask: "np.ndarray",
    ) -> MorphologyResult:
        """brief:
            Classify from image.

        parameter:
            - image: Input value for image.
            - mask: Input value for mask.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        features = self._extractor.extract(image, mask)
        return self.classify(features)

    # ---- 规则引擎 ----

    def _rule_based_classify(self, features: LesionFeatures) -> MorphologyResult:
        """brief:
            Handle rule based classify.

        parameter:
            - features: Input value for features.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        geo = features.geometric
        color = features.color
        texture = features.texture

        # 1. 蒂型判定
        pedicle, pedicle_conf = self._classify_pedicle(geo)

        # 2. 大小分级
        size_grade, est_size = self._classify_size(geo)

        # 3. 形状描述生成
        shape_desc = self._generate_shape_description(geo, color, texture)

        # 综合置信度
        overall_conf = pedicle_conf * 0.6 + 0.4  # 大小分级置信度通常较高

        return MorphologyResult(
            pedicle_type=pedicle,
            size_grade=size_grade,
            estimated_size_mm=est_size,
            surface_pattern=texture.surface_pattern.value,
            dominant_color=color.dominant_color.value,
            shape_description=shape_desc,
            confidence=round(overall_conf, 4),
            used_llm=False,
        )

    def _classify_pedicle(
        self,
        geo: "GeometricFeatures",
    ) -> tuple[PedicleType, float]:
        """brief:
            Classify pedicle.

        parameter:
            - geo: Input value for geo.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        solidity = geo.solidity
        aspect = geo.aspect_ratio
        circularity = geo.circularity

        # 有蒂：低 solidity + 高 aspect_ratio
        if solidity < 0.65 and aspect > 2.0:
            return PedicleType.PEDUNCULATED, 0.85

        # 亚蒂：中等 solidity + 中等 aspect_ratio
        if 0.65 <= solidity < 0.80 and 1.5 < aspect <= 2.0:
            return PedicleType.SUBPEDUNCULATED, 0.75

        # 扁平：高圆度 + 高 solidity
        if circularity > 0.7 and solidity > 0.85:
            return PedicleType.FLAT, 0.80

        # 无蒂：默认
        if solidity >= 0.80:
            return PedicleType.SESSILE, 0.70

        # 不确定
        return PedicleType.UNCERTAIN, 0.40

    def _classify_size(
        self,
        geo: "GeometricFeatures",
    ) -> tuple[SizeGrade, float]:
        """brief:
            Classify size.

        parameter:
            - geo: Input value for geo.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        area_mm2 = geo.area_mm2

        if area_mm2 is None:
            # 无物理尺寸时，用像素面积粗估（假设 0.15 mm/px）
            fallback_px_size = 0.15
            area_mm2 = geo.area_px * (fallback_px_size ** 2)

        # 从面积反算等效直径
        equiv_diam_mm = 2.0 * (area_mm2 / np.pi) ** 0.5 if area_mm2 > 0 else 0.0

        grade = SizeGrade.GIANT  # 默认最大
        for g, threshold in self.SIZE_THRESHOLDS.items():
            if equiv_diam_mm <= threshold:
                grade = g
                break

        return grade, round(equiv_diam_mm, 1)

    @staticmethod
    def _generate_shape_description(
        geo: "GeometricFeatures",
        color: "ColorFeatures",
        texture: "TextureFeatures",
    ) -> str:
        """brief:
            Handle generate shape description.

        parameter:
            - geo: Input value for geo.
            - color: Input value for color.
            - texture: Input value for texture.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        parts: list[str] = []

        # 大小
        if geo.area_mm2 is not None:
            parts.append(f"面积约 {geo.area_mm2:.1f} mm²")

        # 形状
        if geo.circularity > 0.8:
            parts.append("类圆形")
        elif geo.circularity > 0.6:
            parts.append("椭圆形")
        elif geo.aspect_ratio > 2.0:
            parts.append("长条形")
        else:
            parts.append("不规则形")

        # 表面
        pattern_map = {
            "smooth": "表面光滑",
            "irregular": "表面不规则",
            "granular": "表面颗粒状",
            "villous": "表面绒毛状",
        }
        parts.append(pattern_map.get(texture.surface_pattern.value, "表面形态未明确"))

        # 颜色
        color_map = {
            "red": "充血发红",
            "pale": "色泽苍白",
            "brown": "褐色",
            "mixed": "色泽混杂",
            "normal": "色泽接近正常黏膜",
        }
        parts.append(color_map.get(color.dominant_color.value, "色泽未明确"))

        return "，".join(parts) + "。"

    # ---- LLM 增强 ----

    def _llm_classify(self, features: LesionFeatures) -> MorphologyResult | None:
        """brief:
            Handle llm classify.

        parameter:
            - features: Input value for features.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if self.llm_client is None:
            return None

        try:
            prompt = self._build_llm_prompt(features)
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return self._parse_llm_response(response, features)
        except Exception as exc:
            logger.warning("LLM morphology classification failed: %s", exc)
            return None

    def _build_llm_prompt(self, features: LesionFeatures) -> str:
        """brief:
            Build llm prompt.

        parameter:
            - features: Input value for features.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        feat_dict = features.to_dict()
        feat_text = "\n".join(f"  {k}: {v}" for k, v in feat_dict.items() if isinstance(v, dict) for kk, vv in v.items())

        return self._prompt_template.format(
            features=feat_text,
            image_size_hint=f"pixel_size_mm={self.pixel_size_mm}" if self.pixel_size_mm else "pixel_size_mm=unknown",
        )

    @staticmethod
    def _parse_llm_response(
        response: str,
        features: LesionFeatures,
    ) -> MorphologyResult:
        """brief:
            Handle parse llm response.

        parameter:
            - response: Input value for response.
            - features: Input value for features.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        import json

        # 尝试提取 JSON 块
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", text[:200])
            return None

        pedicle_map = {e.value: e for e in PedicleType}
        size_map = {e.value: e for e in SizeGrade}

        return MorphologyResult(
            pedicle_type=pedicle_map.get(data.get("pedicle_type", ""), PedicleType.UNCERTAIN),
            size_grade=size_map.get(data.get("size_grade", ""), SizeGrade.SMALL),
            estimated_size_mm=float(data.get("estimated_size_mm", 0)),
            surface_pattern=data.get("surface_pattern", features.texture.surface_pattern.value),
            dominant_color=data.get("dominant_color", features.color.dominant_color.value),
            shape_description=data.get("shape_description", ""),
            confidence=float(data.get("confidence", 0.7)),
            used_llm=True,
            llm_reasoning=data.get("reasoning", ""),
        )

    @staticmethod
    def _load_prompt(custom_path: Path | None) -> str:
        """brief:
            Load prompt.

        parameter:
            - custom_path: Input value for custom_path.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        path = custom_path or (PROMPT_DIR / "classification.txt")
        if path.exists():
            return path.read_text(encoding="utf-8")

        # 内置 fallback prompt
        return (
            "你是一名消化内镜专家。请根据以下病变定量特征，进行形态学分类。\n\n"
            "## 病变特征\n{features}\n\n"
            "## 图像参数\n{image_size_hint}\n\n"
            "请以 JSON 格式返回：\n"
            "{{\n"
            '  "pedicle_type": "pedunculated|sessile|subpedunculated|flat|uncertain",\n'
            '  "size_grade": "tiny|small|medium|large|giant",\n'
            '  "estimated_size_mm": <float>,\n'
            '  "surface_pattern": "smooth|irregular|granular|villous|unknown",\n'
            '  "dominant_color": "red|pale|brown|mixed|normal",\n'
            '  "shape_description": "<中文自然语言描述>",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "reasoning": "<推理过程>"\n'
            "}}\n"
        )
