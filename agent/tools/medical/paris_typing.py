"""
paris_typing.py — 巴黎分型推理

Paris 分型（Paris Classification）是内镜下浅表性胃肠道肿瘤的形态学分类标准：
  - 0-Ip: 有蒂型（Pedunculated）
  - 0-Is: 无蒂型（Sessile）
  - 0-IIa: 浅表隆起型
  - 0-IIb: 浅表平坦型
  - 0-IIc: 浅表凹陷型
  - 0-III: 凹陷型

本模块结合形态分类结果 + 定量特征进行 Paris 分型推理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .feature_extractor import (
    FeatureExtractor,
    LesionFeatures,
)
from .morphology_classifier import (
    MorphologyClassifier,
    MorphologyResult,
    PedicleType,
    SizeGrade,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts" / "medical"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class ParisType(str, Enum):
    """Paris 分型"""
    IP = "0-Ip"       # 有蒂型
    IS = "0-Is"       # 无蒂型
    IIA = "0-IIa"     # 浅表隆起型
    IIB = "0-IIb"     # 浅表平坦型
    IIC = "0-IIc"     # 浅表凹陷型
    III = "0-III"     # 凹陷型
    MIXED = "0-IIa+IIc"  # 混合型（隆起+凹陷）
    UNCERTAIN = "uncertain"


class InvasionRisk(str, Enum):
    """基于 Paris 分型的浸润风险"""
    LOW = "low"           # 黏膜内 (M)
    MODERATE = "moderate" # 黏膜下层浅层 (SM1)
    HIGH = "high"         # 黏膜下层深层 (SM2+)


@dataclass(slots=True)
class ParisTypingResult:
    """Paris 分型结果"""
    paris_type: ParisType = ParisType.UNCERTAIN
    sub_type: str = ""                # 混合型子分类，如 "0-IIa+IIc"
    invasion_risk: InvasionRisk = InvasionRisk.LOW
    confidence: float = 0.0
    reasoning: str = ""
    used_llm: bool = False
    llm_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paris_type": self.paris_type.value,
            "sub_type": self.sub_type,
            "invasion_risk": self.invasion_risk.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "used_llm": self.used_llm,
            "llm_reasoning": self.llm_reasoning,
        }


# ---------------------------------------------------------------------------
# Paris 分型映射表
# ---------------------------------------------------------------------------

# 蒂型 → Paris 分型的初步映射
_PEDICLE_TO_PARIS: dict[PedicleType, list[tuple[ParisType, float]]] = {
    PedicleType.PEDUNCULATED: [(ParisType.IP, 0.90)],
    PedicleType.SESSILE: [(ParisType.IS, 0.75), (ParisType.IIA, 0.15)],
    PedicleType.SUBPEDUNCULATED: [(ParisType.IS, 0.50), (ParisType.IIA, 0.30), (ParisType.IP, 0.10)],
    PedicleType.FLAT: [(ParisType.IIB, 0.50), (ParisType.IIA, 0.25), (ParisType.IIC, 0.15)],
    PedicleType.UNCERTAIN: [(ParisType.IIA, 0.30), (ParisType.IIB, 0.25), (ParisType.IS, 0.20)],
}

# Paris 分型 → 默认浸润风险
_PARIS_INVASION_RISK: dict[ParisType, InvasionRisk] = {
    ParisType.IP: InvasionRisk.LOW,
    ParisType.IS: InvasionRisk.LOW,
    ParisType.IIA: InvasionRisk.LOW,
    ParisType.IIB: InvasionRisk.LOW,
    ParisType.IIC: InvasionRisk.MODERATE,
    ParisType.III: InvasionRisk.HIGH,
    ParisType.MIXED: InvasionRisk.MODERATE,
    ParisType.UNCERTAIN: InvasionRisk.LOW,
}


# ---------------------------------------------------------------------------
# LLM 接口协议
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Paris 分型引擎
# ---------------------------------------------------------------------------

class ParisTypingEngine:
    """
    Paris 分型推理引擎。

    流程：
      1. 接收 MorphologyResult + LesionFeatures
      2. 基于蒂型 + 几何特征做规则映射
      3. 用凹陷/隆起特征修正（区分 IIa/IIb/IIc）
      4. 可选 LLM 二次确认

    Usage::

        engine = ParisTypingEngine()
        result = engine.infer(morphology_result, lesion_features)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_confidence_threshold: float = 0.65,
        prompt_path: Path | None = None,
    ):
        self.llm_client = llm_client
        self.llm_threshold = llm_confidence_threshold
        self._prompt_template = self._load_prompt(prompt_path)

    # ---- 公共接口 ----

    def infer(
        self,
        morphology: MorphologyResult,
        features: LesionFeatures,
    ) -> ParisTypingResult:
        """
        从形态分类结果 + 定量特征推断 Paris 分型。

        Args:
            morphology: MorphologyClassifier 的输出
            features:   FeatureExtractor 的输出

        Returns:
            ParisTypingResult
        """
        result = self._rule_based_infer(morphology, features)

        if (
            self.llm_client is not None
            and result.confidence < self.llm_threshold
        ):
            llm_result = self._llm_infer(morphology, features)
            if llm_result is not None:
                result.paris_type = llm_result.paris_type
                result.sub_type = llm_result.sub_type
                result.invasion_risk = llm_result.invasion_risk
                result.confidence = llm_result.confidence
                result.used_llm = True
                result.llm_reasoning = llm_result.llm_reasoning

        return result

    def infer_from_image(
        self,
        image: "np.ndarray",
        mask: "np.ndarray",
        pixel_size_mm: float | None = None,
    ) -> ParisTypingResult:
        """从图像 + 掩码直接推断（内部串联 extractor → classifier → paris）"""
        extractor = FeatureExtractor(pixel_size_mm=pixel_size_mm)
        features = extractor.extract(image, mask)

        classifier = MorphologyClassifier(pixel_size_mm=pixel_size_mm)
        morphology = classifier.classify(features)

        return self.infer(morphology, features)

    # ---- 规则引擎 ----

    def _rule_based_infer(
        self,
        morphology: MorphologyResult,
        features: LesionFeatures,
    ) -> ParisTypingResult:
        geo = features.geometric
        color = features.color
        texture = features.texture

        # Step 1: 蒂型 → 初步 Paris 分型
        candidates = _PEDICLE_TO_PARIS.get(
            morphology.pedicle_type,
            _PEDICLE_TO_PARIS[PedicleType.UNCERTAIN],
        )
        best_type, base_conf = candidates[0]

        # Step 2: 用几何 + 颜色特征修正
        best_type, modifier_conf, reasoning = self._refine_with_features(
            best_type, geo, color, texture, morphology
        )

        # Step 3: 综合置信度
        confidence = base_conf * 0.5 + modifier_conf * 0.5

        # Step 4: 浸润风险
        invasion = _PARIS_INVASION_RISK.get(best_type, InvasionRisk.LOW)

        # 大尺寸修正：> 20mm 的 IIc/III 风险升级
        if morphology.size_grade in (SizeGrade.LARGE, SizeGrade.GIANT):
            if invasion == InvasionRisk.LOW:
                invasion = InvasionRisk.MODERATE
            elif invasion == InvasionRisk.MODERATE:
                invasion = InvasionRisk.HIGH

        return ParisTypingResult(
            paris_type=best_type,
            invasion_risk=invasion,
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )

    @staticmethod
    def _refine_with_features(
        initial_type: ParisType,
        geo: "GeometricFeatures",
        color: "ColorFeatures",
        texture: "TextureFeatures",
        morphology: MorphologyResult,
    ) -> tuple[ParisType, float, str]:
        """
        用定量特征修正 Paris 分型。

        关键区分：
          - IIa vs IIb: 隆起高度（用 solidity + 边缘对比度近似）
          - IIb vs IIc: 凹陷特征（用边缘对比度 + 颜色变化近似）
          - 混合型 IIa+IIc: 表面不规则 + 颜色混杂
        """
        reasoning_parts: list[str] = []
        final_type = initial_type
        conf = 0.7

        # 跳过 Ip/Is 的修正（蒂型已经明确）
        if initial_type in (ParisType.IP, ParisType.IS):
            reasoning_parts.append(f"蒂型明确为 {morphology.pedicle_type.value}，直接映射为 {initial_type.value}")
            return final_type, 0.85, "；".join(reasoning_parts)

        # 边缘对比度高 → 可能有凹陷成分
        high_contrast = color.border_contrast > 0.15
        # 颜色混杂 → 混合型
        mixed_color = color.dominant_color.value == "mixed" or color.color_variance > 1500
        # 表面不规则
        irregular_surface = texture.surface_pattern.value == "irregular"

        # 隆起判定：solidity 高 + 边缘对比度中等 → IIa
        is_elevated = geo.solidity > 0.80 and 0.05 < color.border_contrast < 0.20

        # 凹陷判定：边缘对比度高 + 颜色偏红 + 表面不规则 → IIc
        is_depressed = (
            high_contrast
            and color.redness_ratio > 0.45
            and irregular_surface
        )

        if is_elevated and is_depressed:
            final_type = ParisType.MIXED
            conf = 0.65
            reasoning_parts.append("同时存在隆起和凹陷特征，判定为混合型 0-IIa+IIc")
        elif is_depressed:
            final_type = ParisType.IIC
            conf = 0.70
            reasoning_parts.append("边缘对比度高、表面不规则、充血明显，提示浅表凹陷型 0-IIc")
        elif is_elevated:
            final_type = ParisType.IIA
            conf = 0.75
            reasoning_parts.append("凸包填充率高、边缘对比度适中，提示浅表隆起型 0-IIa")
        elif geo.circularity > 0.75 and color.border_contrast < 0.08:
            final_type = ParisType.IIB
            conf = 0.60
            reasoning_parts.append("形态规则、边缘对比度低，提示浅表平坦型 0-IIb")
        else:
            reasoning_parts.append(f"特征不典型，保持初始判定 {initial_type.value}")

        if mixed_color:
            reasoning_parts.append("颜色混杂，需关注混合型可能")

        return final_type, conf, "；".join(reasoning_parts)

    # ---- LLM 增强 ----

    def _llm_infer(
        self,
        morphology: MorphologyResult,
        features: LesionFeatures,
    ) -> ParisTypingResult | None:
        if self.llm_client is None:
            return None

        try:
            prompt = self._build_llm_prompt(morphology, features)
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return self._parse_llm_response(response)
        except Exception as exc:
            logger.warning("LLM Paris typing failed: %s", exc)
            return None

    def _build_llm_prompt(
        self,
        morphology: MorphologyResult,
        features: LesionFeatures,
    ) -> str:
        morph_dict = morphology.to_dict()
        feat_dict = features.to_dict()

        morph_text = "\n".join(f"  {k}: {v}" for k, v in morph_dict.items())
        feat_text = "\n".join(
            f"  {k}: {v}" for k, v in feat_dict.items()
            if isinstance(v, dict) for kk, vv in v.items()
        )

        return self._prompt_template.format(
            morphology=morph_text,
            features=feat_text,
        )

    @staticmethod
    def _parse_llm_response(response: str) -> ParisTypingResult | None:
        import json

        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM Paris response: %s", text[:200])
            return None

        type_map = {e.value: e for e in ParisType}
        risk_map = {e.value: e for e in InvasionRisk}

        return ParisTypingResult(
            paris_type=type_map.get(data.get("paris_type", ""), ParisType.UNCERTAIN),
            sub_type=data.get("sub_type", ""),
            invasion_risk=risk_map.get(data.get("invasion_risk", ""), InvasionRisk.LOW),
            confidence=float(data.get("confidence", 0.7)),
            reasoning=data.get("reasoning", ""),
            used_llm=True,
            llm_reasoning=data.get("reasoning", ""),
        )

    @staticmethod
    def _load_prompt(custom_path: Path | None) -> str:
        path = custom_path or (PROMPT_DIR / "paris_typing.txt")
        if path.exists():
            return path.read_text(encoding="utf-8")

        return (
            "你是一名消化内镜专家，精通 Paris 分型标准。\n\n"
            "## 形态分类结果\n{morphology}\n\n"
            "## 定量特征\n{features}\n\n"
            "请根据以上信息进行 Paris 分型，以 JSON 格式返回：\n"
            "{{\n"
            '  "paris_type": "0-Ip|0-Is|0-IIa|0-IIb|0-IIc|0-III|0-IIa+IIc|uncertain",\n'
            '  "sub_type": "",\n'
            '  "invasion_risk": "low|moderate|high",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "reasoning": "<推理过程>"\n'
            "}}\n"
        )
