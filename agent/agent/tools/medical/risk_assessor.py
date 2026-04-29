"""
risk_assessor.py — 恶性风险评估

综合形态分类、Paris 分型、定量特征进行恶性风险评分。

评分维度：
  1. Paris 分型风险（IIc/III 高风险）
  2. 大小风险（> 10mm 风险上升）
  3. 表面纹理风险（不规则/绒毛状）
  4. 颜色风险（充血、混杂色）
  5. 血管纹理风险（血管密度异常）

输出：
  - 综合风险等级：Low / Intermediate / High
  - 各维度评分明细
  - 处理建议
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from agent.tools.medical.feature_extractor import (
    ColorTone,
    LesionFeatures,
    SurfacePattern,
)
from agent.tools.medical.morphology_classifier import (
    MorphologyResult,
    PedicleType,
    SizeGrade,
)
from agent.tools.medical.paris_typing import (
    InvasionRisk,
    ParisType,
    ParisTypingResult,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts" / "medical"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"


class Disposition(str, Enum):
    """处理建议"""
    MONITOR = "monitor"              # 随访观察
    ENDOSCOPIC_RESECTION = "endoscopic_resection"  # 内镜下切除
    BIOPSY = "biopsy"                # 活检
    SURGICAL_REFERRAL = "surgical_referral"  # 外科会诊
    URGENT_REFERRAL = "urgent_referral"      # 紧急转诊


@dataclass(slots=True)
class DimensionScore:
    """单维度评分"""
    name: str
    score: float       # 0~10
    weight: float      # 权重
    weighted_score: float
    reasoning: str


@dataclass(slots=True)
class RiskAssessmentResult:
    """风险评估结果"""
    risk_level: RiskLevel = RiskLevel.LOW
    total_score: float = 0.0          # 加权总分 0~10
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    disposition: Disposition = Disposition.MONITOR
    disposition_reason: str = ""
    confidence: float = 0.0
    used_llm: bool = False
    llm_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "total_score": round(self.total_score, 2),
            "dimension_scores": [
                {
                    "name": ds.name,
                    "score": round(ds.score, 2),
                    "weight": round(ds.weight, 2),
                    "weighted_score": round(ds.weighted_score, 2),
                    "reasoning": ds.reasoning,
                }
                for ds in self.dimension_scores
            ],
            "disposition": self.disposition.value,
            "disposition_reason": self.disposition_reason,
            "confidence": round(self.confidence, 4),
            "used_llm": self.used_llm,
            "llm_reasoning": self.llm_reasoning,
        }


# ---------------------------------------------------------------------------
# 维度评分规则
# ---------------------------------------------------------------------------

# 各维度权重（总和 = 1.0）
DIMENSION_WEIGHTS = {
    "paris_type": 0.25,
    "size": 0.20,
    "surface_texture": 0.20,
    "color": 0.15,
    "vascularity": 0.10,
    "morphology": 0.10,
}

# Paris 分型 → 基础风险分 (0~10)
PARIS_RISK_SCORES: dict[ParisType, float] = {
    ParisType.IP: 1.0,
    ParisType.IS: 2.0,
    ParisType.IIA: 2.5,
    ParisType.IIB: 1.5,
    ParisType.IIC: 6.0,
    ParisType.III: 8.0,
    ParisType.MIXED: 5.0,
    ParisType.UNCERTAIN: 3.0,
}

# 大小分级 → 风险分 (0~10)
SIZE_RISK_SCORES: dict[SizeGrade, float] = {
    SizeGrade.TINY: 0.5,
    SizeGrade.SMALL: 1.0,
    SizeGrade.MEDIUM: 2.5,
    SizeGrade.LARGE: 5.0,
    SizeGrade.GIANT: 7.5,
}

# 表面纹理 → 风险分 (0~10)
SURFACE_RISK_SCORES: dict[SurfacePattern, float] = {
    SurfacePattern.SMOOTH: 1.0,
    SurfacePattern.GRANULAR: 3.0,
    SurfacePattern.VILLOUS: 5.0,
    SurfacePattern.IRREGULAR: 6.5,
    SurfacePattern.UNKNOWN: 3.0,
}

# 颜色 → 风险分 (0~10)
COLOR_RISK_SCORES: dict[ColorTone, float] = {
    ColorTone.NORMAL: 0.5,
    ColorTone.PALE: 1.5,
    ColorTone.RED: 3.0,
    ColorTone.BROWN: 2.5,
    ColorTone.MIXED: 5.0,
    ColorTone.UNKNOWN: 3.0,
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
# 风险评估器
# ---------------------------------------------------------------------------

class RiskAssessor:
    """
    恶性风险评估器。

    Usage::

        assessor = RiskAssessor()
        result = assessor.assess(morphology_result, paris_result, features)
    """

    # 风险等级阈值
    INTERMEDIATE_THRESHOLD = 3.5
    HIGH_THRESHOLD = 6.0

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

    def assess(
        self,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        features: LesionFeatures,
    ) -> RiskAssessmentResult:
        """综合评估恶性风险"""
        dimensions = self._compute_dimensions(morphology, paris, features)
        total = sum(d.weighted_score for d in dimensions)

        risk_level = self._classify_risk(total)
        disposition, disp_reason = self._recommend_disposition(risk_level, morphology, paris)
        confidence = self._estimate_confidence(dimensions)

        result = RiskAssessmentResult(
            risk_level=risk_level,
            total_score=round(total, 2),
            dimension_scores=dimensions,
            disposition=disposition,
            disposition_reason=disp_reason,
            confidence=round(confidence, 4),
        )

        # LLM 增强
        if (
            self.llm_client is not None
            and confidence < self.llm_threshold
        ):
            llm_result = self._llm_assess(morphology, paris, features)
            if llm_result is not None:
                result.risk_level = llm_result.risk_level
                result.total_score = llm_result.total_score
                result.disposition = llm_result.disposition
                result.disposition_reason = llm_result.disposition_reason
                result.used_llm = True
                result.llm_reasoning = llm_result.llm_reasoning

        return result

    # ---- 维度评分 ----

    def _compute_dimensions(
        self,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        features: LesionFeatures,
    ) -> list[DimensionScore]:
        """计算各维度评分"""
        dimensions: list[DimensionScore] = []

        # 1. Paris 分型
        paris_score = PARIS_RISK_SCORES.get(paris.paris_type, 3.0)
        # 浸润风险修正
        if paris.invasion_risk == InvasionRisk.HIGH:
            paris_score = min(10.0, paris_score + 2.0)
        elif paris.invasion_risk == InvasionRisk.MODERATE:
            paris_score = min(10.0, paris_score + 1.0)

        dimensions.append(DimensionScore(
            name="paris_type",
            score=paris_score,
            weight=DIMENSION_WEIGHTS["paris_type"],
            weighted_score=paris_score * DIMENSION_WEIGHTS["paris_type"],
            reasoning=f"Paris 分型 {paris.paris_type.value}，浸润风险 {paris.invasion_risk.value}",
        ))

        # 2. 大小
        size_score = SIZE_RISK_SCORES.get(morphology.size_grade, 2.0)
        dimensions.append(DimensionScore(
            name="size",
            score=size_score,
            weight=DIMENSION_WEIGHTS["size"],
            weighted_score=size_score * DIMENSION_WEIGHTS["size"],
            reasoning=f"大小分级 {morphology.size_grade.value}，等效直径约 {morphology.estimated_size_mm:.1f} mm",
        ))

        # 3. 表面纹理
        surface_score = SURFACE_RISK_SCORES.get(features.texture.surface_pattern, 3.0)
        dimensions.append(DimensionScore(
            name="surface_texture",
            score=surface_score,
            weight=DIMENSION_WEIGHTS["surface_texture"],
            weighted_score=surface_score * DIMENSION_WEIGHTS["surface_texture"],
            reasoning=f"表面纹理 {features.texture.surface_pattern.value}，熵值 {features.texture.entropy:.2f}",
        ))

        # 4. 颜色
        color_score = COLOR_RISK_SCORES.get(features.color.dominant_color, 3.0)
        # 边缘对比度修正
        if features.color.border_contrast > 0.15:
            color_score = min(10.0, color_score + 1.5)
        dimensions.append(DimensionScore(
            name="color",
            score=color_score,
            weight=DIMENSION_WEIGHTS["color"],
            weighted_score=color_score * DIMENSION_WEIGHTS["color"],
            reasoning=f"颜色基调 {features.color.dominant_color.value}，边缘对比度 {features.color.border_contrast:.2f}",
        ))

        # 5. 血管纹理
        vessel_score = self._score_vascularity(features)
        dimensions.append(DimensionScore(
            name="vascularity",
            score=vessel_score,
            weight=DIMENSION_WEIGHTS["vascularity"],
            weighted_score=vessel_score * DIMENSION_WEIGHTS["vascularity"],
            reasoning=f"血管密度 {features.texture.vessel_density:.4f}",
        ))

        # 6. 形态（蒂型）
        morph_score = self._score_morphology(morphology)
        dimensions.append(DimensionScore(
            name="morphology",
            score=morph_score,
            weight=DIMENSION_WEIGHTS["morphology"],
            weighted_score=morph_score * DIMENSION_WEIGHTS["morphology"],
            reasoning=f"蒂型 {morphology.pedicle_type.value}，圆度 {features.geometric.circularity:.2f}",
        ))

        return dimensions

    @staticmethod
    def _score_vascularity(features: LesionFeatures) -> float:
        """血管纹理评分"""
        density = features.texture.vessel_density

        if density < 0.02:
            return 1.0
        elif density < 0.05:
            return 2.5
        elif density < 0.10:
            return 4.0
        elif density < 0.20:
            return 6.0
        else:
            return 8.0

    @staticmethod
    def _score_morphology(morphology: MorphologyResult) -> float:
        """形态评分"""
        morph_scores = {
            PedicleType.PEDUNCULATED: 1.5,
            PedicleType.SESSILE: 3.0,
            PedicleType.SUBPEDUNCULATED: 2.5,
            PedicleType.FLAT: 3.5,
            PedicleType.UNCERTAIN: 4.0,
        }
        return morph_scores.get(morphology.pedicle_type, 3.0)

    # ---- 风险分级 ----

    def _classify_risk(self, total_score: float) -> RiskLevel:
        if total_score >= self.HIGH_THRESHOLD:
            return RiskLevel.HIGH
        elif total_score >= self.INTERMEDIATE_THRESHOLD:
            return RiskLevel.INTERMEDIATE
        return RiskLevel.LOW

    @staticmethod
    def _recommend_disposition(
        risk_level: RiskLevel,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
    ) -> tuple[Disposition, str]:
        """基于风险等级推荐处理方案"""
        if risk_level == RiskLevel.HIGH:
            if paris.invasion_risk == InvasionRisk.HIGH:
                return (
                    Disposition.SURGICAL_REFERRAL,
                    "高风险病变，浸润风险高，建议外科会诊评估手术指征。",
                )
            return (
                Disposition.BIOPSY,
                "高风险病变，建议先行活检明确病理性质。",
            )

        if risk_level == RiskLevel.INTERMEDIATE:
            if morphology.size_grade in (SizeGrade.LARGE, SizeGrade.GIANT):
                return (
                    Disposition.ENDOSCOPIC_RESECTION,
                    "中等风险、较大病变，建议内镜下切除。",
                )
            return (
                Disposition.BIOPSY,
                "中等风险病变，建议活检或内镜下切除。",
            )

        # Low risk
        if morphology.size_grade in (SizeGrade.TINY, SizeGrade.SMALL):
            return (
                Disposition.MONITOR,
                "低风险小病变，建议定期随访观察。",
            )
        return (
            Disposition.ENDOSCOPIC_RESECTION,
            "低风险但有一定大小的病变，可考虑内镜下预防性切除。",
        )

    @staticmethod
    def _estimate_confidence(dimensions: list[DimensionScore]) -> float:
        """估算综合置信度（基于各维度评分的一致性）"""
        if not dimensions:
            return 0.0

        scores = [d.score for d in dimensions]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # 方差越小 → 各维度越一致 → 置信度越高
        consistency = max(0.0, 1.0 - variance / 10.0)

        # 有不确定因素时降低
        has_uncertain = any(d.score > 6.0 and d.score < 8.0 for d in dimensions)
        if has_uncertain:
            consistency *= 0.85

        return min(1.0, consistency)

    # ---- LLM 增强 ----

    def _llm_assess(
        self,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        features: LesionFeatures,
    ) -> RiskAssessmentResult | None:
        if self.llm_client is None:
            return None

        try:
            prompt = self._build_llm_prompt(morphology, paris, features)
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return self._parse_llm_response(response)
        except Exception as exc:
            logger.warning("LLM risk assessment failed: %s", exc)
            return None

    def _build_llm_prompt(
        self,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        features: LesionFeatures,
    ) -> str:
        morph_text = "\n".join(f"  {k}: {v}" for k, v in morphology.to_dict().items())
        paris_text = "\n".join(f"  {k}: {v}" for k, v in paris.to_dict().items())
        feat_text = "\n".join(
            f"  {k}: {v}" for k, v in features.to_dict().items()
            if isinstance(v, dict) for kk, vv in v.items()
        )

        return self._prompt_template.format(
            morphology=morph_text,
            paris_typing=paris_text,
            features=feat_text,
        )

    @staticmethod
    def _parse_llm_response(response: str) -> RiskAssessmentResult | None:
        import json

        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM risk response: %s", text[:200])
            return None

        level_map = {e.value: e for e in RiskLevel}
        disp_map = {e.value: e for e in Disposition}

        return RiskAssessmentResult(
            risk_level=level_map.get(data.get("risk_level", ""), RiskLevel.LOW),
            total_score=float(data.get("total_score", 3.0)),
            disposition=disp_map.get(data.get("disposition", ""), Disposition.MONITOR),
            disposition_reason=data.get("disposition_reason", ""),
            confidence=float(data.get("confidence", 0.7)),
            used_llm=True,
            llm_reasoning=data.get("reasoning", ""),
        )

    @staticmethod
    def _load_prompt(custom_path: Path | None) -> str:
        path = custom_path or (PROMPT_DIR / "risk_assessment.txt")
        if path.exists():
            return path.read_text(encoding="utf-8")

        return (
            "你是一名消化内镜专家，请根据以下信息评估病变的恶性风险。\n\n"
            "## 形态分类\n{morphology}\n\n"
            "## Paris 分型\n{paris_typing}\n\n"
            "## 定量特征\n{features}\n\n"
            "请以 JSON 格式返回：\n"
            "{{\n"
            '  "risk_level": "low|intermediate|high",\n'
            '  "total_score": <0.0-10.0>,\n'
            '  "disposition": "monitor|endoscopic_resection|biopsy|surgical_referral|urgent_referral",\n'
            '  "disposition_reason": "<处理建议理由>",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "reasoning": "<推理过程>"\n'
            "}}\n"
        )
