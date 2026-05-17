from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .feature_extractor import LesionFeatures
from .morphology_classifier import MorphologyResult
from .paris_typing import ParisTypingResult
from .risk_assessor import Disposition, RiskAssessmentResult, RiskLevel


@dataclass(slots=True)
class ToolParameterSchema:
    name: str
    py_type: type[Any] | tuple[type[Any], ...]
    required: bool = True
    description: str = ""

    @property
    def type_name(self) -> str:
        if isinstance(self.py_type, tuple):
            return " | ".join(t.__name__ for t in self.py_type)
        return self.py_type.__name__


@dataclass(slots=True)
class ReportToolSpec:
    name: str
    description: str
    input_schema: tuple[ToolParameterSchema, ...]


@dataclass(slots=True)
class ReportToolCallLog:
    tool_name: str
    status: str
    duration_ms: float
    input_payload: dict[str, str]
    output_preview: str
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "input_payload": self.input_payload,
            "output_preview": self.output_preview,
            "error_message": self.error_message,
        }


class ReportToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[ReportToolSpec, Callable[..., Any]]] = {}
        self._call_logs: list[ReportToolCallLog] = []

    def register(self, spec: ReportToolSpec, handler: Callable[..., Any]) -> None:
        self._tools[spec.name] = (spec, handler)

    def reset_logs(self) -> None:
        self._call_logs = []

    def get_call_logs(self) -> list[dict[str, Any]]:
        return [log.to_dict() for log in self._call_logs]

    def list_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": [
                    {
                        "name": p.name,
                        "type": p.type_name,
                        "required": p.required,
                        "description": p.description,
                    }
                    for p in spec.input_schema
                ],
            }
            for spec, _ in self._tools.values()
        ]

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._tools:
            raise ValueError(f"Unknown report tool: {tool_name}")

        spec, handler = self._tools[tool_name]
        self._validate_inputs(spec, kwargs)

        started = time.perf_counter()
        try:
            result = handler(**kwargs)
            self._call_logs.append(
                ReportToolCallLog(
                    tool_name=tool_name,
                    status="ok",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    input_payload={k: _short_repr(v) for k, v in kwargs.items()},
                    output_preview=_short_repr(result),
                )
            )
            return result
        except Exception as exc:
            self._call_logs.append(
                ReportToolCallLog(
                    tool_name=tool_name,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    input_payload={k: _short_repr(v) for k, v in kwargs.items()},
                    output_preview="",
                    error_message=str(exc),
                )
            )
            raise

    @staticmethod
    def _validate_inputs(spec: ReportToolSpec, kwargs: dict[str, Any]) -> None:
        for parameter in spec.input_schema:
            if parameter.required and parameter.name not in kwargs:
                raise ValueError(
                    f"Tool '{spec.name}' missing required parameter '{parameter.name}'"
                )

            if parameter.name not in kwargs:
                continue

            value = kwargs[parameter.name]
            if value is None:
                continue

            if not isinstance(value, parameter.py_type):
                expected = parameter.type_name
                actual = type(value).__name__
                raise TypeError(
                    f"Tool '{spec.name}' parameter '{parameter.name}' expects {expected}, got {actual}"
                )


@dataclass(slots=True)
class FindingsComposerTool:
    """Compose structured findings text from model outputs."""

    def compose(
        self,
        *,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        features: LesionFeatures,
    ) -> str:
        findings_parts: list[str] = []

        findings_parts.append(
            f"内镜检查发现病变 {morphology.size_grade.value} 型，"
            f"等效直径约 {morphology.estimated_size_mm:.1f} mm"
        )
        if features.geometric.area_mm2 is not None:
            findings_parts.append(f"（面积约 {features.geometric.area_mm2:.1f} mm²）")

        findings_parts.append("。")
        findings_parts.append(
            f"病变呈{_pedicle_cn(morphology.pedicle_type.value)}，"
            f"{morphology.shape_description}"
        )

        findings_parts.append(f"按 Paris 分型标准，该病变为 {paris.paris_type.value} 型")
        if paris.sub_type:
            findings_parts.append(f"（{paris.sub_type}）")
        findings_parts.append("。")

        findings_parts.append(
            f"表面纹理呈{_surface_cn(features.texture.surface_pattern.value)}，"
            f"血管密度{_vessel_cn(features.texture.vessel_density)}。"
        )

        findings_parts.append(
            f"病变{_color_cn(features.color.dominant_color.value)}，"
            f"边缘对比度{_contrast_cn(features.color.border_contrast)}。"
        )

        return "".join(findings_parts)


@dataclass(slots=True)
class ConclusionComposerTool:
    """Compose diagnosis conclusion and disposition text."""

    def compose(self, *, paris: ParisTypingResult, risk: RiskAssessmentResult) -> str:
        conclusion_parts: list[str] = []

        risk_cn = {
            RiskLevel.LOW: "低",
            RiskLevel.INTERMEDIATE: "中等",
            RiskLevel.HIGH: "高",
        }
        conclusion_parts.append(
            f"综合评估恶性风险为{risk_cn.get(risk.risk_level, '未明确')}风险"
            f"（评分 {risk.total_score:.1f}/10）。"
        )

        disp_cn = {
            Disposition.MONITOR: "定期随访观察",
            Disposition.ENDOSCOPIC_RESECTION: "内镜下切除",
            Disposition.BIOPSY: "活检明确病理",
            Disposition.SURGICAL_REFERRAL: "外科会诊评估",
            Disposition.URGENT_REFERRAL: "紧急转诊",
        }
        conclusion_parts.append(
            f"建议：{disp_cn.get(risk.disposition, '进一步评估')}。"
            f"{risk.disposition_reason}"
        )

        if paris.invasion_risk.value in ("moderate", "high"):
            conclusion_parts.append(
                f"Paris {paris.paris_type.value} 型病变浸润风险为"
                f"{paris.invasion_risk.value}，需重点关注。"
            )

        return "".join(conclusion_parts)


@dataclass(slots=True)
class LayoutSuggestionTool:
    """Generate layout suggestions for frontend report rendering."""

    def compose(
        self,
        *,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
    ) -> str:
        suggestions: list[str] = []

        if risk.risk_level == RiskLevel.HIGH:
            suggestions.append("报告头部使用红色警示标识")
        elif risk.risk_level == RiskLevel.INTERMEDIATE:
            suggestions.append("报告头部使用橙色提醒标识")
        else:
            suggestions.append("报告头部使用常规样式")

        suggestions.append("病变区域截图放置在检查所见段落上方")
        suggestions.append("分割掩码叠加在原图上以半透明红色显示")
        suggestions.append(
            f"风险评分 {risk.total_score:.1f}/10 以仪表盘形式展示，"
            f"颜色从绿（低）到红（高）渐变"
        )

        if risk.dimension_scores:
            dim_names = [ds.name for ds in risk.dimension_scores]
            suggestions.append(f"各维度评分（{', '.join(dim_names)}）以雷达图展示")

        suggestions.append("处理建议以醒目卡片形式置于结论段落下方")

        # Touch unused args to keep signature stable for future tool expansion.
        _ = morphology, paris
        return "；".join(suggestions) + "。"


@dataclass(slots=True)
class ReportKeywordSuggestionTool:
    """Extract concise report keywords for downstream indexing and display."""

    _keyword_patterns: tuple[tuple[str, str], ...] = (
        (r"0-I[p|s]|0-II[a-c]|0-III", "Paris分型"),
        (r"浸润", "浸润风险"),
        (r"血管", "血管异常"),
        (r"颗粒|绒毛", "表面结构异常"),
        (r"切除", "内镜切除建议"),
        (r"随访", "随访建议"),
    )

    def compose(self, *, findings: str, conclusion: str, max_keywords: int = 6) -> list[str]:
        text = f"{findings} {conclusion}".strip()
        if not text:
            return []

        keywords: list[str] = []
        for pattern, label in self._keyword_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE) and label not in keywords:
                keywords.append(label)

        if not keywords:
            fallback_tokens = [token for token in re.split(r"[，。；、\s]+", text) if len(token) >= 2]
            keywords = fallback_tokens[:max_keywords]

        return keywords[:max(1, max_keywords)]


def create_default_report_tool_registry() -> ReportToolRegistry:
    findings_tool = FindingsComposerTool()
    conclusion_tool = ConclusionComposerTool()
    layout_tool = LayoutSuggestionTool()
    keyword_tool = ReportKeywordSuggestionTool()
    analysis_tool = ReportAnalysisTool()
    refinement_tool = ReportRefinementTool()
    scoring_tool = ReportScoringTool()

    registry = ReportToolRegistry()
    registry.register(
        ReportToolSpec(
            name="compose_findings",
            description="Compose structured findings text from morphology, Paris typing and features.",
            input_schema=(
                ToolParameterSchema("morphology", MorphologyResult, True, "Morphology classification result"),
                ToolParameterSchema("paris", ParisTypingResult, True, "Paris typing result"),
                ToolParameterSchema("features", LesionFeatures, True, "Extracted lesion features"),
            ),
        ),
        findings_tool.compose,
    )
    registry.register(
        ReportToolSpec(
            name="compose_conclusion",
            description="Compose diagnosis conclusion with risk level and disposition suggestions.",
            input_schema=(
                ToolParameterSchema("paris", ParisTypingResult, True, "Paris typing result"),
                ToolParameterSchema("risk", RiskAssessmentResult, True, "Risk assessment result"),
            ),
        ),
        conclusion_tool.compose,
    )
    registry.register(
        ReportToolSpec(
            name="suggest_layout",
            description="Generate frontend layout suggestions for report rendering.",
            input_schema=(
                ToolParameterSchema("morphology", MorphologyResult, True, "Morphology classification result"),
                ToolParameterSchema("paris", ParisTypingResult, True, "Paris typing result"),
                ToolParameterSchema("risk", RiskAssessmentResult, True, "Risk assessment result"),
            ),
        ),
        layout_tool.compose,
    )
    registry.register(
        ReportToolSpec(
            name="suggest_report_keywords",
            description="Suggest concise report keywords for indexing and trace display.",
            input_schema=(
                ToolParameterSchema("findings", str, True, "Findings text"),
                ToolParameterSchema("conclusion", str, True, "Conclusion text"),
                ToolParameterSchema("max_keywords", int, False, "Max returned keywords"),
            ),
        ),
        keyword_tool.compose,
    )
    registry.register(
        ReportToolSpec(
            name="analyze_report",
            description="ReAct thinking: Analyze report for issues and improvement suggestions.",
            input_schema=(
                ToolParameterSchema("findings", str, True, "Findings text"),
                ToolParameterSchema("conclusion", str, True, "Conclusion text"),
                ToolParameterSchema("paris", ParisTypingResult, True, "Paris typing result"),
                ToolParameterSchema("risk", RiskAssessmentResult, True, "Risk assessment result"),
            ),
        ),
        analysis_tool.analyze,
    )
    registry.register(
        ReportToolSpec(
            name="refine_report",
            description="ReAct acting: Apply refinements based on analysis suggestions.",
            input_schema=(
                ToolParameterSchema("original_text", str, True, "Original text to refine"),
                ToolParameterSchema("suggestions", list, True, "Refinement suggestions"),
                ToolParameterSchema("refinement_type", str, True, "Type: 'findings' or 'conclusion'"),
            ),
        ),
        refinement_tool.refine,
    )
    registry.register(
        ReportToolSpec(
            name="score_report",
            description="Generate multidimensional report quality scores.",
            input_schema=(
                ToolParameterSchema("findings", str, True, "Findings text"),
                ToolParameterSchema("conclusion", str, True, "Conclusion text"),
                ToolParameterSchema("paris", ParisTypingResult, True, "Paris typing result"),
                ToolParameterSchema("risk", RiskAssessmentResult, True, "Risk assessment result"),
                ToolParameterSchema("analysis_result", dict, True, "Result from analysis tool"),
            ),
        ),
        scoring_tool.score,
    )

    return registry


def _short_repr(value: Any, limit: int = 180) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _pedicle_cn(pedicle: str) -> str:
    return {
        "pedunculated": "有蒂型",
        "sessile": "无蒂型",
        "subpedunculated": "亚蒂型",
        "flat": "扁平型",
        "uncertain": "形态未明确",
    }.get(pedicle, pedicle)


def _surface_cn(surface: str) -> str:
    return {
        "smooth": "光滑",
        "irregular": "不规则",
        "granular": "颗粒状",
        "villous": "绒毛状",
        "unknown": "未明确",
    }.get(surface, surface)


def _vessel_cn(density: float) -> str:
    if density < 0.02:
        return "稀疏"
    if density < 0.05:
        return "较少"
    if density < 0.10:
        return "中等"
    if density < 0.20:
        return "较丰富"
    return "丰富"


def _color_cn(color: str) -> str:
    return {
        "red": "充血发红",
        "pale": "色泽苍白",
        "brown": "褐色",
        "mixed": "色泽混杂",
        "normal": "色泽接近正常黏膜",
        "unknown": "色泽未明确",
    }.get(color, color)


def _contrast_cn(contrast: float) -> str:
    if contrast < 0.05:
        return "低"
    if contrast < 0.10:
        return "中等"
    if contrast < 0.20:
        return "较高"
    return "高"


@dataclass(slots=True)
class ReportAnalysisTool:
    """Analyze generated report and identify potential issues or improvements via ReAct reasoning."""

    def analyze(
        self,
        *,
        findings: str,
        conclusion: str,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
    ) -> dict[str, Any]:
        """
        ReAct Thinking step: Analyze report for accuracy, completeness, clarity.
        Returns issues and improvement suggestions.
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Check for missing key risk factors
        if risk.risk_level == RiskLevel.HIGH and "紧急" not in conclusion:
            issues.append("高风险病灶未突出紧急程度")
            suggestions.append("在结论中强调 '紧急处理' 关键词")

        # Check for incomplete Paris typing description
        if paris.paris_type and paris.paris_type.value not in findings:
            issues.append("病灶 Paris 分型在病灶描述中未明确提及")
            suggestions.append(f"补充 Paris {paris.paris_type.value} 型的特征描述")

        # Check for missing disposition details
        if "建议" not in conclusion:
            issues.append("诊断建议信息缺失")
            suggestions.append("补充明确的诊疗建议（随访/切除/活检等）")

        # Check report length reasonableness
        if len(findings) < 50:
            issues.append("病灶描述过于简洁，可能信息不足")
            suggestions.append("补充更多病灶特征描述")

        if len(findings) > 500:
            issues.append("病灶描述过长，不够精炼")
            suggestions.append("精简表述，突出核心信息")

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.85,
        }


@dataclass(slots=True)
class ReportRefinementTool:
    """Refine findings or conclusion based on analysis suggestions."""

    def refine(
        self,
        *,
        original_text: str,
        suggestions: list[str],
        refinement_type: str,  # "findings" or "conclusion"
    ) -> dict[str, str]:
        """
        ReAct Acting step: Apply refinements to the identified issues.
        Returns refined text and change log.
        """
        refined = original_text
        changes: list[str] = []

        for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
            # Simple keyword injection based on suggestion
            if "紧急" in suggestion and refinement_type == "conclusion":
                if "紧急" not in refined:
                    refined = refined.replace("建议：", "建议（紧急处理）：", 1)
                    changes.append("添加紧急处理标记")

            if "Paris" in suggestion and refinement_type == "findings":
                if "Paris" not in refined:
                    refined = f"{refined}。按 Paris 分型标准评估。"
                    changes.append("补充 Paris 分型评估")

            if "诊疗建议" in suggestion or "建议" in suggestion:
                if refinement_type == "conclusion" and "建议" not in refined:
                    refined = f"{refined}。诊疗建议：进一步评估。"
                    changes.append("补充诊疗建议")

        return {
            "refined_text": refined,
            "changes": changes,
            "refinement_count": len(changes),
        }


@dataclass(slots=True)
class ReportScoringTool:
    """Score the report across multiple dimensions."""

    def score(
        self,
        *,
        findings: str,
        conclusion: str,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        analysis_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate multidimensional report scores and confidence.
        Returns overall score and breakdown by dimension.
        """
        # Dimension scoring (0-10 scale)
        accuracy_score = 8.5 if not analysis_result.get("has_issues") else 7.0
        completeness_score = 9.0 if len(findings) > 80 and len(conclusion) > 50 else 7.5
        clarity_score = 8.0 if "。" in findings and "。" in conclusion else 6.5
        risk_recognition_score = 9.0 if (
            risk.risk_level == RiskLevel.HIGH and "高" in conclusion
        ) else 7.5

        dimensions = {
            "accuracy": round(accuracy_score, 1),  # 准确性
            "completeness": round(completeness_score, 1),  # 完整性
            "clarity": round(clarity_score, 1),  # 清晰度
            "risk_recognition": round(risk_recognition_score, 1),  # 风险识别度
        }

        overall_score = round(sum(dimensions.values()) / len(dimensions), 1)
        quality_level = (
            "excellent" if overall_score >= 8.5
            else "good" if overall_score >= 7.5
            else "fair" if overall_score >= 6.5
            else "poor"
        )

        return {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "dimensions": dimensions,
            "confidence": 0.8,
            "timestamp": None,  # Set by caller
        }
