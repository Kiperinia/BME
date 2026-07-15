"""
report_generator.py — 报告文本生成

将形态分类、Paris 分型、风险评估结果整合为结构化诊断报告文本。

输出格式对齐前端 ReportBuilderView 的数据结构：
  - findings:      检查所见
  - conclusion:    诊断结论
  - layoutSuggestion: 报告排版建议
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .feature_extractor import LesionFeatures
from .morphology_classifier import MorphologyResult
from .paris_typing import ParisTypingResult
from .report_tools import ReportToolRegistry, create_default_report_tool_registry
from .risk_assessor import (
    Disposition,
    RiskAssessmentResult,
    RiskLevel,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts" / "medical"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReportData:
    """brief:
        Represent ReportData state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    patient_id: str = ""
    study_id: str = ""
    exam_date: str = ""
    findings: str = ""               # 检查所见
    conclusion: str = ""             # 诊断结论
    layout_suggestion: str = ""      # 排版建议
    lesion_summary: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""
    model_version: str = "medical-pipeline-v1"
    # ReAct 相关字段
    react_analysis: dict[str, Any] = field(default_factory=dict)     # 反思结果
    react_refinement: dict[str, Any] = field(default_factory=dict)   # 精修结果
    report_score: dict[str, Any] = field(default_factory=dict)       # 报告评分

    def to_dict(self) -> dict[str, Any]:
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "exam_date": self.exam_date,
            "findings": self.findings,
            "conclusion": self.conclusion,
            "layoutSuggestion": self.layout_suggestion,
            "lesion_summary": self.lesion_summary,
            "risk_summary": self.risk_summary,
            "tool_calls": self.tool_calls,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "react_analysis": self.react_analysis,
            "react_refinement": self.react_refinement,
            "report_score": self.report_score,
        }

    def to_api_response(self) -> dict[str, str]:
        """brief:
            Handle to api response.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "findings": self.findings,
            "conclusion": self.conclusion,
            "layoutSuggestion": self.layout_suggestion,
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
        temperature: float = 0.4,
        max_tokens: int = 1024,
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
# 报告生成器
# ---------------------------------------------------------------------------

class ReportGenerator:
    """brief:
        Represent ReportGenerator state and behavior.

    parameter:
        - llm_client: Input value for llm_client.
        - use_llm: Input value for use_llm.
        - prompt_path: Input value for prompt_path.
        - report_tool_registry: Input value for report_tool_registry.

    retrival:
        - Provides instances used by the surrounding workflow.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm: bool = False,
        prompt_path: Path | None = None,
        report_tool_registry: ReportToolRegistry | None = None,
    ):
        """brief:
            Initialize this object.

        parameter:
            - llm_client: Input value for llm_client.
            - use_llm: Input value for use_llm.
            - prompt_path: Input value for prompt_path.
            - report_tool_registry: Input value for report_tool_registry.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.llm_client = llm_client
        self.use_llm = use_llm
        self._prompt_template = self._load_prompt(prompt_path)
        # Pass llm_client to registry for ReAct tools
        self.report_tool_registry = report_tool_registry or create_default_report_tool_registry(llm_client=llm_client)

    # ---- 公共接口 ----

    def generate(
        self,
        patient_id: str,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        features: LesionFeatures,
        study_id: str = "",
        exam_date: str = "",
    ) -> ReportData:
        """brief:
            Handle generate.

        parameter:
            - patient_id: Input value for patient_id.
            - morphology: Input value for morphology.
            - paris: Input value for paris.
            - risk: Input value for risk.
            - features: Input value for features.
            - study_id: Input value for study_id.
            - exam_date: Input value for exam_date.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if self.use_llm and self.llm_client is not None:
            return self._generate_with_llm(
                patient_id, morphology, paris, risk, features, study_id, exam_date
            )

        return self._generate_with_template(
            patient_id, morphology, paris, risk, features, study_id, exam_date
        )

    # ---- 模板模式 ----

    def _generate_with_template(
        self,
        patient_id: str,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        features: LesionFeatures,
        study_id: str,
        exam_date: str,
    ) -> ReportData:
        """brief:
            Handle generate with template.

        parameter:
            - patient_id: Input value for patient_id.
            - morphology: Input value for morphology.
            - paris: Input value for paris.
            - risk: Input value for risk.
            - features: Input value for features.
            - study_id: Input value for study_id.
            - exam_date: Input value for exam_date.

        retrival:
            - Returns the computed value for the caller or workflow.
        """

        self.report_tool_registry.reset_logs()
        findings = self.report_tool_registry.call(
            "compose_findings",
            morphology=morphology,
            paris=paris,
            features=features,
        )
        conclusion = self.report_tool_registry.call(
            "compose_conclusion",
            paris=paris,
            risk=risk,
        )
        layout = self.report_tool_registry.call(
            "suggest_layout",
            morphology=morphology,
            paris=paris,
            risk=risk,
        )
        keywords = self.report_tool_registry.call(
            "suggest_report_keywords",
            findings=findings,
            conclusion=conclusion,
            max_keywords=6,
        )
        tool_calls = self.report_tool_registry.get_call_logs()

        # ---- 初步报告 ----
        report_data = ReportData(
            patient_id=patient_id,
            study_id=study_id,
            exam_date=exam_date or datetime.now().strftime("%Y-%m-%d"),
            findings=findings,
            conclusion=conclusion,
            layout_suggestion=layout,
            lesion_summary=morphology.to_dict(),
            risk_summary={
                **risk.to_dict(),
                "suggested_keywords": keywords,
            },
            tool_calls=tool_calls,
            generated_at=datetime.now().isoformat(),
        )

        # ---- ReAct 反思与精修 ----
        return self._apply_react_refinement(
            report_data,
            morphology=morphology,
            paris=paris,
            risk=risk,
        )

    # ---- LLM 模式 ----

    def _generate_with_llm(
        self,
        patient_id: str,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        features: LesionFeatures,
        study_id: str,
        exam_date: str,
    ) -> ReportData:
        """brief:
            Handle generate with llm.

        parameter:
            - patient_id: Input value for patient_id.
            - morphology: Input value for morphology.
            - paris: Input value for paris.
            - risk: Input value for risk.
            - features: Input value for features.
            - study_id: Input value for study_id.
            - exam_date: Input value for exam_date.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        prompt = self._build_llm_prompt(
            patient_id, morphology, paris, risk, features, study_id, exam_date
        )
        self.report_tool_registry.reset_logs()

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )
            parsed = self._parse_llm_report(response)
            findings = parsed.get("findings", "")
            conclusion = parsed.get("conclusion", "")
            layout_suggestion = parsed.get("layoutSuggestion", "")

            if not layout_suggestion.strip():
                layout_suggestion = self.report_tool_registry.call(
                    "suggest_layout",
                    morphology=morphology,
                    paris=paris,
                    risk=risk,
                )

            keywords = self.report_tool_registry.call(
                "suggest_report_keywords",
                findings=findings,
                conclusion=conclusion,
                max_keywords=6,
            )
            tool_calls = self.report_tool_registry.get_call_logs()

            # ---- 初步报告 ----
            report_data = ReportData(
                patient_id=patient_id,
                study_id=study_id,
                exam_date=exam_date or datetime.now().strftime("%Y-%m-%d"),
                findings=findings,
                conclusion=conclusion,
                layout_suggestion=layout_suggestion,
                lesion_summary=morphology.to_dict(),
                risk_summary={
                    **risk.to_dict(),
                    "suggested_keywords": keywords,
                },
                tool_calls=tool_calls,
                generated_at=datetime.now().isoformat(),
            )

            # ---- ReAct 反思与精修 ----
            return self._apply_react_refinement(
                report_data,
                morphology=morphology,
                paris=paris,
                risk=risk,
            )
        except Exception as exc:
            logger.warning("LLM report generation failed, falling back to template: %s", exc)
            return self._generate_with_template(
                patient_id, morphology, paris, risk, features, study_id, exam_date
            )

    def _apply_react_refinement(
        self,
        report_data: ReportData,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
    ) -> ReportData:
        """brief:
            Handle apply react refinement.

        parameter:
            - report_data: Input value for report_data.
            - morphology: Input value for morphology.
            - paris: Input value for paris.
            - risk: Input value for risk.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        try:
            # ---- 第一步：ReAct Thinking - LLM 反思与分析 ----
            analysis_result = self.report_tool_registry.call(
                "analyze_report",
                findings=report_data.findings,
                conclusion=report_data.conclusion,
                paris=paris,
                risk=risk,
            )
            report_data.react_analysis = analysis_result
            logger.info(f"ReAct analysis: has_issues={analysis_result.get('has_issues')}, confidence={analysis_result.get('confidence')}")

            # ---- 第二步：ReAct Acting - LLM 根据分析精修报告 ----
            if analysis_result.get("has_issues") and analysis_result.get("suggestions"):
                refinement_findings = self.report_tool_registry.call(
                    "refine_report",
                    original_text=report_data.findings,
                    analysis_result=analysis_result,
                    text_type="findings",
                )
                report_data.findings = refinement_findings.get("refined_text", report_data.findings)
                logger.info(f"Refined findings: {len(refinement_findings.get('changes', []))} changes made")

                refinement_conclusion = self.report_tool_registry.call(
                    "refine_report",
                    original_text=report_data.conclusion,
                    analysis_result=analysis_result,
                    text_type="conclusion",
                )
                report_data.conclusion = refinement_conclusion.get("refined_text", report_data.conclusion)
                logger.info(f"Refined conclusion: {len(refinement_conclusion.get('changes', []))} changes made")

                report_data.react_refinement = {
                    "findings_refinement": refinement_findings,
                    "conclusion_refinement": refinement_conclusion,
                }

            # ---- 第三步：评分 ----
            score_result = self.report_tool_registry.call(
                "score_report",
                findings=report_data.findings,
                conclusion=report_data.conclusion,
                paris=paris,
                risk=risk,
                analysis_result=analysis_result,
            )
            report_data.report_score = score_result
            logger.info(f"Report score: {score_result.get('overall_score')}/10 ({score_result.get('quality_level')})")

            # ---- 更新 tool_calls 来包含所有 ReAct 工具调用 ----
            report_data.tool_calls = self.report_tool_registry.get_call_logs()

            return report_data

        except Exception as exc:
            logger.warning("ReAct refinement failed, returning original report: %s", exc)
            return report_data

    def _build_llm_prompt(
        self,
        patient_id: str,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        features: LesionFeatures,
        study_id: str,
        exam_date: str,
    ) -> str:
        """brief:
            Build llm prompt.

        parameter:
            - patient_id: Input value for patient_id.
            - morphology: Input value for morphology.
            - paris: Input value for paris.
            - risk: Input value for risk.
            - features: Input value for features.
            - study_id: Input value for study_id.
            - exam_date: Input value for exam_date.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        morph_text = "\n".join(f"  {k}: {v}" for k, v in morphology.to_dict().items())
        paris_text = "\n".join(f"  {k}: {v}" for k, v in paris.to_dict().items())
        risk_text = "\n".join(f"  {k}: {v}" for k, v in risk.to_dict().items())
        feat_text = "\n".join(
            f"  {k}: {v}" for k, v in features.to_dict().items()
            if isinstance(v, dict) for kk, vv in v.items()
        )

        return self._prompt_template.format(
            patient_id=patient_id,
            study_id=study_id or "未提供",
            exam_date=exam_date or datetime.now().strftime("%Y-%m-%d"),
            morphology=morph_text,
            paris_typing=paris_text,
            risk_assessment=risk_text,
            features=feat_text,
        )

    @staticmethod
    def _parse_llm_report(response: str) -> dict[str, str]:
        """brief:
            Handle parse llm report.

        parameter:
            - response: Input value for response.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        import json

        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM report: %s", text[:200])
            return {"findings": text, "conclusion": "", "layoutSuggestion": ""}

    # ---- 排版建议 ----

    @staticmethod
    def _generate_layout_suggestion(
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
    ) -> str:
        """brief:
            Handle generate layout suggestion.

        parameter:
            - morphology: Input value for morphology.
            - paris: Input value for paris.
            - risk: Input value for risk.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        suggestions: list[str] = []

        # 风险等级决定整体色调
        if risk.risk_level == RiskLevel.HIGH:
            suggestions.append("报告头部使用红色警示标识")
        elif risk.risk_level == RiskLevel.INTERMEDIATE:
            suggestions.append("报告头部使用橙色提醒标识")
        else:
            suggestions.append("报告头部使用常规样式")

        # 病变图像展示
        suggestions.append("病变区域截图放置在检查所见段落上方")
        suggestions.append("分割掩码叠加在原图上以半透明红色显示")

        # 风险评分可视化
        suggestions.append(
            f"风险评分 {risk.total_score:.1f}/10 以仪表盘形式展示，"
            f"颜色从绿（低）到红（高）渐变"
        )

        # 维度评分
        if risk.dimension_scores:
            dim_names = [ds.name for ds in risk.dimension_scores]
            suggestions.append(
                f"各维度评分（{', '.join(dim_names)}）以雷达图展示"
            )

        # 处理建议
        suggestions.append("处理建议以醒目卡片形式置于结论段落下方")

        return "；".join(suggestions) + "。"

    # ---- 中文映射 ----

    @staticmethod
    def _pedicle_cn(pedicle: str) -> str:
        """brief:
            Handle pedicle cn.

        parameter:
            - pedicle: Input value for pedicle.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "pedunculated": "有蒂型",
            "sessile": "无蒂型",
            "subpedunculated": "亚蒂型",
            "flat": "扁平型",
            "uncertain": "形态未明确",
        }.get(pedicle, pedicle)

    @staticmethod
    def _surface_cn(surface: str) -> str:
        """brief:
            Handle surface cn.

        parameter:
            - surface: Input value for surface.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "smooth": "光滑",
            "irregular": "不规则",
            "granular": "颗粒状",
            "villous": "绒毛状",
            "unknown": "未明确",
        }.get(surface, surface)

    @staticmethod
    def _vessel_cn(density: float) -> str:
        """brief:
            Handle vessel cn.

        parameter:
            - density: Input value for density.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if density < 0.02:
            return "稀疏"
        elif density < 0.05:
            return "较少"
        elif density < 0.10:
            return "中等"
        elif density < 0.20:
            return "较丰富"
        return "丰富"

    @staticmethod
    def _color_cn(color: str) -> str:
        """brief:
            Handle color cn.

        parameter:
            - color: Input value for color.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "red": "充血发红",
            "pale": "色泽苍白",
            "brown": "褐色",
            "mixed": "色泽混杂",
            "normal": "色泽接近正常黏膜",
            "unknown": "色泽未明确",
        }.get(color, color)

    @staticmethod
    def _contrast_cn(contrast: float) -> str:
        """brief:
            Handle contrast cn.

        parameter:
            - contrast: Input value for contrast.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if contrast < 0.05:
            return "低"
        elif contrast < 0.10:
            return "中等"
        elif contrast < 0.20:
            return "较高"
        return "高"

    @staticmethod
    def _load_prompt(custom_path: Path | None) -> str:
        """brief:
            Load prompt.

        parameter:
            - custom_path: Input value for custom_path.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        path = custom_path or (PROMPT_DIR / "report_generation.txt")
        if path.exists():
            return path.read_text(encoding="utf-8")

        return (
            "你是一名消化内镜诊断报告撰写专家。请根据以下分析结果，生成规范的中文内镜诊断报告。\n\n"
            "## 患者信息\n"
            "患者ID: {patient_id}\n"
            "检查编号: {study_id}\n"
            "检查日期: {exam_date}\n\n"
            "## 形态分类\n{morphology}\n\n"
            "## Paris 分型\n{paris_typing}\n\n"
            "## 风险评估\n{risk_assessment}\n\n"
            "## 定量特征\n{features}\n\n"
            "请以 JSON 格式返回：\n"
            "{{\n"
            '  "findings": "<检查所见，200-400字中文>",\n'
            '  "conclusion": "<诊断结论与建议，100-200字中文>",\n'
            '  "layoutSuggestion": "<报告排版建议，供前端渲染参考>"\n'
            "}}\n\n"
            "要求：\n"
            "1. findings 应包含病变位置、大小、形态、表面特征、颜色、血管等描述\n"
            "2. conclusion 应包含诊断意见和处理建议\n"
            "3. 使用规范的医学术语，语句通顺\n"
            "4. 避免使用不确定的表述（如'可能'），除非确实无法确定\n"
        )
