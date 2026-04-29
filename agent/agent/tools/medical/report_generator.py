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

from agent.tools.medical.feature_extractor import LesionFeatures
from agent.tools.medical.morphology_classifier import MorphologyResult
from agent.tools.medical.paris_typing import ParisTypingResult
from agent.tools.medical.risk_assessor import (
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
    """完整报告数据"""
    patient_id: str = ""
    study_id: str = ""
    exam_date: str = ""
    findings: str = ""               # 检查所见
    conclusion: str = ""             # 诊断结论
    layout_suggestion: str = ""      # 排版建议
    lesion_summary: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    model_version: str = "medical-pipeline-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "exam_date": self.exam_date,
            "findings": self.findings,
            "conclusion": self.conclusion,
            "layoutSuggestion": self.layout_suggestion,
            "lesion_summary": self.lesion_summary,
            "risk_summary": self.risk_summary,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
        }

    def to_api_response(self) -> dict[str, str]:
        """对齐前端 GenerateReportDraftResponse 格式"""
        return {
            "findings": self.findings,
            "conclusion": self.conclusion,
            "layoutSuggestion": self.layout_suggestion,
        }


# ---------------------------------------------------------------------------
# LLM 接口协议
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> str: ...


# ---------------------------------------------------------------------------
# 报告生成器
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    诊断报告生成器。

    两种模式：
      1. 模板模式（默认）：基于规则拼接结构化报告
      2. LLM 模式：调用 LLM 生成更自然的报告文本

    Usage::

        generator = ReportGenerator()
        report = generator.generate(
            patient_id="PATIENT_001",
            morphology=morph_result,
            paris=paris_result,
            risk=risk_result,
            features=features,
        )
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm: bool = False,
        prompt_path: Path | None = None,
    ):
        """
        Args:
            llm_client: LLM 客户端。
            use_llm: 是否使用 LLM 生成报告（否则用模板）。
            prompt_path: 自定义 prompt 路径。
        """
        self.llm_client = llm_client
        self.use_llm = use_llm
        self._prompt_template = self._load_prompt(prompt_path)

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
        """
        生成完整诊断报告。

        Args:
            patient_id: 患者ID
            morphology: 形态分类结果
            paris: Paris 分型结果
            risk: 风险评估结果
            features: 定量特征
            study_id: 检查编号
            exam_date: 检查日期

        Returns:
            ReportData
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
        """基于模板生成结构化报告"""

        # ---- 检查所见 (findings) ----
        findings_parts: list[str] = []

        # 基本描述
        findings_parts.append(
            f"内镜检查发现病变 {morphology.size_grade.value} 型，"
            f"等效直径约 {morphology.estimated_size_mm:.1f} mm"
        )
        if features.geometric.area_mm2 is not None:
            findings_parts.append(f"（面积约 {features.geometric.area_mm2:.1f} mm²）")

        findings_parts.append("。")

        # 形态描述
        findings_parts.append(
            f"病变呈{self._pedicle_cn(morphology.pedicle_type.value)}，"
            f"{morphology.shape_description}"
        )

        # Paris 分型
        findings_parts.append(
            f"按 Paris 分型标准，该病变为 {paris.paris_type.value} 型"
        )
        if paris.sub_type:
            findings_parts.append(f"（{paris.sub_type}）")
        findings_parts.append("。")

        # 表面与血管
        findings_parts.append(
            f"表面纹理呈{self._surface_cn(features.texture.surface_pattern.value)}，"
            f"血管密度{self._vessel_cn(features.texture.vessel_density)}。"
        )

        # 颜色
        findings_parts.append(
            f"病变{self._color_cn(features.color.dominant_color.value)}，"
            f"边缘对比度{self._contrast_cn(features.color.border_contrast)}。"
        )

        findings = "".join(findings_parts)

        # ---- 诊断结论 (conclusion) ----
        conclusion_parts: list[str] = []

        # 风险等级
        risk_cn = {
            RiskLevel.LOW: "低",
            RiskLevel.INTERMEDIATE: "中等",
            RiskLevel.HIGH: "高",
        }
        conclusion_parts.append(
            f"综合评估恶性风险为{risk_cn.get(risk.risk_level, '未明确')}风险"
            f"（评分 {risk.total_score:.1f}/10）。"
        )

        # 处理建议
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

        # Paris 分型提示
        if paris.invasion_risk.value in ("moderate", "high"):
            conclusion_parts.append(
                f"Paris {paris.paris_type.value} 型病变浸润风险为"
                f"{paris.invasion_risk.value}，需重点关注。"
            )

        conclusion = "".join(conclusion_parts)

        # ---- 排版建议 (layoutSuggestion) ----
        layout = self._generate_layout_suggestion(morphology, paris, risk)

        # ---- 汇总 ----
        return ReportData(
            patient_id=patient_id,
            study_id=study_id,
            exam_date=exam_date or datetime.now().strftime("%Y-%m-%d"),
            findings=findings,
            conclusion=conclusion,
            layout_suggestion=layout,
            lesion_summary=morphology.to_dict(),
            risk_summary=risk.to_dict(),
            generated_at=datetime.now().isoformat(),
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
        """调用 LLM 生成自然语言报告"""
        prompt = self._build_llm_prompt(
            patient_id, morphology, paris, risk, features, study_id, exam_date
        )

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )
            parsed = self._parse_llm_report(response)

            return ReportData(
                patient_id=patient_id,
                study_id=study_id,
                exam_date=exam_date or datetime.now().strftime("%Y-%m-%d"),
                findings=parsed.get("findings", ""),
                conclusion=parsed.get("conclusion", ""),
                layout_suggestion=parsed.get("layoutSuggestion", ""),
                lesion_summary=morphology.to_dict(),
                risk_summary=risk.to_dict(),
                generated_at=datetime.now().isoformat(),
            )
        except Exception as exc:
            logger.warning("LLM report generation failed, falling back to template: %s", exc)
            return self._generate_with_template(
                patient_id, morphology, paris, risk, features, study_id, exam_date
            )

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
        """生成报告排版建议（供前端 ReportPreviewPanel 使用）"""
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
        return {
            "pedunculated": "有蒂型",
            "sessile": "无蒂型",
            "subpedunculated": "亚蒂型",
            "flat": "扁平型",
            "uncertain": "形态未明确",
        }.get(pedicle, pedicle)

    @staticmethod
    def _surface_cn(surface: str) -> str:
        return {
            "smooth": "光滑",
            "irregular": "不规则",
            "granular": "颗粒状",
            "villous": "绒毛状",
            "unknown": "未明确",
        }.get(surface, surface)

    @staticmethod
    def _vessel_cn(density: float) -> str:
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
        if contrast < 0.05:
            return "低"
        elif contrast < 0.10:
            return "中等"
        elif contrast < 0.20:
            return "较高"
        return "高"

    @staticmethod
    def _load_prompt(custom_path: Path | None) -> str:
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
