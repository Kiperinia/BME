from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.agent_workflow import AgentWorkflowLesionSchema, AgentWorkflowSchema
from app.schemas.workspace import (
    AgentTraceStepSchema,
    ExpertConfigurationSchema,
    WorkspaceFeatureTagSchema,
    WorkspaceReportRequestSchema,
    WorkspaceReportResponseSchema,
    WorkspaceSegmentationSchema,
)
from app.services.agent_workflow_service import AgentWorkflowService
from app.services.sam3_runtime import SAM3Engine


class WorkspaceService:
    _MORPHOLOGY_GROUP_LABELS = {
        "elevated": "隆起型",
        "flat": "平坦型",
        "depressed": "凹陷型",
    }

    def __init__(self, settings: Settings, sam3_engine: SAM3Engine):
        self.settings = settings
        self.sam3_engine = sam3_engine
        self.agent_runtime = AgentWorkflowService(settings=settings, sam3_engine=sam3_engine)

    def generate_report(self, payload: WorkspaceReportRequestSchema) -> WorkspaceReportResponseSchema:
        image_bytes = self._decode_image_source(payload.image.dataUrl)
        image = self._decode_image_bytes(image_bytes)
        mask, points, bbox, uses_full_frame_fallback = self._resolve_mask_inputs(payload.segmentation, image)

        diagnosis = self.agent_runtime.agent.diagnose_single_sync(
            image=image,
            mask=mask,
            bbox=bbox,
            lesion_id="workspace-lesion-1",
            context={
                "patient_id": payload.patient.patientId,
                "study_id": payload.image.filename,
                "exam_date": payload.patient.examDate,
            },
        )

        findings = self._compose_findings(
            base_findings=diagnosis.report.findings,
            expert_config=payload.expertConfig,
            segmentation=payload.segmentation,
        )
        conclusion = self._compose_conclusion(
            base_conclusion=diagnosis.report.conclusion,
            expert_config=payload.expertConfig,
            diagnosis=diagnosis,
        )
        recommendation = self._compose_recommendation(diagnosis=diagnosis, expert_config=payload.expertConfig)
        workflow = self._build_workflow(
            payload=payload,
            diagnosis=diagnosis,
            uses_full_frame_fallback=uses_full_frame_fallback,
        )
        report_markdown = self._build_report_markdown(
            payload=payload,
            findings=findings,
            conclusion=conclusion,
            recommendation=recommendation,
            workflow=workflow,
            diagnosis=diagnosis,
        )
        feature_tags = self._extract_feature_tags(
            payload=payload,
            diagnosis=diagnosis,
            findings=findings,
            conclusion=conclusion,
            recommendation=recommendation,
        )
        agent_trace = self._build_agent_trace(
            payload=payload,
            diagnosis=diagnosis,
            findings=findings,
            feature_tags=feature_tags,
        )

        return WorkspaceReportResponseSchema(
            findings=findings,
            conclusion=conclusion,
            recommendation=recommendation,
            reportMarkdown=report_markdown,
            featureTags=feature_tags,
            agentTrace=agent_trace,
            workflow=workflow,
        )

    def _resolve_mask_inputs(
        self,
        segmentation: WorkspaceSegmentationSchema,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int, int], bool]:
        if segmentation.maskDataUrl.strip():
            mask = self._decode_mask_source(segmentation.maskDataUrl, image_width=image.shape[1], image_height=image.shape[0])
            points, bbox = self._extract_points_and_bbox_from_mask(mask)
            if points:
                return mask, points, bbox, False

        if segmentation.maskCoordinates and len(segmentation.maskCoordinates) >= 3:
            points = self._normalize_points(
                segmentation.maskCoordinates,
                image_width=image.shape[1],
                image_height=image.shape[0],
            )
            mask = self._polygon_points_to_mask(points, width=image.shape[1], height=image.shape[0])
            bbox = self._resolve_bbox(segmentation, points)
            return mask, points, bbox, False

        image_height, image_width = image.shape[:2]
        points = [
            (0, 0),
            (image_width - 1, 0),
            (image_width - 1, image_height - 1),
            (0, image_height - 1),
        ]
        mask = np.full((image_height, image_width), 255, dtype=np.uint8)
        bbox = (0, 0, image_width - 1, image_height - 1)
        return mask, points, bbox, True

    def _build_workflow(
        self,
        *,
        payload: WorkspaceReportRequestSchema,
        diagnosis: Any,
        uses_full_frame_fallback: bool,
    ) -> AgentWorkflowSchema:
        report_tool_calls = list(getattr(diagnosis.report, "tool_calls", []) or [])
        report_tool_names = [
            str(call.get("tool_name", "")).strip()
            for call in report_tool_calls
            if call.get("status") == "ok" and str(call.get("tool_name", "")).strip()
        ]
        unique_report_tools = list(dict.fromkeys(report_tool_names))

        lesion = AgentWorkflowLesionSchema(
            lesionId=diagnosis.lesion_id,
            sourceLabel=payload.image.filename,
            label=diagnosis.label,
            confidence=round(diagnosis.confidence, 4),
            bbox=tuple(int(value) for value in diagnosis.bbox),
            parisType=diagnosis.paris_typing.paris_type.value,
            invasionRisk=diagnosis.paris_typing.invasion_risk.value,
            riskLevel=diagnosis.risk_assessment.risk_level.value,
            totalScore=round(diagnosis.risk_assessment.total_score, 2),
            disposition=diagnosis.risk_assessment.disposition.value,
            estimatedSizeMm=round(diagnosis.morphology.estimated_size_mm, 1),
            shapeDescription=diagnosis.morphology.shape_description,
            usedLlm=(
                diagnosis.morphology.used_llm
                or diagnosis.paris_typing.used_llm
                or diagnosis.risk_assessment.used_llm
            ),
        )

        warnings = list(self.agent_runtime.runtime_warnings)
        if not payload.expertConfig.parisClassification.strip():
            warnings.append("Expert Paris classification is empty; report uses model-derived morphology cues only.")
        if not payload.expertConfig.pathologyClassification.strip():
            warnings.append("Pathology classification is empty; downstream bank scoring will treat the sample as partially labeled.")
        if uses_full_frame_fallback:
            warnings.append("未提供有效分割掩码，已使用整帧区域生成报告；建议后续补充分割结果以提升定位准确性。")

        return AgentWorkflowSchema(
            agentName="workspace-report-agent",
            description="Generate a structured report from uploaded image segmentation and expert annotations.",
            pipeline="upload -> segmentation -> expert morphology -> feature -> morphology -> paris -> risk -> report -> keyword tags",
            llmConfigured=bool(self.agent_runtime.agent.summary().get("llm_configured", False)),
            workflowMode=self.agent_runtime.workflow_mode,
            generatedAt=datetime.now(timezone.utc),
            lesionCount=1,
            highestRiskLesionId=diagnosis.lesion_id,
            modelVersion=diagnosis.report.model_version,
            steps=[
                f"Loaded local image {payload.image.filename} ({payload.image.width}x{payload.image.height}).",
                (
                    "No valid segmentation provided; used full-frame ROI fallback for diagnosis."
                    if uses_full_frame_fallback
                    else f"Reused frontend segmentation mask with {payload.segmentation.pointCount or len(payload.segmentation.maskCoordinates)} contour points."
                ),
                f"Applied doctor-selected Paris morphology reference: {payload.expertConfig.parisClassification or 'not provided'}.",
                "Ran the MedicalSAM3-compatible diagnosis agent on the segmented lesion region.",
                (
                    f"Report tool registry executed: {', '.join(unique_report_tools)}."
                    if unique_report_tools
                    else "Merged expert morphology, pathology, and notes into a formal markdown diagnostic report and extracted feature index tags."
                ),
            ],
            warnings=warnings,
            lesions=[lesion],
        )

    def _compose_findings(
        self,
        *,
        base_findings: str,
        expert_config: ExpertConfigurationSchema,
        segmentation: WorkspaceSegmentationSchema,
    ) -> str:
        paris_statement = self._build_paris_statement(expert_config=expert_config)
        additions: list[str] = [base_findings.strip()]
        if expert_config.lesionType.strip():
            additions.append(f"专家类型判断为{expert_config.lesionType.strip()}。")
        if paris_statement:
            additions.append(f"Paris 形态学参考为{paris_statement}。")
        if expert_config.pathologyClassification.strip():
            additions.append(f"病理/类型分类参考为{expert_config.pathologyClassification.strip()}。")
        additions.append(
            f"本次分割轮廓点数为{segmentation.pointCount or len(segmentation.maskCoordinates)}，掩码面积占比约{segmentation.maskAreaRatio:.3f}。"
        )
        return " ".join(part for part in additions if part)

    def _compose_conclusion(self, *, base_conclusion: str, expert_config: ExpertConfigurationSchema, diagnosis: Any) -> str:
        additions: list[str] = [base_conclusion.strip()]
        if expert_config.pathologyClassification.strip():
            additions.append(f"建议结合{expert_config.pathologyClassification.strip()}方向进行病理复核。")
        if expert_config.expertNotes.strip():
            additions.append(f"专家备注提示：{expert_config.expertNotes.strip()}")
        additions.append(
            f"综合风险等级为{diagnosis.risk_assessment.risk_level.value}，评分{diagnosis.risk_assessment.total_score:.1f}/10。"
        )
        return " ".join(part for part in additions if part)

    @staticmethod
    def _compose_recommendation(*, diagnosis: Any, expert_config: ExpertConfigurationSchema) -> str:
        recommendation = diagnosis.risk_assessment.disposition_reason.strip() or diagnosis.report.layout_suggestion.strip()
        if expert_config.surfacePattern.strip():
            recommendation = f"{recommendation} 表面模式参考：{expert_config.surfacePattern.strip()}。"
        return recommendation.strip() or "建议结合内镜医师复核意见及后续病理结果进行综合判断。"

    def _build_report_markdown(
        self,
        *,
        payload: WorkspaceReportRequestSchema,
        findings: str,
        conclusion: str,
        recommendation: str,
        workflow: AgentWorkflowSchema,
        diagnosis: Any,
    ) -> str:
        paris_statement = self._build_paris_statement(expert_config=payload.expertConfig)
        lesion = workflow.lesions[0]
        return "\n".join(
            [
                "# 正式内镜诊断报告",
                "",
                "> 本报告由 MedicalSAM3 分割结果、专家配置和诊断 Agent 联合生成，请由临床医生最终审核。",
                "",
                "## 一、患者与检查信息",
                f"- 患者编号：{payload.patient.patientId}",
                f"- 患者姓名：{payload.patient.patientName or '未填写'}",
                f"- 检查日期：{payload.patient.examDate or '未填写'}",
                f"- 影像文件：{payload.image.filename}",
                "",
                "## 二、分割与病灶概况",
                f"- 分割边界框：`{list(payload.segmentation.boundingBox)}`",
                f"- 掩码轮廓点数：{payload.segmentation.pointCount or len(payload.segmentation.maskCoordinates)}",
                f"- 掩码面积占比：{payload.segmentation.maskAreaRatio:.3f}",
                f"- 估计病灶大小：{lesion.estimatedSizeMm:.1f} mm",
                f"- 形态描述：{lesion.shapeDescription}",
                "",
                "## 三、Paris 分型参考",
                f"- 形态学分组：{self._MORPHOLOGY_GROUP_LABELS.get(payload.expertConfig.parisDetail.morphologyGroup, payload.expertConfig.parisDetail.morphologyGroup)}",
                f"- 细分编码：{payload.expertConfig.parisDetail.subtypeCode}",
                f"- 表型标签：{payload.expertConfig.parisDetail.displayLabel or '未填写'}",
                f"- 特征描述：{payload.expertConfig.parisDetail.featureSummary or '未填写'}",
                f"- 形态学参考：{payload.expertConfig.parisDetail.featureReference or '未填写'}",
                f"- 汇总结论：{paris_statement or '未填写'}",
                "",
                "## 四、诊断所见",
                findings,
                "",
                "## 五、诊断结论",
                conclusion,
                "",
                "## 六、处理建议",
                recommendation,
                "",
                "## 七、专家补充信息",
                f"- 类型分类：{payload.expertConfig.lesionType or '未填写'}",
                f"- 病理/类型分类：{payload.expertConfig.pathologyClassification or '未填写'}",
                f"- 表面模式：{payload.expertConfig.surfacePattern or '未填写'}",
                f"- 专家备注：{payload.expertConfig.expertNotes or '无'}",
                "",
                "## 八、Agent 评估摘要",
                f"- Agent 模式：{workflow.workflowMode}",
                f"- 风险等级：{lesion.riskLevel}",
                f"- 风险评分：{lesion.totalScore:.2f}",
                f"- Paris 推断：{diagnosis.paris_typing.paris_type.value}",
                f"- 处置建议：{lesion.disposition}",
                "",
                "## 九、工作流记录",
                *[f"- {step}" for step in workflow.steps],
            ]
        )

    def _extract_feature_tags(
        self,
        *,
        payload: WorkspaceReportRequestSchema,
        diagnosis: Any,
        findings: str,
        conclusion: str,
        recommendation: str,
    ) -> list[WorkspaceFeatureTagSchema]:
        tags: list[WorkspaceFeatureTagSchema] = []

        def append_tag(label: str, category: str, tone: str) -> None:
            normalized = label.strip()
            if not normalized:
                return

            tag_id = f"{category}-{re.sub('[^0-9A-Za-z\u4e00-\u9fff]+', '-', normalized).strip('-').lower()}"
            if any(existing.id == tag_id for existing in tags):
                return
            tags.append(
                WorkspaceFeatureTagSchema(
                    id=tag_id[:128] or f"{category}-tag",
                    label=normalized[:64],
                    category=category[:64],
                    tone=tone,  # type: ignore[arg-type]
                )
            )

        paris_detail = payload.expertConfig.parisDetail
        append_tag(paris_detail.subtypeCode, "Paris 分型", "sky")
        append_tag(paris_detail.displayLabel or self._MORPHOLOGY_GROUP_LABELS.get(paris_detail.morphologyGroup, ""), "Paris 形态", "sky")
        append_tag(self._MORPHOLOGY_GROUP_LABELS.get(paris_detail.morphologyGroup, ""), "形态分组", "violet")
        append_tag(diagnosis.risk_assessment.risk_level.value, "风险等级", "rose")
        append_tag(diagnosis.risk_assessment.disposition.value, "处置建议", "amber")
        append_tag(payload.expertConfig.lesionType, "类型分类", "emerald")
        append_tag(payload.expertConfig.pathologyClassification, "病理分类", "rose")
        append_tag(payload.expertConfig.surfacePattern, "表面模式", "amber")

        lesion_size_label = f"{round(diagnosis.morphology.estimated_size_mm, 1)}mm"
        append_tag(lesion_size_label, "病灶大小", "violet")
        if payload.segmentation.maskAreaRatio >= 0.2:
            append_tag("较大掩码面积", "分割特征", "amber")
        elif payload.segmentation.maskAreaRatio > 0:
            append_tag("局灶掩码面积", "分割特征", "sky")

        text_blob = " ".join([findings, conclusion, recommendation, paris_detail.featureSummary, paris_detail.featureReference])
        keyword_patterns = [
            (r"轻微隆起", "轻微隆起", "形态关键词", "sky"),
            (r"完全平坦", "完全平坦", "形态关键词", "sky"),
            (r"轻微凹陷", "轻微凹陷", "形态关键词", "sky"),
            (r"充血", "充血", "表面征象", "rose"),
            (r"糜烂", "糜烂", "表面征象", "rose"),
            (r"不规则血管|血管", "血管异常", "表面征象", "amber"),
            (r"颗粒", "颗粒样表面", "表面征象", "emerald"),
            (r"浸润", "浸润风险", "风险征象", "rose"),
        ]
        for pattern, label, category, tone in keyword_patterns:
            if re.search(pattern, text_blob):
                append_tag(label, category, tone)

        return tags

    def _build_agent_trace(
        self,
        *,
        payload: WorkspaceReportRequestSchema,
        diagnosis: Any,
        findings: str,
        feature_tags: list[WorkspaceFeatureTagSchema],
    ) -> list[AgentTraceStepSchema]:
        llm_mode = "LLM-assisted" if self.agent_runtime.workflow_mode == "llm" else "rule-only"
        paris_detail = payload.expertConfig.parisDetail
        report_tool_calls = list(getattr(diagnosis.report, "tool_calls", []) or [])

        trace_steps: list[AgentTraceStepSchema] = [
            AgentTraceStepSchema(
                id="thought-input",
                kind="thought",
                title="Review task context",
                detail=(
                    f"Use uploaded image {payload.image.filename}, the segmentation polygon, and the doctor-provided "
                    f"Paris reference {payload.expertConfig.parisClassification or paris_detail.subtypeCode} to build a formal report."
                ),
                status=llm_mode,
            ),
            AgentTraceStepSchema(
                id="tool-feature-extractor",
                kind="tool_call",
                title="Call feature extractor",
                detail="Measure lesion size, contour geometry, and local visual features from the segmented region.",
                toolName="FeatureExtractor",
                status="running",
            ),
            AgentTraceStepSchema(
                id="tool-feature-result",
                kind="tool_result",
                title="Feature extractor result",
                detail=(
                    f"Estimated lesion size {diagnosis.morphology.estimated_size_mm:.1f} mm with bbox {list(diagnosis.bbox)} "
                    f"and shape description '{diagnosis.morphology.shape_description}'."
                ),
                toolName="FeatureExtractor",
                status="ok",
            ),
            AgentTraceStepSchema(
                id="tool-morphology",
                kind="tool_call",
                title="Call morphology + Paris typing tools",
                detail="Combine model features with expert Paris morphology reference before assigning Paris-related morphology labels.",
                toolName="MorphologyClassifier + ParisTypingEngine",
                status="running",
            ),
            AgentTraceStepSchema(
                id="tool-morphology-result",
                kind="tool_result",
                title="Morphology + Paris result",
                detail=(
                    f"Model Paris result is {diagnosis.paris_typing.paris_type.value}; "
                    f"expert slider reference is {paris_detail.subtypeCode} {paris_detail.displayLabel}."
                ),
                toolName="MorphologyClassifier + ParisTypingEngine",
                status="ok",
            ),
            AgentTraceStepSchema(
                id="tool-risk",
                kind="tool_call",
                title="Call risk assessor",
                detail="Assess invasion and management risk from morphology, Paris typing, and lesion features.",
                toolName="RiskAssessor",
                status="running",
            ),
            AgentTraceStepSchema(
                id="tool-risk-result",
                kind="tool_result",
                title="Risk assessor result",
                detail=(
                    f"Risk level {diagnosis.risk_assessment.risk_level.value} with score "
                    f"{diagnosis.risk_assessment.total_score:.2f} and disposition {diagnosis.risk_assessment.disposition.value}."
                ),
                toolName="RiskAssessor",
                status="ok",
            ),
            AgentTraceStepSchema(
                id="tool-report",
                kind="tool_call",
                title="Call report tool registry",
                detail="Invoke registered report tools with validated input schema to assemble findings, conclusion and layout suggestions.",
                toolName="ReportToolRegistry",
                status="running",
            ),
        ]

        if report_tool_calls:
            for index, call in enumerate(report_tool_calls, start=1):
                tool_name = str(call.get("tool_name", "report-tool"))
                status = str(call.get("status", "ok"))
                duration = call.get("duration_ms")
                duration_label = f"{duration}ms" if duration is not None else "n/a"
                output_preview = str(call.get("output_preview", ""))
                error_message = str(call.get("error_message", ""))

                detail_parts = [
                    f"Validated schema inputs and executed {tool_name} in {duration_label}.",
                ]
                if output_preview:
                    detail_parts.append(f"Output preview: {output_preview}")
                if error_message:
                    detail_parts.append(f"Error: {error_message}")

                trace_steps.append(
                    AgentTraceStepSchema(
                        id=f"tool-report-result-{index}",
                        kind="tool_result",
                        title=f"Report tool result #{index}",
                        detail=" ".join(detail_parts)[:3900],
                        toolName=tool_name,
                        status=status,
                    )
                )
        else:
            trace_steps.append(
                AgentTraceStepSchema(
                    id="tool-report-result",
                    kind="tool_result",
                    title="Report generator result",
                    detail=f"Generated report findings preview: {findings[:180]}",
                    toolName="ReportGenerator",
                    status="ok",
                )
            )

        trace_steps.append(
            AgentTraceStepSchema(
                id="final-tags",
                kind="final",
                title="Finalize feature index tags",
                detail=(
                    f"Extracted {len(feature_tags)} feature tags for this patient case, including Paris subtype, risk level, "
                    f"pathology cues, and surface morphology keywords."
                ),
                status="completed",
            )
        )

        return trace_steps

    def _build_paris_statement(self, *, expert_config: ExpertConfigurationSchema) -> str:
        if expert_config.parisClassification.strip():
            return expert_config.parisClassification.strip()

        detail = expert_config.parisDetail
        group_label = self._MORPHOLOGY_GROUP_LABELS.get(detail.morphologyGroup, detail.morphologyGroup)
        if not detail.subtypeCode.strip():
            return ""
        return f"{group_label} / {detail.subtypeCode} {detail.displayLabel}：{detail.featureSummary}"

    @staticmethod
    def _decode_image_source(image_source: str) -> bytes:
        if not image_source.startswith("data:"):
            raise AppException(400, 40051, "workspace image must be sent as a data URL")

        try:
            _, encoded = image_source.split(",", 1)
        except ValueError as exc:
            raise AppException(400, 40052, "invalid image data URL payload") from exc

        try:
            return base64.b64decode(encoded)
        except Exception as exc:
            raise AppException(400, 40053, "failed to decode uploaded image data") from exc

    @staticmethod
    def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise AppException(400, 40054, "failed to decode uploaded image")
        return image

    @staticmethod
    def _decode_mask_source(data_url: str, *, image_width: int, image_height: int) -> np.ndarray:
        mask_bytes = WorkspaceService._decode_image_source(data_url)
        image_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise AppException(400, 40066, "failed to decode mask image payload")

        if decoded.ndim == 2:
            raw_mask = decoded
        elif decoded.ndim == 3 and decoded.shape[2] == 4:
            raw_mask = decoded[:, :, 3]
        else:
            raw_mask = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)

        binary = (raw_mask > 20).astype(np.uint8) * 255
        if binary.shape[1] != image_width or binary.shape[0] != image_height:
            binary = cv2.resize(binary, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        return binary

    @staticmethod
    def _extract_points_and_bbox_from_mask(mask: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], (0, 0, 0, 0)

        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        if polygon.shape[0] < 3:
            polygon = contour

        points = [tuple(map(int, point)) for point in polygon.reshape(-1, 2).tolist()]
        x, y, width, height = cv2.boundingRect(contour)
        bbox = (int(x), int(y), int(x + max(width - 1, 0)), int(y + max(height - 1, 0)))
        return points, bbox

    @staticmethod
    def _normalize_points(
        points: list[tuple[int, int]],
        *,
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int]]:
        normalized: list[tuple[int, int]] = []
        for x_value, y_value in points:
            normalized.append(
                (
                    min(max(int(x_value), 0), image_width - 1),
                    min(max(int(y_value), 0), image_height - 1),
                )
            )
        return normalized

    @staticmethod
    def _polygon_points_to_mask(points: list[tuple[int, int]], *, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    @staticmethod
    def _resolve_bbox(
        segmentation: WorkspaceSegmentationSchema,
        points: list[tuple[int, int]],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = segmentation.boundingBox
        if x2 > x1 and y2 > y1:
            return (int(x1), int(y1), int(x2), int(y2))

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs), min(ys), max(xs), max(ys))
