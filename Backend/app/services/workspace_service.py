from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.agent_workflow import AgentWorkflowLesionSchema, AgentWorkflowSchema
from app.schemas.workspace import (
    ExpertConfigurationSchema,
    WorkspaceReportRequestSchema,
    WorkspaceReportResponseSchema,
    WorkspaceSegmentationSchema,
)
from app.services.agent_workflow_service import AgentWorkflowService
from app.services.sam3_runtime import SAM3Engine


class WorkspaceService:
    def __init__(self, settings: Settings, sam3_engine: SAM3Engine):
        self.settings = settings
        self.sam3_engine = sam3_engine
        self.agent_runtime = AgentWorkflowService(settings=settings, sam3_engine=sam3_engine)

    def generate_report(self, payload: WorkspaceReportRequestSchema) -> WorkspaceReportResponseSchema:
        if len(payload.segmentation.maskCoordinates) < 3:
            raise AppException(422, 42241, "segmentation mask must contain at least 3 points")

        image_bytes = self._decode_image_source(payload.image.dataUrl)
        image = self._decode_image_bytes(image_bytes)
        points = self._normalize_points(
            payload.segmentation.maskCoordinates,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        mask = self._polygon_points_to_mask(points, width=image.shape[1], height=image.shape[0])
        bbox = self._resolve_bbox(payload.segmentation, points)

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
        workflow = self._build_workflow(payload=payload, diagnosis=diagnosis)
        report_markdown = self._build_report_markdown(
            payload=payload,
            findings=findings,
            conclusion=conclusion,
            recommendation=recommendation,
            workflow=workflow,
        )

        return WorkspaceReportResponseSchema(
            findings=findings,
            conclusion=conclusion,
            recommendation=recommendation,
            reportMarkdown=report_markdown,
            workflow=workflow,
        )

    def _build_workflow(
        self,
        *,
        payload: WorkspaceReportRequestSchema,
        diagnosis: Any,
    ) -> AgentWorkflowSchema:
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

        return AgentWorkflowSchema(
            agentName="workspace-report-agent",
            description="Generate a structured report from uploaded image segmentation and expert annotations.",
            pipeline="upload -> segmentation -> feature -> morphology -> paris -> risk -> report",
            llmConfigured=bool(self.agent_runtime.agent.summary().get("llm_configured", False)),
            workflowMode=self.agent_runtime.workflow_mode,
            generatedAt=datetime.now(timezone.utc),
            lesionCount=1,
            highestRiskLesionId=diagnosis.lesion_id,
            modelVersion=diagnosis.report.model_version,
            steps=[
                f"Loaded local image {payload.image.filename} ({payload.image.width}x{payload.image.height}).",
                f"Reused frontend segmentation mask with {payload.segmentation.pointCount or len(payload.segmentation.maskCoordinates)} contour points.",
                "Ran the MedicalSAM3-compatible diagnosis agent on the segmented lesion region.",
                "Merged expert Paris classification, pathology/type labels, and notes into the final report text.",
                "Prepared a structured report for doctor review and downstream exemplar-bank evaluation.",
            ],
            warnings=warnings,
            lesions=[lesion],
        )

    @staticmethod
    def _compose_findings(
        *,
        base_findings: str,
        expert_config: ExpertConfigurationSchema,
        segmentation: WorkspaceSegmentationSchema,
    ) -> str:
        additions: list[str] = [base_findings.strip()]
        if expert_config.lesionType.strip():
            additions.append(f"Expert lesion type: {expert_config.lesionType.strip()}.")
        if expert_config.parisClassification.strip():
            additions.append(f"Expert Paris classification: {expert_config.parisClassification.strip()}.")
        if expert_config.pathologyClassification.strip():
            additions.append(f"Pathology/type note: {expert_config.pathologyClassification.strip()}.")
        additions.append(
            f"Segmentation summary: {segmentation.pointCount or len(segmentation.maskCoordinates)} contour points, mask area ratio {segmentation.maskAreaRatio:.3f}."
        )
        return " ".join(part for part in additions if part)

    @staticmethod
    def _compose_conclusion(*, base_conclusion: str, expert_config: ExpertConfigurationSchema, diagnosis: Any) -> str:
        additions: list[str] = [base_conclusion.strip()]
        if expert_config.pathologyClassification.strip():
            additions.append(f"Pathology impression should be reviewed against {expert_config.pathologyClassification.strip()}.")
        if expert_config.expertNotes.strip():
            additions.append(f"Expert note: {expert_config.expertNotes.strip()}")
        additions.append(
            f"Risk level is {diagnosis.risk_assessment.risk_level.value} with score {diagnosis.risk_assessment.total_score:.1f}/10."
        )
        return " ".join(part for part in additions if part)

    @staticmethod
    def _compose_recommendation(*, diagnosis: Any, expert_config: ExpertConfigurationSchema) -> str:
        recommendation = diagnosis.risk_assessment.disposition_reason.strip() or diagnosis.report.layout_suggestion.strip()
        if expert_config.surfacePattern.strip():
            recommendation = f"{recommendation} Surface pattern note: {expert_config.surfacePattern.strip()}."
        return recommendation.strip() or "Recommend specialist review of the segmented lesion and final pathology correlation."

    @staticmethod
    def _build_report_markdown(
        *,
        payload: WorkspaceReportRequestSchema,
        findings: str,
        conclusion: str,
        recommendation: str,
        workflow: AgentWorkflowSchema,
    ) -> str:
        expert_lines = [
            f"- Paris classification: {payload.expertConfig.parisClassification or 'N/A'}",
            f"- Lesion type: {payload.expertConfig.lesionType or 'N/A'}",
            f"- Pathology/type classification: {payload.expertConfig.pathologyClassification or 'N/A'}",
            f"- Surface pattern: {payload.expertConfig.surfacePattern or 'N/A'}",
        ]
        if payload.expertConfig.expertNotes.strip():
            expert_lines.append(f"- Expert notes: {payload.expertConfig.expertNotes.strip()}")

        return "\n".join(
            [
                "# Endoscopy Diagnostic Report",
                "",
                "## Case",
                f"- Patient ID: {payload.patient.patientId}",
                f"- Patient name: {payload.patient.patientName or 'N/A'}",
                f"- Exam date: {payload.patient.examDate or 'N/A'}",
                f"- Image: {payload.image.filename}",
                "",
                "## Segmentation",
                f"- Bounding box: {list(payload.segmentation.boundingBox)}",
                f"- Mask point count: {payload.segmentation.pointCount or len(payload.segmentation.maskCoordinates)}",
                f"- Mask area ratio: {payload.segmentation.maskAreaRatio:.3f}",
                "",
                "## Findings",
                findings,
                "",
                "## Conclusion",
                conclusion,
                "",
                "## Recommendation",
                recommendation,
                "",
                "## Expert Configuration",
                *expert_lines,
                "",
                "## Workflow",
                *[f"- {step}" for step in workflow.steps],
            ]
        )

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
