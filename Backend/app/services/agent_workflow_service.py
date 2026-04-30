from __future__ import annotations

import base64
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from app.core.config import WORKSPACE_DIR, Settings
from app.core.exceptions import AppException
from app.schemas.agent_workflow import (
    AgentWorkflowLesionSchema,
    AgentWorkflowSchema,
    AnnotationTagSchema,
    FetchAnnotationTagsRequestSchema,
    GenerateReportDraftRequestSchema,
    PolygonMaskSchema,
    ReportDraftRecordSchema,
    SaveReportDraftRequestSchema,
)
from app.services.sam3_runtime import SAM3Engine


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedLesion:
    lesion_id: str
    source_label: str
    image: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


class AgentWorkflowService:
    def __init__(self, settings: Settings, sam3_engine: SAM3Engine):
        self.settings = settings
        self.sam3_engine = sam3_engine
        self.agent, self.workflow_mode, self.runtime_warnings = self._build_agent()

    def generate_report_draft(
        self,
        payload: GenerateReportDraftRequestSchema,
    ) -> dict[str, Any]:
        workflow = self._run_agent_workflow(
            context_data=payload.contextData,
            report_snippet=payload.contextData.initialOpinion or payload.contextData.reportSnippet,
        )

        report = workflow["batch_result"].report
        return {
            "findings": report.get("findings", ""),
            "conclusion": report.get("conclusion", ""),
            "layoutSuggestion": report.get("layoutSuggestion", ""),
            "workflow": workflow["workflow"],
            "streamMessages": workflow["workflow"].steps,
        }

    def infer_annotation_tags(
        self,
        payload: FetchAnnotationTagsRequestSchema,
    ) -> dict[str, Any]:
        workflow = self._run_agent_workflow(
            context_data=payload.contextData,
            report_snippet=payload.reportSnippet,
        )
        tags = self._build_annotation_tags(
            workflow=workflow["workflow"],
            timestamp=payload.contextData.videoFrameData.timestamp,
            location_label=payload.contextData.videoFrameData.suspectedLocation,
        )
        return {
            "tags": tags,
            "workflow": workflow["workflow"],
        }

    def save_report_draft(
        self,
        payload: SaveReportDraftRequestSchema,
    ) -> ReportDraftRecordSchema:
        return ReportDraftRecordSchema(
            reportId=payload.reportId or f"draft-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            patientId=payload.patientId,
            findings=payload.findings,
            conclusion=payload.conclusion,
            layoutSuggestion=payload.layoutSuggestion,
            updatedAt=datetime.now(timezone.utc),
        )

    def _run_agent_workflow(self, context_data: Any, report_snippet: str) -> dict[str, Any]:
        prepared_lesions, segmentation_warnings = self._prepare_lesions(context_data)
        if not prepared_lesions:
            raise AppException(400, 40031, "no usable lesion images were provided for agent workflow")

        batch_result = self.agent.diagnose_batch_sync(
            lesions=[
                {
                    "image": lesion.image,
                    "mask": lesion.mask,
                    "bbox": lesion.bbox,
                    "lesion_id": lesion.lesion_id,
                    "metadata": {"source_label": lesion.source_label},
                }
                for lesion in prepared_lesions
            ],
            patient_context={
                "patient_id": context_data.patient.patientId,
                "study_id": context_data.videoFrameData.sourceId,
                "exam_date": context_data.patient.examDate,
                "report_snippet": report_snippet,
            },
        )

        workflow = self._build_workflow_summary(
            batch_result=batch_result,
            prepared_lesions=prepared_lesions,
            segmentation_warnings=segmentation_warnings,
        )
        return {
            "batch_result": batch_result,
            "workflow": workflow,
        }

    def _build_workflow_summary(
        self,
        batch_result: Any,
        prepared_lesions: list[PreparedLesion],
        segmentation_warnings: list[str],
    ) -> AgentWorkflowSchema:
        agent_summary = self.agent.summary()
        report = batch_result.report
        lesion_summaries: list[AgentWorkflowLesionSchema] = []

        for lesion_result, prepared in zip(batch_result.lesions, prepared_lesions, strict=False):
            lesion_summaries.append(
                AgentWorkflowLesionSchema(
                    lesionId=lesion_result.lesion_id,
                    sourceLabel=prepared.source_label,
                    label=lesion_result.label,
                    confidence=round(lesion_result.confidence, 4),
                    bbox=tuple(int(value) for value in lesion_result.bbox),
                    parisType=lesion_result.paris_typing.paris_type.value,
                    invasionRisk=lesion_result.paris_typing.invasion_risk.value,
                    riskLevel=lesion_result.risk_assessment.risk_level.value,
                    totalScore=round(lesion_result.risk_assessment.total_score, 2),
                    disposition=lesion_result.risk_assessment.disposition.value,
                    estimatedSizeMm=round(lesion_result.morphology.estimated_size_mm, 1),
                    shapeDescription=lesion_result.morphology.shape_description,
                    usedLlm=(
                        lesion_result.morphology.used_llm
                        or lesion_result.paris_typing.used_llm
                        or lesion_result.risk_assessment.used_llm
                    ),
                )
            )

        warnings = [*self.runtime_warnings, *segmentation_warnings]
        steps = [
            f"已从前端上下文装配 {len(prepared_lesions)} 个候选病灶。",
            f"Medical SAM 3 已完成 {len(prepared_lesions)} 张图像的分割。",
            f"Agent pipeline: {agent_summary.get('metadata', {}).get('pipeline', 'feature -> morphology -> paris -> risk -> report')}",
            (
                "已启用 LLM 报告生成与推理增强。"
                if self.workflow_mode == "llm"
                else "LLM 未就绪，当前回退到规则驱动 Agent 流程。"
            ),
            f"批量诊断完成，主病灶为 {report.get('highest_risk_lesion_id', lesion_summaries[0].lesionId if lesion_summaries else 'unknown')}。",
        ]

        return AgentWorkflowSchema(
            agentName=agent_summary.get("name", "medical-diagnosis-agent"),
            description=agent_summary.get("description", ""),
            pipeline=agent_summary.get("metadata", {}).get("pipeline", "feature -> morphology -> paris -> risk -> report"),
            llmConfigured=bool(agent_summary.get("llm_configured", False)),
            workflowMode=self.workflow_mode,
            generatedAt=datetime.now(timezone.utc),
            lesionCount=len(lesion_summaries),
            highestRiskLesionId=report.get("highest_risk_lesion_id"),
            modelVersion=(batch_result.lesions[0].report.model_version if batch_result.lesions else "medical-pipeline-v1"),
            steps=steps,
            warnings=warnings,
            lesions=lesion_summaries,
        )

    def _build_annotation_tags(
        self,
        workflow: AgentWorkflowSchema,
        timestamp: float,
        location_label: str,
    ) -> list[AnnotationTagSchema]:
        if not workflow.lesions:
            return []

        primary = next(
            (lesion for lesion in workflow.lesions if lesion.lesionId == workflow.highestRiskLesionId),
            workflow.lesions[0],
        )
        base_time = max(timestamp, 0.0)
        tags = [
            AnnotationTagSchema(
                id=f"{primary.lesionId}-label",
                label=primary.label,
                confidence=primary.confidence,
                targetTime=round(base_time, 1),
                locationLabel=location_label,
                needsReview=primary.confidence < 0.75,
            ),
            AnnotationTagSchema(
                id=f"{primary.lesionId}-paris",
                label=primary.parisType,
                confidence=max(primary.confidence - 0.04, 0.0),
                targetTime=round(base_time + 0.2, 1),
                locationLabel=location_label,
                needsReview=False,
            ),
            AnnotationTagSchema(
                id=f"{primary.lesionId}-risk",
                label=f"{primary.riskLevel} risk",
                confidence=max(min(primary.totalScore / 10.0, 1.0), 0.0),
                targetTime=round(base_time + 0.4, 1),
                locationLabel=location_label,
                needsReview=primary.riskLevel == "high",
            ),
        ]

        if primary.usedLlm or primary.confidence < 0.72:
            tags.append(
                AnnotationTagSchema(
                    id=f"{primary.lesionId}-review",
                    label="建议人工复核",
                    confidence=max(primary.confidence - 0.1, 0.0),
                    targetTime=round(base_time + 0.6, 1),
                    locationLabel=location_label,
                    needsReview=True,
                )
            )

        return tags

    def _prepare_lesions(self, context_data: Any) -> tuple[list[PreparedLesion], list[str]]:
        lesions: list[PreparedLesion] = []
        warnings: list[str] = []
        seen_sources: set[str] = set()
        source_candidates: list[tuple[str, str, list[PolygonMaskSchema]]] = []

        source_candidates.append(
            (
                "tumor-focus",
                context_data.tumorFocus.tumorImageSrc,
                context_data.tumorFocus.maskData if isinstance(context_data.tumorFocus.maskData, list) else [],
            )
        )
        for index, image_src in enumerate(context_data.captureImageSrcs[:2], start=1):
            source_candidates.append((f"capture-{index}", image_src, context_data.maskData))

        for lesion_id, image_source, fallback_polygons in source_candidates:
            if not image_source or image_source in seen_sources:
                continue
            seen_sources.add(image_source)

            try:
                image_bytes = self._decode_image_source(image_source)
                image = self._decode_image_bytes(image_bytes)
                mask_points, bbox = self._segment_or_fallback(
                    image_bytes=image_bytes,
                    image=image,
                    fallback_polygons=fallback_polygons,
                    filename=f"{lesion_id}.png",
                )
                mask = self._polygon_points_to_mask(mask_points, image.shape[1], image.shape[0])
                lesions.append(
                    PreparedLesion(
                        lesion_id=lesion_id,
                        source_label=lesion_id,
                        image=image,
                        mask=mask,
                        bbox=bbox,
                    )
                )
            except AppException as exc:
                warnings.append(exc.message)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.exception("failed to prepare lesion input for %s", lesion_id)
                warnings.append(f"{lesion_id} 预处理失败：{exc}")

        return lesions, warnings

    def _segment_or_fallback(
        self,
        image_bytes: bytes,
        image: np.ndarray,
        fallback_polygons: Iterable[PolygonMaskSchema],
        filename: str,
    ) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        try:
            result = self.sam3_engine.predict_bytes(image_bytes, filename)
            mask_points = [tuple(map(int, point)) for point in result.get("mask_coordinates", [])]
            bbox = tuple(map(int, result.get("bounding_box", (0, 0, 0, 0))))
            if mask_points:
                return mask_points, bbox
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("SAM3 segmentation failed for %s: %s", filename, exc)

        scaled_fallback = self._scale_polygons(fallback_polygons, width=width, height=height)
        if not scaled_fallback:
            raise AppException(422, 42231, f"{filename} 无法获得可用分割结果")

        bbox = self._bounding_box_from_points(scaled_fallback)
        return scaled_fallback, bbox

    @staticmethod
    def _decode_image_source(image_source: str) -> bytes:
        if not image_source.startswith("data:"):
            raise AppException(400, 40032, "agent workflow expects rasterized image data URLs from frontend")

        try:
            _, encoded = image_source.split(",", 1)
        except ValueError as exc:
            raise AppException(400, 40033, "invalid image data url payload") from exc

        try:
            return base64.b64decode(encoded)
        except Exception as exc:
            raise AppException(400, 40034, "failed to decode image data url") from exc

    @staticmethod
    def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise AppException(400, 40035, "failed to decode raster image for agent workflow")
        return image

    @staticmethod
    def _scale_polygons(
        polygons: Iterable[PolygonMaskSchema],
        *,
        width: int,
        height: int,
    ) -> list[tuple[int, int]]:
        scaled_points: list[tuple[int, int]] = []
        for polygon in polygons:
            if not polygon.points:
                continue
            scale_x = width / polygon.frameWidth
            scale_y = height / polygon.frameHeight
            scaled_points.extend(
                (int(round(point[0] * scale_x)), int(round(point[1] * scale_y)))
                for point in polygon.points
            )
        return scaled_points

    @staticmethod
    def _polygon_points_to_mask(points: list[tuple[int, int]], width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(points, dtype=np.int32)
        if polygon.size == 0:
            return mask
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    @staticmethod
    def _bounding_box_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        if not points:
            return (0, 0, 0, 0)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _build_agent(self) -> tuple[Any, str, list[str]]:
        agent_root = (WORKSPACE_DIR / "agent").resolve()
        agent_root_str = str(agent_root)
        if agent_root_str not in sys.path:
            sys.path.insert(0, agent_root_str)

        from core.agent import build_minimal_agent

        requested_use_llm = self.settings.agent_use_llm
        requested_use_llm_report = self.settings.agent_use_llm_report
        pixel_size_mm = self.settings.agent_pixel_size_mm

        if not requested_use_llm and not requested_use_llm_report:
            return (
                build_minimal_agent(
                    use_llm=False,
                    use_llm_report=False,
                    pixel_size_mm=pixel_size_mm,
                ),
                "rule-only",
                ["系统设置已禁用 LLM 增强，当前仅使用规则驱动 Agent。"],
            )

        try:
            return (
                build_minimal_agent(
                    use_llm=requested_use_llm,
                    use_llm_report=requested_use_llm_report,
                    pixel_size_mm=pixel_size_mm,
                ),
                "llm",
                [],
            )
        except Exception as exc:
            logger.warning("LLM agent bootstrap failed, falling back to rule-only mode: %s", exc)
            warning = "LLM 配置未就绪，当前已回退到规则驱动 Agent 工作流。"
            return (
                build_minimal_agent(
                    use_llm=False,
                    use_llm_report=False,
                    pixel_size_mm=pixel_size_mm,
                ),
                "rule-only",
                [warning],
            )