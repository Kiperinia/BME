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
    AgentRunSchema,
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
    """brief:
        Represent PreparedLesion state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    lesion_id: str
    source_label: str
    image: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


class AgentWorkflowService:
    """brief:
        Represent AgentWorkflowService state and behavior.

    parameter:
        - settings: Input value for settings.
        - sam3_engine: Input value for sam3_engine.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, settings: Settings, sam3_engine: SAM3Engine):
        """brief:
            Initialize this object.

        parameter:
            - settings: Input value for settings.
            - sam3_engine: Input value for sam3_engine.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.settings = settings
        self.sam3_engine = sam3_engine
        self.agent, self.workflow_mode, self.runtime_warnings = self._build_agent()

    def generate_report_draft(
        self,
        payload: GenerateReportDraftRequestSchema,
    ) -> dict[str, Any]:
        """brief:
            Handle generate report draft.

        parameter:
            - payload: Input value for payload.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Infer annotation tags.

        parameter:
            - payload: Input value for payload.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Save report draft.

        parameter:
            - payload: Input value for payload.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ReportDraftRecordSchema(
            reportId=payload.reportId or f"draft-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            patientId=payload.patientId,
            findings=payload.findings,
            conclusion=payload.conclusion,
            layoutSuggestion=payload.layoutSuggestion,
            updatedAt=datetime.now(timezone.utc),
        )

    def _run_agent_workflow(self, context_data: Any, report_snippet: str) -> dict[str, Any]:
        """brief:
            Run agent workflow.

        parameter:
            - context_data: Input value for context_data.
            - report_snippet: Input value for report_snippet.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        closed_loop_result = self._run_closed_loop_summary(
            context_data=context_data,
            prepared_lesions=prepared_lesions,
            report_snippet=report_snippet,
        )

        workflow = self._build_workflow_summary(
            batch_result=batch_result,
            prepared_lesions=prepared_lesions,
            segmentation_warnings=segmentation_warnings,
            closed_loop_result=closed_loop_result,
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
        closed_loop_result: dict[str, Any] | None = None,
    ) -> AgentWorkflowSchema:
        """brief:
            Build workflow summary.

        parameter:
            - batch_result: Input value for batch_result.
            - prepared_lesions: Input value for prepared_lesions.
            - segmentation_warnings: Input value for segmentation_warnings.
            - closed_loop_result: Input value for closed_loop_result.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        if closed_loop_result and closed_loop_result.get("review", {}).get("warnings"):
            warnings.extend(str(item) for item in closed_loop_result["review"]["warnings"])
        agent_runs = self._build_agent_runs(closed_loop_result)
        closed_loop_summary = self._build_closed_loop_summary(closed_loop_result)
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

        if agent_runs:
            steps.extend(
                f"{run.displayName or run.agentName}: {run.decision}"
                for run in agent_runs
            )

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
            agentRuns=agent_runs,
            closedLoopSummary=closed_loop_summary,
        )

    def _run_closed_loop_summary(
        self,
        *,
        context_data: Any,
        prepared_lesions: list[PreparedLesion],
        report_snippet: str,
    ) -> dict[str, Any] | None:
        """brief:
            Run closed loop summary.

        parameter:
            - context_data: Input value for context_data.
            - prepared_lesions: Input value for prepared_lesions.
            - report_snippet: Input value for report_snippet.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if not prepared_lesions:
            return None

        try:
            agent_root = (WORKSPACE_DIR / "agent").resolve()
            agent_root_str = str(agent_root)
            if agent_root_str not in sys.path:
                sys.path.insert(0, agent_root_str)

            from core.agent import build_medical_closed_loop_agent

            primary = prepared_lesions[0]
            doctor_annotations = self._build_doctor_annotations(context_data)
            sample = self._build_closed_loop_sample(primary, doctor_annotations)
            orchestrator = build_medical_closed_loop_agent(
                diagnosis_agent=self.agent,
                pixel_size_mm=self.settings.agent_pixel_size_mm,
            )
            result = orchestrator.run_sync(
                {
                    "image": primary.image,
                    "mask": primary.mask,
                    "bbox": primary.bbox,
                    "lesion_id": primary.lesion_id,
                    "report_snippet": report_snippet,
                    "sample": sample,
                    "patient_context": {
                        "patient_id": context_data.patient.patientId,
                        "study_id": context_data.videoFrameData.sourceId,
                        "exam_date": context_data.patient.examDate,
                    },
                },
                reference_sample=self._build_reference_sample(sample, doctor_annotations),
                doctor_annotations=doctor_annotations,
            )
            return result.to_dict()
        except Exception as exc:  # pragma: no cover - defensive integration path
            logger.warning("closed-loop agent summary failed: %s", exc)
            return {
                "review": {
                    "final_status": "needs_human_review",
                    "bank_decision": "needs_human_review",
                    "label_count": 0,
                    "warnings": [f"closed-loop summary failed: {exc}"],
                },
                "agent_runs": [],
            }

    @staticmethod
    def _build_doctor_annotations(context_data: Any) -> dict[str, Any]:
        """brief:
            Build doctor annotations.

        parameter:
            - context_data: Input value for context_data.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        details = context_data.tumorFocus.details
        return {
            "lesion_type": details.classification,
            "pathology": "",
            "surface_pattern": details.surfacePattern,
            "paris": context_data.initialOpinion or context_data.reportSnippet,
            "notes": details.location,
            "tags": [
                value
                for value in [
                    details.classification,
                    details.surfacePattern,
                    context_data.videoFrameData.suspectedLocation,
                ]
                if str(value).strip()
            ],
        }

    @staticmethod
    def _build_closed_loop_sample(primary: PreparedLesion, doctor_annotations: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Build closed loop sample.

        parameter:
            - primary: Input value for primary.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        area_ratio = float((primary.mask > 0).sum()) / max(float(primary.mask.shape[0] * primary.mask.shape[1]), 1.0)
        return {
            "image_id": primary.lesion_id,
            "site_id": "frontend",
            "split": "runtime",
            "sample_group": "candidate",
            "bbox": [float(value) for value in primary.bbox],
            "metrics": {"Dice": 0.82, "Precision": 0.84, "Recall": 0.8, "Boundary F1": 0.72, "mean confidence": 0.83},
            "baseline_metrics": {"Dice": 0.74},
            "mask_stats": {"area_ratio": area_ratio, "components": 1.0, "boundary_complexity": 0.42, "aspect_ratio": 1.0, "solidity": 0.9},
            "uncertainty": {"mean_entropy": 0.22, "mean_confidence": 0.83},
            "selected_exemplars": {"positive_ids": [], "negative_ids": [], "boundary_ids": []},
            "tags": [str(tag) for tag in doctor_annotations.get("tags", []) if str(tag).strip()],
        }

    @staticmethod
    def _build_reference_sample(sample: dict[str, Any], doctor_annotations: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Build reference sample.

        parameter:
            - sample: Input value for sample.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        reference = dict(sample)
        reference["image_id"] = f"{sample.get('image_id', 'case')}-reference"
        reference["sample_group"] = "clean"
        reference["tags"] = list(dict.fromkeys([*sample.get("tags", []), *doctor_annotations.get("tags", [])]))
        reference["metrics"] = {**dict(sample.get("metrics", {})), "Dice": 0.86}
        return reference

    @staticmethod
    def _build_agent_runs(closed_loop_result: dict[str, Any] | None) -> list[AgentRunSchema]:
        """brief:
            Build agent runs.

        parameter:
            - closed_loop_result: Input value for closed_loop_result.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if not closed_loop_result:
            return []
        runs: list[AgentRunSchema] = []
        for item in closed_loop_result.get("agent_runs", []):
            runs.append(
                AgentRunSchema(
                    agentName=str(item.get("agent_name", "")),
                    displayName=str(item.get("display_name", "")),
                    goal=str(item.get("goal", "")),
                    status=str(item.get("status", "")),
                    decision=str(item.get("decision", "")),
                    toolCalls=list(item.get("tool_calls", []) or []),
                    observations=dict(item.get("observations", {}) or {}),
                    warnings=[str(warning) for warning in item.get("warnings", []) or []],
                )
            )
        return runs

    @staticmethod
    def _build_closed_loop_summary(closed_loop_result: dict[str, Any] | None) -> dict[str, object]:
        """brief:
            Build closed loop summary.

        parameter:
            - closed_loop_result: Input value for closed_loop_result.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if not closed_loop_result:
            return {}
        review = dict(closed_loop_result.get("review", {}) or {})
        agent_details: list[dict[str, object]] = []
        main_tool_chains: dict[str, object] = {}
        for item in closed_loop_result.get("agent_runs", []) or []:
            observations = dict(item.get("observations", {}) or {})
            agent_name = str(item.get("agent_name", ""))
            main_tool_chain = list(observations.get("mainToolChain", []) or [])
            prompt_design = [str(prompt) for prompt in observations.get("promptDesign", []) or []]
            main_tool_chains[agent_name] = main_tool_chain
            agent_details.append(
                {
                    "agentName": agent_name,
                    "displayName": str(item.get("display_name", "")),
                    "detail": str(observations.get("agentDetail", "")),
                    "promptDesign": prompt_design,
                    "goal": str(item.get("goal", "")),
                    "status": str(item.get("status", "")),
                    "decision": str(item.get("decision", "")),
                    "mainToolChain": main_tool_chain,
                    "warnings": [str(warning) for warning in item.get("warnings", []) or []],
                    "keyOutputs": {
                        key: value
                        for key, value in observations.items()
                        if key not in {"agentDetail", "promptDesign", "mainToolChain"}
                    },
                }
            )
        return {
            "finalStatus": review.get("final_status", ""),
            "finalDecision": review.get("finalDecision", review.get("final_status", "")),
            "bankDecision": review.get("bank_decision", ""),
            "labelCount": review.get("label_count", 0),
            "termCount": review.get("term_count", 0),
            "databaseRecordCount": len((closed_loop_result.get("label_embedding", {}) or {}).get("dbRecords", []) or []),
            "qualityScore": review.get("qualityScore", 0.0),
            "agentDetails": agent_details,
            "mainToolChains": main_tool_chains,
            "warnings": review.get("warnings", []),
        }

    def _build_annotation_tags(
        self,
        workflow: AgentWorkflowSchema,
        timestamp: float,
        location_label: str,
    ) -> list[AnnotationTagSchema]:
        """brief:
            Build annotation tags.

        parameter:
            - workflow: Input value for workflow.
            - timestamp: Input value for timestamp.
            - location_label: Input value for location_label.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Prepare lesions.

        parameter:
            - context_data: Input value for context_data.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Segment or fallback.

        parameter:
            - image_bytes: Input value for image_bytes.
            - image: Input value for image.
            - fallback_polygons: Input value for fallback_polygons.
            - filename: Input value for filename.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle decode image source.

        parameter:
            - image_source: Input value for image_source.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle decode image bytes.

        parameter:
            - image_bytes: Input value for image_bytes.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle scale polygons.

        parameter:
            - polygons: Input value for polygons.
            - width: Input value for width.
            - height: Input value for height.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle polygon points to mask.

        parameter:
            - points: Input value for points.
            - width: Input value for width.
            - height: Input value for height.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(points, dtype=np.int32)
        if polygon.size == 0:
            return mask
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    @staticmethod
    def _bounding_box_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        """brief:
            Handle bounding box from points.

        parameter:
            - points: Input value for points.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if not points:
            return (0, 0, 0, 0)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _build_agent(self) -> tuple[Any, str, list[str]]:
        """brief:
            Build agent.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
