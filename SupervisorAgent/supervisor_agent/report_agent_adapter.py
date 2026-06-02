from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .context import SupervisorContext
from .models import (
    Report,
    ReportToolCallLog,
    ReportWorkflowLesion,
    ReportWorkflowSummary,
)


def build_context_from_report_agent(
    report_payload: Any,
    *,
    workflow_payload: Optional[Any] = None,
    tool_logs: Optional[Iterable[dict[str, Any]]] = None,
    patient_context: Optional[dict[str, Any]] = None,
    report_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    stream_messages: Optional[Iterable[str]] = None,
) -> SupervisorContext:
    report = report_from_agent_payload(report_payload, report_id=report_id)
    resolved_tool_logs = tool_logs or report.tool_calls
    context = SupervisorContext(
        report=report,
        tool_logs=_tool_logs_from_payload(resolved_tool_logs),
        patient_context=patient_context or {},
        correlation_id=correlation_id,
    )
    if workflow_payload is not None:
        context.report_workflow = workflow_from_agent_payload(workflow_payload)
    if stream_messages is not None:
        context.stream_messages = list(stream_messages)
    return context


def report_from_agent_payload(payload: Any, report_id: Optional[str] = None) -> Report:
    if isinstance(payload, Report):
        return payload
    data = _payload_to_dict(payload)
    findings = str(data.get("findings", ""))
    conclusion = str(data.get("conclusion", ""))
    layout_suggestion = str(data.get("layoutSuggestion") or data.get("layout_suggestion") or "")
    resolved_report_id = report_id or str(data.get("report_id") or data.get("study_id") or "")
    report = Report(
        report_id=resolved_report_id,
        patient_id=str(data.get("patient_id", "")),
        study_id=str(data.get("study_id", "")),
        exam_date=str(data.get("exam_date", "")),
        findings=findings,
        conclusion=conclusion,
        layout_suggestion=layout_suggestion,
        report_text=str(data.get("report_text", "")),
        sections=data.get("sections") or {},
        structured_findings=data.get("structured_findings") or {},
        lesion_summary=data.get("lesion_summary") or {},
        risk_summary=data.get("risk_summary") or {},
        tool_calls=data.get("tool_calls") or [],
        generated_at=str(data.get("generated_at", "")),
        model_version=str(data.get("model_version", "")),
        react_analysis=data.get("react_analysis") or {},
        react_refinement=data.get("react_refinement") or {},
        report_score=data.get("report_score") or {},
    )
    if not report.sections:
        report.sections = {
            "findings": findings,
            "conclusion": conclusion,
            "layoutSuggestion": layout_suggestion,
        }
    return report


def workflow_from_agent_payload(payload: Any) -> ReportWorkflowSummary:
    if isinstance(payload, ReportWorkflowSummary):
        return payload
    data = _payload_to_dict(payload)
    lesions_payload = data.get("lesions", []) or []
    lesions = [_workflow_lesion_from_payload(item) for item in lesions_payload]
    return ReportWorkflowSummary(
        agent_name=str(data.get("agentName", "")),
        description=str(data.get("description", "")),
        pipeline=str(data.get("pipeline", "")),
        llm_configured=bool(data.get("llmConfigured", False)),
        workflow_mode=str(data.get("workflowMode", "")),
        generated_at=str(data.get("generatedAt", "")),
        lesion_count=int(data.get("lesionCount", 0) or 0),
        highest_risk_lesion_id=data.get("highestRiskLesionId"),
        model_version=str(data.get("modelVersion", "")),
        steps=[str(step) for step in data.get("steps", []) or []],
        warnings=[str(step) for step in data.get("warnings", []) or []],
        lesions=lesions,
    )


def _workflow_lesion_from_payload(payload: Any) -> ReportWorkflowLesion:
    data = _payload_to_dict(payload)
    bbox_payload = data.get("bbox")
    bbox = tuple(bbox_payload) if isinstance(bbox_payload, (list, tuple)) else None
    return ReportWorkflowLesion(
        lesion_id=str(data.get("lesionId", "")),
        source_label=str(data.get("sourceLabel", "")),
        label=str(data.get("label", "")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        bbox=bbox,  # type: ignore[arg-type]
        paris_type=str(data.get("parisType", "")),
        invasion_risk=str(data.get("invasionRisk", "")),
        risk_level=str(data.get("riskLevel", "")),
        total_score=float(data.get("totalScore", 0.0) or 0.0),
        disposition=str(data.get("disposition", "")),
        estimated_size_mm=float(data.get("estimatedSizeMm", 0.0) or 0.0),
        shape_description=str(data.get("shapeDescription", "")),
        used_llm=bool(data.get("usedLlm", False)),
    )


def _tool_logs_from_payload(payload: Iterable[dict[str, Any]]) -> List[ReportToolCallLog]:
    logs: List[ReportToolCallLog] = []
    for item in payload:
        if isinstance(item, ReportToolCallLog):
            logs.append(item)
            continue
        data = _payload_to_dict(item)
        logs.append(
            ReportToolCallLog(
                tool_name=str(data.get("tool_name") or data.get("toolName") or ""),
                status=str(data.get("status", "")),
                duration_ms=float(data.get("duration_ms") or data.get("durationMs") or 0.0),
                input_payload=dict(data.get("input_payload") or data.get("inputPayload") or {}),
                output_preview=str(data.get("output_preview") or data.get("outputPreview") or ""),
                error_message=str(data.get("error_message") or data.get("errorMessage") or ""),
            )
        )
    return logs


def _payload_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    return {}
