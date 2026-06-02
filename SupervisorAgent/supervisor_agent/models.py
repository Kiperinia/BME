from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Issue:
    type: str
    severity: str
    message: str
    location: Optional[str] = None
    evidence_refs: Optional[List[str]] = None


@dataclass
class Report:
    report_id: str = ""
    patient_id: str = ""
    study_id: str = ""
    exam_date: str = ""
    findings: str = ""
    conclusion: str = ""
    layout_suggestion: str = ""
    report_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    structured_findings: Dict[str, Any] = field(default_factory=dict)
    lesion_summary: Dict[str, Any] = field(default_factory=dict)
    risk_summary: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""
    model_version: str = ""
    react_analysis: Dict[str, Any] = field(default_factory=dict)
    react_refinement: Dict[str, Any] = field(default_factory=dict)
    report_score: Dict[str, Any] = field(default_factory=dict)

    def full_text(self) -> str:
        if self.report_text:
            return self.report_text
        parts = [self.findings.strip(), self.conclusion.strip()]
        return "\n".join(part for part in parts if part)


@dataclass
class ReportToolCallLog:
    tool_name: str
    status: str
    duration_ms: float
    input_payload: Dict[str, Any] = field(default_factory=dict)
    output_preview: str = ""
    error_message: str = ""


@dataclass
class ReportWorkflowLesion:
    lesion_id: str
    source_label: str = ""
    label: str = ""
    confidence: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None
    paris_type: str = ""
    invasion_risk: str = ""
    risk_level: str = ""
    total_score: float = 0.0
    disposition: str = ""
    estimated_size_mm: float = 0.0
    shape_description: str = ""
    used_llm: bool = False


@dataclass
class ReportWorkflowSummary:
    agent_name: str = ""
    description: str = ""
    pipeline: str = ""
    llm_configured: bool = False
    workflow_mode: str = ""
    generated_at: str = ""
    lesion_count: int = 0
    highest_risk_lesion_id: Optional[str] = None
    model_version: str = ""
    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    lesions: List[ReportWorkflowLesion] = field(default_factory=list)


@dataclass
class ReportDraft:
    findings: str
    conclusion: str
    layout_suggestion: str
    workflow: Optional[ReportWorkflowSummary] = None
    stream_messages: List[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    source: str
    citation: str
    facts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    tool_name: str
    inputs_hash: str
    outputs_hash: str
    duration_ms: int


@dataclass
class AuditRecord:
    audit_id: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    created_at: Optional[str] = None


@dataclass
class Decision:
    report_id: str
    status: DecisionStatus
    risk_level: RiskLevel
    issues: List[Issue]
    audit_id: str
    rationale: Optional[str] = None
    routing: List[str] = field(default_factory=list)
    hard_case: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    report_id: str
    labels: List[str]
    reviewer_notes: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None
