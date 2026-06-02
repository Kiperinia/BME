from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import PolicyConfig
from .models import EvidenceItem, Report, ReportToolCallLog, ReportWorkflowSummary


@dataclass
class SupervisorContext:
    report: Report
    evidence: List[EvidenceItem] = field(default_factory=list)
    tool_logs: List[ReportToolCallLog] = field(default_factory=list)
    report_workflow: Optional[ReportWorkflowSummary] = None
    stream_messages: List[str] = field(default_factory=list)
    patient_context: Dict[str, Any] = field(default_factory=dict)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
