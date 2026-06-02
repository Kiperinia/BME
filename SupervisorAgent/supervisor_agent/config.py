from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskThresholds:
    low: float = 0.3
    medium: float = 0.6
    high: float = 0.85


@dataclass
class PolicyConfig:
    min_quality_score: float = 0.8
    borderline_quality_score: float = 0.7
    max_unsupported_claims: int = 0
    allow_borderline_to_review: bool = True
    min_report_chars: int = 200
    min_findings_chars: int = 0
    min_conclusion_chars: int = 0
    require_layout_suggestion: bool = False
    required_sections: List[str] = field(default_factory=lambda: ["findings", "conclusion"])
    required_patient_fields: List[str] = field(default_factory=list)
    require_evidence: bool = True
    require_findings_evidence: bool = True
    require_tool_logs: bool = False
    risk_keywords: List[str] = field(default_factory=list)
    high_risk_keywords: List[str] = field(default_factory=list)
    hard_case_risk_levels: List[str] = field(default_factory=lambda: ["high", "critical"])
    hard_case_issue_severities: List[str] = field(default_factory=lambda: ["critical"])
    hard_case_keywords: List[str] = field(default_factory=list)
    hard_case_min_quality_score: float = 0.75
    hard_case_tool_error_ratio: float = 0.1
    hard_case_destination: str = "human_review"
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    guidelines_version: str = "latest"
    tool_timeout_ms: int = 10_000
    max_tool_retries: int = 2
