from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .context import SupervisorContext
from .models import Issue, RiskLevel


@dataclass
class HardCaseDecision:
    should_route: bool
    reasons: List[str]
    destination: str = "human_review"
    tags: List[str] = field(default_factory=list)


class HardCaseRouter:
    def route(
        self,
        context: SupervisorContext,
        issues: List[Issue],
        risk_level: RiskLevel,
        quality_score: Optional[float] = None,
    ) -> HardCaseDecision:
        policy = context.policy
        reasons: List[str] = []
        tags: List[str] = []

        if any(issue.severity in policy.hard_case_issue_severities for issue in issues):
            reasons.append("issue_severity")
            tags.append("severity")

        if risk_level.value in policy.hard_case_risk_levels:
            reasons.append("risk_level")
            tags.append("risk")

        text = context.report.full_text().lower()
        if policy.hard_case_keywords and any(keyword.lower() in text for keyword in policy.hard_case_keywords):
            reasons.append("keyword_hit")
            tags.append("keyword")

        if quality_score is not None and quality_score < policy.hard_case_min_quality_score:
            reasons.append("low_quality_score")
            tags.append("quality")

        if context.tool_logs:
            error_count = 0
            for log in context.tool_logs:
                status = getattr(log, "status", None)
                if status is None and isinstance(log, dict):
                    status = log.get("status")
                if status != "ok":
                    error_count += 1
            error_ratio = error_count / max(len(context.tool_logs), 1)
            if error_ratio >= policy.hard_case_tool_error_ratio:
                reasons.append("tool_error_ratio")
                tags.append("tool_error")

        destination = policy.hard_case_destination
        return HardCaseDecision(bool(reasons), reasons, destination=destination, tags=tags)
