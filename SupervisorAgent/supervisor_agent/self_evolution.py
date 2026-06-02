from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from .config import PolicyConfig
from .feedback_memory import FeedbackRecord


@dataclass
class PolicyUpdate:
    field: str
    suggested_value: object
    reason: str


@dataclass
class EvolutionPlan:
    policy_updates: List[PolicyUpdate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    hard_case_samples: List[str] = field(default_factory=list)


class SelfEvolutionEngine:
    def propose(self, records: Iterable[FeedbackRecord], policy: PolicyConfig) -> EvolutionPlan:
        records_list = list(records)
        plan = EvolutionPlan()
        if not records_list:
            plan.notes.append("no_feedback_records")
            return plan

        issue_counts: dict[str, int] = {}
        hard_case_ids: List[str] = []
        for record in records_list:
            if record.hard_case:
                hard_case_ids.append(record.report_id)
            for issue in record.issues:
                issue_counts[issue.type] = issue_counts.get(issue.type, 0) + 1

        hard_case_rate = len(hard_case_ids) / max(len(records_list), 1)
        if hard_case_rate > 0.25:
            plan.notes.append("hard_case_rate_high")
            suggested = min(policy.min_quality_score + 0.05, 0.95)
            plan.policy_updates.append(
                PolicyUpdate("min_quality_score", suggested, "hard_case_rate_high")
            )

        if issue_counts.get("quality.min_report_length", 0) >= 5:
            suggested = max(policy.min_report_chars, 220)
            plan.policy_updates.append(
                PolicyUpdate("min_report_chars", suggested, "frequent_short_reports")
            )

        if issue_counts.get("validation.missing_findings", 0) >= 3:
            plan.policy_updates.append(
                PolicyUpdate("required_sections", ["findings", "conclusion"], "missing_findings")
            )

        if issue_counts.get("hallucination.evidence_required", 0) >= 3:
            plan.policy_updates.append(
                PolicyUpdate("require_evidence", True, "missing_evidence")
            )

        plan.hard_case_samples = hard_case_ids[:10]
        if not plan.policy_updates:
            plan.notes.append("no_policy_updates_suggested")
        return plan
