from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, TYPE_CHECKING

from .models import Issue

if TYPE_CHECKING:
    from .context import SupervisorContext


@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    severity: str
    message: str
    location: Optional[str] = None
    evidence_refs: Optional[List[str]] = None


class Rule(Protocol):
    rule_id: str

    def evaluate(self, context: "SupervisorContext") -> RuleResult:
        ...


class RuleEngine:
    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules = list(rules)

    def evaluate(self, context: "SupervisorContext") -> List[RuleResult]:
        return [rule.evaluate(context) for rule in self._rules]

    @staticmethod
    def to_issues(results: Iterable[RuleResult]) -> List[Issue]:
        issues: List[Issue] = []
        for result in results:
            if result.passed:
                continue
            issues.append(
                Issue(
                    type=result.rule_id,
                    severity=result.severity,
                    message=result.message,
                    location=result.location,
                    evidence_refs=result.evidence_refs,
                )
            )
        return issues

    @staticmethod
    def score(results: Iterable[RuleResult]) -> float:
        results_list = list(results)
        if not results_list:
            return 1.0
        failed = sum(1 for result in results_list if not result.passed)
        return max(0.0, 1.0 - failed / len(results_list))
