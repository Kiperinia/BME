from __future__ import annotations

from typing import Iterable, List, Optional

from .config import PolicyConfig
from .context import SupervisorContext
from .hard_case_routing import HardCaseDecision, HardCaseRouter
from .models import Decision, DecisionStatus, Issue, RiskLevel
from .orchestrator import SupervisorAgent
from .quality import QualityScorer, StyleAndCompletenessChecker
from .clinical import ClinicalConsistencyChecker
from .hallucination import HallucinationDetector
from .risk import RiskAssessor, SafetyPolicyGate
from .rules import Rule, RuleEngine, RuleResult
from .state_machine import State, StateMachineEngine, TransitionPolicy


class MinReportLengthRule:
    rule_id = "quality.min_report_length"

    def __init__(self, min_chars: int) -> None:
        self._min_chars = min_chars

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if self._min_chars <= 0:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        text = context.report.full_text()
        length = len(text.strip())
        passed = length >= self._min_chars
        message = f"Report length below minimum {self._min_chars}." if not passed else "ok"
        severity = "warn" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class FindingsLengthRule:
    rule_id = "quality.min_findings_length"

    def __init__(self, min_chars: int) -> None:
        self._min_chars = min_chars

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if self._min_chars <= 0:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        length = len((context.report.findings or "").strip())
        passed = length >= self._min_chars
        message = f"Findings length below minimum {self._min_chars}." if not passed else "ok"
        severity = "warn" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class ConclusionLengthRule:
    rule_id = "quality.min_conclusion_length"

    def __init__(self, min_chars: int) -> None:
        self._min_chars = min_chars

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if self._min_chars <= 0:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        length = len((context.report.conclusion or "").strip())
        passed = length >= self._min_chars
        message = f"Conclusion length below minimum {self._min_chars}." if not passed else "ok"
        severity = "warn" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class RequiredSectionsRule:
    rule_id = "quality.required_sections"

    def __init__(self, required_sections: Iterable[str]) -> None:
        self._required_sections = [section for section in required_sections if section]

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if not self._required_sections:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        sections = context.report.sections or {}
        if not sections:
            sections = {
                "findings": context.report.findings,
                "conclusion": context.report.conclusion,
                "layoutSuggestion": context.report.layout_suggestion,
                "layout_suggestion": context.report.layout_suggestion,
            }
        missing = [name for name in self._required_sections if not sections.get(name)]
        passed = not missing
        message = "Missing required sections: " + ", ".join(missing) if not passed else "ok"
        severity = "error" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class LayoutSuggestionRequiredRule:
    rule_id = "quality.layout_suggestion_required"

    def __init__(self, required: bool) -> None:
        self._required = required

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if not self._required:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        passed = bool((context.report.layout_suggestion or "").strip())
        message = "Layout suggestion required but missing." if not passed else "ok"
        severity = "warn" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class ToolLogsRequiredRule:
    rule_id = "quality.tool_logs_required"

    def __init__(self, required: bool) -> None:
        self._required = required

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if not self._required:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        passed = bool(context.tool_logs)
        message = "Tool logs required but missing." if not passed else "ok"
        severity = "warn" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class RequiredPatientFieldsRule:
    rule_id = "clinical.required_patient_fields"

    def __init__(self, required_fields: Iterable[str]) -> None:
        self._required_fields = [field for field in required_fields if field]

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if not self._required_fields:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        patient_context = context.patient_context or {}
        missing = [field for field in self._required_fields if not patient_context.get(field)]
        passed = not missing
        message = "Missing patient fields: " + ", ".join(missing) if not passed else "ok"
        severity = "error" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class EvidenceRequiredRule:
    rule_id = "hallucination.evidence_required"

    def __init__(self, required: bool) -> None:
        self._required = required

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if not self._required:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        passed = bool(context.evidence)
        message = "Evidence list is empty but required." if not passed else "ok"
        severity = "error" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


class FindingsEvidenceRule:
    rule_id = "hallucination.findings_evidence"

    def __init__(self, required: bool) -> None:
        self._required = required

    def evaluate(self, context: SupervisorContext) -> RuleResult:
        if not self._required:
            return RuleResult(self.rule_id, True, "info", "not_applicable")
        has_findings = bool(context.report.structured_findings)
        has_evidence = bool(context.evidence)
        passed = not (has_findings and not has_evidence)
        message = "Structured findings present without evidence." if not passed else "ok"
        severity = "error" if not passed else "info"
        return RuleResult(self.rule_id, passed, severity, message)


def _quality_rules(policy: PolicyConfig) -> List[Rule]:
    return [
        MinReportLengthRule(policy.min_report_chars),
        FindingsLengthRule(policy.min_findings_chars),
        ConclusionLengthRule(policy.min_conclusion_chars),
        LayoutSuggestionRequiredRule(policy.require_layout_suggestion),
        RequiredSectionsRule(policy.required_sections),
        ToolLogsRequiredRule(policy.require_tool_logs),
    ]


def _clinical_rules(policy: PolicyConfig) -> List[Rule]:
    return [
        RequiredPatientFieldsRule(policy.required_patient_fields),
    ]


def _hallucination_rules(policy: PolicyConfig) -> List[Rule]:
    return [
        EvidenceRequiredRule(policy.require_evidence),
        FindingsEvidenceRule(policy.require_findings_evidence),
    ]


class RuleBasedQualityScorer(QualityScorer):
    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules = list(rules) if rules is not None else None

    def score(self, context: SupervisorContext) -> float:
        rules = self._rules or _quality_rules(context.policy)
        engine = RuleEngine(rules)
        return RuleEngine.score(engine.evaluate(context))


class RuleBasedStyleAndCompletenessChecker(StyleAndCompletenessChecker):
    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules = list(rules) if rules is not None else None

    def check(self, context: SupervisorContext) -> List[Issue]:
        rules = self._rules or _quality_rules(context.policy)
        engine = RuleEngine(rules)
        return RuleEngine.to_issues(engine.evaluate(context))


class RuleBasedClinicalConsistencyChecker(ClinicalConsistencyChecker):
    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules = list(rules) if rules is not None else None

    def validate(self, context: SupervisorContext) -> List[Issue]:
        rules = self._rules or _clinical_rules(context.policy)
        engine = RuleEngine(rules)
        return RuleEngine.to_issues(engine.evaluate(context))


class RuleBasedHallucinationDetector(HallucinationDetector):
    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules = list(rules) if rules is not None else None

    def detect(self, context: SupervisorContext) -> List[Issue]:
        rules = self._rules or _hallucination_rules(context.policy)
        engine = RuleEngine(rules)
        return RuleEngine.to_issues(engine.evaluate(context))


class RuleBasedRiskAssessor(RiskAssessor):
    def assess(self, context: SupervisorContext) -> RiskLevel:
        text = context.report.full_text()
        high_hits = _keyword_hits(text, context.policy.high_risk_keywords)
        risk_hits = _keyword_hits(text, context.policy.risk_keywords)
        if high_hits >= 2:
            return RiskLevel.CRITICAL
        score = 0.0
        if high_hits > 0:
            score = 0.9
        elif risk_hits > 0:
            score = 0.6
        thresholds = context.policy.risk_thresholds
        if score >= thresholds.high:
            return RiskLevel.HIGH
        if score >= thresholds.medium:
            return RiskLevel.MEDIUM
        if score >= thresholds.low:
            return RiskLevel.LOW
        return RiskLevel.LOW


class RuleBasedSafetyPolicyGate(SafetyPolicyGate):
    def allow(self, context: SupervisorContext, risk_level: RiskLevel) -> bool:
        return risk_level == RiskLevel.LOW


class SequentialTransitionPolicy(TransitionPolicy):
    _order = [
        State.INIT,
        State.INGEST,
        State.QUALITY_CHECK,
        State.CLINICAL_CONSISTENCY,
        State.HALLUCINATION_CHECK,
        State.RISK_GATING,
    ]

    def next_state(self, current: State, context: SupervisorContext) -> State:
        if current not in self._order:
            return State.FAILED
        index = self._order.index(current)
        if index >= len(self._order) - 1:
            return State.RISK_GATING
        return self._order[index + 1]


class RuleBasedStateMachineEngine(StateMachineEngine):
    def __init__(
        self,
        quality_scorer: QualityScorer,
        style_checker: StyleAndCompletenessChecker,
        clinical_checker: ClinicalConsistencyChecker,
        hallucination_detector: HallucinationDetector,
        risk_assessor: RiskAssessor,
        safety_gate: SafetyPolicyGate,
        hard_case_router: Optional[HardCaseRouter] = None,
        transition_policy: Optional[TransitionPolicy] = None,
    ) -> None:
        super().__init__(transition_policy or SequentialTransitionPolicy())
        self._quality_scorer = quality_scorer
        self._style_checker = style_checker
        self._clinical_checker = clinical_checker
        self._hallucination_detector = hallucination_detector
        self._risk_assessor = risk_assessor
        self._safety_gate = safety_gate
        self._hard_case_router = hard_case_router

    def run(self, context: SupervisorContext) -> Decision:
        issues: List[Issue] = []
        state = State.INIT

        while True:
            _record_state(context, state)

            if state == State.INIT:
                state = self._transition_policy.next_state(state, context)
                continue

            if state == State.INGEST:
                ingest_issues = _validate_context(context)
                issues.extend(ingest_issues)
                if ingest_issues:
                    return _decision(context, DecisionStatus.REJECTED, issues)
                state = self._transition_policy.next_state(state, context)
                continue

            if state == State.QUALITY_CHECK:
                quality_issues = self._style_checker.check(context)
                issues.extend(quality_issues)
                score = self._quality_scorer.score(context)
                context.metadata["quality_score"] = score
                if score < context.policy.borderline_quality_score:
                    return _decision(context, DecisionStatus.REJECTED, issues)
                if score < context.policy.min_quality_score:
                    if context.policy.allow_borderline_to_review:
                        return _decision(context, DecisionStatus.HUMAN_REVIEW, issues)
                    return _decision(context, DecisionStatus.REJECTED, issues)
                state = self._transition_policy.next_state(state, context)
                continue

            if state == State.CLINICAL_CONSISTENCY:
                clinical_issues = self._clinical_checker.validate(context)
                issues.extend(clinical_issues)
                if _has_severity(clinical_issues, "critical"):
                    return _decision(context, DecisionStatus.REJECTED, issues)
                if _has_severity(clinical_issues, "error"):
                    return _decision(context, DecisionStatus.HUMAN_REVIEW, issues)
                state = self._transition_policy.next_state(state, context)
                continue

            if state == State.HALLUCINATION_CHECK:
                hallucination_issues = self._hallucination_detector.detect(context)
                issues.extend(hallucination_issues)
                if len(hallucination_issues) > context.policy.max_unsupported_claims:
                    return _decision(context, DecisionStatus.HUMAN_REVIEW, issues)
                state = self._transition_policy.next_state(state, context)
                continue

            if state == State.RISK_GATING:
                risk_level = self._risk_assessor.assess(context)
                context.metadata["risk_level"] = risk_level.value
                hard_case = self._route_hard_case(context, issues, risk_level)
                if hard_case is not None:
                    return _decision(
                        context,
                        DecisionStatus.HUMAN_REVIEW,
                        issues,
                        risk_level,
                        hard_case=hard_case,
                    )
                if not self._safety_gate.allow(context, risk_level):
                    return _decision(context, DecisionStatus.HUMAN_REVIEW, issues, risk_level)
                return _decision(context, DecisionStatus.APPROVED, issues, risk_level)

            return _decision(context, DecisionStatus.FAILED, issues)

    def _route_hard_case(
        self,
        context: SupervisorContext,
        issues: List[Issue],
        risk_level: RiskLevel,
    ) -> Optional[HardCaseDecision]:
        if self._hard_case_router is None:
            return None
        quality_score = context.metadata.get("quality_score")
        decision = self._hard_case_router.route(
            context=context,
            issues=issues,
            risk_level=risk_level,
            quality_score=quality_score if isinstance(quality_score, (int, float)) else None,
        )
        return decision if decision.should_route else None


def build_default_agent(policy: Optional[PolicyConfig] = None) -> SupervisorAgent:
    engine = RuleBasedStateMachineEngine(
        quality_scorer=RuleBasedQualityScorer(),
        style_checker=RuleBasedStyleAndCompletenessChecker(),
        clinical_checker=RuleBasedClinicalConsistencyChecker(),
        hallucination_detector=RuleBasedHallucinationDetector(),
        risk_assessor=RuleBasedRiskAssessor(),
        safety_gate=RuleBasedSafetyPolicyGate(),
        hard_case_router=HardCaseRouter(),
    )
    return SupervisorAgent(engine, policy)


def _decision(
    context: SupervisorContext,
    status: DecisionStatus,
    issues: List[Issue],
    risk_level: Optional[RiskLevel] = None,
    hard_case: Optional[HardCaseDecision] = None,
) -> Decision:
    resolved_risk = risk_level or _risk_from_issues(
        issues,
        RiskLevel.LOW if status == DecisionStatus.APPROVED else RiskLevel.MEDIUM,
    )
    audit_id = context.metadata.get("audit_id", "audit-pending")
    metadata = dict(context.metadata)
    routing: List[str] = []
    hard_case_flag = False
    if hard_case is not None:
        routing = [hard_case.destination]
        hard_case_flag = True
        metadata["hard_case_reasons"] = hard_case.reasons
        if hard_case.tags:
            metadata["hard_case_tags"] = hard_case.tags
    return Decision(
        report_id=context.report.report_id,
        status=status,
        risk_level=resolved_risk,
        issues=issues,
        audit_id=audit_id,
        routing=routing,
        hard_case=hard_case_flag,
        metadata=metadata,
    )


def _record_state(context: SupervisorContext, state: State) -> None:
    history = context.metadata.setdefault("state_history", [])
    history.append(state.value)


def _validate_context(context: SupervisorContext) -> List[Issue]:
    issues: List[Issue] = []
    if context.report is None:
        issues.append(
            Issue(
                type="validation.missing_report",
                severity="critical",
                message="Report is missing.",
            )
        )
        return issues
    if not context.report.report_id:
        issues.append(
            Issue(
                type="validation.missing_report_id",
                severity="error",
                message="Report ID is missing.",
            )
        )
    required_sections = set(context.policy.required_sections or [])
    if "findings" in required_sections and not context.report.findings:
        issues.append(
            Issue(
                type="validation.missing_findings",
                severity="error",
                message="Report findings are missing.",
            )
        )
    if "conclusion" in required_sections and not context.report.conclusion:
        issues.append(
            Issue(
                type="validation.missing_conclusion",
                severity="error",
                message="Report conclusion is missing.",
            )
        )
    return issues


def _risk_from_issues(issues: List[Issue], default: RiskLevel) -> RiskLevel:
    severities = {issue.severity for issue in issues}
    if "critical" in severities:
        return RiskLevel.CRITICAL
    if "error" in severities:
        return RiskLevel.HIGH
    if "warn" in severities:
        return RiskLevel.MEDIUM
    return default


def _has_severity(issues: List[Issue], severity: str) -> bool:
    return any(issue.severity == severity for issue in issues)


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    hits = 0
    for keyword in keywords:
        if not keyword:
            continue
        if keyword.lower() in lowered:
            hits += 1
    return hits
