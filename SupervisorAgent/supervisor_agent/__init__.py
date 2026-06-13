"""SupervisorAgent package stubs."""

from .config import PolicyConfig, RiskThresholds
from .context import SupervisorContext
from .feedback_memory import FeedbackMemory, FeedbackRecord, InMemoryFeedbackMemory, JsonlFeedbackMemory
from .hard_case_routing import HardCaseDecision, HardCaseRouter
from .models import (
    Decision,
    DecisionStatus,
    EvidenceItem,
    Feedback,
    Issue,
    Report,
    ReportDraft,
    ReportToolCallLog,
    ReportWorkflowLesion,
    ReportWorkflowSummary,
    RiskLevel,
)
from .orchestrator import SupervisorAgent
from .report_agent_adapter import build_context_from_report_agent
from .self_evolution import EvolutionPlan, PolicyUpdate, SelfEvolutionEngine
from .rule_based import (
    RuleBasedClinicalConsistencyChecker,
    RuleBasedHallucinationDetector,
    RuleBasedQualityScorer,
    RuleBasedRiskAssessor,
    RuleBasedSafetyPolicyGate,
    RuleBasedStateMachineEngine,
    RuleBasedStyleAndCompletenessChecker,
    build_default_agent,
)
from .rules import Rule, RuleEngine, RuleResult

__all__ = [
    "Decision",
    "DecisionStatus",
    "EvidenceItem",
    "EvolutionPlan",
    "Feedback",
    "FeedbackMemory",
    "FeedbackRecord",
    "HardCaseDecision",
    "HardCaseRouter",
    "InMemoryFeedbackMemory",
    "Issue",
    "JsonlFeedbackMemory",
    "PolicyConfig",
    "PolicyUpdate",
    "Report",
    "ReportDraft",
    "ReportToolCallLog",
    "ReportWorkflowLesion",
    "ReportWorkflowSummary",
    "RiskLevel",
    "RiskThresholds",
    "Rule",
    "RuleEngine",
    "RuleResult",
    "RuleBasedClinicalConsistencyChecker",
    "RuleBasedHallucinationDetector",
    "RuleBasedQualityScorer",
    "RuleBasedRiskAssessor",
    "RuleBasedSafetyPolicyGate",
    "RuleBasedStateMachineEngine",
    "RuleBasedStyleAndCompletenessChecker",
    "SelfEvolutionEngine",
    "SupervisorAgent",
    "SupervisorContext",
    "build_context_from_report_agent",
    "build_default_agent",
]
