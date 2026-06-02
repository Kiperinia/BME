from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .context import SupervisorContext
from .models import Decision, Issue


class State(str, Enum):
    INIT = "init"
    INGEST = "ingest"
    QUALITY_CHECK = "quality_check"
    CLINICAL_CONSISTENCY = "clinical_consistency"
    HALLUCINATION_CHECK = "hallucination_check"
    RISK_GATING = "risk_gating"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"
    FEEDBACK = "feedback"


@dataclass
class StateResult:
    next_state: State
    issues: List[Issue]
    decision: Optional[Decision] = None


class TransitionPolicy:
    def next_state(self, current: State, context: SupervisorContext) -> State:
        raise NotImplementedError


class StateMachineEngine:
    def __init__(self, transition_policy: TransitionPolicy) -> None:
        self._transition_policy = transition_policy

    def run(self, context: SupervisorContext) -> Decision:
        raise NotImplementedError
