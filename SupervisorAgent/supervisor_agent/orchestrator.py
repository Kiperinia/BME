from __future__ import annotations

from typing import Optional

from .config import PolicyConfig
from .context import SupervisorContext
from .feedback_memory import FeedbackMemory, FeedbackRecord
from .models import Decision, Feedback
from .self_evolution import EvolutionPlan, SelfEvolutionEngine
from .state_machine import StateMachineEngine


class SupervisorAgent:
    def __init__(
        self,
        state_machine: StateMachineEngine,
        policy: Optional[PolicyConfig] = None,
        feedback_memory: Optional[FeedbackMemory] = None,
        self_evolution: Optional[SelfEvolutionEngine] = None,
    ) -> None:
        self._state_machine = state_machine
        self._policy = policy or PolicyConfig()
        self._feedback_memory = feedback_memory
        self._self_evolution = self_evolution

    def evaluate(self, context: SupervisorContext, feedback: Optional[Feedback] = None) -> Decision:
        context.policy = self._policy
        decision = self._state_machine.run(context)
        if self._feedback_memory is not None:
            record = FeedbackRecord.from_decision(
                decision,
                feedback=feedback,
                metadata=context.metadata,
            )
            self._feedback_memory.record(record)
        return decision

    def propose_self_evolution(self, limit: int = 200) -> Optional[EvolutionPlan]:
        if self._self_evolution is None or self._feedback_memory is None:
            return None
        records = self._feedback_memory.list_recent(limit=limit)
        return self._self_evolution.propose(records, self._policy)
