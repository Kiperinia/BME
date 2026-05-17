from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from hello_agents import HelloAgentsLLM
from hello_agents.core.agent import Agent as HelloAgent

from core.llm import RuleOnlyLLM
from tools.medical.exemplar_bank_memory import ExemplarMemoryManager
from tools.medical.exemplar_bank_quality import (
    PrototypeEvolutionDecision,
    PrototypeEvolutionManager,
    PrototypeQualityController,
)
from tools.medical.exemplar_bank_retrieval import ExemplarRetrievalPipeline, RetrievedFeatureSet
from tools.medical.exemplar_bank_schemas import (
    ExemplarLifecycleState,
    ExemplarPolarity,
    MemoryBankSnapshot,
    MedicalExemplarRecord,
    QueryFeatureBatch,
    RetrievalPackage,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExemplarAgentTrace:
    step: str
    status: str
    detail: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class ExemplarAgentResult:
    bank_id: str
    traces: list[ExemplarAgentTrace] = field(default_factory=list)
    retrieval: RetrievalPackage | None = None
    stored_record: MedicalExemplarRecord | None = None
    evolution: PrototypeEvolutionDecision | None = None


class ExemplarBankAgent(HelloAgent):
    def __init__(
        self,
        llm: HelloAgentsLLM | None = None,
        *,
        memory_root: str | Path = "agent/memory/exemplar_agent_bank",
        hidden_dim: int = 256,
    ) -> None:
        resolved_llm = llm or RuleOnlyLLM(model="exemplar-bank-rule-only", provider="rule-only")
        super().__init__(
            name="exemplar-bank-agent",
            llm=resolved_llm,
            system_prompt="You manage medical exemplar memory for retrieval-enhanced SAM3 segmentation.",
        )
        self.memory = ExemplarMemoryManager(memory_root)
        self.quality = PrototypeQualityController()
        self.evolution = PrototypeEvolutionManager(self.quality)
        self.retrieval_pipeline = ExemplarRetrievalPipeline(dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def ingest(self, record: MedicalExemplarRecord, bank_id: str = "default-bank") -> ExemplarAgentResult:
        snapshot = self.memory.load(bank_id=bank_id)
        traces = [ExemplarAgentTrace("ingest", "started", f"ingesting exemplar {record.exemplar_id}")]
        decision = self.quality.decide(record)
        record.state = decision.next_state
        if record.state == ExemplarLifecycleState.ARCHIVED:
            traces.append(ExemplarAgentTrace("quality_gate", "rejected", ",".join(decision.reasons)))
            return ExemplarAgentResult(bank_id=bank_id, traces=traces, stored_record=record, evolution=decision)

        all_records = [*snapshot.positive_items, *snapshot.negative_items, *snapshot.boundary_items, record]
        positive, negative, boundary = self.memory.partition(self.memory.deduplicate(all_records))
        snapshot.positive_items = positive
        snapshot.negative_items = negative
        snapshot.boundary_items = boundary
        snapshot.stats = self._build_stats(snapshot)
        self.memory.save(snapshot)
        self.memory.append_audit_log(
            {
                "bank_id": bank_id,
                "event": "ingest",
                "exemplar_id": record.exemplar_id,
                "state": record.state.value,
                "quality": decision.quality.overall,
            }
        )
        traces.append(ExemplarAgentTrace("ingest", "completed", f"stored {record.exemplar_id} as {record.state.value}"))
        return ExemplarAgentResult(bank_id=bank_id, traces=traces, stored_record=record, evolution=decision)

    def retrieve_prior(
        self,
        query: QueryFeatureBatch,
        retrieved: RetrievedFeatureSet,
        bank_id: str = "default-bank",
    ) -> ExemplarAgentResult:
        traces = [ExemplarAgentTrace("retrieve_prior", "started", f"query={query.query_id}")]
        package = self.retrieval_pipeline(query, retrieved)
        traces.append(
            ExemplarAgentTrace(
                "retrieve_prior",
                "completed",
                f"confidence={float(package.confidence.mean().item()):.4f}, uncertainty={float(package.uncertainty.mean().item()):.4f}",
            )
        )
        return ExemplarAgentResult(bank_id=bank_id, traces=traces, retrieval=package)

    def update_with_feedback(
        self,
        exemplar_id: str,
        feedback: dict[str, Any],
        bank_id: str = "default-bank",
    ) -> ExemplarAgentResult:
        snapshot = self.memory.load(bank_id=bank_id)
        traces = [ExemplarAgentTrace("feedback", "started", f"updating {exemplar_id}")]
        record = self._find_record(snapshot, exemplar_id)
        if record is None:
            traces.append(ExemplarAgentTrace("feedback", "missing", f"{exemplar_id} not found"))
            return ExemplarAgentResult(bank_id=bank_id, traces=traces)

        decision = self.evolution.evolve_record(record, feedback)
        record.state = decision.next_state
        snapshot.stats = self._build_stats(snapshot)
        self.memory.save(snapshot)
        self.memory.append_audit_log(
            {
                "bank_id": bank_id,
                "event": "feedback",
                "exemplar_id": exemplar_id,
                "failure_mode": feedback.get("failure_mode", ""),
                "next_state": decision.next_state.value,
                "quality": decision.quality.overall,
            }
        )
        traces.append(ExemplarAgentTrace("feedback", "completed", f"state -> {decision.next_state.value}"))
        return ExemplarAgentResult(bank_id=bank_id, traces=traces, stored_record=record, evolution=decision)

    def run(self, input_text: str, **kwargs: Any) -> str:
        payload = {"input_text": input_text, **kwargs}
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _find_record(snapshot: MemoryBankSnapshot, exemplar_id: str) -> MedicalExemplarRecord | None:
        for record in [*snapshot.positive_items, *snapshot.negative_items, *snapshot.boundary_items]:
            if record.exemplar_id == exemplar_id:
                return record
        return None

    @staticmethod
    def _build_stats(snapshot: MemoryBankSnapshot) -> dict[str, Any]:
        return {
            "positive": len(snapshot.positive_items),
            "negative": len(snapshot.negative_items),
            "boundary": len(snapshot.boundary_items),
            "active": sum(
                item.state == ExemplarLifecycleState.ACTIVE
                for item in [*snapshot.positive_items, *snapshot.negative_items, *snapshot.boundary_items]
            ),
        }


__all__ = [
    "ExemplarAgentTrace",
    "ExemplarAgentResult",
    "ExemplarBankAgent",
]
