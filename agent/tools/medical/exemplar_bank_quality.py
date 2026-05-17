from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp
from typing import Any

import torch

from .exemplar_bank_schemas import ExemplarLifecycleState, MedicalExemplarRecord, PrototypeClusterRecord


@dataclass(slots=True)
class PrototypeQualityScore:
    overall: float
    clinical_quality: float
    boundary_quality: float
    novelty: float
    hard_value: float
    uncertainty_resolution: float
    retrieval_success: float
    usage_efficiency: float
    freshness: float


@dataclass(slots=True)
class PrototypeEvolutionConfig:
    ema_momentum: float = 0.95
    prototype_momentum: float = 0.90
    keep_threshold: float = 0.75
    decay_threshold: float = 0.55
    prune_threshold: float = 0.35
    max_age_days: int = 90
    curriculum_temperature: float = 0.7


@dataclass(slots=True)
class PrototypeEvolutionDecision:
    exemplar_id: str
    next_state: ExemplarLifecycleState
    quality: PrototypeQualityScore
    reasons: list[str] = field(default_factory=list)


class PrototypeQualityController:
    def __init__(self, config: PrototypeEvolutionConfig | None = None) -> None:
        self.config = config or PrototypeEvolutionConfig()

    def score(self, record: MedicalExemplarRecord) -> PrototypeQualityScore:
        freshness = self._freshness_score(record.updated_at)
        retrieval_success = self._safe_ratio(
            record.retrieval_statistics.positive_hits + record.retrieval_statistics.boundary_hits,
            max(record.retrieval_statistics.retrieval_count, 1),
        )
        usage_efficiency = min(record.usage_frequency / 25.0, 1.0)
        hard_value = min(
            (
                record.retrieval_statistics.false_positive_count
                + record.retrieval_statistics.false_negative_count
                + record.retrieval_statistics.failure_count
            )
            / 12.0,
            1.0,
        )
        uncertainty_resolution = max(0.0, 1.0 - record.uncertainty_score)
        overall = (
            0.22 * record.quality_score
            + 0.18 * record.boundary_complexity
            + 0.16 * self._novelty_score(record)
            + 0.12 * hard_value
            + 0.10 * uncertainty_resolution
            + 0.10 * retrieval_success
            + 0.07 * usage_efficiency
            + 0.05 * freshness
        )
        return PrototypeQualityScore(
            overall=round(float(overall), 4),
            clinical_quality=round(record.quality_score, 4),
            boundary_quality=round(record.boundary_complexity, 4),
            novelty=round(self._novelty_score(record), 4),
            hard_value=round(hard_value, 4),
            uncertainty_resolution=round(uncertainty_resolution, 4),
            retrieval_success=round(retrieval_success, 4),
            usage_efficiency=round(usage_efficiency, 4),
            freshness=round(freshness, 4),
        )

    def decide(self, record: MedicalExemplarRecord) -> PrototypeEvolutionDecision:
        quality = self.score(record)
        if quality.overall >= self.config.keep_threshold:
            return PrototypeEvolutionDecision(record.exemplar_id, ExemplarLifecycleState.ACTIVE, quality, ["high_quality"])
        if quality.overall >= self.config.decay_threshold:
            return PrototypeEvolutionDecision(record.exemplar_id, ExemplarLifecycleState.INDEXED, quality, ["keep_with_neutral_weight"])
        if quality.overall >= self.config.prune_threshold:
            return PrototypeEvolutionDecision(record.exemplar_id, ExemplarLifecycleState.AGED, quality, ["decay_and_review"])
        return PrototypeEvolutionDecision(record.exemplar_id, ExemplarLifecycleState.ARCHIVED, quality, ["prune"])

    @staticmethod
    def update_ema_prototype(previous: torch.Tensor, current: torch.Tensor, momentum: float) -> torch.Tensor:
        return momentum * previous + (1.0 - momentum) * current

    @staticmethod
    def update_cluster_centroid(cluster: PrototypeClusterRecord, semantic_centroid: list[float], momentum: float) -> PrototypeClusterRecord:
        if not cluster.member_ids:
            return cluster
        cluster.prototype_quality = momentum * cluster.prototype_quality + (1.0 - momentum) * float(sum(semantic_centroid) / max(len(semantic_centroid), 1))
        cluster.updated_at = datetime.now(timezone.utc).isoformat()
        return cluster

    @staticmethod
    def curriculum_weight(record: MedicalExemplarRecord, temperature: float) -> float:
        difficulty = max(0.0, min(record.difficulty_score, 1.0))
        return float(exp(-(difficulty / max(temperature, 1e-4))))

    @staticmethod
    def _novelty_score(record: MedicalExemplarRecord) -> float:
        return min((len(record.morphology_tags) + len(record.semantic_tags) + len(record.pathology_tags)) / 12.0, 1.0)

    @staticmethod
    def _freshness_score(updated_at: str) -> float:
        if not updated_at:
            return 0.0
        age_days = max((datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)).days, 0)
        return max(0.0, 1.0 - age_days / 90.0)

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return 0.0 if denominator <= 0 else min(max(numerator / denominator, 0.0), 1.0)


class PrototypeEvolutionManager:
    def __init__(self, controller: PrototypeQualityController | None = None) -> None:
        self.controller = controller or PrototypeQualityController()

    def evolve_record(self, record: MedicalExemplarRecord, feedback: dict[str, Any]) -> PrototypeEvolutionDecision:
        failure_mode = str(feedback.get("failure_mode", "")).lower()
        if failure_mode == "false_positive":
            record.state = ExemplarLifecycleState.HARD_NEGATIVE
            record.retrieval_statistics.false_positive_count += 1
        elif failure_mode == "false_negative":
            record.state = ExemplarLifecycleState.HARD_POSITIVE
            record.retrieval_statistics.false_negative_count += 1
        elif failure_mode == "uncertain":
            record.state = ExemplarLifecycleState.UNCERTAIN
            record.retrieval_statistics.failure_count += 1

        record.quality_score = max(record.quality_score, float(feedback.get("quality_score", record.quality_score)))
        record.uncertainty_score = float(feedback.get("uncertainty", record.uncertainty_score))
        record.updated_at = datetime.now(timezone.utc).isoformat()
        return self.controller.decide(record)
