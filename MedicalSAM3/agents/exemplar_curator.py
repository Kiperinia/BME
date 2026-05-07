"""Human-verified exemplar curation logic."""

from __future__ import annotations

from dataclasses import asdict

from MedicalSAM3.exemplar.curator import ExemplarScoreBreakdown, compute_exemplar_score
from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank


class ExemplarCurator:
    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def score(self, breakdown: ExemplarScoreBreakdown) -> float:
        return breakdown.score

    def curate(
        self,
        item: ExemplarItem,
        breakdown: ExemplarScoreBreakdown,
        memory_bank: ExemplarMemoryBank,
    ) -> tuple[bool, float]:
        score = compute_exemplar_score(**asdict(breakdown))
        if score >= self.threshold and item.human_verified:
            memory_bank.add_item(item)
            return True, score
        memory_bank.reject_item(item.item_id, f"curation_score_below_threshold:{score:.4f}")
        return False, score
