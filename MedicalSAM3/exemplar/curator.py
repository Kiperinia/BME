"""Shared exemplar scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExemplarScoreBreakdown:
    mask_quality: float
    boundary_quality: float
    hard_case_value: float
    diversity_gain: float
    domain_shift_value: float
    false_positive_risk: float

    @property
    def score(self) -> float:
        return compute_exemplar_score(
            mask_quality=self.mask_quality,
            boundary_quality=self.boundary_quality,
            hard_case_value=self.hard_case_value,
            diversity_gain=self.diversity_gain,
            domain_shift_value=self.domain_shift_value,
            false_positive_risk=self.false_positive_risk,
        )


def compute_exemplar_score(
    mask_quality: float,
    boundary_quality: float,
    hard_case_value: float,
    diversity_gain: float,
    domain_shift_value: float,
    false_positive_risk: float,
) -> float:
    return (
        0.30 * mask_quality
        + 0.20 * boundary_quality
        + 0.20 * hard_case_value
        + 0.15 * diversity_gain
        + 0.15 * domain_shift_value
        - 0.30 * false_positive_risk
    )
