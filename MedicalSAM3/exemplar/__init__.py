"""Exemplar memory and prototype modules for MedEx-SAM3."""

from .curator import ExemplarScoreBreakdown, compute_exemplar_score
from .exemplar_encoder import ExemplarEncoder
from .losses import (
    BoundaryBandDiceLoss,
    ExemplarConsistencyLoss,
    ExemplarInfoNCELoss,
    NegativeSuppressionLoss,
    PrototypeVarianceLoss,
    SoftHausdorffLoss,
)
from .memory_bank import ExemplarItem, ExemplarMemoryBank
from .prototype_builder import PrototypeBuilder
from .sampler import ExemplarSampler

__all__ = [
    "BoundaryBandDiceLoss",
    "ExemplarConsistencyLoss",
    "ExemplarEncoder",
    "ExemplarInfoNCELoss",
    "ExemplarItem",
    "ExemplarMemoryBank",
    "ExemplarSampler",
    "ExemplarScoreBreakdown",
    "NegativeSuppressionLoss",
    "PrototypeBuilder",
    "PrototypeVarianceLoss",
    "SoftHausdorffLoss",
    "compute_exemplar_score",
]
