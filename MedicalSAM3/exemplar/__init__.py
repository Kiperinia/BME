"""MedEx-SAM3 示例记忆与原型模块。

本包提供示例（exemplar）的编码、记忆库管理、原型构建、采样、
评分以及示例感知损失等组件，用于支持 MedEx-SAM3 的训练与推理流程。
"""

from .curator import ExemplarScoreBreakdown, compute_exemplar_score
from .exemplar_encoder import ExemplarEncoder
from .losses import (
    BoundaryBandDiceLoss,
    CrossDomainConsistencyLoss,
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
    "CrossDomainConsistencyLoss",
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
