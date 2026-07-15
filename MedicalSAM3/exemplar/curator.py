"""示例评分共享辅助工具。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExemplarScoreBreakdown:
    """示例评分明细数据类。

    汇总示例在掩码质量、边界质量、困难样本价值、多样性增益、
    域迁移价值、假阳性风险等维度上的分项评分，并可通过 score 属性得到综合评分。
    """
    mask_quality: float
    boundary_quality: float
    hard_case_value: float
    diversity_gain: float
    domain_shift_value: float
    false_positive_risk: float

    @property
    def score(self) -> float:
        """计算示例综合评分。

        参数：
            - 无。

        返回：
            - 基于各分项加权得到的综合评分浮点值。
        """
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
    """根据各维度分项计算示例综合评分。

    参数：
        - mask_quality: 掩码质量分项。
        - boundary_quality: 边界质量分项。
        - hard_case_value: 困难样本价值分项。
        - diversity_gain: 多样性增益分项。
        - domain_shift_value: 域迁移价值分项。
        - false_positive_risk: 假阳性风险分项（作为惩罚项）。

    返回：
        - 加权综合评分浮点值。
    """
    return (
        0.30 * mask_quality
        + 0.20 * boundary_quality
        + 0.20 * hard_case_value
        + 0.15 * diversity_gain
        + 0.15 * domain_shift_value
        - 0.30 * false_positive_risk
    )
