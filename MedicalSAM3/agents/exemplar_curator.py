"""Human-verified exemplar curation logic."""

from __future__ import annotations

from dataclasses import asdict

from MedicalSAM3.exemplar.curator import ExemplarScoreBreakdown, compute_exemplar_score
from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank


class ExemplarCurator:
    """基于阈值的人体验证范例策管器。

    负责根据打分结果和人工验证标志，决定将范例存入或拒绝记忆库。
    """
    def __init__(self, threshold: float = 0.4) -> None:
        """初始化策管器。

        参数：
            - threshold: 分数阈值，低于此值的范例将被拒绝（默认 0.4）

        返回：
            - None
        """
        self.threshold = threshold

    def score(self, breakdown: ExemplarScoreBreakdown) -> float:
        """返回打分配置项的分数值。

        参数：
            - breakdown: 范例分数拆解对象

        返回：
            - 浮点数分数值
        """
        return breakdown.score

    def curate(
        self,
        item: ExemplarItem,
        breakdown: ExemplarScoreBreakdown,
        memory_bank: ExemplarMemoryBank,
    ) -> tuple[bool, float]:
        """判断并执行范例的入库或拒绝操作。

        当分数达到阈值且人工已验证时存入记忆库，否则拒绝并记录原因。

        参数：
            - item: 范例条目对象
            - breakdown: 范例分数拆解对象
            - memory_bank: 范例记忆库对象

        返回：
            - (是否入库, 分数值) 元组
        """
        score = compute_exemplar_score(**asdict(breakdown))
        if score >= self.threshold and item.human_verified:
            memory_bank.add_item(item)
            return True, score
        memory_bank.reject_item(item.item_id, f"curation_score_below_threshold:{score:.4f}")
        return False, score
