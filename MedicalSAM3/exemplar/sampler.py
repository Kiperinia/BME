"""示例检索用的采样辅助工具。"""

from __future__ import annotations

import random
from typing import Optional

from .memory_bank import ExemplarItem, ExemplarMemoryBank


class ExemplarSampler:
    """示例采样器。

    提供按类型采样与均衡采样能力，使用可设置随机种子的随机数发生器保证可复现性。

    参数：
        - seed: 随机数种子。
    """
    def __init__(self, seed: int = 42) -> None:
        """初始化示例采样器。

        参数：
            - seed: 随机数种子。

        返回：
            - 无返回值，仅完成对象初始化。
        """
        self.random = random.Random(seed)

    def sample_by_type(
        self,
        memory_bank: ExemplarMemoryBank,
        exemplar_type: str,
        count: int,
        human_verified: bool = True,
    ) -> list[ExemplarItem]:
        """按示例类型随机采样指定数量的示例条目。

        若请求数量不少于可用数量，则直接返回全部候选。

        参数：
            - memory_bank: 示例记忆库实例。
            - exemplar_type: 示例类型（如 positive、negative、boundary）。
            - count: 期望采样的数量。
            - human_verified: 是否仅从人工校验条目中采样。

        返回：
            - 采样得到的示例条目列表。
        """
        items = memory_bank.get_items(type=exemplar_type, human_verified=human_verified)
        if count >= len(items):
            return items
        return self.random.sample(items, k=count)

    def sample_balanced(
        self,
        memory_bank: ExemplarMemoryBank,
        positive_count: int,
        negative_count: int,
        boundary_count: int,
    ) -> dict[str, list[ExemplarItem]]:
        """均衡采样正例、负例与边界三类示例。

        参数：
            - memory_bank: 示例记忆库实例。
            - positive_count: 正例采样数量。
            - negative_count: 负例采样数量。
            - boundary_count: 边界采样数量。

        返回：
            - 包含 positive、negative、boundary 三类采样结果的字典。
        """
        return {
            "positive": self.sample_by_type(memory_bank, "positive", positive_count),
            "negative": self.sample_by_type(memory_bank, "negative", negative_count),
            "boundary": self.sample_by_type(memory_bank, "boundary", boundary_count),
        }
