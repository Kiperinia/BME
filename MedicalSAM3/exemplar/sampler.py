"""Sampling helpers for exemplar retrieval."""

from __future__ import annotations

import random
from typing import Optional

from .memory_bank import ExemplarItem, ExemplarMemoryBank


class ExemplarSampler:
    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def sample_by_type(
        self,
        memory_bank: ExemplarMemoryBank,
        exemplar_type: str,
        count: int,
        human_verified: bool = True,
    ) -> list[ExemplarItem]:
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
        return {
            "positive": self.sample_by_type(memory_bank, "positive", positive_count),
            "negative": self.sample_by_type(memory_bank, "negative", negative_count),
            "boundary": self.sample_by_type(memory_bank, "boundary", boundary_count),
        }
