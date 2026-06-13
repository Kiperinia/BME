from __future__ import annotations

from .models import Decision


class HumanReviewRouter:
    def route(self, decision: Decision) -> str:
        raise NotImplementedError


class ReviewQueueClient:
    def submit(self, decision: Decision) -> str:
        raise NotImplementedError
