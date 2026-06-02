from __future__ import annotations

from .context import SupervisorContext


class QualityScorer:
    def score(self, context: SupervisorContext) -> float:
        raise NotImplementedError


class StyleAndCompletenessChecker:
    def check(self, context: SupervisorContext) -> list:
        raise NotImplementedError
