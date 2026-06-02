from __future__ import annotations

from typing import List

from .context import SupervisorContext
from .models import Issue


class ClinicalConsistencyChecker:
    def validate(self, context: SupervisorContext) -> List[Issue]:
        raise NotImplementedError


class GuidelineValidator:
    def validate(self, context: SupervisorContext) -> List[Issue]:
        raise NotImplementedError
