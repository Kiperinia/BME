from __future__ import annotations

from typing import List

from .context import SupervisorContext
from .models import Issue


class HallucinationDetector:
    def detect(self, context: SupervisorContext) -> List[Issue]:
        raise NotImplementedError


class EvidenceLinker:
    def link(self, context: SupervisorContext) -> List[Issue]:
        raise NotImplementedError
