from __future__ import annotations

from .models import AuditRecord, Decision


class DecisionStore:
    def save(self, decision: Decision) -> None:
        raise NotImplementedError


class AuditLogStore:
    def save(self, audit: AuditRecord) -> None:
        raise NotImplementedError


class MetricsEmitter:
    def emit(self, name: str, value: float) -> None:
        raise NotImplementedError
