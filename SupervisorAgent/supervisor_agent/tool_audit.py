from __future__ import annotations

from .models import AuditRecord


class ToolAuditCollector:
    def record(self, audit: AuditRecord) -> str:
        raise NotImplementedError


class ToolAuditAnalyzer:
    def analyze(self, audit: AuditRecord) -> list:
        raise NotImplementedError
