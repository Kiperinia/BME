from __future__ import annotations

from .context import SupervisorContext
from .models import RiskLevel


class RiskAssessor:
    def assess(self, context: SupervisorContext) -> RiskLevel:
        raise NotImplementedError


class SafetyPolicyGate:
    def allow(self, context: SupervisorContext, risk_level: RiskLevel) -> bool:
        raise NotImplementedError
