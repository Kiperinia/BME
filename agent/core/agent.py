"""业务侧 Agent 入口：复用 hello_agents.Agent，并提供当前项目的工厂方法。"""

from __future__ import annotations

from typing import Any

from hello_agents.core.agent import Agent

from agents.diagnosis_agent import DiagnosisAgent
from agents.exemplar_bank_agent import ExemplarBankAgent


def build_minimal_agent(
    *,
    use_llm: bool = False,
    pixel_size_mm: float | None = 0.15,
    use_llm_report: bool = False,
    **kwargs: Any,
) -> DiagnosisAgent:
    return DiagnosisAgent.from_env(
        use_llm=use_llm,
        pixel_size_mm=pixel_size_mm,
        use_llm_report=use_llm_report,
        **kwargs,
    )


def build_exemplar_bank_agent(
    *,
    memory_root: str = "agent/memory/exemplar_bank",
    hidden_dim: int = 256,
    **kwargs: Any,
) -> ExemplarBankAgent:
    return ExemplarBankAgent(
        memory_root=memory_root,
        hidden_dim=hidden_dim,
        **kwargs,
    )


__all__ = ["Agent", "DiagnosisAgent", "ExemplarBankAgent", "build_minimal_agent", "build_exemplar_bank_agent"]
