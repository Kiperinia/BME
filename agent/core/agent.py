"""业务侧 Agent 入口：复用 hello_agents.Agent，并提供当前项目的工厂方法。"""

from __future__ import annotations

from typing import Any

from hello_agents.core.agent import Agent

from agents.diagnosis_agent import DiagnosisAgent
from agents.exemplar_bank_agent import ExemplarBankAgent
from agents.medical_closed_loop_agents import MedicalClosedLoopOrchestrator


def build_minimal_agent(
    *,
    use_llm: bool = False,
    pixel_size_mm: float | None = 0.15,
    use_llm_report: bool = False,
    **kwargs: Any,
) -> DiagnosisAgent:
    """brief:
        Build minimal agent.

    parameter:
        - use_llm: Input value for use_llm.
        - pixel_size_mm: Input value for pixel_size_mm.
        - use_llm_report: Input value for use_llm_report.
        - **kwargs: Input value for kwargs.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
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
    """brief:
        Build exemplar bank agent.

    parameter:
        - memory_root: Input value for memory_root.
        - hidden_dim: Input value for hidden_dim.
        - **kwargs: Input value for kwargs.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return ExemplarBankAgent(
        memory_root=memory_root,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def build_medical_closed_loop_agent(
    *,
    diagnosis_agent: DiagnosisAgent | None = None,
    pixel_size_mm: float | None = 0.15,
    **kwargs: Any,
) -> MedicalClosedLoopOrchestrator:
    """brief:
        Build medical closed loop agent.

    parameter:
        - diagnosis_agent: Input value for diagnosis_agent.
        - pixel_size_mm: Input value for pixel_size_mm.
        - **kwargs: Input value for kwargs.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return MedicalClosedLoopOrchestrator(
        diagnosis_agent=diagnosis_agent,
        pixel_size_mm=pixel_size_mm,
        **kwargs,
    )


__all__ = [
    "Agent",
    "DiagnosisAgent",
    "ExemplarBankAgent",
    "MedicalClosedLoopOrchestrator",
    "build_minimal_agent",
    "build_exemplar_bank_agent",
    "build_medical_closed_loop_agent",
]
