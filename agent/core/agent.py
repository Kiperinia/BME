"""最小可运行 agent 工厂。"""

from __future__ import annotations

from typing import Any

from agents.diagnosis_agent import DiagnosisAgent


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


__all__ = ["DiagnosisAgent", "build_minimal_agent"]
