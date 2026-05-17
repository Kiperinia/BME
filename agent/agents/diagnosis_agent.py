from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from hello_agents import HelloAgentsLLM
from hello_agents.core.agent import Agent as HelloAgent
from hello_agents.tools import ToolRegistry

from core.config import Config
from core.llm import MyLLM, RuleOnlyLLM
from tools.medical.feature_extractor import FeatureExtractor, LesionFeatures
from tools.medical.morphology_classifier import MorphologyClassifier, MorphologyResult
from tools.medical.paris_typing import ParisTypingEngine, ParisTypingResult
from tools.medical.report_generator import ReportData, ReportGenerator
from tools.medical.risk_assessor import RiskAssessmentResult, RiskAssessor, RiskLevel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LesionInput:
    image: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None = None
    lesion_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiagnosisResult:
    lesion_id: str
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    features: LesionFeatures
    morphology: MorphologyResult
    paris_typing: ParisTypingResult
    risk_assessment: RiskAssessmentResult
    report: ReportData

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesion_id": self.lesion_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "features": self.features.to_dict(),
            "morphology": self.morphology.to_dict(),
            "paris_typing": self.paris_typing.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "report": self.report.to_dict(),
            "reasoning": [
                self.morphology.to_dict(),
                self.paris_typing.to_dict(),
                self.risk_assessment.to_dict(),
            ],
        }


@dataclass(slots=True)
class BatchDiagnosisResult:
    lesions: list[DiagnosisResult]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesions": [lesion.to_dict() for lesion in self.lesions],
            "report": self.report,
        }


class DiagnosisAgent(HelloAgent):
    """业务侧医学诊断 Agent，底层复用 hello_agents.Agent。"""

    _RISK_PRIORITY = {
        RiskLevel.LOW: 0,
        RiskLevel.INTERMEDIATE: 1,
        RiskLevel.HIGH: 2,
    }

    def __init__(
        self,
        llm: HelloAgentsLLM | None = None,
        *,
        config: Config | None = None,
        pixel_size_mm: float | None = 0.15,
        use_llm_report: bool = False,
        feature_extractor: FeatureExtractor | None = None,
        tool_registry: ToolRegistry | None = None,
        enable_report_reflection: bool = True,
        reflection_max_iterations: int = 3,
    ):
        resolved_config = config or Config.from_env()
        resolved_llm = llm or RuleOnlyLLM(
            model=resolved_config.default_model,
            provider=resolved_config.default_provider,
        )

        super().__init__(
            name="medical-diagnosis-agent",
            llm=resolved_llm,
            system_prompt="你是一名结构化的消化内镜病灶诊断助手。",
            config=resolved_config,
            tool_registry=tool_registry,
        )

        self.description = "基于提示词和医学工具链的最小 HelloAgent 诊断编排器"
        self.metadata = {"pipeline": "feature -> morphology -> paris -> risk -> report -> [reflection]"}
        self.pixel_size_mm = pixel_size_mm
        llm_client = None if isinstance(resolved_llm, RuleOnlyLLM) else resolved_llm
        self.feature_extractor = feature_extractor or FeatureExtractor(pixel_size_mm=pixel_size_mm)
        self.morphology_classifier = MorphologyClassifier(
            pixel_size_mm=pixel_size_mm,
            llm_client=llm_client,
        )
        self.paris_typing_engine = ParisTypingEngine(llm_client=llm_client)
        self.risk_assessor = RiskAssessor(llm_client=llm_client)
        self.report_generator = ReportGenerator(
            llm_client=llm_client,
            use_llm=use_llm_report and llm_client is not None,
        )
        
        # Initialize reflection agent if enabled and LLM is available
        self.enable_report_reflection = enable_report_reflection and llm_client is not None
        if self.enable_report_reflection:
            from agents.report_reflection_agent import ReportReflectionAgent
            self.reflection_agent = ReportReflectionAgent(
                llm=llm_client,
                max_iterations=reflection_max_iterations,
                quality_threshold=8.0,
            )
        else:
            self.reflection_agent = None

    @classmethod
    def from_env(cls, use_llm: bool = False, **kwargs: Any) -> "DiagnosisAgent":
        config = kwargs.pop("config", None) or Config.from_env()
        llm_kwargs = kwargs.pop("llm_kwargs", {})

        if not use_llm:
            llm = RuleOnlyLLM(model=config.default_model, provider=config.default_provider)
            return cls(llm=llm, config=config, **kwargs)

        try:
            llm = MyLLM(config=config, **llm_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize LLM. Fill agent/config/llm_profiles.json or configure LLM_API_KEY/LLM_BASE_URL or MODELSCOPE_API_KEY before enabling LLM mode."
            ) from exc

        return cls(llm=llm, config=config, **kwargs)

    def run(self, input_text: str, **kwargs: Any) -> str:
        payload = kwargs.pop("input_data", None)
        if payload is None:
            if kwargs:
                payload = dict(kwargs)
                if input_text:
                    payload.setdefault("input_text", input_text)
            elif input_text:
                payload = self._parse_input_text(input_text)
            else:
                payload = {}

        result = self.run_payload(payload)
        return json.dumps(result, ensure_ascii=False)

    def run_payload(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        payload = dict(input_data or {})
        payload.update(kwargs)

        patient_context = payload.get("patient_context") or payload.get("context")
        lesions = payload.get("lesions")
        if lesions is not None:
            return self.diagnose_batch_sync(lesions, patient_context).to_dict()

        image = payload.get("image")
        mask = payload.get("mask")
        if image is None or mask is None:
            raise ValueError("Single-run payload must contain image and mask.")

        result = self.diagnose_single_sync(
            image=image,
            mask=mask,
            bbox=payload.get("bbox"),
            context=patient_context,
            lesion_id=payload.get("lesion_id", "lesion-1"),
        )
        return result.to_dict()

    async def arun_payload(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        payload = dict(input_data or {})
        payload.update(kwargs)

        patient_context = payload.get("patient_context") or payload.get("context")
        lesions = payload.get("lesions")
        if lesions is not None:
            return (await self.diagnose_batch(lesions, patient_context)).to_dict()

        image = payload.get("image")
        mask = payload.get("mask")
        if image is None or mask is None:
            raise ValueError("Single-run payload must contain image and mask.")

        result = await self.diagnose_single(
            image=image,
            mask=mask,
            bbox=payload.get("bbox"),
            context=patient_context,
            lesion_id=payload.get("lesion_id", "lesion-1"),
        )
        return result.to_dict()

    def run_sync(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.run_payload(input_data, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "llm_configured": not isinstance(self.llm, RuleOnlyLLM),
            "metadata": self.metadata,
        }

    async def diagnose_single(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
        context: dict[str, Any] | None = None,
        lesion_id: str = "lesion-1",
    ) -> DiagnosisResult:
        normalized_context = self._normalize_context(context)
        features = self.feature_extractor.extract(image, mask)
        morphology = self.morphology_classifier.classify(features)
        paris_typing = self.paris_typing_engine.infer(morphology, features)
        risk_assessment = self.risk_assessor.assess(morphology, paris_typing, features)
        report = self.report_generator.generate(
            patient_id=normalized_context["patient_id"],
            study_id=normalized_context["study_id"],
            exam_date=normalized_context["exam_date"],
            morphology=morphology,
            paris=paris_typing,
            risk=risk_assessment,
            features=features,
        )
        
        # Apply report reflection (ReAct agent thinking) if enabled
        if self.enable_report_reflection and self.reflection_agent is not None:
            logger.info(f"Applying report reflection for lesion {lesion_id}...")
            reflection_result = self.reflection_agent.reflect(
                report=report,
                morphology=morphology,
                paris=paris_typing,
                risk=risk_assessment,
            )
            report = reflection_result.final_report
            logger.info(f"Reflection completed: {reflection_result.total_iterations} iterations, quality={reflection_result.final_quality_score}")

        return DiagnosisResult(
            lesion_id=lesion_id,
            label=self._build_label(paris_typing, risk_assessment),
            confidence=self._combine_confidence(morphology, paris_typing, risk_assessment),
            bbox=tuple(bbox) if bbox else tuple(features.geometric.bbox),
            features=features,
            morphology=morphology,
            paris_typing=paris_typing,
            risk_assessment=risk_assessment,
            report=report,
        )

    async def diagnose_batch(
        self,
        lesions: Iterable[LesionInput | dict[str, Any]],
        patient_context: dict[str, Any] | None = None,
    ) -> BatchDiagnosisResult:
        normalized_context = self._normalize_context(patient_context)
        normalized_lesions = [
            self._normalize_lesion(lesion, index)
            for index, lesion in enumerate(lesions, start=1)
        ]

        results = await asyncio.gather(
            *[
                self.diagnose_single(
                    image=lesion.image,
                    mask=lesion.mask,
                    bbox=lesion.bbox,
                    context=normalized_context,
                    lesion_id=lesion.lesion_id or f"lesion-{index}",
                )
                for index, lesion in enumerate(normalized_lesions, start=1)
            ]
        )

        return BatchDiagnosisResult(
            lesions=results,
            report=self._build_batch_report(results, normalized_context),
        )

    def diagnose_single_sync(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
        context: dict[str, Any] | None = None,
        lesion_id: str = "lesion-1",
    ) -> DiagnosisResult:
        return self._run_coroutine(
            self.diagnose_single(
                image=image,
                mask=mask,
                bbox=bbox,
                context=context,
                lesion_id=lesion_id,
            )
        )

    def diagnose_batch_sync(
        self,
        lesions: Iterable[LesionInput | dict[str, Any]],
        patient_context: dict[str, Any] | None = None,
    ) -> BatchDiagnosisResult:
        return self._run_coroutine(self.diagnose_batch(lesions, patient_context))

    @staticmethod
    def _normalize_context(context: dict[str, Any] | None) -> dict[str, Any]:
        normalized = {
            "patient_id": "demo-patient",
            "study_id": "demo-study",
            "exam_date": "",
        }
        if context:
            normalized.update(dict(context))
        return normalized

    @staticmethod
    def _normalize_lesion(lesion: LesionInput | dict[str, Any], index: int) -> LesionInput:
        if isinstance(lesion, LesionInput):
            return lesion
        if not isinstance(lesion, dict):
            raise TypeError(f"Unsupported lesion input at index {index}: {type(lesion)!r}")

        image = lesion.get("image")
        mask = lesion.get("mask")
        if image is None or mask is None:
            raise ValueError(f"Lesion at index {index} must contain image and mask.")

        bbox = lesion.get("bbox")
        return LesionInput(
            image=image,
            mask=mask,
            bbox=tuple(bbox) if bbox else None,
            lesion_id=lesion.get("lesion_id", f"lesion-{index}"),
            metadata=dict(lesion.get("metadata", {})),
        )

    @classmethod
    def _build_batch_report(cls, results: list[DiagnosisResult], context: dict[str, Any]) -> dict[str, Any]:
        if not results:
            return {
                "patient_id": context["patient_id"],
                "study_id": context["study_id"],
                "findings": "未提供病灶输入。",
                "conclusion": "无法生成批量报告。",
                "layoutSuggestion": "使用常规布局展示空结果。",
            }

        highest_risk = max(
            results,
            key=lambda item: (
                cls._RISK_PRIORITY[item.risk_assessment.risk_level],
                item.risk_assessment.total_score,
            ),
        )
        findings = (
            f"本次共评估 {len(results)} 处病灶。最高风险病灶为 {highest_risk.lesion_id}，"
            f"{highest_risk.report.findings}"
        )
        conclusion = (
            f"批量判读以 {highest_risk.lesion_id} 为主要关注对象。"
            f"{highest_risk.report.conclusion}"
        )

        return {
            "patient_id": context["patient_id"],
            "study_id": context["study_id"],
            "exam_date": context["exam_date"],
            "highest_risk_lesion_id": highest_risk.lesion_id,
            "findings": findings,
            "conclusion": conclusion,
            "layoutSuggestion": highest_risk.report.layout_suggestion,
        }

    @staticmethod
    def _build_label(paris_typing: ParisTypingResult, risk_assessment: RiskAssessmentResult) -> str:
        return f"{paris_typing.paris_type.value}/{risk_assessment.risk_level.value}"

    @staticmethod
    def _combine_confidence(
        morphology: MorphologyResult,
        paris_typing: ParisTypingResult,
        risk_assessment: RiskAssessmentResult,
    ) -> float:
        return round(
            (morphology.confidence + paris_typing.confidence + risk_assessment.confidence) / 3.0,
            4,
        )

    @staticmethod
    def _parse_input_text(input_text: str) -> dict[str, Any]:
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            return {"input_text": input_text}

    @staticmethod
    def _run_coroutine(coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        raise RuntimeError("Synchronous helper cannot run inside an active event loop; use await instead.")


__all__ = [
    "LesionInput",
    "DiagnosisResult",
    "BatchDiagnosisResult",
    "DiagnosisAgent",
]
