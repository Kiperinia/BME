from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from hello_agents import HelloAgent, HelloAgentsLLM
from agent.tools.medical.feature_extractor import FeatureExtractor, LesionFeatures
from agent.tools.medical.morphology_classifier import MorphologyClassifier, MorphologyResult
from agent.tools.medical.paris_typing import ParisTypingEngine, ParisTypingResult
from agent.tools.medical.report_generator import ReportData, ReportGenerator
from agent.tools.medical.risk_assessor import RiskAssessmentResult, RiskAssessor, RiskLevel


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
    """
    最小可运行诊断 Agent。

    使用固定推理管线：
    1. extract_features
    2. morphology_classify
    3. paris_typing
    4. assess_risk
    5. generate_report
    """

    _RISK_PRIORITY = {
        RiskLevel.LOW: 0,
        RiskLevel.INTERMEDIATE: 1,
        RiskLevel.HIGH: 2,
    }

    def __init__(
        self,
        llm: HelloAgentsLLM | None = None,
        *,
        pixel_size_mm: float | None = 0.15,
        use_llm_report: bool = False,
        feature_extractor: FeatureExtractor | None = None,
    ):
        super().__init__(
            name="medical-diagnosis-agent",
            description="基于提示词和医学工具链的最小 HelloAgent 诊断编排器",
            llm=llm,
            metadata={"pipeline": "feature -> morphology -> paris -> risk -> report"},
        )
        self.pixel_size_mm = pixel_size_mm
        self.feature_extractor = feature_extractor or FeatureExtractor(pixel_size_mm=pixel_size_mm)
        self.morphology_classifier = MorphologyClassifier(
            pixel_size_mm=pixel_size_mm,
            llm_client=llm,
        )
        self.paris_typing_engine = ParisTypingEngine(llm_client=llm)
        self.risk_assessor = RiskAssessor(llm_client=llm)
        self.report_generator = ReportGenerator(
            llm_client=llm,
            use_llm=use_llm_report and llm is not None,
        )

    @classmethod
    def from_env(cls, use_llm: bool = False, **kwargs: Any) -> "DiagnosisAgent":
        if not use_llm:
            return cls(llm=None, **kwargs)

        from core.llm import MyLLM

        llm_kwargs = kwargs.pop("llm_kwargs", {})
        try:
            llm = MyLLM(**llm_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize LLM. Configure OPENAI_API_KEY or MODELSCOPE_API_KEY before enabling LLM mode."
            ) from exc

        return cls(llm=llm, **kwargs)

    async def run(self, input_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        payload = dict(input_data or {})
        payload.update(kwargs)

        patient_context = payload.get("patient_context") or payload.get("context")
        lesions = payload.get("lesions")
        if lesions is not None:
            result = await self.diagnose_batch(lesions, patient_context)
            return result.to_dict()

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
    def _normalize_lesion(
        lesion: LesionInput | dict[str, Any],
        index: int,
    ) -> LesionInput:
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
    def _build_batch_report(
        cls,
        results: list[DiagnosisResult],
        context: dict[str, Any],
    ) -> dict[str, Any]:
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
    def _build_label(
        paris_typing: ParisTypingResult,
        risk_assessment: RiskAssessmentResult,
    ) -> str:
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
    def _run_coroutine(coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        raise RuntimeError("Synchronous helper cannot run inside an active event loop; use await instead.")
