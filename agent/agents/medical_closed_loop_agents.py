from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from agents.diagnosis_agent import DiagnosisAgent
from tools.medical.sample_library_toolsets import create_sample_library_tool_registry


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(slots=True)
class ClosedLoopAgentRun:
    agent_name: str
    display_name: str
    goal: str
    status: str
    decision: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(slots=True)
class ClosedLoopResult:
    preprocess: dict[str, Any]
    sample_audit: dict[str, Any]
    report: dict[str, Any]
    label_embedding: dict[str, Any]
    review: dict[str, Any]
    agent_runs: list[ClosedLoopAgentRun]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess": _plain(self.preprocess),
            "sample_audit": _plain(self.sample_audit),
            "report": _plain(self.report),
            "label_embedding": _plain(self.label_embedding),
            "review": _plain(self.review),
            "agent_runs": [run.to_dict() for run in self.agent_runs],
        }


class _ClosedLoopAgent:
    agent_name = ""
    display_name = ""
    goal = ""

    def __init__(self, registry: Any):
        self.registry = registry

    def _finish(
        self,
        *,
        status: str,
        decision: str,
        observations: dict[str, Any],
        warnings: list[str] | None = None,
    ) -> ClosedLoopAgentRun:
        return ClosedLoopAgentRun(
            agent_name=self.agent_name,
            display_name=self.display_name,
            goal=self.goal,
            status=status,
            decision=decision,
            tool_calls=self.registry.get_call_logs(),
            observations=observations,
            warnings=warnings or [],
        )


class SegmentationPreprocessAgent(_ClosedLoopAgent):
    agent_name = "segmentation_preprocess_agent"
    display_name = "分割预处理智能体"
    goal = "Prepare normalized image and prompt hints before segmentation diagnosis."

    def run(self, case: dict[str, Any], sample: dict[str, Any]) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        self.registry.reset_logs()
        normalized = self.registry.call("normalize_image_plan", sample=sample, target_size=int(case.get("target_size", 1024)))
        bbox_request = self.registry.call("build_bbox_cache_request", sample=sample, detector="yolo")
        prompt_package = self.registry.call("package_prompts", sample=sample, use_text=True, use_box=True, use_exemplar=True)
        uncertainty = self.registry.call("scan_region_uncertainty", sample=sample)
        small_guard = self.registry.call("guard_small_lesion", sample=sample)
        large_gate = self.registry.call("gate_large_mask", sample=sample)
        trace = self.registry.call(
            "trace_preprocess",
            sample=sample,
            steps=[normalized, bbox_request, prompt_package, uncertainty, small_guard, large_gate],
        )

        warnings: list[str] = []
        if not bbox_request.get("use_cached"):
            warnings.append("YOLO bbox is not available in smoke mode; using mask-derived or full-frame bbox fallback.")
        if large_gate.get("is_large_mask"):
            warnings.append("Large mask gate triggered; downstream review should be stricter.")

        result = {
            "normalization": normalized,
            "bbox_request": bbox_request,
            "prompt_package": prompt_package,
            "uncertainty": uncertainty,
            "small_lesion_guard": small_guard,
            "large_mask_gate": large_gate,
            "trace": trace,
        }
        decision = "use_yolo_bbox" if bbox_request.get("use_cached") else "use_bbox_fallback"
        return result, self._finish(status="ok", decision=decision, observations=result, warnings=warnings)


class SampleAuditAgent(_ClosedLoopAgent):
    agent_name = "sample_audit_agent"
    display_name = "样本审核智能体"
    goal = "Decide whether a segmented sample is valuable enough for the sample bank."

    def run(
        self,
        sample: dict[str, Any],
        reference_sample: dict[str, Any] | None,
        doctor_annotations: dict[str, Any],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        self.registry.reset_logs()
        identity = self.registry.call("check_identity", sample=sample, known_ids=[])
        mask_consistency = self.registry.call("check_label_mask_consistency", sample=sample)
        hard_case = self.registry.call("mine_hard_case", sample=sample)
        boundary_case = self.registry.call("detect_boundary_case", sample=sample)
        quiz = self.registry.call(
            "run_reference_label_quiz",
            sample=sample,
            reference_sample=reference_sample or {},
            doctor_annotations=doctor_annotations,
        )
        grade = self.registry.call("assign_sample_grade", sample=sample)
        review_item = self.registry.call(
            "build_review_queue_item",
            sample=sample,
            audit_results=[identity, mask_consistency, hard_case, boundary_case, quiz],
        )

        accepted = bool(identity["valid"] and mask_consistency["valid"] and quiz["passed"] and grade["grade"] != "reject")
        if accepted:
            bank_decision = "accept"
        elif grade["grade"] == "reject" or not mask_consistency["valid"]:
            bank_decision = "reject"
        else:
            bank_decision = "needs_human_review"

        result = {
            "identity": identity,
            "mask_consistency": mask_consistency,
            "hard_case": hard_case,
            "boundary_case": boundary_case,
            "reference_quiz": quiz,
            "grade": grade,
            "review_item": review_item,
            "accepted": accepted,
            "bank_decision": bank_decision,
        }
        return result, self._finish(status="ok", decision=bank_decision, observations=result)


class ReportGenerationAgent(_ClosedLoopAgent):
    agent_name = "report_generation_agent"
    display_name = "报告生成智能体"
    goal = "Generate a structured report from segmentation evidence and doctor annotations."

    def __init__(self, registry: Any, diagnosis_agent: DiagnosisAgent):
        super().__init__(registry)
        self.diagnosis_agent = diagnosis_agent

    def run(
        self,
        case: dict[str, Any],
        sample: dict[str, Any],
        doctor_annotations: dict[str, Any],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        self.registry.reset_logs()
        context = self.registry.call("assemble_case_context", sample=sample, similar_cases=[], review_summary={})
        template = self.registry.call("compose_report_template", context=context, report_type="clinical")
        findings_evidence = self.registry.call("narrate_findings", context=context)
        uncertainty = self.registry.call("explain_uncertainty", context=context)

        report = self._diagnose(case)
        doctor_notes = self._doctor_note(doctor_annotations)
        if doctor_notes:
            report["findings"] = f"{report.get('findings', '').strip()} Doctor annotation reference: {doctor_notes}".strip()
            report.setdefault("doctor_annotations", doctor_annotations)

        risks = self.registry.call("flag_report_risks", context={**context, "sample_group": sample.get("sample_group", "")})
        evidence = self.registry.call(
            "bind_evidence",
            context=context,
            statements=[
                report.get("findings", ""),
                report.get("conclusion", ""),
            ],
        )

        result = {
            **report,
            "case_context": context,
            "template": template,
            "finding_evidence": findings_evidence,
            "uncertainty_summary": uncertainty,
            "risk_flags": risks,
            "evidence": evidence,
        }
        decision = "needs_human_review" if risks.get("needs_human_review") else "report_ready"
        return result, self._finish(status="ok", decision=decision, observations=result)

    def _diagnose(self, case: dict[str, Any]) -> dict[str, Any]:
        image = case.get("image")
        mask = case.get("mask")
        if image is None or mask is None:
            return {
                "findings": str(case.get("report_snippet", "No image and mask were provided.")),
                "conclusion": "Insufficient visual input; human review is required.",
                "layoutSuggestion": "Show report text with review warning.",
            }

        diagnosis = self.diagnosis_agent.diagnose_single_sync(
            image=image,
            mask=mask,
            bbox=case.get("bbox"),
            lesion_id=str(case.get("lesion_id", "closed-loop-lesion")),
            context=case.get("patient_context") or {},
        )
        report = diagnosis.report.to_dict()
        report["diagnosis"] = diagnosis.to_dict()
        return report

    @staticmethod
    def _doctor_note(doctor_annotations: dict[str, Any]) -> str:
        pairs = [
            ("Paris", doctor_annotations.get("paris")),
            ("lesion", doctor_annotations.get("lesion_type")),
            ("pathology", doctor_annotations.get("pathology")),
            ("surface", doctor_annotations.get("surface_pattern")),
            ("notes", doctor_annotations.get("notes")),
        ]
        return "; ".join(f"{key}={value}" for key, value in pairs if str(value or "").strip())


class LabelEmbeddingAgent(_ClosedLoopAgent):
    agent_name = "label_embedding_agent"
    display_name = "标签嵌入智能体"
    goal = "Extract report labels and build searchable lightweight embeddings."

    def run(
        self,
        sample: dict[str, Any],
        report: dict[str, Any],
        doctor_annotations: dict[str, Any],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        self.registry.reset_logs()
        labels = self.registry.call(
            "extract_report_feature_labels",
            report=report,
            doctor_annotations=doctor_annotations,
            max_labels=12,
        )
        mask_shape = self.registry.call("embed_mask_shape", sample=sample)
        text_embedding = self.registry.call("embed_text_label", labels=labels["labels"])
        boundary = self.registry.call("encode_boundary_features", sample=sample)
        hard_signature = self.registry.call("build_hard_case_signature", sample=sample)
        route = self.registry.call("route_site_aware_embedding", sample=sample)
        drift = self.registry.call(
            "monitor_embedding_drift",
            embedding=mask_shape.get("vector", []),
            centroid=[0.05, 1.0, 0.4, 0.9, 0.1],
            threshold=0.6,
        )

        result = {
            "labels": labels["labels"],
            "label_count": labels["label_count"],
            "mask_shape_embedding": mask_shape,
            "text_embedding": text_embedding,
            "boundary_signature": boundary,
            "hard_case_signature": hard_signature,
            "index_route": route,
            "drift": drift,
        }
        decision = "index_ready" if labels["label_count"] else "needs_human_review"
        return result, self._finish(status="ok", decision=decision, observations=result)


class ResultReviewAgent(_ClosedLoopAgent):
    agent_name = "result_review_agent"
    display_name = "结果复核智能体"
    goal = "Review the whole closed loop and decide the final disposition."

    def run(
        self,
        sample: dict[str, Any],
        sample_audit: dict[str, Any],
        report: dict[str, Any],
        label_embedding: dict[str, Any],
        prior_warnings: list[str],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        self.registry.reset_logs()
        delta = self.registry.call("analyze_metric_delta", sample=sample)
        failure = self.registry.call("classify_failure_case", sample=sample)
        confidence = self.registry.call("check_confidence_consistency", sample=sample)
        sanity = self.registry.call("review_mask_sanity", sample=sample)
        regression = self.registry.call("detect_regression", sample=sample)
        exemplar_effect = self.registry.call("audit_exemplar_effect", sample=sample)
        continual = self.registry.call(
            "update_continual_bank_item",
            sample=sample,
            review={**sanity, **regression},
        )

        warnings = list(prior_warnings)
        if not sample_audit.get("accepted"):
            warnings.append("Sample audit did not fully accept the case.")
        if not report.get("findings") or not report.get("conclusion"):
            warnings.append("Report is incomplete.")
        if not label_embedding.get("labels"):
            warnings.append("No searchable labels were extracted.")
        if confidence.get("inconsistent"):
            warnings.append("Confidence and result quality are inconsistent.")
        if regression.get("is_regression"):
            warnings.append("Meaningful metric regression detected.")

        if not sanity.get("sane") or sample_audit.get("bank_decision") == "reject":
            final_status = "rejected"
        elif warnings:
            final_status = "needs_human_review"
        else:
            final_status = "approved"

        result = {
            "final_status": final_status,
            "bank_decision": sample_audit.get("bank_decision", "needs_human_review"),
            "label_count": len(label_embedding.get("labels", [])),
            "warnings": warnings,
            "metric_delta": delta,
            "failure_mode": failure,
            "confidence_consistency": confidence,
            "mask_sanity": sanity,
            "regression": regression,
            "exemplar_effect": exemplar_effect,
            "continual_bank_item": continual,
        }
        return result, self._finish(status="ok", decision=final_status, observations=result, warnings=warnings)


class MedicalClosedLoopOrchestrator:
    def __init__(self, diagnosis_agent: DiagnosisAgent | None = None, *, pixel_size_mm: float | None = 0.15):
        self.registry = create_sample_library_tool_registry()
        self.diagnosis_agent = diagnosis_agent or DiagnosisAgent.from_env(
            use_llm=False,
            use_llm_report=False,
            pixel_size_mm=pixel_size_mm,
        )
        self.preprocess_agent = SegmentationPreprocessAgent(self.registry)
        self.sample_audit_agent = SampleAuditAgent(self.registry)
        self.report_agent = ReportGenerationAgent(self.registry, self.diagnosis_agent)
        self.label_agent = LabelEmbeddingAgent(self.registry)
        self.review_agent = ResultReviewAgent(self.registry)

    def run_sync(
        self,
        case: dict[str, Any],
        reference_sample: dict[str, Any] | None = None,
        doctor_annotations: dict[str, Any] | None = None,
    ) -> ClosedLoopResult:
        annotations = dict(doctor_annotations or {})
        sample = self._build_sample(case, annotations)
        runs: list[ClosedLoopAgentRun] = []

        preprocess, run = self.preprocess_agent.run(case, sample)
        runs.append(run)
        sample = self._merge_preprocess_sample(sample, preprocess)

        sample_audit, run = self.sample_audit_agent.run(sample, reference_sample, annotations)
        runs.append(run)

        report, run = self.report_agent.run(case, sample, annotations)
        runs.append(run)

        label_embedding, run = self.label_agent.run(sample, report, annotations)
        runs.append(run)

        prior_warnings = [warning for item in runs for warning in item.warnings]
        review, run = self.review_agent.run(sample, sample_audit, report, label_embedding, prior_warnings)
        runs.append(run)

        return ClosedLoopResult(
            preprocess=preprocess,
            sample_audit=sample_audit,
            report=report,
            label_embedding=label_embedding,
            review=review,
            agent_runs=runs,
        )

    def _build_sample(self, case: dict[str, Any], doctor_annotations: dict[str, Any]) -> dict[str, Any]:
        source = copy.deepcopy(case.get("sample") or {})
        image = case.get("image")
        mask = case.get("mask")
        height, width = self._image_shape(image)
        bbox = list(case.get("bbox") or source.get("bbox") or self._bbox_from_mask(mask, width, height))
        mask_stats = {**self._mask_stats(mask, width, height), **dict(source.get("mask_stats", {}))}
        tags = list(dict.fromkeys([*source.get("tags", []), *self._annotation_tags(doctor_annotations)]))
        return {
            "image_id": str(case.get("lesion_id") or source.get("image_id") or "closed-loop-lesion"),
            "site_id": str(source.get("site_id") or case.get("site_id") or "runtime"),
            "split": str(source.get("split") or "runtime"),
            "sample_group": str(source.get("sample_group") or "candidate"),
            "image_path": str(source.get("image_path") or ""),
            "mask_path": str(source.get("mask_path") or ""),
            "bbox": [float(value) for value in bbox],
            "metrics": {"Dice": 0.82, "Precision": 0.84, "Recall": 0.8, "Boundary F1": 0.72, **dict(source.get("metrics", {}))},
            "baseline_metrics": {"Dice": 0.74, **dict(source.get("baseline_metrics", {}))},
            "mask_stats": mask_stats,
            "uncertainty": {"mean_entropy": 0.22, "mean_confidence": 0.83, **dict(source.get("uncertainty", {}))},
            "selected_exemplars": dict(source.get("selected_exemplars", {"positive_ids": [], "negative_ids": [], "boundary_ids": []})),
            "tags": tags,
            "metadata": {**dict(source.get("metadata", {})), "doctor_annotations": doctor_annotations},
        }

    @staticmethod
    def _image_shape(image: Any) -> tuple[int, int]:
        if isinstance(image, np.ndarray) and image.ndim >= 2:
            return int(image.shape[0]), int(image.shape[1])
        return 256, 256

    @staticmethod
    def _bbox_from_mask(mask: Any, width: int, height: int) -> list[int]:
        if isinstance(mask, np.ndarray) and mask.size:
            binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                return [int(x), int(y), int(x + max(w - 1, 0)), int(y + max(h - 1, 0))]
        return [0, 0, max(width - 1, 0), max(height - 1, 0)]

    @staticmethod
    def _mask_stats(mask: Any, width: int, height: int) -> dict[str, float]:
        if not isinstance(mask, np.ndarray) or not mask.size:
            return {"area_ratio": 1.0, "aspect_ratio": 1.0, "boundary_complexity": 0.0, "solidity": 1.0, "components": 1.0}
        binary = (mask > 0).astype(np.uint8)
        area = float(binary.sum())
        area_ratio = area / max(float(width * height), 1.0)
        components = max(int(cv2.connectedComponents(binary)[0]) - 1, 0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"area_ratio": 0.0, "aspect_ratio": 1.0, "boundary_complexity": 0.0, "solidity": 0.0, "components": float(components)}
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        solidity = contour_area / max(float(cv2.contourArea(hull)), 1.0)
        perimeter = float(cv2.arcLength(contour, True))
        boundary_complexity = min(perimeter / max(np.sqrt(contour_area) * 8.0, 1.0), 1.0)
        return {
            "area_ratio": round(area_ratio, 6),
            "aspect_ratio": round(float(w) / max(float(h), 1.0), 4),
            "boundary_complexity": round(boundary_complexity, 4),
            "solidity": round(solidity, 4),
            "components": float(components),
        }

    @staticmethod
    def _annotation_tags(doctor_annotations: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for value in [
            doctor_annotations.get("paris"),
            doctor_annotations.get("lesion_type"),
            doctor_annotations.get("pathology"),
            doctor_annotations.get("surface_pattern"),
            *list(doctor_annotations.get("tags", []) or []),
        ]:
            normalized = str(value or "").strip()
            if normalized:
                tags.append(normalized)
        return tags

    @staticmethod
    def _merge_preprocess_sample(sample: dict[str, Any], preprocess: dict[str, Any]) -> dict[str, Any]:
        updated = dict(sample)
        bbox = preprocess.get("bbox_request", {}).get("bbox")
        if bbox:
            updated["bbox"] = bbox
        return updated


def build_medical_closed_loop_agent(
    *,
    diagnosis_agent: DiagnosisAgent | None = None,
    pixel_size_mm: float | None = 0.15,
) -> MedicalClosedLoopOrchestrator:
    return MedicalClosedLoopOrchestrator(diagnosis_agent=diagnosis_agent, pixel_size_mm=pixel_size_mm)


__all__ = [
    "ClosedLoopAgentRun",
    "ClosedLoopResult",
    "MedicalClosedLoopOrchestrator",
    "build_medical_closed_loop_agent",
]
