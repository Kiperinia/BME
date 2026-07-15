from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from agents.diagnosis_agent import DiagnosisAgent
from tools.medical.sample_library_toolsets import create_sample_library_tool_registry, get_primary_agent_tool_chains


PRIMARY_AGENT_TOOL_CHAINS = get_primary_agent_tool_chains()


def _plain(value: Any) -> Any:
    """brief:
        Handle plain.

    parameter:
        - value: Input value for value.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
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
    """brief:
        Represent ClosedLoopAgentRun state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
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
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return _plain(asdict(self))


@dataclass(slots=True)
class ClosedLoopResult:
    """brief:
        Represent ClosedLoopResult state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    preprocess: dict[str, Any]
    sample_audit: dict[str, Any]
    report: dict[str, Any]
    label_embedding: dict[str, Any]
    review: dict[str, Any]
    agent_runs: list[ClosedLoopAgentRun]

    def to_dict(self) -> dict[str, Any]:
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "preprocess": _plain(self.preprocess),
            "sample_audit": _plain(self.sample_audit),
            "report": _plain(self.report),
            "label_embedding": _plain(self.label_embedding),
            "review": _plain(self.review),
            "agent_runs": [run.to_dict() for run in self.agent_runs],
        }


class _ClosedLoopAgent:
    """brief:
        Represent ClosedLoopAgent state and behavior.

    parameter:
        - registry: Input value for registry.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = ""
    display_name = ""
    goal = ""

    def __init__(self, registry: Any):
        """brief:
            Initialize this object.

        parameter:
            - registry: Input value for registry.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.registry = registry

    def _annotate(self, observations: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle annotate.

        parameter:
            - observations: Input value for observations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        metadata = copy.deepcopy(PRIMARY_AGENT_TOOL_CHAINS.get(self.agent_name, {}))
        return {
            **observations,
            "agentDetail": metadata.get("agentDetail", ""),
            "promptDesign": metadata.get("promptDesign", []),
            "mainToolChain": metadata.get("mainToolChain", []),
        }

    def _finish(
        self,
        *,
        status: str,
        decision: str,
        observations: dict[str, Any],
        warnings: list[str] | None = None,
    ) -> ClosedLoopAgentRun:
        """brief:
            Handle finish.

        parameter:
            - status: Input value for status.
            - decision: Input value for decision.
            - observations: Input value for observations.
            - warnings: Input value for warnings.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
    """brief:
        Represent SegmentationPreprocessAgent state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "segmentation_preprocess_agent"
    display_name = "分割预处理智能体"
    goal = "Prepare normalized image and prompt hints before segmentation diagnosis."

    def run(self, case: dict[str, Any], sample: dict[str, Any]) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        """brief:
            Handle run.

        parameter:
            - case: Input value for case.
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        self.registry.reset_logs()
        bbox_request = self.registry.call("BuildBboxRequest", sample=sample, detector="yolo")
        normalized = self.registry.call("NormalizeImagePlan", sample=sample, target_size=int(case.get("target_size", 1024)))
        trace = self.registry.call(
            "TracePreprocess",
            sample=sample,
            steps=[bbox_request, normalized],
        )
        prompt_package = self.registry.call("PackagePrompts", sample=sample, use_text=True, use_box=True, use_exemplar=True)

        uncertainty_payload = dict(sample.get("uncertainty", {}) or {})
        mask_stats = dict(sample.get("mask_stats", {}) or {})
        area_ratio = float(mask_stats.get("area_ratio", 0.0) or 0.0)
        mean_entropy = float(uncertainty_payload.get("mean_entropy", 0.0) or 0.0)
        mean_confidence = float(uncertainty_payload.get("mean_confidence", 1.0) or 1.0)
        uncertainty = {
            "needs_region_attention": mean_entropy > 0.35 or mean_confidence < 0.65,
            "mean_entropy": mean_entropy,
            "mean_confidence": mean_confidence,
        }
        small_guard = {"is_small_lesion": 0.0 < area_ratio < 0.002, "recommended_scale": 1.5 if 0.0 < area_ratio < 0.002 else 1.0}
        large_gate = {"is_large_mask": area_ratio > 0.35, "use_exemplar_guard": area_ratio > 0.35, "area_ratio": area_ratio}

        warnings: list[str] = []
        if not bbox_request.get("use_cached"):
            warnings.append("YOLO bbox is not available in smoke mode; using mask-derived or full-frame bbox fallback.")
        if large_gate.get("is_large_mask"):
            warnings.append("Large mask gate triggered; downstream review should be stricter.")

        result = self._annotate({
            "normalization": normalized,
            "bbox_request": bbox_request,
            "prompt_package": prompt_package,
            "uncertainty": uncertainty,
            "small_lesion_guard": small_guard,
            "large_mask_gate": large_gate,
            "trace": trace,
        })
        decision = "use_yolo_bbox" if bbox_request.get("use_cached") else "use_bbox_fallback"
        return result, self._finish(status="ok", decision=decision, observations=result, warnings=warnings)


class SampleAuditAgent(_ClosedLoopAgent):
    """brief:
        Represent SampleAuditAgent state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "sample_audit_agent"
    display_name = "样本审核智能体"
    goal = "Decide whether a segmented sample is valuable enough for the sample bank."

    def run(
        self,
        sample: dict[str, Any],
        reference_sample: dict[str, Any] | None,
        doctor_annotations: dict[str, Any],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        """brief:
            Handle run.

        parameter:
            - sample: Input value for sample.
            - reference_sample: Input value for reference_sample.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        self.registry.reset_logs()
        review_bundle = self.registry.call("BuildReviewQueueItem", sample=sample, known_ids=[])
        quiz = self.registry.call(
            "RunReferenceLabelQuiz",
            sample=sample,
            reference_sample=reference_sample or {},
            doctor_annotations=doctor_annotations,
        )

        identity = review_bundle["identity"]
        mask_consistency = review_bundle["mask_consistency"]
        hard_case = review_bundle["hard_case"]
        boundary_case = review_bundle["boundary_case"]
        grade = review_bundle["grade"]
        review_item = {
            **review_bundle["review_item"],
            "quiz_passed": quiz["passed"],
            "quiz_score": quiz["score"],
        }

        accepted = bool(identity["valid"] and mask_consistency["valid"] and quiz["passed"] and grade["grade"] != "reject")
        if accepted:
            bank_decision = "accept"
        elif grade["grade"] == "reject" or not mask_consistency["valid"]:
            bank_decision = "reject"
        else:
            bank_decision = "needs_human_review"

        result = self._annotate({
            "identity": identity,
            "mask_consistency": mask_consistency,
            "hard_case": hard_case,
            "boundary_case": boundary_case,
            "reference_quiz": quiz,
            "grade": grade,
            "review_item": review_item,
            "accepted": accepted,
            "bank_decision": bank_decision,
        })
        return result, self._finish(status="ok", decision=bank_decision, observations=result)


class ReportGenerationAgent(_ClosedLoopAgent):
    """brief:
        Represent ReportGenerationAgent state and behavior.

    parameter:
        - registry: Input value for registry.
        - diagnosis_agent: Input value for diagnosis_agent.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "report_generation_agent"
    display_name = "报告生成智能体"
    goal = "Generate a structured report from segmentation evidence and doctor annotations."

    def __init__(self, registry: Any, diagnosis_agent: DiagnosisAgent):
        """brief:
            Initialize this object.

        parameter:
            - registry: Input value for registry.
            - diagnosis_agent: Input value for diagnosis_agent.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__(registry)
        self.diagnosis_agent = diagnosis_agent

    def run(
        self,
        case: dict[str, Any],
        sample: dict[str, Any],
        doctor_annotations: dict[str, Any],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        """brief:
            Handle run.

        parameter:
            - case: Input value for case.
            - sample: Input value for sample.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        self.registry.reset_logs()
        context = self.registry.call("CaseContextAssembler", sample=sample, similar_cases=[], review_summary={})
        uncertainty = self.registry.call("UncertaintyExplainer", context=context)
        template = self.registry.call("ReportTemplateComposer", context=context, report_type="clinical")

        report = self._diagnose(case)
        doctor_notes = self._doctor_note(doctor_annotations)
        if doctor_notes:
            report["findings"] = f"{report.get('findings', '').strip()} Doctor annotation reference: {doctor_notes}".strip()
            report.setdefault("doctor_annotations", doctor_annotations)

        dice = float(context.get("dice", 0.0))
        delta = float(context.get("delta_dice", 0.0))
        area_ratio = float(context.get("mask_stats", {}).get("area_ratio", 0.0))
        findings_evidence = {
            "quality_band": "high" if dice >= 0.85 else "moderate" if dice >= 0.65 else "low",
            "delta_direction": "improved" if delta > 0.03 else "regressed" if delta < -0.03 else "stable",
            "finding_facts": [
                f"Dice={dice:.4f}",
                f"baseline Dice={float(context.get('baseline_dice', 0.0)):.4f}",
                f"delta Dice={delta:.4f}",
                f"mask area ratio={area_ratio:.4f}",
            ],
        }
        risk_flags: list[str] = []
        if dice < 0.5:
            risk_flags.append("low_result_dice")
        if delta <= -0.03:
            risk_flags.append("regression")
        if sample.get("sample_group") in {"ambiguous", "reject"}:
            risk_flags.append("sample_quality_risk")
        risks = {"risk_flags": risk_flags, "needs_human_review": bool(risk_flags)}
        evidence = [
            {
                "statement": statement,
                "image_id": context.get("image_id", ""),
                "evidence_refs": ["metrics", "mask_stats", "selected_exemplars"],
                "similar_case_ids": [case.get("image_id", "") for case in context.get("similar_cases", [])[:3]],
            }
            for statement in [
                report.get("findings", ""),
                report.get("conclusion", ""),
            ]
        ]

        result = self._annotate({
            **report,
            "case_context": context,
            "template": template,
            "finding_evidence": findings_evidence,
            "uncertainty_summary": uncertainty,
            "risk_flags": risks,
            "evidence": evidence,
        })
        decision = "needs_human_review" if risks.get("needs_human_review") else "report_ready"
        return result, self._finish(status="ok", decision=decision, observations=result)

    def _diagnose(self, case: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle diagnose.

        parameter:
            - case: Input value for case.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle doctor note.

        parameter:
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        pairs = [
            ("Paris", doctor_annotations.get("paris")),
            ("lesion", doctor_annotations.get("lesion_type")),
            ("pathology", doctor_annotations.get("pathology")),
            ("surface", doctor_annotations.get("surface_pattern")),
            ("notes", doctor_annotations.get("notes")),
        ]
        return "; ".join(f"{key}={value}" for key, value in pairs if str(value or "").strip())


class DatabaseTermAgent(_ClosedLoopAgent):
    """brief:
        Represent DatabaseTermAgent state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "label_embedding_agent"
    display_name = "标签嵌入智能体"
    goal = "Generate normalized database filter terms from report text and doctor annotations."

    def run(
        self,
        sample: dict[str, Any],
        report: dict[str, Any],
        doctor_annotations: dict[str, Any],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        """brief:
            Handle run.

        parameter:
            - sample: Input value for sample.
            - report: Input value for report.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        self.registry.reset_logs()
        extracted = self.registry.call(
            "ExtractReportTerms",
            report=report,
            doctor_annotations=doctor_annotations,
            max_terms=16,
        )
        normalized = self.registry.call("NormalizeMedicalTerms", terms=extracted["terms"])
        deduped = self.registry.call("DeduplicateTerms", terms=normalized["terms"])
        db_records = self.registry.call(
            "Build_db_TermRecords",
            terms=deduped["terms"],
            report=report,
            doctor_annotations=doctor_annotations,
            report_id=str(report.get("study_id", sample.get("image_id", ""))),
            patient_id=str(report.get("patient_id", "")),
        )

        result = self._annotate({
            "terms": [record["normalizedTerm"] for record in db_records["dbRecords"]],
            "labels": [record["normalizedTerm"] for record in db_records["dbRecords"]],
            "term_count": db_records["record_count"],
            "label_count": db_records["record_count"],
            "extracted": extracted,
            "normalized": normalized,
            "deduplicated": deduped,
            "dbRecords": db_records["dbRecords"],
            "validation": db_records["validation"],
            "routes": db_records["routes"],
            "upsert": db_records["upsert"],
            "facets": db_records["facets"],
            "coverage": db_records["coverage"],
        })
        if not db_records["dbRecords"]:
            decision = "insufficient_terms"
        elif not db_records["validation"]["valid"] or db_records["coverage"]["needsReview"]:
            decision = "needs_term_review"
        else:
            decision = "ready_to_index"
        result["decision"] = decision
        return result, self._finish(status="ok", decision=decision, observations=result)


class CrossAgentResultReviewAgent(_ClosedLoopAgent):
    """brief:
        Represent CrossAgentResultReviewAgent state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "result_review_agent"
    display_name = "结果复核智能体"
    goal = "Review every upstream agent output and decide the final closed-loop action."

    def run(
        self,
        preprocess: dict[str, Any],
        sample: dict[str, Any],
        sample_audit: dict[str, Any],
        report: dict[str, Any],
        label_embedding: dict[str, Any],
        upstream_agent_runs: list[ClosedLoopAgentRun],
    ) -> tuple[dict[str, Any], ClosedLoopAgentRun]:
        """brief:
            Handle run.

        parameter:
            - preprocess: Input value for preprocess.
            - sample: Input value for sample.
            - sample_audit: Input value for sample_audit.
            - report: Input value for report.
            - label_embedding: Input value for label_embedding.
            - upstream_agent_runs: Input value for upstream_agent_runs.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        self.registry.reset_logs()
        _ = sample
        agent_outputs = {
            "segmentation_preprocess_agent": preprocess,
            "sample_audit_agent": sample_audit,
            "report_generation_agent": report,
            "label_embedding_agent": label_embedding,
        }
        agent_run_dicts = [run.to_dict() for run in upstream_agent_runs]
        review_package = self.registry.call(
            "CollectAgentOutputs",
            agent_outputs=agent_outputs,
            agent_runs=agent_run_dicts,
        )
        completeness = review_package["completeness"]
        preprocess_audit = self.registry.call("AuditPreprocessResult", preprocess=preprocess)
        sample_audit_review = self.registry.call("AuditSampleAuditResult", sample_audit=sample_audit)
        report_audit = self.registry.call("AuditReportResult", report=report)
        term_audit = self.registry.call("AuditTermResult", term_payload=label_embedding)

        named_audits = {
            "workflow": completeness,
            "preprocess": preprocess_audit,
            "sampleAudit": sample_audit_review,
            "report": report_audit,
            "terms": term_audit,
        }
        blocking: list[str] = []
        passed_count = 0
        for audit_name, audit_result in named_audits.items():
            passed = bool(audit_result.get("passed", audit_result.get("complete", False)))
            if passed:
                passed_count += 1
                continue
            issues = [
                *audit_result.get("issues", []),
                *audit_result.get("missing_agents", []),
                *audit_result.get("missing_outputs", []),
            ]
            blocking.extend(f"{audit_name}:{issue}" for issue in (issues or ["failed_audit"]))

        quality = {
            "qualityScore": round(passed_count / max(len(named_audits), 1), 4),
            "blockingIssues": sorted(set(blocking)),
            "warnings": [],
        }

        if not completeness.get("complete"):
            final_status = "needs_human_review"
        elif not preprocess_audit.get("passed"):
            final_status = "retry_preprocess"
        elif not sample_audit_review.get("passed"):
            final_status = "retry_sample_audit"
        elif not report_audit.get("passed"):
            final_status = "retry_report_generation"
        elif not term_audit.get("passed"):
            final_status = "retry_term_embedding"
        elif quality["qualityScore"] >= 1.0:
            final_status = "approved"
        else:
            final_status = "approved_with_warnings"

        retry_targets = {
            "retry_preprocess": "segmentation_preprocess_agent",
            "retry_sample_audit": "sample_audit_agent",
            "retry_report_generation": "report_generation_agent",
            "retry_term_embedding": "label_embedding_agent",
        }
        route = {
            "shouldRetry": final_status in retry_targets,
            "targetAgent": retry_targets.get(final_status, ""),
            "humanReviewRequired": final_status not in {"approved", "approved_with_warnings"},
            "reason": "; ".join(quality["blockingIssues"][:3]),
        }
        if route["shouldRetry"]:
            review_report = f"闭环复核未通过，建议重跑 {route['targetAgent']}。原因：{route['reason']}"
        elif route["humanReviewRequired"]:
            review_report = f"闭环复核需要人工确认。问题：{route['reason']}"
        else:
            review_report = f"闭环复核通过，最终决策为 {final_status}。"

        warnings = list(quality.get("warnings", []))
        warnings.extend(quality.get("blockingIssues", []))
        result = self._annotate({
            "final_status": final_status,
            "finalDecision": final_status,
            "bank_decision": sample_audit.get("bank_decision", "needs_human_review"),
            "label_count": label_embedding.get("label_count", 0),
            "term_count": label_embedding.get("term_count", 0),
            "qualityScore": quality["qualityScore"],
            "blockingIssues": quality["blockingIssues"],
            "warnings": warnings,
            "humanReviewRequired": route["humanReviewRequired"],
            "retryPlan": route,
            "reviewReport": review_report,
            "audits": {
                **named_audits,
            },
        })
        return result, self._finish(status="ok", decision=final_status, observations=result, warnings=warnings)


class MedicalClosedLoopOrchestrator:
    """brief:
        Represent MedicalClosedLoopOrchestrator state and behavior.

    parameter:
        - diagnosis_agent: Input value for diagnosis_agent.
        - pixel_size_mm: Input value for pixel_size_mm.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, diagnosis_agent: DiagnosisAgent | None = None, *, pixel_size_mm: float | None = 0.15):
        """brief:
            Initialize this object.

        parameter:
            - diagnosis_agent: Input value for diagnosis_agent.
            - pixel_size_mm: Input value for pixel_size_mm.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self.registry = create_sample_library_tool_registry()
        self.diagnosis_agent = diagnosis_agent or DiagnosisAgent.from_env(
            use_llm=False,
            use_llm_report=False,
            pixel_size_mm=pixel_size_mm,
        )
        self.preprocess_agent = SegmentationPreprocessAgent(self.registry)
        self.sample_audit_agent = SampleAuditAgent(self.registry)
        self.report_agent = ReportGenerationAgent(self.registry, self.diagnosis_agent)
        self.label_agent = DatabaseTermAgent(self.registry)
        self.review_agent = CrossAgentResultReviewAgent(self.registry)

    def run_sync(
        self,
        case: dict[str, Any],
        reference_sample: dict[str, Any] | None = None,
        doctor_annotations: dict[str, Any] | None = None,
    ) -> ClosedLoopResult:
        """brief:
            Run sync.

        parameter:
            - case: Input value for case.
            - reference_sample: Input value for reference_sample.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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

        review, run = self.review_agent.run(preprocess, sample, sample_audit, report, label_embedding, runs)
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
        """brief:
            Build sample.

        parameter:
            - case: Input value for case.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle image shape.

        parameter:
            - image: Input value for image.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if isinstance(image, np.ndarray) and image.ndim >= 2:
            return int(image.shape[0]), int(image.shape[1])
        return 256, 256

    @staticmethod
    def _bbox_from_mask(mask: Any, width: int, height: int) -> list[int]:
        """brief:
            Handle bbox from mask.

        parameter:
            - mask: Input value for mask.
            - width: Input value for width.
            - height: Input value for height.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if isinstance(mask, np.ndarray) and mask.size:
            binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                return [int(x), int(y), int(x + max(w - 1, 0)), int(y + max(h - 1, 0))]
        return [0, 0, max(width - 1, 0), max(height - 1, 0)]

    @staticmethod
    def _mask_stats(mask: Any, width: int, height: int) -> dict[str, float]:
        """brief:
            Handle mask stats.

        parameter:
            - mask: Input value for mask.
            - width: Input value for width.
            - height: Input value for height.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle annotation tags.

        parameter:
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
        """brief:
            Handle merge preprocess sample.

        parameter:
            - sample: Input value for sample.
            - preprocess: Input value for preprocess.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
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
    """brief:
        Build medical closed loop agent.

    parameter:
        - diagnosis_agent: Input value for diagnosis_agent.
        - pixel_size_mm: Input value for pixel_size_mm.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return MedicalClosedLoopOrchestrator(diagnosis_agent=diagnosis_agent, pixel_size_mm=pixel_size_mm)


__all__ = [
    "ClosedLoopAgentRun",
    "ClosedLoopResult",
    "MedicalClosedLoopOrchestrator",
    "build_medical_closed_loop_agent",
]
