from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import build_medical_closed_loop_agent


def _synthetic_case() -> dict[str, object]:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (30, 52, 88)
    cv2.circle(image, (128, 128), 42, (25, 55, 220), -1)
    cv2.ellipse(image, (128, 128), (52, 28), 12, 0, 360, (12, 30, 235), 4)

    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 42, 255, -1)

    return {
        "image": image,
        "mask": mask,
        "bbox": (86, 86, 170, 170),
        "lesion_id": "closed-loop-case-001",
        "patient_context": {
            "patient_id": "demo-patient",
            "study_id": "demo-study",
            "exam_date": "2026-06-12",
        },
        "sample": {
            "image_id": "closed-loop-case-001",
            "site_id": "C1",
            "split": "runtime",
            "sample_group": "candidate",
            "bbox": [86, 86, 170, 170],
            "metrics": {
                "Dice": 0.84,
                "Precision": 0.86,
                "Recall": 0.82,
                "Boundary F1": 0.73,
                "mean confidence": 0.84,
            },
            "baseline_metrics": {"Dice": 0.73},
            "mask_stats": {
                "area_ratio": 0.084,
                "aspect_ratio": 1.0,
                "boundary_complexity": 0.42,
                "solidity": 0.92,
                "components": 1,
            },
            "uncertainty": {
                "mean_entropy": 0.2,
                "mean_confidence": 0.84,
            },
            "selected_exemplars": {
                "positive_ids": ["pos-1"],
                "negative_ids": [],
                "boundary_ids": ["bd-1"],
            },
            "tags": ["polyp", "flat", "0-IIa"],
        },
    }


def main() -> None:
    doctor_annotations = {
        "paris": "0-IIa",
        "lesion_type": "polyp",
        "pathology": "adenoma",
        "surface_pattern": "smooth",
        "notes": "doctor marked as useful training sample",
        "tags": ["polyp", "flat", "adenoma"],
    }
    reference_sample = {
        "image_id": "objective-reference-001",
        "site_id": "C1",
        "split": "reference",
        "sample_group": "clean",
        "metrics": {"Dice": 0.86},
        "baseline_metrics": {"Dice": 0.78},
        "mask_stats": {"area_ratio": 0.08, "boundary_complexity": 0.38, "components": 1},
        "tags": ["polyp", "flat", "adenoma", "0-IIa"],
    }

    orchestrator = build_medical_closed_loop_agent(pixel_size_mm=0.15)
    result = orchestrator.run_sync(
        _synthetic_case(),
        reference_sample=reference_sample,
        doctor_annotations=doctor_annotations,
    ).to_dict()

    assert set(result) == {
        "preprocess",
        "sample_audit",
        "report",
        "label_embedding",
        "review",
        "agent_runs",
    }
    assert [run["agent_name"] for run in result["agent_runs"]] == [
        "segmentation_preprocess_agent",
        "sample_audit_agent",
        "report_generation_agent",
        "label_embedding_agent",
        "result_review_agent",
    ]
    expected_tool_chains = {
        "segmentation_preprocess_agent": ["BuildBboxRequest", "NormalizeImagePlan", "TracePreprocess", "PackagePrompts"],
        "sample_audit_agent": ["BuildReviewQueueItem", "RunReferenceLabelQuiz"],
        "report_generation_agent": ["CaseContextAssembler", "UncertaintyExplainer", "ReportTemplateComposer"],
        "label_embedding_agent": ["ExtractReportTerms", "NormalizeMedicalTerms", "DeduplicateTerms", "Build_db_TermRecords"],
        "result_review_agent": ["CollectAgentOutputs", "AuditPreprocessResult", "AuditSampleAuditResult", "AuditReportResult", "AuditTermResult"],
    }
    for run in result["agent_runs"]:
        tool_names = [call["tool_name"] for call in run["tool_calls"]]
        assert tool_names == expected_tool_chains[run["agent_name"]]
        assert run["observations"]["agentDetail"]
        assert run["observations"]["promptDesign"]
        assert [tool["name"] for tool in run["observations"]["mainToolChain"]] == expected_tool_chains[run["agent_name"]]
    assert result["report"]["findings"]
    assert result["report"]["conclusion"]
    assert result["label_embedding"]["labels"]
    assert result["label_embedding"]["dbRecords"]
    assert result["label_embedding"]["facets"]
    assert result["label_embedding"]["decision"] in {"ready_to_index", "needs_term_review", "insufficient_terms"}
    assert result["sample_audit"]["reference_quiz"]["passed"]
    assert result["review"]["finalDecision"] in {
        "approved",
        "approved_with_warnings",
        "needs_human_review",
        "retry_preprocess",
        "retry_sample_audit",
        "retry_report_generation",
        "retry_term_embedding",
        "rejected",
    }
    assert result["review"]["audits"]["workflow"]["complete"]

    print("medical-closed-loop-agents-smoke: ok")


if __name__ == "__main__":
    main()
