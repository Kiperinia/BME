from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.medical.sample_library_toolsets import (
    ResultReviewToolSet,
    create_sample_library_tool_registry,
    group_tool_specs_by_agent,
)


def _sample(image_id: str, baseline: float, result: float, group: str = "hard") -> dict[str, object]:
    return {
        "image_id": image_id,
        "site_id": "C1",
        "split": "external",
        "sample_group": group,
        "bbox": [10, 20, 100, 120],
        "metrics": {
            "Dice": result,
            "Precision": 0.82,
            "Recall": 0.75,
            "Boundary F1": 0.61,
            "mean confidence": 0.88,
        },
        "baseline_metrics": {"Dice": baseline},
        "mask_stats": {
            "area_ratio": 0.045,
            "boundary_complexity": 0.62,
            "components": 2,
        },
        "uncertainty": {
            "mean_entropy": 0.28,
            "mean_confidence": 0.88,
        },
        "selected_exemplars": {
            "positive_ids": ["pos-1"],
            "negative_ids": ["neg-1"],
            "boundary_ids": ["bd-1"],
        },
        "tags": ["polyp", "flat"],
    }


def main() -> None:
    registry = create_sample_library_tool_registry()
    specs = registry.list_tool_specs()
    grouped = group_tool_specs_by_agent(specs)

    assert len(specs) == 41
    assert set(grouped) == {
        "report_generation_agent",
        "sample_audit_agent",
        "segmentation_preprocess_agent",
        "label_embedding_agent",
        "result_review_agent",
    }

    sample = _sample("case-001", 0.22, 0.64)
    grade = registry.call("assign_sample_grade", sample=sample)
    assert grade["grade"] == "hard"

    quiz = registry.call(
        "run_reference_label_quiz",
        sample=sample,
        reference_sample={**sample, "image_id": "reference-001", "tags": ["polyp", "flat"]},
        doctor_annotations={"tags": ["polyp", "flat"], "lesion_type": "polyp"},
    )
    assert quiz["passed"]

    prompt_package = registry.call("package_prompts", sample=sample, use_exemplar=True)
    assert prompt_package["prompts"]["exemplars"]["positive_ids"] == ["pos-1"]

    labels = registry.call(
        "extract_report_feature_labels",
        report={"findings": "Paris 0-IIa flat lesion with vessel feature", "conclusion": "low risk resection"},
        doctor_annotations={"tags": ["polyp"]},
    )
    assert "Paris type" in labels["labels"]

    effect = registry.call("audit_exemplar_effect", sample=sample)
    assert effect["effect"] == "helpful"

    rows = [
        _sample("case-001", 0.22, 0.64),
        _sample("case-002", 0.81, 0.80, "clean"),
        _sample("case-003", 0.10, 0.12),
    ]
    report = ResultReviewToolSet.generate_hard_case_delta_report(rows=rows)
    assert report["count"] == 3
    assert report["threshold_subsets"]["baseline_dice<0.3"]["count"] == 2
    assert report["overall"]["positive_delta_rate"] == 2 / 3

    print("sample-library-toolsets-smoke: ok")


if __name__ == "__main__":
    main()
