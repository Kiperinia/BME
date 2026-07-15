from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools.medical.sample_library_toolsets import (
    create_sample_library_tool_registry,
    get_primary_agent_tool_chains,
    group_tool_specs_by_agent,
)


def _sample(image_id: str, baseline: float, result: float, group: str = "hard") -> dict[str, object]:
    """brief:
        Handle sample.

    parameter:
        - image_id: Input value for image_id.
        - baseline: Input value for baseline.
        - result: Input value for result.
        - group: Input value for group.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
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
    """brief:
        Run the command-line entry point for this module.

    parameter:
        - None.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    registry = create_sample_library_tool_registry()
    specs = registry.list_tool_specs()
    grouped = group_tool_specs_by_agent(specs)

    assert len(specs) == 66
    assert set(grouped) == {
        "report_generation_agent",
        "sample_audit_agent",
        "segmentation_preprocess_agent",
        "label_embedding_agent",
        "result_review_agent",
    }
    primary_tool_descriptions = {
        "segmentation_preprocess_agent": {
            "BuildBboxRequest": "准备 YOLO 请求",
            "NormalizeImagePlan": "生成图像归一化方案，包括目标尺寸、颜色空间等",
            "TracePreprocess": "记录预处理步骤",
            "PackagePrompts": "打包 SAM3 prompt",
        },
        "sample_audit_agent": {
            "BuildReviewQueueItem": "把可疑样本生成医生/人工复核队列项",
            "RunReferenceLabelQuiz": "把候选样本作为提示词分割并与标准结果比对，输出结果",
        },
        "report_generation_agent": {
            "CaseContextAssembler": "汇总病例与相似样本上下文",
            "UncertaintyExplainer": "解释分割置信度风险来源",
            "ReportTemplateComposer": "生成结构化诊疗报告模板",
        },
        "label_embedding_agent": {
            "ExtractReportTerms": "从 findings、conclusion、医生标注中抽取候选词条",
            "NormalizeMedicalTerms": "把同义词、大小写、中英文混写统一成标准词条",
            "DeduplicateTerms": "合并重复词条和同义词，避免数据库冗余",
            "Build_db_TermRecords": "词条记录构建",
        },
        "result_review_agent": {
            "CollectAgentOutputs": "收集四个上游智能体的输出、决策、warnings、tool calls",
            "AuditPreprocessResult": "检查预处理结果是否合理",
            "AuditSampleAuditResult": "检查样本审核结果是否合理",
            "AuditReportResult": "检查报告生成结果是否合理",
            "AuditTermResult": "检查标签词条结果是否合理",
        },
    }
    for agent_name, descriptions in primary_tool_descriptions.items():
        specs_by_name = {spec["name"]: spec for spec in grouped[agent_name]}
        assert list(descriptions) == [spec["name"] for spec in grouped[agent_name][: len(descriptions)]]
        for tool_name, description in descriptions.items():
            assert specs_by_name[tool_name]["purpose"] == description
    primary_chains = get_primary_agent_tool_chains()
    assert primary_chains["segmentation_preprocess_agent"]["promptDesign"][0].startswith("目标：")
    assert "不新增诊断或反思智能体" in primary_chains["report_generation_agent"]["promptDesign"][1]
    assert "数据库检索而非样本库" in primary_chains["label_embedding_agent"]["promptDesign"][2]

    sample = _sample("case-001", 0.22, 0.64)
    bbox = registry.call("BuildBboxRequest", sample=sample)
    normalization = registry.call("NormalizeImagePlan", sample=sample)
    trace = registry.call("TracePreprocess", sample=sample, steps=[bbox, normalization])
    prompt_package = registry.call("PackagePrompts", sample=sample, use_exemplar=True)
    assert bbox["bbox"] == [10.0, 20.0, 100.0, 120.0]
    assert normalization["target_size"] == 1024
    assert trace["step_count"] == 2
    assert prompt_package["prompts"]["exemplars"]["positive_ids"] == ["pos-1"]

    review_bundle = registry.call("BuildReviewQueueItem", sample=sample)
    quiz_main = registry.call(
        "RunReferenceLabelQuiz",
        sample=sample,
        reference_sample={**sample, "image_id": "reference-001", "tags": ["polyp", "flat"]},
        doctor_annotations={"tags": ["polyp", "flat"], "lesion_type": "polyp"},
    )
    assert review_bundle["review_item"]["priority"] in {"low", "medium", "high"}
    assert quiz_main["passed"]

    context = registry.call("CaseContextAssembler", sample=sample, similar_cases=[], review_summary={})
    uncertainty = registry.call("UncertaintyExplainer", context=context)
    template = registry.call("ReportTemplateComposer", context=context, report_type="clinical")
    assert context["image_id"] == "case-001"
    assert uncertainty["uncertainty_level"] in {"low", "medium", "high"}
    assert template["sections"] == ["finding", "impression", "risk_note", "evidence"]

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

    report_payload = {
        "report_id": "report-001",
        "patient_id": "patient-001",
        "findings": "Paris 0-IIa flat lesion with vessel feature",
        "conclusion": "low risk resection recommendation",
    }
    terms = registry.call(
        "ExtractReportTerms",
        report=report_payload,
        doctor_annotations={"tags": ["polyp"], "paris": "0-IIa", "lesion_type": "polyp"},
    )
    assert terms["term_count"] >= 3

    normalized = registry.call("NormalizeMedicalTerms", terms=terms["terms"])
    deduped = registry.call("DeduplicateTerms", terms=normalized["terms"])
    records = registry.call(
        "Build_db_TermRecords",
        terms=deduped["terms"],
        report=report_payload,
        doctor_annotations={"tags": ["polyp"], "paris": "0-IIa", "lesion_type": "polyp"},
        report_id=report_payload["report_id"],
        patient_id=report_payload["patient_id"],
    )

    assert records["record_count"] == len(records["dbRecords"])
    assert records["validation"]["valid"]
    assert records["routes"]
    assert records["upsert"]["dryRun"] and records["upsert"]["planned"] == records["record_count"]
    assert records["facets"]
    assert 0.0 <= records["coverage"]["coverage"] <= 1.0

    agent_runs = [
        {"agent_name": "segmentation_preprocess_agent", "decision": "preprocessed"},
        {"agent_name": "sample_audit_agent", "decision": "accept"},
        {"agent_name": "report_generation_agent", "decision": "report_ready"},
        {"agent_name": "label_embedding_agent", "decision": "ready_to_index"},
    ]
    agent_outputs = {
        "segmentation_preprocess_agent": {
            "bbox_request": {"bbox": [10, 20, 100, 120]},
            "prompt_package": {"prompts": {"bbox": [10, 20, 100, 120]}},
            "large_mask_gate": {"is_large_mask": False},
        },
        "sample_audit_agent": {
            "accepted": True,
            "bank_decision": "accept",
            "reference_quiz": {"passed": True},
            "mask_consistency": {"valid": True},
        },
        "report_generation_agent": {**report_payload, "report_score": {"overall_score": 8.5}},
        "label_embedding_agent": {
            "dbRecords": records["dbRecords"],
            "validation": records["validation"],
            "coverage": records["coverage"],
            "decision": "ready_to_index",
        },
    }
    package = registry.call("CollectAgentOutputs", agent_outputs=agent_outputs, agent_runs=agent_runs)
    workflow = package["completeness"]
    preprocess_audit = registry.call("AuditPreprocessResult", preprocess=agent_outputs["segmentation_preprocess_agent"])
    sample_audit = registry.call("AuditSampleAuditResult", sample_audit=agent_outputs["sample_audit_agent"])
    report_audit = registry.call("AuditReportResult", report=agent_outputs["report_generation_agent"])
    term_audit = registry.call("AuditTermResult", term_payload=agent_outputs["label_embedding_agent"])

    assert package["agent_count"] == 4
    assert workflow["complete"]
    assert preprocess_audit["passed"]
    assert sample_audit["passed"]
    assert report_audit["passed"]
    assert term_audit["passed"] or term_audit["issues"]

    print("sample-library-toolsets-smoke: ok")


if __name__ == "__main__":
    main()
