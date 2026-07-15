from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Callable, Iterable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """brief:
        Handle clamp.

    parameter:
        - value: Input value for value.
        - low: Input value for low.
        - high: Input value for high.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return max(low, min(high, value))


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    """brief:
        Handle safe float.

    parameter:
        - payload: Input value for payload.
        - key: Input value for key.
        - default: Input value for default.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    value = payload.get(key, default)
    if value is None:
        return default
    return float(value)


def _dice(row: dict[str, Any], field: str = "metrics") -> float:
    """brief:
        Handle dice.

    parameter:
        - row: Input value for row.
        - field: Input value for field.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    payload = row.get(field, {})
    if isinstance(payload, dict):
        return _safe_float(payload, "Dice")
    return 0.0


def _as_list(value: Any) -> list[Any]:
    """brief:
        Handle as list.

    parameter:
        - value: Input value for value.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass(slots=True)
class SampleLibraryRecord:
    """brief:
        Represent SampleLibraryRecord state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    image_id: str
    site_id: str = ""
    split: str = ""
    fold: int | None = None
    sample_group: str = "candidate"
    image_path: str = ""
    mask_path: str = ""
    bbox: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    mask_stats: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    selected_exemplars: dict[str, list[str]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SampleLibraryRecord":
        """brief:
            Handle from mapping.

        parameter:
            - payload: Input value for payload.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return cls(
            image_id=str(payload.get("image_id", "")),
            site_id=str(payload.get("site_id", payload.get("site", ""))),
            split=str(payload.get("split", "")),
            fold=payload.get("fold"),
            sample_group=str(payload.get("sample_group", "candidate")),
            image_path=str(payload.get("image_path", "")),
            mask_path=str(payload.get("mask_path", "")),
            bbox=[float(v) for v in _as_list(payload.get("bbox"))],
            metrics=dict(payload.get("metrics", {})),
            baseline_metrics=dict(payload.get("baseline_metrics", {})),
            mask_stats=dict(payload.get("mask_stats", {})),
            uncertainty=dict(payload.get("uncertainty", {})),
            selected_exemplars=dict(payload.get("selected_exemplars", {})),
            tags=[str(tag) for tag in _as_list(payload.get("tags"))],
            metadata=dict(payload.get("metadata", {})),
        )

    @property
    def baseline_dice(self) -> float:
        """brief:
            Handle baseline dice.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return _safe_float(self.baseline_metrics, "Dice")

    @property
    def result_dice(self) -> float:
        """brief:
            Handle result dice.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return _safe_float(self.metrics, "Dice")

    @property
    def delta_dice(self) -> float:
        """brief:
            Handle delta dice.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return self.result_dice - self.baseline_dice

    def to_dict(self) -> dict[str, Any]:
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return asdict(self)


@dataclass(slots=True)
class ToolExplanation:
    """brief:
        Represent ToolExplanation state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    name: str
    agent: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    sample_library_role: str

    def to_dict(self) -> dict[str, Any]:
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return asdict(self)


@dataclass(slots=True)
class ToolCallLog:
    """brief:
        Represent ToolCallLog state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    tool_name: str
    status: str
    duration_ms: float
    output_preview: str
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """brief:
            Handle to dict.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "output_preview": self.output_preview[:240],
            "error_message": self.error_message,
        }


PRIMARY_AGENT_TOOL_CHAINS: dict[str, dict[str, Any]] = {
    "segmentation_preprocess_agent": {
        "displayName": "分割预处理智能体",
        "agentDetail": "准备 YOLO bbox 请求、归一化计划和 prompt 包",
        "promptDesign": [
            "目标：围绕 polyp lesion 组织视觉提示，不生成诊断或报告文字。",
            "空间：优先使用 YOLO bbox；缺失时使用 mask-derived/full-frame bbox。",
            "参考：把 positive/boundary exemplar ids 放入 SAM3 prompt 包，服务边界定位。",
        ],
        "mainToolChain": [
            {"name": "BuildBboxRequest", "description": "准备 YOLO 请求"},
            {"name": "NormalizeImagePlan", "description": "生成图像归一化方案，包括目标尺寸、颜色空间等"},
            {"name": "TracePreprocess", "description": "记录预处理步骤"},
            {"name": "PackagePrompts", "description": "打包 SAM3 prompt"},
        ],
    },
    "sample_audit_agent": {
        "displayName": "样本审核智能体",
        "agentDetail": "判断分割样本是否值得进入样本库作为视觉提示词，输出 accept/reject",
        "promptDesign": [
            "输入：候选样本标签、标准样本标签、医生标注和 mask 质量指标。",
            "任务：像参考标签测验一样比对候选样本是否可作为视觉提示词。",
            "输出：passed、score、reasons，并汇总为 accept/reject/human review。",
        ],
        "mainToolChain": [
            {"name": "BuildReviewQueueItem", "description": "把可疑样本生成医生/人工复核队列项"},
            {"name": "RunReferenceLabelQuiz", "description": "把候选样本作为提示词分割并与标准结果比对，输出结果"},
        ],
    },
    "report_generation_agent": {
        "displayName": "报告生成智能体",
        "agentDetail": "生成结构化报告，结合分割证据和医生标注",
        "promptDesign": [
            "输入：病例上下文、分割指标、mask 统计、医生标注和不确定性。",
            "任务：只生成 findings、conclusion、layoutSuggestion，不新增诊断或反思智能体。",
            "约束：每个结论必须能回到分割证据、医生标注或病例上下文。",
        ],
        "mainToolChain": [
            {"name": "CaseContextAssembler", "description": "汇总病例与相似样本上下文"},
            {"name": "UncertaintyExplainer", "description": "解释分割置信度风险来源"},
            {"name": "ReportTemplateComposer", "description": "生成结构化诊疗报告模板"},
        ],
    },
    "label_embedding_agent": {
        "displayName": "标签嵌入智能体",
        "agentDetail": "从报告和医生标注提取标签",
        "promptDesign": [
            "输入：findings、conclusion 和医生标注。",
            "任务：抽取可检索医学词条，统一中英文、同义词和大小写。",
            "输出：dbRecords、routes、facets、validation，目标是数据库检索而非样本库。",
        ],
        "mainToolChain": [
            {"name": "ExtractReportTerms", "description": "从 findings、conclusion、医生标注中抽取候选词条"},
            {"name": "NormalizeMedicalTerms", "description": "把同义词、大小写、中英文混写统一成标准词条"},
            {"name": "DeduplicateTerms", "description": "合并重复词条和同义词，避免数据库冗余"},
            {"name": "Build_db_TermRecords", "description": "词条记录构建"},
        ],
    },
    "result_review_agent": {
        "displayName": "结果复核智能体",
        "agentDetail": "复核全链路输出质量",
        "promptDesign": [
            "输入：四个上游智能体输出、决策、warnings 和 tool calls。",
            "任务：审核链路完整性、预处理、样本审核、报告和词条结果。",
            "输出：finalDecision、qualityScore、blockingIssues、retryPlan，不做疾病诊断。",
        ],
        "mainToolChain": [
            {"name": "CollectAgentOutputs", "description": "收集四个上游智能体的输出、决策、warnings、tool calls"},
            {"name": "AuditPreprocessResult", "description": "检查预处理结果是否合理"},
            {"name": "AuditSampleAuditResult", "description": "检查样本审核结果是否合理"},
            {"name": "AuditReportResult", "description": "检查报告生成结果是否合理"},
            {"name": "AuditTermResult", "description": "检查标签词条结果是否合理"},
        ],
    },
}


def get_primary_agent_tool_chains() -> dict[str, dict[str, Any]]:
    """brief:
        Get primary agent tool chains.

    parameter:
        - None.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return {
        agent_name: {
            "displayName": str(metadata["displayName"]),
            "agentDetail": str(metadata["agentDetail"]),
            "promptDesign": [str(item) for item in metadata["promptDesign"]],
            "mainToolChain": [dict(tool) for tool in metadata["mainToolChain"]],
        }
        for agent_name, metadata in PRIMARY_AGENT_TOOL_CHAINS.items()
    }


class SampleLibraryToolRegistry:
    """brief:
        Represent SampleLibraryToolRegistry state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self) -> None:
        """brief:
            Initialize this object.

        parameter:
            - None.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self._tools: dict[str, tuple[ToolExplanation, Callable[..., Any]]] = {}
        self._logs: list[ToolCallLog] = []

    def register(self, explanation: ToolExplanation, handler: Callable[..., Any]) -> None:
        """brief:
            Handle register.

        parameter:
            - explanation: Input value for explanation.
            - handler: Input value for handler.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self._tools[explanation.name] = (explanation, handler)

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """brief:
            Handle call.

        parameter:
            - tool_name: Input value for tool_name.
            - **kwargs: Input value for kwargs.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown sample-library tool: {tool_name}")
        _, handler = self._tools[tool_name]
        started = time.perf_counter()
        try:
            result = handler(**kwargs)
            self._logs.append(
                ToolCallLog(
                    tool_name=tool_name,
                    status="ok",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    output_preview=repr(result),
                )
            )
            return result
        except Exception as exc:
            self._logs.append(
                ToolCallLog(
                    tool_name=tool_name,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    output_preview="",
                    error_message=str(exc),
                )
            )
            raise

    def list_tool_specs(self) -> list[dict[str, Any]]:
        """brief:
            List tool specs.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return [explanation.to_dict() for explanation, _ in self._tools.values()]

    def get_call_logs(self) -> list[dict[str, Any]]:
        """brief:
            Get call logs.

        parameter:
            - None.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return [log.to_dict() for log in self._logs]

    def reset_logs(self) -> None:
        """brief:
            Handle reset logs.

        parameter:
            - None.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        self._logs = []


class ReportGenerationToolSet:
    """brief:
        Represent ReportGenerationToolSet state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "report_generation_agent"

    @staticmethod
    def assemble_case_context(
        *,
        sample: dict[str, Any],
        similar_cases: list[dict[str, Any]] | None = None,
        review_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """brief:
            Handle assemble case context.

        parameter:
            - sample: Input value for sample.
            - similar_cases: Input value for similar_cases.
            - review_summary: Input value for review_summary.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {
            "image_id": record.image_id,
            "site_id": record.site_id,
            "split": record.split,
            "sample_group": record.sample_group,
            "dice": record.result_dice,
            "baseline_dice": record.baseline_dice,
            "delta_dice": record.delta_dice,
            "mask_stats": record.mask_stats,
            "uncertainty": record.uncertainty,
            "selected_exemplars": record.selected_exemplars,
            "similar_case_count": len(similar_cases or []),
            "similar_cases": similar_cases or [],
            "review_summary": review_summary or {},
        }

    @staticmethod
    def retrieve_similar_cases(
        *,
        query: dict[str, Any],
        library: list[dict[str, Any]],
        top_k: int = 5,
        prefer_groups: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """brief:
            Handle retrieve similar cases.

        parameter:
            - query: Input value for query.
            - library: Input value for library.
            - top_k: Input value for top_k.
            - prefer_groups: Input value for prefer_groups.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        query_record = SampleLibraryRecord.from_mapping(query)
        preferred = set(prefer_groups or ["hard", "boundary", "positive"])
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in library:
            record = SampleLibraryRecord.from_mapping(item)
            tag_overlap = len(set(query_record.tags) & set(record.tags))
            dice_distance = abs(query_record.baseline_dice - record.baseline_dice)
            group_bonus = 0.25 if record.sample_group in preferred else 0.0
            site_bonus = 0.10 if record.site_id and record.site_id == query_record.site_id else 0.0
            score = group_bonus + site_bonus + 0.08 * tag_overlap - dice_distance
            ranked.append((score, {**record.to_dict(), "similarity_score": round(score, 4)}))
        return [payload for _, payload in sorted(ranked, key=lambda item: item[0], reverse=True)[:top_k]]

    @staticmethod
    def compose_report_template(*, context: dict[str, Any], report_type: str = "segmentation_review") -> dict[str, Any]:
        """brief:
            Handle compose report template.

        parameter:
            - context: Input value for context.
            - report_type: Input value for report_type.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        sections = ["case_summary", "segmentation_result", "uncertainty", "evidence", "review_recommendation"]
        if report_type == "clinical":
            sections = ["finding", "impression", "risk_note", "evidence"]
        return {
            "report_type": report_type,
            "image_id": context.get("image_id", ""),
            "sections": sections,
            "required_evidence": ["mask", "metrics", "similar_cases"],
        }

    @staticmethod
    def narrate_findings(*, context: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle narrate findings.

        parameter:
            - context: Input value for context.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        dice = float(context.get("dice", 0.0))
        delta = float(context.get("delta_dice", 0.0))
        area_ratio = float(context.get("mask_stats", {}).get("area_ratio", 0.0))
        quality = "high" if dice >= 0.85 else "moderate" if dice >= 0.65 else "low"
        direction = "improved" if delta > 0.03 else "regressed" if delta < -0.03 else "stable"
        return {
            "quality_band": quality,
            "delta_direction": direction,
            "finding_facts": [
                f"Dice={dice:.4f}",
                f"baseline Dice={float(context.get('baseline_dice', 0.0)):.4f}",
                f"delta Dice={delta:.4f}",
                f"mask area ratio={area_ratio:.4f}",
            ],
        }

    @staticmethod
    def explain_uncertainty(*, context: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle explain uncertainty.

        parameter:
            - context: Input value for context.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        uncertainty = context.get("uncertainty", {})
        mean_entropy = float(uncertainty.get("mean_entropy", 0.0))
        confidence = float(uncertainty.get("mean_confidence", context.get("metrics", {}).get("mean confidence", 0.0)))
        reasons: list[str] = []
        if confidence < 0.65:
            reasons.append("low_confidence")
        if mean_entropy > 0.35:
            reasons.append("high_entropy")
        if float(context.get("baseline_dice", 1.0)) < 0.5:
            reasons.append("low_baseline_dice")
        return {
            "uncertainty_level": "high" if len(reasons) >= 2 else "medium" if reasons else "low",
            "reasons": reasons,
            "mean_entropy": mean_entropy,
            "mean_confidence": confidence,
        }

    @staticmethod
    def bind_evidence(*, context: dict[str, Any], statements: list[str]) -> list[dict[str, Any]]:
        """brief:
            Handle bind evidence.

        parameter:
            - context: Input value for context.
            - statements: Input value for statements.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        evidence = []
        for statement in statements:
            evidence.append(
                {
                    "statement": statement,
                    "image_id": context.get("image_id", ""),
                    "evidence_refs": ["metrics", "mask_stats", "selected_exemplars"],
                    "similar_case_ids": [case.get("image_id", "") for case in context.get("similar_cases", [])[:3]],
                }
            )
        return evidence

    @staticmethod
    def flag_report_risks(*, context: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle flag report risks.

        parameter:
            - context: Input value for context.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        flags: list[str] = []
        if float(context.get("dice", 0.0)) < 0.5:
            flags.append("low_result_dice")
        if float(context.get("delta_dice", 0.0)) <= -0.03:
            flags.append("regression")
        if context.get("sample_group") in {"ambiguous", "reject"}:
            flags.append("sample_quality_risk")
        return {"risk_flags": flags, "needs_human_review": bool(flags)}


class CaseContextAssembler:
    """brief:
        Represent CaseContextAssembler state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    description = "汇总病例与相似样本上下文"

    def __call__(
        self,
        *,
        sample: dict[str, Any],
        similar_cases: list[dict[str, Any]] | None = None,
        review_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - similar_cases: Input value for similar_cases.
            - review_summary: Input value for review_summary.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ReportGenerationToolSet.assemble_case_context(
            sample=sample,
            similar_cases=similar_cases,
            review_summary=review_summary,
        )


class UncertaintyExplainer:
    """brief:
        Represent UncertaintyExplainer state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    description = "解释分割置信度风险来源"

    def __call__(self, *, context: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - context: Input value for context.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ReportGenerationToolSet.explain_uncertainty(context=context)


class ReportTemplateComposer:
    """brief:
        Represent ReportTemplateComposer state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    description = "生成结构化诊疗报告模板"

    def __call__(self, *, context: dict[str, Any], report_type: str = "segmentation_review") -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - context: Input value for context.
            - report_type: Input value for report_type.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ReportGenerationToolSet.compose_report_template(context=context, report_type=report_type)


def _register_report_generation_primary_tools(registry: SampleLibraryToolRegistry) -> None:
    """brief:
        Register report generation primary tools.

    parameter:
        - registry: Input value for registry.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    tools: list[tuple[str, Callable[..., Any], list[str], list[str], str]] = [
        (
            "CaseContextAssembler",
            CaseContextAssembler(),
            ["sample", "similar_cases", "review_summary"],
            ["case_context"],
            "Builds report-ready case evidence from sample metadata and related cases.",
        ),
        (
            "UncertaintyExplainer",
            UncertaintyExplainer(),
            ["context"],
            ["uncertainty_summary"],
            "Explains confidence and entropy risks before report writing.",
        ),
        (
            "ReportTemplateComposer",
            ReportTemplateComposer(),
            ["context", "report_type"],
            ["template"],
            "Chooses the structured report skeleton for the case.",
        ),
    ]
    for name, handler, inputs, outputs, role in tools:
        registry.register(
            ToolExplanation(
                name=name,
                agent=ReportGenerationToolSet.agent_name,
                purpose=str(getattr(handler, "description")),
                inputs=inputs,
                outputs=outputs,
                sample_library_role=role,
            ),
            handler,
        )


class SampleAuditToolSet:
    """brief:
        Represent SampleAuditToolSet state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "sample_audit_agent"

    @staticmethod
    def check_identity(*, sample: dict[str, Any], known_ids: list[str] | None = None) -> dict[str, Any]:
        """brief:
            Handle check identity.

        parameter:
            - sample: Input value for sample.
            - known_ids: Input value for known_ids.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        issues: list[str] = []
        if not record.image_id:
            issues.append("missing_image_id")
        if known_ids and record.image_id in set(known_ids):
            issues.append("duplicate_image_id")
        if not record.site_id:
            issues.append("missing_site_id")
        return {"valid": not issues, "issues": issues, "image_id": record.image_id}

    @staticmethod
    def check_label_mask_consistency(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle check label mask consistency.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        components = int(record.mask_stats.get("components", 1) or 1)
        issues: list[str] = []
        if area_ratio <= 0.0:
            issues.append("empty_mask")
        if area_ratio > 0.55:
            issues.append("oversized_mask")
        if components > 8:
            issues.append("fragmented_mask")
        return {"valid": not issues, "issues": issues, "area_ratio": area_ratio, "components": components}

    @staticmethod
    def audit_site_leakage(*, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle audit site leakage.

        parameter:
            - samples: Input value for samples.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        seen: dict[str, set[str]] = {}
        for item in samples:
            record = SampleLibraryRecord.from_mapping(item)
            if not record.site_id:
                continue
            seen.setdefault(record.site_id, set()).add(record.split)
        leakage = {site: sorted(splits) for site, splits in seen.items() if len(splits - {""}) > 1}
        return {"leakage_found": bool(leakage), "site_split_map": {k: sorted(v) for k, v in seen.items()}, "leakage": leakage}

    @staticmethod
    def mine_hard_case(*, sample: dict[str, Any], dice_threshold: float = 0.7, harm_threshold: float = -0.03) -> dict[str, Any]:
        """brief:
            Handle mine hard case.

        parameter:
            - sample: Input value for sample.
            - dice_threshold: Input value for dice_threshold.
            - harm_threshold: Input value for harm_threshold.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        reasons: list[str] = []
        if record.baseline_dice < dice_threshold:
            reasons.append("low_baseline_dice")
        if record.result_dice < dice_threshold:
            reasons.append("low_result_dice")
        if record.delta_dice <= harm_threshold:
            reasons.append("regression")
        return {"is_hard_case": bool(reasons), "reasons": reasons, "delta_dice": record.delta_dice}

    @staticmethod
    def detect_boundary_case(*, sample: dict[str, Any], boundary_threshold: float = 0.55) -> dict[str, Any]:
        """brief:
            Handle detect boundary case.

        parameter:
            - sample: Input value for sample.
            - boundary_threshold: Input value for boundary_threshold.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        complexity = _safe_float(record.mask_stats, "boundary_complexity")
        boundary_f1 = _safe_float(record.metrics, "Boundary F1", 1.0)
        is_boundary = complexity >= boundary_threshold or boundary_f1 < 0.65
        return {
            "is_boundary_case": is_boundary,
            "boundary_complexity": complexity,
            "boundary_f1": boundary_f1,
            "reasons": [reason for reason, active in {"complex_boundary": complexity >= boundary_threshold, "low_boundary_f1": boundary_f1 < 0.65}.items() if active],
        }

    @staticmethod
    def validate_negative_sample(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Validate negative sample.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        confidence = _safe_float(record.uncertainty, "mean_confidence")
        suspicious = area_ratio > 0.002 or confidence > 0.75
        return {"is_valid_negative": not suspicious, "suspicious": suspicious, "area_ratio": area_ratio, "confidence": confidence}

    @staticmethod
    def build_review_queue_item(*, sample: dict[str, Any], audit_results: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Build review queue item.

        parameter:
            - sample: Input value for sample.
            - audit_results: Input value for audit_results.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        reasons = [reason for result in audit_results for reason in result.get("issues", []) + result.get("reasons", [])]
        priority = "high" if any(reason in {"empty_mask", "regression", "duplicate_image_id"} for reason in reasons) else "medium" if reasons else "low"
        return {"image_id": record.image_id, "priority": priority, "reasons": sorted(set(reasons)), "sample_group": record.sample_group}

    @staticmethod
    def assign_sample_grade(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle assign sample grade.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        boundary_complexity = _safe_float(record.mask_stats, "boundary_complexity")
        if area_ratio <= 0.0:
            grade = "reject"
        elif record.baseline_dice < 0.5 or record.delta_dice <= -0.03:
            grade = "hard"
        elif boundary_complexity >= 0.55:
            grade = "boundary"
        elif record.split == "external":
            grade = "external-only"
        else:
            grade = "clean"
        return {"image_id": record.image_id, "grade": grade}

    @staticmethod
    def run_reference_label_quiz(
        *,
        sample: dict[str, Any],
        reference_sample: dict[str, Any] | None = None,
        doctor_annotations: dict[str, Any] | None = None,
        pass_threshold: float = 0.55,
    ) -> dict[str, Any]:
        """brief:
            Run reference label quiz.

        parameter:
            - sample: Input value for sample.
            - reference_sample: Input value for reference_sample.
            - doctor_annotations: Input value for doctor_annotations.
            - pass_threshold: Input value for pass_threshold.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        reference = SampleLibraryRecord.from_mapping(reference_sample or {})
        annotations = doctor_annotations or {}

        candidate_tags = {tag.lower() for tag in record.tags}
        reference_tags = {tag.lower() for tag in reference.tags}
        doctor_tags = {
            str(value).strip().lower()
            for value in [
                annotations.get("lesion_type"),
                annotations.get("pathology"),
                annotations.get("surface_pattern"),
                annotations.get("paris"),
                *list(annotations.get("tags", []) or []),
            ]
            if str(value).strip()
        }

        target_tags = reference_tags | doctor_tags
        overlap = len(candidate_tags & target_tags)
        tag_score = overlap / max(len(target_tags), 1) if target_tags else 0.6
        mask_score = 1.0
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        if area_ratio <= 0.0 or area_ratio > 0.55:
            mask_score = 0.0
        elif area_ratio > 0.35:
            mask_score = 0.45

        reference_delta = 0.0
        if reference.image_id:
            reference_delta = abs(record.result_dice - reference.result_dice)
        quiz_score = _clamp(0.5 * tag_score + 0.35 * mask_score + 0.15 * max(0.0, 1.0 - reference_delta))

        reasons: list[str] = []
        if not candidate_tags:
            reasons.append("missing_candidate_tags")
        if target_tags and overlap == 0:
            reasons.append("no_overlap_with_reference_or_doctor_labels")
        if mask_score < 0.5:
            reasons.append("mask_quality_failed")
        if reference.image_id and reference_delta > 0.35:
            reasons.append("reference_performance_gap")

        return {
            "image_id": record.image_id,
            "passed": quiz_score >= pass_threshold and not any(reason == "mask_quality_failed" for reason in reasons),
            "score": round(quiz_score, 4),
            "pass_threshold": pass_threshold,
            "matched_tag_count": overlap,
            "candidate_tags": sorted(candidate_tags),
            "target_tags": sorted(target_tags),
            "reasons": reasons,
        }


class SegmentationPreprocessToolSet:
    """brief:
        Represent SegmentationPreprocessToolSet state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "segmentation_preprocess_agent"

    @staticmethod
    def normalize_image_plan(*, sample: dict[str, Any], target_size: int = 1024) -> dict[str, Any]:
        """brief:
            Handle normalize image plan.

        parameter:
            - sample: Input value for sample.
            - target_size: Input value for target_size.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "target_size": target_size, "color_space": "RGB", "scale_mode": "long_side_pad"}

    @staticmethod
    def build_bbox_cache_request(*, sample: dict[str, Any], detector: str = "yolo") -> dict[str, Any]:
        """brief:
            Build bbox cache request.

        parameter:
            - sample: Input value for sample.
            - detector: Input value for detector.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "detector": detector, "use_cached": bool(record.bbox), "bbox": record.bbox}

    @staticmethod
    def package_prompts(*, sample: dict[str, Any], use_text: bool = True, use_box: bool = True, use_exemplar: bool = False) -> dict[str, Any]:
        """brief:
            Handle package prompts.

        parameter:
            - sample: Input value for sample.
            - use_text: Input value for use_text.
            - use_box: Input value for use_box.
            - use_exemplar: Input value for use_exemplar.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        prompts: dict[str, Any] = {}
        if use_text:
            prompts["text"] = "polyp lesion"
        if use_box and record.bbox:
            prompts["box"] = record.bbox
        if use_exemplar:
            prompts["exemplars"] = record.selected_exemplars
        return {"image_id": record.image_id, "prompts": prompts, "prompt_modes": sorted(prompts)}

    @staticmethod
    def generate_mask_prior_plan(*, sample: dict[str, Any], similar_cases: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle generate mask prior plan.

        parameter:
            - sample: Input value for sample.
            - similar_cases: Input value for similar_cases.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        selected = [case.get("image_id", "") for case in similar_cases if case.get("mask_path")][:3]
        return {"image_id": record.image_id, "prior_type": "similar_case_mask", "source_case_ids": selected, "enabled": bool(selected)}

    @staticmethod
    def scan_region_uncertainty(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle scan region uncertainty.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        entropy = _safe_float(record.uncertainty, "mean_entropy")
        confidence = _safe_float(record.uncertainty, "mean_confidence", 1.0)
        return {"needs_region_attention": entropy > 0.35 or confidence < 0.65, "mean_entropy": entropy, "mean_confidence": confidence}

    @staticmethod
    def guard_small_lesion(*, sample: dict[str, Any], min_area_ratio: float = 0.002) -> dict[str, Any]:
        """brief:
            Handle guard small lesion.

        parameter:
            - sample: Input value for sample.
            - min_area_ratio: Input value for min_area_ratio.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        return {"is_small_lesion": 0.0 < area_ratio < min_area_ratio, "recommended_scale": 1.5 if 0.0 < area_ratio < min_area_ratio else 1.0}

    @staticmethod
    def gate_large_mask(*, sample: dict[str, Any], max_area_ratio: float = 0.35) -> dict[str, Any]:
        """brief:
            Handle gate large mask.

        parameter:
            - sample: Input value for sample.
            - max_area_ratio: Input value for max_area_ratio.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        return {"is_large_mask": area_ratio > max_area_ratio, "use_exemplar_guard": area_ratio > max_area_ratio, "area_ratio": area_ratio}

    @staticmethod
    def trace_preprocess(*, sample: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle trace preprocess.

        parameter:
            - sample: Input value for sample.
            - steps: Input value for steps.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "step_count": len(steps), "steps": steps}


class LabelEmbeddingToolSet:
    """brief:
        Represent LabelEmbeddingToolSet state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "label_embedding_agent"

    _report_label_patterns: tuple[tuple[str, str], ...] = (
        (r"0-I[p|s]|0-II[a-c]|0-III", "Paris type"),
        (r"flat|扁平|平坦", "flat morphology"),
        (r"elevated|隆起", "elevated morphology"),
        (r"depressed|凹陷", "depressed morphology"),
        (r"red|充血|发红", "erythema"),
        (r"vessel|血管", "vascular feature"),
        (r"浸润|invasion", "invasion risk"),
        (r"切除|resection", "resection recommendation"),
        (r"随访|follow", "follow-up recommendation"),
        (r"低风险|low risk", "low risk"),
        (r"中等|intermediate", "intermediate risk"),
        (r"高风险|high risk", "high risk"),
    )

    @classmethod
    def extract_report_terms(
        cls,
        *,
        report: dict[str, Any],
        doctor_annotations: dict[str, Any] | None = None,
        max_terms: int = 16,
    ) -> dict[str, Any]:
        """brief:
            Extract report terms.

        parameter:
            - report: Input value for report.
            - doctor_annotations: Input value for doctor_annotations.
            - max_terms: Input value for max_terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        labels = cls.extract_report_feature_labels(
            report=report,
            doctor_annotations=doctor_annotations,
            max_labels=max_terms,
        )["labels"]
        terms = [{"term": label, "source": "report+doctor_annotations"} for label in labels]
        return {"terms": terms, "term_count": len(terms)}

    @staticmethod
    def normalize_medical_terms(*, terms: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle normalize medical terms.

        parameter:
            - terms: Input value for terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        synonym_map = {
            "polyp": "息肉",
            "adenoma": "腺瘤",
            "flat morphology": "扁平型",
            "elevated morphology": "隆起型",
            "depressed morphology": "凹陷型",
            "paris type": "Paris分型",
            "low risk": "低风险",
            "intermediate risk": "中等风险",
            "high risk": "高风险",
            "resection recommendation": "内镜下切除",
            "follow-up recommendation": "随访",
        }
        normalized: list[dict[str, Any]] = []
        for item in terms:
            raw = str(item.get("term", "")).strip()
            if not raw:
                continue
            key = raw.lower()
            normalized_term = synonym_map.get(key, raw)
            normalized.append({**item, "rawTerm": raw, "normalizedTerm": normalized_term})
        return {"terms": normalized, "term_count": len(normalized)}

    @staticmethod
    def classify_term_category(*, terms: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Classify term category.

        parameter:
            - terms: Input value for terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        def category_for(term: str) -> str:
            """brief:
                Handle category for.

            parameter:
                - term: Input value for term.

            retrival:
                - Returns the computed value for the caller or workflow.
            """
            text = term.lower()
            if "0-i" in text or "paris" in text:
                return "Paris分型"
            if any(token in text for token in ["扁平", "隆起", "凹陷", "flat", "elevated", "depressed"]):
                return "病灶形态"
            if "风险" in text or "risk" in text:
                return "风险等级"
            if any(token in text for token in ["切除", "随访", "活检", "resection", "follow"]):
                return "处理建议"
            if any(token in text for token in ["血管", "vascular", "充血", "erythema"]):
                return "表面/血管特征"
            if any(token in text for token in ["息肉", "腺瘤", "polyp", "adenoma"]):
                return "病理/类型"
            return "科研筛选标签"

        classified = []
        for item in terms:
            normalized = str(item.get("normalizedTerm", item.get("term", ""))).strip()
            if not normalized:
                continue
            classified.append({**item, "category": category_for(normalized)})
        return {"terms": classified, "categories": sorted({item["category"] for item in classified})}

    @staticmethod
    def score_term_confidence(*, terms: list[dict[str, Any]], doctor_annotations: dict[str, Any] | None = None) -> dict[str, Any]:
        """brief:
            Handle score term confidence.

        parameter:
            - terms: Input value for terms.
            - doctor_annotations: Input value for doctor_annotations.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        annotation_blob = " ".join(str(value) for value in (doctor_annotations or {}).values()).lower()
        scored = []
        for item in terms:
            term = str(item.get("rawTerm", item.get("term", ""))).lower()
            source = str(item.get("source", ""))
            score = 0.72
            if "doctor" in source:
                score += 0.12
            if term and term in annotation_blob:
                score += 0.14
            scored.append({**item, "confidence": round(_clamp(score), 4), "needsReview": score < 0.7})
        return {"terms": scored}

    @staticmethod
    def deduplicate_terms(*, terms: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle deduplicate terms.

        parameter:
            - terms: Input value for terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in terms:
            key = (
                str(item.get("category", "")).lower(),
                str(item.get("normalizedTerm", item.get("term", ""))).lower(),
            )
            if key not in deduped or float(item.get("confidence", 0.0)) > float(deduped[key].get("confidence", 0.0)):
                deduped[key] = item
        values = list(deduped.values())
        return {"terms": values, "term_count": len(values)}

    @staticmethod
    def bind_terms_to_report(*, terms: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle bind terms to report.

        parameter:
            - terms: Input value for terms.
            - report: Input value for report.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        bindings = []
        fields = {
            "findings": str(report.get("findings", "")),
            "conclusion": str(report.get("conclusion", "")),
            "layoutSuggestion": str(report.get("layoutSuggestion", report.get("layout_suggestion", ""))),
        }
        for item in terms:
            raw = str(item.get("rawTerm", item.get("term", ""))).lower()
            matched_field = "doctor_annotations"
            matched_text = ""
            for field_name, field_text in fields.items():
                if raw and raw in field_text.lower():
                    matched_field = field_name
                    matched_text = field_text[:240]
                    break
            bindings.append({**item, "sourceField": matched_field, "sourceText": matched_text})
        return {"terms": bindings}

    @staticmethod
    def build_db_term_records(
        *,
        terms: list[dict[str, Any]],
        report_id: str = "",
        patient_id: str = "",
    ) -> dict[str, Any]:
        """brief:
            Build db term records.

        parameter:
            - terms: Input value for terms.
            - report_id: Input value for report_id.
            - patient_id: Input value for patient_id.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        records = []
        for index, item in enumerate(terms, start=1):
            category = str(item.get("category", "科研筛选标签"))
            normalized = str(item.get("normalizedTerm", item.get("term", "")))
            slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", f"{category}-{normalized}").strip("-").lower()
            records.append(
                {
                    "termId": f"term-{report_id or 'runtime'}-{index:03d}-{slug[:48]}",
                    "reportId": report_id,
                    "patientId": patient_id,
                    "term": item.get("rawTerm", item.get("term", "")),
                    "normalizedTerm": normalized,
                    "category": category,
                    "sourceField": item.get("sourceField", ""),
                    "sourceText": item.get("sourceText", ""),
                    "confidence": float(item.get("confidence", 0.0)),
                    "needsReview": bool(item.get("needsReview", False)),
                }
            )
        return {"dbRecords": records, "record_count": len(records)}

    @staticmethod
    def validate_term_records(*, records: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Validate term records.

        parameter:
            - records: Input value for records.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        issues: list[str] = []
        valid_categories = {"Paris分型", "病灶形态", "风险等级", "处理建议", "表面/血管特征", "病理/类型", "科研筛选标签"}
        for record in records:
            if not record.get("normalizedTerm"):
                issues.append("missing_normalized_term")
            if record.get("category") not in valid_categories:
                issues.append("invalid_category")
            confidence = float(record.get("confidence", 0.0))
            if confidence < 0.0 or confidence > 1.0:
                issues.append("confidence_out_of_range")
        return {"valid": not issues, "issues": sorted(set(issues)), "record_count": len(records)}

    @staticmethod
    def route_term_index(*, records: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle route term index.

        parameter:
            - records: Input value for records.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        routes: dict[str, list[str]] = {}
        for record in records:
            category = str(record.get("category", "科研筛选标签"))
            table_name = {
                "Paris分型": "report_terms_paris",
                "病灶形态": "report_terms_morphology",
                "风险等级": "report_terms_risk",
                "处理建议": "report_terms_disposition",
            }.get(category, "report_terms_general")
            routes.setdefault(table_name, []).append(str(record.get("termId", "")))
        return {"routes": routes}

    @staticmethod
    def upsert_report_terms(*, records: list[dict[str, Any]], dry_run: bool = True) -> dict[str, Any]:
        """brief:
            Handle upsert report terms.

        parameter:
            - records: Input value for records.
            - dry_run: Input value for dry_run.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "dryRun": dry_run,
            "upserted": 0 if dry_run else len(records),
            "planned": len(records),
            "recordIds": [str(record.get("termId", "")) for record in records],
        }

    @staticmethod
    def build_filter_facets(*, records: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Build filter facets.

        parameter:
            - records: Input value for records.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        facets: dict[str, list[str]] = {}
        for record in records:
            category = str(record.get("category", "科研筛选标签"))
            term = str(record.get("normalizedTerm", ""))
            if term:
                facets.setdefault(category, [])
                if term not in facets[category]:
                    facets[category].append(term)
        return {"facets": facets}

    @staticmethod
    def audit_term_coverage(*, records: list[dict[str, Any]], required_categories: list[str] | None = None) -> dict[str, Any]:
        """brief:
            Handle audit term coverage.

        parameter:
            - records: Input value for records.
            - required_categories: Input value for required_categories.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        required = required_categories or ["Paris分型", "风险等级", "处理建议"]
        present = {str(record.get("category", "")) for record in records}
        missing = [category for category in required if category not in present]
        return {
            "coverage": 1.0 - len(missing) / max(len(required), 1),
            "missingCategories": missing,
            "needsReview": bool(missing),
        }

    @classmethod
    def extract_report_feature_labels(
        cls,
        *,
        report: dict[str, Any],
        doctor_annotations: dict[str, Any] | None = None,
        max_labels: int = 12,
    ) -> dict[str, Any]:
        """brief:
            Extract report feature labels.

        parameter:
            - report: Input value for report.
            - doctor_annotations: Input value for doctor_annotations.
            - max_labels: Input value for max_labels.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        annotations = doctor_annotations or {}
        text = " ".join(
            str(report.get(key, ""))
            for key in ("findings", "conclusion", "layoutSuggestion", "layout_suggestion")
        )
        labels: list[str] = []
        for pattern, label in cls._report_label_patterns:
            if label not in labels and re.search(pattern, text, flags=re.IGNORECASE):
                labels.append(label)

        for value in [
            annotations.get("lesion_type"),
            annotations.get("pathology"),
            annotations.get("surface_pattern"),
            annotations.get("paris"),
            *list(annotations.get("tags", []) or []),
        ]:
            normalized = str(value).strip()
            if normalized and normalized not in labels:
                labels.append(normalized)

        return {
            "labels": labels[: max(1, max_labels)],
            "source": "report+doctor_annotations",
            "label_count": min(len(labels), max(1, max_labels)),
        }

    @staticmethod
    def embed_mask_shape(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle embed mask shape.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        stats = record.mask_stats
        vector = [
            _safe_float(stats, "area_ratio"),
            _safe_float(stats, "aspect_ratio"),
            _safe_float(stats, "boundary_complexity"),
            _safe_float(stats, "solidity", 1.0),
            _safe_float(stats, "components", 1.0) / 10.0,
        ]
        return {"image_id": record.image_id, "embedding_type": "mask_shape", "vector": [_clamp(v, 0.0, 2.0) for v in vector]}

    @staticmethod
    def embed_visual_region_request(*, sample: dict[str, Any], crop_padding: float = 0.15) -> dict[str, Any]:
        """brief:
            Handle embed visual region request.

        parameter:
            - sample: Input value for sample.
            - crop_padding: Input value for crop_padding.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "image_path": record.image_path, "bbox": record.bbox, "crop_padding": crop_padding}

    @staticmethod
    def embed_text_label(*, labels: list[str]) -> dict[str, Any]:
        """brief:
            Handle embed text label.

        parameter:
            - labels: Input value for labels.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        tokens = [label.strip().lower() for label in labels if label.strip()]
        vocabulary = sorted(set(tokens))
        vector = [tokens.count(token) / max(len(tokens), 1) for token in vocabulary]
        return {"embedding_type": "text_label_bow", "vocabulary": vocabulary, "vector": vector}

    @staticmethod
    def encode_boundary_features(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle encode boundary features.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        boundary_f1 = _safe_float(record.metrics, "Boundary F1", 1.0)
        complexity = _safe_float(record.mask_stats, "boundary_complexity")
        return {
            "image_id": record.image_id,
            "boundary_signature": {
                "complexity": complexity,
                "boundary_f1": boundary_f1,
                "risk": "high" if complexity > 0.6 or boundary_f1 < 0.55 else "medium" if complexity > 0.4 else "low",
            },
        }

    @staticmethod
    def build_hard_case_signature(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Build hard case signature.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        tags: list[str] = []
        if record.baseline_dice < 0.5:
            tags.append("low_baseline")
        if record.delta_dice <= -0.03:
            tags.append("regression")
        if _safe_float(record.uncertainty, "mean_entropy") > 0.35:
            tags.append("high_entropy")
        if _safe_float(record.mask_stats, "area_ratio") < 0.002:
            tags.append("small_target")
        return {"image_id": record.image_id, "hard_case_signature": "+".join(tags) or "not_hard", "tags": tags}

    @staticmethod
    def index_polarity_groups(*, samples: list[dict[str, Any]]) -> dict[str, list[str]]:
        """brief:
            Handle index polarity groups.

        parameter:
            - samples: Input value for samples.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        groups = {"positive": [], "negative": [], "boundary": [], "hard": []}
        for item in samples:
            record = SampleLibraryRecord.from_mapping(item)
            group = record.sample_group if record.sample_group in groups else "positive"
            groups[group].append(record.image_id)
        return groups

    @staticmethod
    def route_site_aware_embedding(*, sample: dict[str, Any], default_index: str = "global") -> dict[str, Any]:
        """brief:
            Handle route site aware embedding.

        parameter:
            - sample: Input value for sample.
            - default_index: Input value for default_index.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        index_name = f"site_{record.site_id}" if record.site_id else default_index
        if record.sample_group in {"hard", "boundary"}:
            index_name = f"{index_name}_{record.sample_group}"
        return {"image_id": record.image_id, "index_name": index_name}

    @staticmethod
    def monitor_embedding_drift(*, embedding: list[float], centroid: list[float], threshold: float = 0.35) -> dict[str, Any]:
        """brief:
            Handle monitor embedding drift.

        parameter:
            - embedding: Input value for embedding.
            - centroid: Input value for centroid.
            - threshold: Input value for threshold.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if not embedding or not centroid or len(embedding) != len(centroid):
            return {"drift": 0.0, "is_outlier": False, "reason": "missing_or_mismatched_embedding"}
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding, centroid))) / math.sqrt(len(embedding))
        return {"drift": distance, "is_outlier": distance > threshold}


class ResultReviewToolSet:
    """brief:
        Represent ResultReviewToolSet state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    agent_name = "result_review_agent"

    @staticmethod
    def collect_agent_outputs(*, agent_outputs: dict[str, Any], agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle collect agent outputs.

        parameter:
            - agent_outputs: Input value for agent_outputs.
            - agent_runs: Input value for agent_runs.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return {
            "agent_outputs": agent_outputs,
            "agent_runs": agent_runs,
            "agent_count": len(agent_runs),
            "agent_names": [str(run.get("agent_name", "")) for run in agent_runs],
        }

    @staticmethod
    def check_workflow_completeness(*, review_package: dict[str, Any], required_agents: list[str] | None = None) -> dict[str, Any]:
        """brief:
            Handle check workflow completeness.

        parameter:
            - review_package: Input value for review_package.
            - required_agents: Input value for required_agents.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        required = required_agents or [
            "segmentation_preprocess_agent",
            "sample_audit_agent",
            "report_generation_agent",
            "label_embedding_agent",
        ]
        names = set(review_package.get("agent_names", []))
        outputs = dict(review_package.get("agent_outputs", {}) or {})
        missing_agents = [agent for agent in required if agent not in names]
        missing_outputs = [agent for agent in required if not outputs.get(agent)]
        return {
            "complete": not missing_agents and not missing_outputs,
            "missing_agents": missing_agents,
            "missing_outputs": missing_outputs,
        }

    @staticmethod
    def audit_preprocess_result(*, preprocess: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle audit preprocess result.

        parameter:
            - preprocess: Input value for preprocess.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        bbox = preprocess.get("bbox_request", {}).get("bbox", [])
        prompts = preprocess.get("prompt_package", {}).get("prompts", {})
        large_mask = preprocess.get("large_mask_gate", {}).get("is_large_mask", False)
        issues: list[str] = []
        if not bbox or len(bbox) != 4:
            issues.append("missing_bbox")
        if not prompts:
            issues.append("missing_prompts")
        if large_mask:
            issues.append("large_mask_gate_triggered")
        return {"passed": not issues, "issues": issues}

    @staticmethod
    def audit_sample_audit_result(*, sample_audit: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle audit sample audit result.

        parameter:
            - sample_audit: Input value for sample_audit.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        issues: list[str] = []
        if sample_audit.get("accepted") and not sample_audit.get("reference_quiz", {}).get("passed"):
            issues.append("accepted_without_reference_quiz_pass")
        if sample_audit.get("accepted") and not sample_audit.get("mask_consistency", {}).get("valid"):
            issues.append("accepted_with_invalid_mask")
        if sample_audit.get("bank_decision") == "reject" and sample_audit.get("accepted"):
            issues.append("reject_accept_conflict")
        return {"passed": not issues, "issues": issues}

    @staticmethod
    def audit_report_result(*, report: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle audit report result.

        parameter:
            - report: Input value for report.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        findings = str(report.get("findings", "")).strip()
        conclusion = str(report.get("conclusion", "")).strip()
        report_score = float(report.get("report_score", {}).get("overall_score", 8.0) or 0.0)
        issues: list[str] = []
        if len(findings) < 20:
            issues.append("findings_too_short")
        if len(conclusion) < 10:
            issues.append("conclusion_too_short")
        if report_score < 6.5:
            issues.append("low_report_score")
        return {"passed": not issues, "issues": issues, "report_score": report_score}

    @staticmethod
    def audit_term_records(*, term_payload: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle audit term records.

        parameter:
            - term_payload: Input value for term_payload.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        validation = term_payload.get("validation", {})
        coverage = term_payload.get("coverage", {})
        issues = list(validation.get("issues", []))
        issues.extend(f"missing_{category}" for category in coverage.get("missingCategories", []))
        return {
            "passed": bool(validation.get("valid", False)) and not coverage.get("needsReview", False),
            "issues": sorted(set(issues)),
            "record_count": len(term_payload.get("dbRecords", [])),
        }

    @staticmethod
    def check_cross_agent_consistency(*, agent_outputs: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle check cross agent consistency.

        parameter:
            - agent_outputs: Input value for agent_outputs.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        issues: list[str] = []
        report = agent_outputs.get("report_generation_agent", {})
        terms = agent_outputs.get("label_embedding_agent", {})
        sample_audit = agent_outputs.get("sample_audit_agent", {})

        report_text = f"{report.get('findings', '')} {report.get('conclusion', '')}".lower()
        term_text = " ".join(str(record.get("normalizedTerm", "")) for record in terms.get("dbRecords", [])).lower()
        if "high risk" in report_text and "低风险" in term_text:
            issues.append("report_high_risk_term_low_risk_conflict")
        if sample_audit.get("bank_decision") == "reject" and terms.get("decision") == "ready_to_index":
            issues.append("rejected_sample_has_index_ready_terms")
        return {"consistent": not issues, "issues": issues}

    @staticmethod
    def detect_decision_conflicts(*, agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle detect decision conflicts.

        parameter:
            - agent_runs: Input value for agent_runs.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        decisions = {str(run.get("agent_name", "")): str(run.get("decision", "")) for run in agent_runs}
        conflicts: list[str] = []
        if decisions.get("sample_audit_agent") == "reject" and decisions.get("label_embedding_agent") == "ready_to_index":
            conflicts.append("sample_rejected_but_terms_ready")
        if decisions.get("report_generation_agent") == "needs_human_review" and decisions.get("label_embedding_agent") == "ready_to_index":
            conflicts.append("report_needs_review_but_terms_ready")
        return {"has_conflicts": bool(conflicts), "conflicts": conflicts, "decisions": decisions}

    @staticmethod
    def score_pipeline_quality(*, audit_results: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle score pipeline quality.

        parameter:
            - audit_results: Input value for audit_results.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        if not audit_results:
            return {"qualityScore": 0.0, "blockingIssues": ["missing_audit_results"], "warnings": []}
        blocking: list[str] = []
        warnings: list[str] = []
        passed_count = 0
        for result in audit_results:
            passed = bool(result.get("passed", result.get("complete", result.get("consistent", True))))
            if passed:
                passed_count += 1
            issues = [str(issue) for issue in result.get("issues", []) + result.get("missing_agents", []) + result.get("missing_outputs", []) + result.get("conflicts", [])]
            if not passed:
                blocking.extend(issues or ["failed_audit"])
            else:
                warnings.extend(issues)
        return {
            "qualityScore": round(passed_count / max(len(audit_results), 1), 4),
            "blockingIssues": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
        }

    @staticmethod
    def assign_review_action(*, quality: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle assign review action.

        parameter:
            - quality: Input value for quality.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        blocking = list(quality.get("blockingIssues", []))
        score = float(quality.get("qualityScore", 0.0))
        if any("missing_bbox" in issue or "large_mask" in issue for issue in blocking):
            decision = "retry_preprocess"
        elif any("reference_quiz" in issue or "invalid_mask" in issue for issue in blocking):
            decision = "retry_sample_audit"
        elif any("findings" in issue or "conclusion" in issue or "report" in issue for issue in blocking):
            decision = "retry_report_generation"
        elif any("missing_" in issue or "term" in issue for issue in blocking):
            decision = "retry_term_embedding"
        elif blocking:
            decision = "needs_human_review"
        elif score >= 0.95:
            decision = "approved"
        else:
            decision = "approved_with_warnings"
        return {"finalDecision": decision, "humanReviewRequired": decision not in {"approved", "approved_with_warnings"}}

    @staticmethod
    def route_retry_or_human_review(*, review_action: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle route retry or human review.

        parameter:
            - review_action: Input value for review_action.
            - quality: Input value for quality.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        decision = str(review_action.get("finalDecision", "needs_human_review"))
        retry_targets = {
            "retry_preprocess": "segmentation_preprocess_agent",
            "retry_sample_audit": "sample_audit_agent",
            "retry_report_generation": "report_generation_agent",
            "retry_term_embedding": "label_embedding_agent",
        }
        target = retry_targets.get(decision, "")
        return {
            "shouldRetry": bool(target),
            "targetAgent": target,
            "humanReviewRequired": bool(review_action.get("humanReviewRequired", False)),
            "reason": "; ".join(quality.get("blockingIssues", [])[:3]),
        }

    @staticmethod
    def build_review_report(*, review_action: dict[str, Any], quality: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Build review report.

        parameter:
            - review_action: Input value for review_action.
            - quality: Input value for quality.
            - route: Input value for route.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        decision = review_action.get("finalDecision", "needs_human_review")
        if route.get("shouldRetry"):
            text = f"闭环复核未通过，建议重跑 {route.get('targetAgent')}。原因：{route.get('reason')}"
        elif review_action.get("humanReviewRequired"):
            text = f"闭环复核需要人工确认。问题：{'; '.join(quality.get('blockingIssues', [])[:3])}"
        else:
            text = f"闭环复核通过，最终决策为 {decision}。"
        return {"reviewReport": text}

    @staticmethod
    def analyze_metric_delta(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle analyze metric delta.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {
            "image_id": record.image_id,
            "baseline_dice": record.baseline_dice,
            "result_dice": record.result_dice,
            "delta_dice": record.delta_dice,
            "direction": "gain" if record.delta_dice > 0.0 else "harm" if record.delta_dice < 0.0 else "flat",
        }

    @staticmethod
    def generate_hard_case_delta_report(
        *,
        rows: list[dict[str, Any]],
        thresholds: list[float] | None = None,
        quantiles: list[float] | None = None,
        min_gain: float = 0.03,
        rescue_threshold: float = 0.5,
        hard_weight_gamma: float = 2.0,
    ) -> dict[str, Any]:
        """brief:
            Handle generate hard case delta report.

        parameter:
            - rows: Input value for rows.
            - thresholds: Input value for thresholds.
            - quantiles: Input value for quantiles.
            - min_gain: Input value for min_gain.
            - rescue_threshold: Input value for rescue_threshold.
            - hard_weight_gamma: Input value for hard_weight_gamma.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        thresholds = thresholds or [0.3, 0.5, 0.7]
        quantiles = quantiles or [0.1, 0.2]

        def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
            """brief:
                Handle summarize.

            parameter:
                - subset: Input value for subset.

            retrival:
                - Returns the computed value for the caller or workflow.
            """
            if not subset:
                return {
                    "count": 0,
                    "baseline_dice_mean": 0.0,
                    "result_dice_mean": 0.0,
                    "mean_delta_dice": 0.0,
                    "median_delta_dice": 0.0,
                    "positive_delta_rate": 0.0,
                    "negative_delta_rate": 0.0,
                    "rescue_rate": 0.0,
                    "meaningful_gain_rate": 0.0,
                    "low_dice_error_reduction": 0.0,
                    "severe_harm_rate": 0.0,
                }
            baseline = [_dice(row, "baseline_metrics") for row in subset]
            result = [_dice(row, "metrics") for row in subset]
            delta = [res - base for base, res in zip(baseline, result)]
            reductions = [(res - base) / max(1.0 - base, 1e-6) for base, res in zip(baseline, result)]
            return {
                "count": len(subset),
                "baseline_dice_mean": sum(baseline) / len(subset),
                "result_dice_mean": sum(result) / len(subset),
                "mean_delta_dice": sum(delta) / len(subset),
                "median_delta_dice": median(delta),
                "positive_delta_rate": sum(1 for value in delta if value > 0.0) / len(subset),
                "negative_delta_rate": sum(1 for value in delta if value < 0.0) / len(subset),
                "rescue_rate": sum(1 for base, res in zip(baseline, result) if base < rescue_threshold <= res) / len(subset),
                "meaningful_gain_rate": sum(1 for value in delta if value >= min_gain) / len(subset),
                "low_dice_error_reduction": sum(reductions) / len(subset),
                "severe_harm_rate": sum(1 for value in delta if value <= -min_gain) / len(subset),
            }

        ranked = sorted(rows, key=lambda row: _dice(row, "baseline_metrics"))
        threshold_subsets = {
            f"baseline_dice<{threshold:g}": summarize([row for row in rows if _dice(row, "baseline_metrics") < threshold])
            for threshold in thresholds
        }
        quantile_subsets = {}
        for quantile in quantiles:
            count = max(1, int(round(len(ranked) * quantile))) if ranked else 0
            quantile_subsets[f"bottom_{int(quantile * 100)}pct_by_baseline_dice"] = summarize(ranked[:count])

        numerator = 0.0
        denominator = 0.0
        for row in rows:
            baseline = _dice(row, "baseline_metrics")
            delta = _dice(row, "metrics") - baseline
            weight = max(1.0 - baseline, 0.0) ** hard_weight_gamma
            numerator += weight * delta
            denominator += weight

        return {
            "count": len(rows),
            "overall": summarize(rows),
            "weighted_hard_case_gain": numerator / max(denominator, 1e-6),
            "threshold_subsets": threshold_subsets,
            "quantile_subsets": quantile_subsets,
        }

    @staticmethod
    def classify_failure_case(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Classify failure case.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        precision = _safe_float(record.metrics, "Precision", 1.0)
        recall = _safe_float(record.metrics, "Recall", 1.0)
        boundary_f1 = _safe_float(record.metrics, "Boundary F1", 1.0)
        if recall < 0.5:
            mode = "under_segmentation"
        elif precision < 0.5:
            mode = "over_segmentation"
        elif boundary_f1 < 0.55:
            mode = "boundary_error"
        elif record.delta_dice <= -0.03:
            mode = "method_regression"
        else:
            mode = "no_major_failure"
        return {"image_id": record.image_id, "failure_mode": mode}

    @staticmethod
    def check_confidence_consistency(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle check confidence consistency.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        confidence = _safe_float(record.uncertainty, "mean_confidence", _safe_float(record.metrics, "mean confidence"))
        dice = record.result_dice
        inconsistent = confidence > 0.85 and dice < 0.5
        return {"image_id": record.image_id, "inconsistent": inconsistent, "confidence": confidence, "dice": dice}

    @staticmethod
    def review_mask_sanity(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle review mask sanity.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        components = int(record.mask_stats.get("components", 1) or 1)
        issues: list[str] = []
        if area_ratio == 0.0:
            issues.append("empty_prediction")
        if area_ratio > 0.6:
            issues.append("mask_too_large")
        if components > 10:
            issues.append("too_fragmented")
        return {"image_id": record.image_id, "sane": not issues, "issues": issues}

    @staticmethod
    def detect_regression(*, sample: dict[str, Any], min_harm: float = -0.03) -> dict[str, Any]:
        """brief:
            Handle detect regression.

        parameter:
            - sample: Input value for sample.
            - min_harm: Input value for min_harm.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "is_regression": record.delta_dice <= min_harm, "delta_dice": record.delta_dice}

    @staticmethod
    def audit_exemplar_effect(*, sample: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle audit exemplar effect.

        parameter:
            - sample: Input value for sample.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        used = any(record.selected_exemplars.values())
        effect = "helpful" if used and record.delta_dice >= 0.03 else "harmful" if used and record.delta_dice <= -0.03 else "neutral"
        return {"image_id": record.image_id, "used_exemplar": used, "effect": effect, "delta_dice": record.delta_dice}

    @staticmethod
    def update_continual_bank_item(*, sample: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Update continual bank item.

        parameter:
            - sample: Input value for sample.
            - review: Input value for review.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        record = SampleLibraryRecord.from_mapping(sample)
        accepted = review.get("sane", True) and not review.get("is_regression", False)
        target_group = "hard" if record.baseline_dice < 0.7 else record.sample_group
        return {"image_id": record.image_id, "accepted": accepted, "target_group": target_group, "review": review}


class BuildBboxRequest:
    """brief:
        Represent BuildBboxRequest state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, sample: dict[str, Any], detector: str = "yolo") -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - detector: Input value for detector.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return SegmentationPreprocessToolSet.build_bbox_cache_request(sample=sample, detector=detector)


class NormalizeImagePlan:
    """brief:
        Represent NormalizeImagePlan state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, sample: dict[str, Any], target_size: int = 1024) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - target_size: Input value for target_size.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return SegmentationPreprocessToolSet.normalize_image_plan(sample=sample, target_size=target_size)


class TracePreprocess:
    """brief:
        Represent TracePreprocess state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, sample: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - steps: Input value for steps.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return SegmentationPreprocessToolSet.trace_preprocess(sample=sample, steps=steps)


class PackagePrompts:
    """brief:
        Represent PackagePrompts state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(
        self,
        *,
        sample: dict[str, Any],
        use_text: bool = True,
        use_box: bool = True,
        use_exemplar: bool = True,
    ) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - use_text: Input value for use_text.
            - use_box: Input value for use_box.
            - use_exemplar: Input value for use_exemplar.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return SegmentationPreprocessToolSet.package_prompts(
            sample=sample,
            use_text=use_text,
            use_box=use_box,
            use_exemplar=use_exemplar,
        )


class BuildReviewQueueItem:
    """brief:
        Represent BuildReviewQueueItem state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, sample: dict[str, Any], known_ids: list[str] | None = None) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - known_ids: Input value for known_ids.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        identity = SampleAuditToolSet.check_identity(sample=sample, known_ids=known_ids or [])
        mask_consistency = SampleAuditToolSet.check_label_mask_consistency(sample=sample)
        hard_case = SampleAuditToolSet.mine_hard_case(sample=sample)
        boundary_case = SampleAuditToolSet.detect_boundary_case(sample=sample)
        grade = SampleAuditToolSet.assign_sample_grade(sample=sample)
        review_item = SampleAuditToolSet.build_review_queue_item(
            sample=sample,
            audit_results=[identity, mask_consistency, hard_case, boundary_case],
        )
        return {
            "review_item": review_item,
            "identity": identity,
            "mask_consistency": mask_consistency,
            "hard_case": hard_case,
            "boundary_case": boundary_case,
            "grade": grade,
        }


class RunReferenceLabelQuiz:
    """brief:
        Represent RunReferenceLabelQuiz state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(
        self,
        *,
        sample: dict[str, Any],
        reference_sample: dict[str, Any] | None = None,
        doctor_annotations: dict[str, Any] | None = None,
        pass_threshold: float = 0.55,
    ) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample: Input value for sample.
            - reference_sample: Input value for reference_sample.
            - doctor_annotations: Input value for doctor_annotations.
            - pass_threshold: Input value for pass_threshold.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return SampleAuditToolSet.run_reference_label_quiz(
            sample=sample,
            reference_sample=reference_sample,
            doctor_annotations=doctor_annotations,
            pass_threshold=pass_threshold,
        )


class ExtractReportTerms:
    """brief:
        Represent ExtractReportTerms state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(
        self,
        *,
        report: dict[str, Any],
        doctor_annotations: dict[str, Any] | None = None,
        max_terms: int = 16,
    ) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - report: Input value for report.
            - doctor_annotations: Input value for doctor_annotations.
            - max_terms: Input value for max_terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return LabelEmbeddingToolSet.extract_report_terms(
            report=report,
            doctor_annotations=doctor_annotations,
            max_terms=max_terms,
        )


class NormalizeMedicalTerms:
    """brief:
        Represent NormalizeMedicalTerms state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, terms: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - terms: Input value for terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return LabelEmbeddingToolSet.normalize_medical_terms(terms=terms)


class DeduplicateTerms:
    """brief:
        Represent DeduplicateTerms state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, terms: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - terms: Input value for terms.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return LabelEmbeddingToolSet.deduplicate_terms(terms=terms)


class BuildDbTermRecords:
    """brief:
        Represent BuildDbTermRecords state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(
        self,
        *,
        terms: list[dict[str, Any]],
        report: dict[str, Any],
        doctor_annotations: dict[str, Any] | None = None,
        report_id: str = "",
        patient_id: str = "",
    ) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - terms: Input value for terms.
            - report: Input value for report.
            - doctor_annotations: Input value for doctor_annotations.
            - report_id: Input value for report_id.
            - patient_id: Input value for patient_id.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        classified = LabelEmbeddingToolSet.classify_term_category(terms=terms)
        scored = LabelEmbeddingToolSet.score_term_confidence(
            terms=classified["terms"],
            doctor_annotations=doctor_annotations,
        )
        bound = LabelEmbeddingToolSet.bind_terms_to_report(terms=scored["terms"], report=report)
        records = LabelEmbeddingToolSet.build_db_term_records(
            terms=bound["terms"],
            report_id=report_id,
            patient_id=patient_id,
        )
        validation = LabelEmbeddingToolSet.validate_term_records(records=records["dbRecords"])
        routes = LabelEmbeddingToolSet.route_term_index(records=records["dbRecords"])
        upsert = LabelEmbeddingToolSet.upsert_report_terms(records=records["dbRecords"], dry_run=True)
        facets = LabelEmbeddingToolSet.build_filter_facets(records=records["dbRecords"])
        coverage = LabelEmbeddingToolSet.audit_term_coverage(records=records["dbRecords"])
        return {
            **records,
            "categorized_terms": classified["terms"],
            "scored_terms": scored["terms"],
            "bound_terms": bound["terms"],
            "validation": validation,
            "routes": routes["routes"],
            "upsert": upsert,
            "facets": facets["facets"],
            "coverage": coverage,
        }


class CollectAgentOutputs:
    """brief:
        Represent CollectAgentOutputs state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, agent_outputs: dict[str, Any], agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - agent_outputs: Input value for agent_outputs.
            - agent_runs: Input value for agent_runs.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        package = ResultReviewToolSet.collect_agent_outputs(agent_outputs=agent_outputs, agent_runs=agent_runs)
        completeness = ResultReviewToolSet.check_workflow_completeness(review_package=package)
        package["completeness"] = completeness
        return package


class AuditPreprocessResult:
    """brief:
        Represent AuditPreprocessResult state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, preprocess: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - preprocess: Input value for preprocess.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ResultReviewToolSet.audit_preprocess_result(preprocess=preprocess)


class AuditSampleAuditResult:
    """brief:
        Represent AuditSampleAuditResult state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, sample_audit: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - sample_audit: Input value for sample_audit.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ResultReviewToolSet.audit_sample_audit_result(sample_audit=sample_audit)


class AuditReportResult:
    """brief:
        Represent AuditReportResult state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, report: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - report: Input value for report.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ResultReviewToolSet.audit_report_result(report=report)


class AuditTermResult:
    """brief:
        Represent AuditTermResult state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __call__(self, *, term_payload: dict[str, Any]) -> dict[str, Any]:
        """brief:
            Handle call.

        parameter:
            - term_payload: Input value for term_payload.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return ResultReviewToolSet.audit_term_records(term_payload=term_payload)


def _register_primary_agent_tools(registry: SampleLibraryToolRegistry) -> None:
    """brief:
        Register primary agent tools.

    parameter:
        - registry: Input value for registry.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    handlers: dict[str, Callable[..., Any]] = {
        "BuildBboxRequest": BuildBboxRequest(),
        "NormalizeImagePlan": NormalizeImagePlan(),
        "TracePreprocess": TracePreprocess(),
        "PackagePrompts": PackagePrompts(),
        "BuildReviewQueueItem": BuildReviewQueueItem(),
        "RunReferenceLabelQuiz": RunReferenceLabelQuiz(),
        "CaseContextAssembler": CaseContextAssembler(),
        "UncertaintyExplainer": UncertaintyExplainer(),
        "ReportTemplateComposer": ReportTemplateComposer(),
        "ExtractReportTerms": ExtractReportTerms(),
        "NormalizeMedicalTerms": NormalizeMedicalTerms(),
        "DeduplicateTerms": DeduplicateTerms(),
        "Build_db_TermRecords": BuildDbTermRecords(),
        "CollectAgentOutputs": CollectAgentOutputs(),
        "AuditPreprocessResult": AuditPreprocessResult(),
        "AuditSampleAuditResult": AuditSampleAuditResult(),
        "AuditReportResult": AuditReportResult(),
        "AuditTermResult": AuditTermResult(),
    }
    input_map: dict[str, list[str]] = {
        "BuildBboxRequest": ["sample", "detector"],
        "NormalizeImagePlan": ["sample", "target_size"],
        "TracePreprocess": ["sample", "steps"],
        "PackagePrompts": ["sample", "use_text", "use_box", "use_exemplar"],
        "BuildReviewQueueItem": ["sample", "known_ids"],
        "RunReferenceLabelQuiz": ["sample", "reference_sample", "doctor_annotations", "pass_threshold"],
        "CaseContextAssembler": ["sample", "similar_cases", "review_summary"],
        "UncertaintyExplainer": ["context"],
        "ReportTemplateComposer": ["context", "report_type"],
        "ExtractReportTerms": ["report", "doctor_annotations", "max_terms"],
        "NormalizeMedicalTerms": ["terms"],
        "DeduplicateTerms": ["terms"],
        "Build_db_TermRecords": ["terms", "report", "doctor_annotations", "report_id", "patient_id"],
        "CollectAgentOutputs": ["agent_outputs", "agent_runs"],
        "AuditPreprocessResult": ["preprocess"],
        "AuditSampleAuditResult": ["sample_audit"],
        "AuditReportResult": ["report"],
        "AuditTermResult": ["term_payload"],
    }
    output_map: dict[str, list[str]] = {
        "BuildBboxRequest": ["bbox_request"],
        "NormalizeImagePlan": ["normalization_plan"],
        "TracePreprocess": ["preprocess_trace"],
        "PackagePrompts": ["prompt_package"],
        "BuildReviewQueueItem": ["review_item"],
        "RunReferenceLabelQuiz": ["quiz_result"],
        "CaseContextAssembler": ["case_context"],
        "UncertaintyExplainer": ["uncertainty_summary"],
        "ReportTemplateComposer": ["template"],
        "ExtractReportTerms": ["terms"],
        "NormalizeMedicalTerms": ["normalized_terms"],
        "DeduplicateTerms": ["deduplicated_terms"],
        "Build_db_TermRecords": ["db_records"],
        "CollectAgentOutputs": ["review_package"],
        "AuditPreprocessResult": ["preprocess_audit"],
        "AuditSampleAuditResult": ["sample_audit_review"],
        "AuditReportResult": ["report_audit"],
        "AuditTermResult": ["term_audit"],
    }
    for agent_name, metadata in PRIMARY_AGENT_TOOL_CHAINS.items():
        for tool in metadata["mainToolChain"]:
            tool_name = str(tool["name"])
            registry.register(
                ToolExplanation(
                    name=tool_name,
                    agent=agent_name,
                    purpose=str(tool["description"]),
                    inputs=input_map[tool_name],
                    outputs=output_map[tool_name],
                    sample_library_role=str(metadata["agentDetail"]),
                ),
                handlers[tool_name],
            )


def _register_many(
    registry: SampleLibraryToolRegistry,
    agent: str,
    toolset: Any,
    specs: list[tuple[str, str, list[str], list[str], str]],
) -> None:
    """brief:
        Register many.

    parameter:
        - registry: Input value for registry.
        - agent: Input value for agent.
        - toolset: Input value for toolset.
        - specs: Input value for specs.

    retrival:
        - Returns None; performs side effects described in the brief section.
    """
    for method_name, purpose, inputs, outputs, role in specs:
        registry.register(
            ToolExplanation(
                name=method_name,
                agent=agent,
                purpose=purpose,
                inputs=inputs,
                outputs=outputs,
                sample_library_role=role,
            ),
            getattr(toolset, method_name),
        )


def create_sample_library_tool_registry() -> SampleLibraryToolRegistry:
    """brief:
        Create sample library tool registry.

    parameter:
        - None.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    registry = SampleLibraryToolRegistry()
    _register_primary_agent_tools(registry)
    _register_many(
        registry,
        ReportGenerationToolSet.agent_name,
        ReportGenerationToolSet,
        [
            ("assemble_case_context", "Merge case metrics, retrieval evidence, and review summary.", ["sample", "similar_cases", "review_summary"], ["case_context"], "Turns one sample row into report-ready evidence."),
            ("retrieve_similar_cases", "Rank sample-library neighbors for report evidence.", ["query", "library", "top_k", "prefer_groups"], ["similar_cases"], "Uses hard, boundary, and positive cases as report references."),
            ("compose_report_template", "Choose report sections for a case.", ["context", "report_type"], ["template"], "Keeps generated reports consistent across sample groups."),
            ("narrate_findings", "Convert metrics into structured finding facts.", ["context"], ["finding_facts"], "Transforms Dice, delta, and mask stats into report material."),
            ("explain_uncertainty", "Explain confidence and entropy risks.", ["context"], ["uncertainty_summary"], "Surfaces why a case should be read cautiously."),
            ("bind_evidence", "Attach evidence references to statements.", ["context", "statements"], ["evidence_bindings"], "Makes report claims traceable to samples and metrics."),
            ("flag_report_risks", "Flag report-level review risks.", ["context"], ["risk_flags"], "Routes risky reports to human review."),
        ],
    )
    _register_many(
        registry,
        SampleAuditToolSet.agent_name,
        SampleAuditToolSet,
        [
            ("check_identity", "Check sample identity and duplicates.", ["sample", "known_ids"], ["identity_audit"], "Protects the bank from duplicate or incomplete records."),
            ("check_label_mask_consistency", "Check label mask area and fragmentation.", ["sample"], ["mask_audit"], "Separates clean, ambiguous, and reject samples."),
            ("audit_site_leakage", "Detect site-level split leakage.", ["samples"], ["leakage_report"], "Keeps train, validation, and external banks isolated."),
            ("mine_hard_case", "Mark low-Dice or regressed samples.", ["sample", "dice_threshold", "harm_threshold"], ["hard_case_flag"], "Feeds the hard-case bank."),
            ("detect_boundary_case", "Mark boundary-complex samples.", ["sample", "boundary_threshold"], ["boundary_case_flag"], "Feeds the boundary-case bank."),
            ("validate_negative_sample", "Audit whether a negative sample is suspicious.", ["sample"], ["negative_audit"], "Protects the negative bank from false negatives."),
            ("build_review_queue_item", "Create a human-review queue entry.", ["sample", "audit_results"], ["review_item"], "Prioritizes samples that need manual judgment."),
            ("assign_sample_grade", "Assign clean, hard, boundary, reject, or external-only grade.", ["sample"], ["sample_grade"], "Maps samples to their correct library partition."),
            ("run_reference_label_quiz", "Compare a candidate sample against a labeled reference and doctor annotations.", ["sample", "reference_sample", "doctor_annotations", "pass_threshold"], ["quiz_result"], "Uses an objective reference task to decide whether the sample is worth keeping."),
        ],
    )
    _register_many(
        registry,
        SegmentationPreprocessToolSet.agent_name,
        SegmentationPreprocessToolSet,
        [
            ("normalize_image_plan", "Build a deterministic image normalization plan.", ["sample", "target_size"], ["normalization_plan"], "Standardizes sample input before segmentation."),
            ("build_bbox_cache_request", "Prepare a YOLO/bbox cache request.", ["sample", "detector"], ["bbox_request"], "Connects sample records to spatial prompts."),
            ("package_prompts", "Package text, box, and exemplar prompts.", ["sample", "use_text", "use_box", "use_exemplar"], ["prompt_package"], "Chooses prompt modes from sample type and metadata."),
            ("generate_mask_prior_plan", "Create a similar-case mask prior plan.", ["sample", "similar_cases"], ["mask_prior_plan"], "Reuses sample-library masks as priors."),
            ("scan_region_uncertainty", "Detect whether uncertainty needs region attention.", ["sample"], ["uncertainty_scan"], "Triggers region-aware retrieval for uncertain cases."),
            ("guard_small_lesion", "Recommend safeguards for tiny targets.", ["sample", "min_area_ratio"], ["small_lesion_guard"], "Protects small-lesion samples from preprocessing loss."),
            ("gate_large_mask", "Gate suspiciously large masks.", ["sample", "max_area_ratio"], ["large_mask_gate"], "Prevents large-mask cases from poisoning prompts."),
            ("trace_preprocess", "Record preprocessing decisions.", ["sample", "steps"], ["preprocess_trace"], "Makes preprocessing reproducible per sample."),
        ],
    )
    _register_many(
        registry,
        LabelEmbeddingToolSet.agent_name,
        LabelEmbeddingToolSet,
        [
            ("extract_report_terms", "Extract candidate database filter terms from report text and doctor annotations.", ["report", "doctor_annotations", "max_terms"], ["terms"], "Builds research/search terms from generated reports."),
            ("normalize_medical_terms", "Normalize synonyms and mixed Chinese/English medical terms.", ["terms"], ["normalized_terms"], "Keeps database filters consistent."),
            ("classify_term_category", "Assign each term to a database filter category.", ["terms"], ["categorized_terms"], "Routes terms to Paris, morphology, risk, pathology, or disposition facets."),
            ("score_term_confidence", "Score each term from report and doctor annotation evidence.", ["terms", "doctor_annotations"], ["scored_terms"], "Marks low-confidence terms for review."),
            ("deduplicate_terms", "Merge duplicate terms and synonyms.", ["terms"], ["deduplicated_terms"], "Prevents redundant database rows."),
            ("bind_terms_to_report", "Bind terms back to report fields and source text.", ["terms", "report"], ["bound_terms"], "Makes filter terms auditable."),
            ("build_db_term_records", "Build structured database records for report terms.", ["terms", "report_id", "patient_id"], ["db_records"], "Creates payloads ready for persistence."),
            ("validate_term_records", "Validate term records before insertion.", ["records"], ["validation"], "Protects database writes from malformed terms."),
            ("route_term_index", "Route term records to logical index tables.", ["records"], ["routes"], "Supports category-specific filtering indexes."),
            ("upsert_report_terms", "Plan or execute report-term upsert.", ["records", "dry_run"], ["upsert_result"], "Avoids duplicate report terms."),
            ("build_filter_facets", "Build frontend/research filter facets from records.", ["records"], ["facets"], "Exposes selectable query dimensions."),
            ("audit_term_coverage", "Check whether terms cover required report concepts.", ["records", "required_categories"], ["coverage"], "Flags missing Paris/risk/disposition terms."),
        ],
    )
    _register_many(
        registry,
        ResultReviewToolSet.agent_name,
        ResultReviewToolSet,
        [
            ("collect_agent_outputs", "Collect upstream agent outputs and traces.", ["agent_outputs", "agent_runs"], ["review_package"], "Builds the full package for final review."),
            ("check_workflow_completeness", "Check that all upstream agents ran and produced outputs.", ["review_package", "required_agents"], ["completeness"], "Guards closed-loop execution integrity."),
            ("audit_preprocess_result", "Audit segmentation preprocessing result.", ["preprocess"], ["preprocess_audit"], "Checks bbox, prompts, and mask gates."),
            ("audit_sample_audit_result", "Audit sample-review decision consistency.", ["sample_audit"], ["sample_audit_review"], "Ensures accepted samples passed mask and reference checks."),
            ("audit_report_result", "Audit report completeness and score.", ["report"], ["report_audit"], "Checks findings, conclusion, and report quality."),
            ("audit_term_records", "Audit database term payloads and coverage.", ["term_payload"], ["term_audit"], "Checks that filter terms are valid and cover key concepts."),
            ("check_cross_agent_consistency", "Check consistency across upstream agent outputs.", ["agent_outputs"], ["consistency"], "Finds contradictions between report, sample decisions, and terms."),
            ("detect_decision_conflicts", "Detect conflicting upstream decisions.", ["agent_runs"], ["decision_conflicts"], "Prevents rejected samples from being silently indexed."),
            ("score_pipeline_quality", "Score the whole closed-loop pipeline.", ["audit_results"], ["quality"], "Aggregates completeness, audits, and conflicts."),
            ("assign_review_action", "Assign final review action.", ["quality"], ["review_action"], "Chooses approve, retry, reject, or human review."),
            ("route_retry_or_human_review", "Route retry or human review.", ["review_action", "quality"], ["route"], "Selects which upstream agent should rerun."),
            ("build_review_report", "Build a final human-readable review report.", ["review_action", "quality", "route"], ["review_report"], "Explains the final quality-control decision."),
        ],
    )
    return registry


def explain_sample_library_toolsets() -> list[dict[str, Any]]:
    """brief:
        Handle explain sample library toolsets.

    parameter:
        - None.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    return create_sample_library_tool_registry().list_tool_specs()


def group_tool_specs_by_agent(specs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """brief:
        Handle group tool specs by agent.

    parameter:
        - specs: Input value for specs.

    retrival:
        - Returns the computed value for the caller or workflow.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(str(spec.get("agent", "")), []).append(spec)
    return grouped
