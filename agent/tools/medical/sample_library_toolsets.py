from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Callable, Iterable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None:
        return default
    return float(value)


def _dice(row: dict[str, Any], field: str = "metrics") -> float:
    payload = row.get(field, {})
    if isinstance(payload, dict):
        return _safe_float(payload, "Dice")
    return 0.0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass(slots=True)
class SampleLibraryRecord:
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
        return _safe_float(self.baseline_metrics, "Dice")

    @property
    def result_dice(self) -> float:
        return _safe_float(self.metrics, "Dice")

    @property
    def delta_dice(self) -> float:
        return self.result_dice - self.baseline_dice

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolExplanation:
    name: str
    agent: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    sample_library_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolCallLog:
    tool_name: str
    status: str
    duration_ms: float
    output_preview: str
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "output_preview": self.output_preview[:240],
            "error_message": self.error_message,
        }


class SampleLibraryToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolExplanation, Callable[..., Any]]] = {}
        self._logs: list[ToolCallLog] = []

    def register(self, explanation: ToolExplanation, handler: Callable[..., Any]) -> None:
        self._tools[explanation.name] = (explanation, handler)

    def call(self, tool_name: str, **kwargs: Any) -> Any:
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
        return [explanation.to_dict() for explanation, _ in self._tools.values()]

    def get_call_logs(self) -> list[dict[str, Any]]:
        return [log.to_dict() for log in self._logs]

    def reset_logs(self) -> None:
        self._logs = []


class ReportGenerationToolSet:
    agent_name = "report_generation_agent"

    @staticmethod
    def assemble_case_context(
        *,
        sample: dict[str, Any],
        similar_cases: list[dict[str, Any]] | None = None,
        review_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        flags: list[str] = []
        if float(context.get("dice", 0.0)) < 0.5:
            flags.append("low_result_dice")
        if float(context.get("delta_dice", 0.0)) <= -0.03:
            flags.append("regression")
        if context.get("sample_group") in {"ambiguous", "reject"}:
            flags.append("sample_quality_risk")
        return {"risk_flags": flags, "needs_human_review": bool(flags)}


class SampleAuditToolSet:
    agent_name = "sample_audit_agent"

    @staticmethod
    def check_identity(*, sample: dict[str, Any], known_ids: list[str] | None = None) -> dict[str, Any]:
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
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        confidence = _safe_float(record.uncertainty, "mean_confidence")
        suspicious = area_ratio > 0.002 or confidence > 0.75
        return {"is_valid_negative": not suspicious, "suspicious": suspicious, "area_ratio": area_ratio, "confidence": confidence}

    @staticmethod
    def build_review_queue_item(*, sample: dict[str, Any], audit_results: list[dict[str, Any]]) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        reasons = [reason for result in audit_results for reason in result.get("issues", []) + result.get("reasons", [])]
        priority = "high" if any(reason in {"empty_mask", "regression", "duplicate_image_id"} for reason in reasons) else "medium" if reasons else "low"
        return {"image_id": record.image_id, "priority": priority, "reasons": sorted(set(reasons)), "sample_group": record.sample_group}

    @staticmethod
    def assign_sample_grade(*, sample: dict[str, Any]) -> dict[str, Any]:
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


class SegmentationPreprocessToolSet:
    agent_name = "segmentation_preprocess_agent"

    @staticmethod
    def normalize_image_plan(*, sample: dict[str, Any], target_size: int = 1024) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "target_size": target_size, "color_space": "RGB", "scale_mode": "long_side_pad"}

    @staticmethod
    def build_bbox_cache_request(*, sample: dict[str, Any], detector: str = "yolo") -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "detector": detector, "use_cached": bool(record.bbox), "bbox": record.bbox}

    @staticmethod
    def package_prompts(*, sample: dict[str, Any], use_text: bool = True, use_box: bool = True, use_exemplar: bool = False) -> dict[str, Any]:
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
        record = SampleLibraryRecord.from_mapping(sample)
        selected = [case.get("image_id", "") for case in similar_cases if case.get("mask_path")][:3]
        return {"image_id": record.image_id, "prior_type": "similar_case_mask", "source_case_ids": selected, "enabled": bool(selected)}

    @staticmethod
    def scan_region_uncertainty(*, sample: dict[str, Any]) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        entropy = _safe_float(record.uncertainty, "mean_entropy")
        confidence = _safe_float(record.uncertainty, "mean_confidence", 1.0)
        return {"needs_region_attention": entropy > 0.35 or confidence < 0.65, "mean_entropy": entropy, "mean_confidence": confidence}

    @staticmethod
    def guard_small_lesion(*, sample: dict[str, Any], min_area_ratio: float = 0.002) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        return {"is_small_lesion": 0.0 < area_ratio < min_area_ratio, "recommended_scale": 1.5 if 0.0 < area_ratio < min_area_ratio else 1.0}

    @staticmethod
    def gate_large_mask(*, sample: dict[str, Any], max_area_ratio: float = 0.35) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        area_ratio = _safe_float(record.mask_stats, "area_ratio")
        return {"is_large_mask": area_ratio > max_area_ratio, "use_exemplar_guard": area_ratio > max_area_ratio, "area_ratio": area_ratio}

    @staticmethod
    def trace_preprocess(*, sample: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "step_count": len(steps), "steps": steps}


class LabelEmbeddingToolSet:
    agent_name = "label_embedding_agent"

    @staticmethod
    def embed_mask_shape(*, sample: dict[str, Any]) -> dict[str, Any]:
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
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "image_path": record.image_path, "bbox": record.bbox, "crop_padding": crop_padding}

    @staticmethod
    def embed_text_label(*, labels: list[str]) -> dict[str, Any]:
        tokens = [label.strip().lower() for label in labels if label.strip()]
        vocabulary = sorted(set(tokens))
        vector = [tokens.count(token) / max(len(tokens), 1) for token in vocabulary]
        return {"embedding_type": "text_label_bow", "vocabulary": vocabulary, "vector": vector}

    @staticmethod
    def encode_boundary_features(*, sample: dict[str, Any]) -> dict[str, Any]:
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
        groups = {"positive": [], "negative": [], "boundary": [], "hard": []}
        for item in samples:
            record = SampleLibraryRecord.from_mapping(item)
            group = record.sample_group if record.sample_group in groups else "positive"
            groups[group].append(record.image_id)
        return groups

    @staticmethod
    def route_site_aware_embedding(*, sample: dict[str, Any], default_index: str = "global") -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        index_name = f"site_{record.site_id}" if record.site_id else default_index
        if record.sample_group in {"hard", "boundary"}:
            index_name = f"{index_name}_{record.sample_group}"
        return {"image_id": record.image_id, "index_name": index_name}

    @staticmethod
    def monitor_embedding_drift(*, embedding: list[float], centroid: list[float], threshold: float = 0.35) -> dict[str, Any]:
        if not embedding or not centroid or len(embedding) != len(centroid):
            return {"drift": 0.0, "is_outlier": False, "reason": "missing_or_mismatched_embedding"}
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding, centroid))) / math.sqrt(len(embedding))
        return {"drift": distance, "is_outlier": distance > threshold}


class ResultReviewToolSet:
    agent_name = "result_review_agent"

    @staticmethod
    def analyze_metric_delta(*, sample: dict[str, Any]) -> dict[str, Any]:
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
        thresholds = thresholds or [0.3, 0.5, 0.7]
        quantiles = quantiles or [0.1, 0.2]

        def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
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
        record = SampleLibraryRecord.from_mapping(sample)
        confidence = _safe_float(record.uncertainty, "mean_confidence", _safe_float(record.metrics, "mean confidence"))
        dice = record.result_dice
        inconsistent = confidence > 0.85 and dice < 0.5
        return {"image_id": record.image_id, "inconsistent": inconsistent, "confidence": confidence, "dice": dice}

    @staticmethod
    def review_mask_sanity(*, sample: dict[str, Any]) -> dict[str, Any]:
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
        record = SampleLibraryRecord.from_mapping(sample)
        return {"image_id": record.image_id, "is_regression": record.delta_dice <= min_harm, "delta_dice": record.delta_dice}

    @staticmethod
    def audit_exemplar_effect(*, sample: dict[str, Any]) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        used = any(record.selected_exemplars.values())
        effect = "helpful" if used and record.delta_dice >= 0.03 else "harmful" if used and record.delta_dice <= -0.03 else "neutral"
        return {"image_id": record.image_id, "used_exemplar": used, "effect": effect, "delta_dice": record.delta_dice}

    @staticmethod
    def update_continual_bank_item(*, sample: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        record = SampleLibraryRecord.from_mapping(sample)
        accepted = review.get("sane", True) and not review.get("is_regression", False)
        target_group = "hard" if record.baseline_dice < 0.7 else record.sample_group
        return {"image_id": record.image_id, "accepted": accepted, "target_group": target_group, "review": review}


def _register_many(
    registry: SampleLibraryToolRegistry,
    agent: str,
    toolset: Any,
    specs: list[tuple[str, str, list[str], list[str], str]],
) -> None:
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
    registry = SampleLibraryToolRegistry()
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
            ("embed_mask_shape", "Build a compact mask-shape vector.", ["sample"], ["shape_embedding"], "Makes shape-based retrieval possible."),
            ("embed_visual_region_request", "Describe the visual crop needed for embedding.", ["sample", "crop_padding"], ["visual_embedding_request"], "Connects image crops to visual indexers."),
            ("embed_text_label", "Build a simple text-label vector.", ["labels"], ["text_embedding"], "Indexes semantic labels without requiring a model."),
            ("encode_boundary_features", "Encode boundary risk features.", ["sample"], ["boundary_signature"], "Feeds boundary-specialized retrieval."),
            ("build_hard_case_signature", "Create a hard-case signature string.", ["sample"], ["hard_case_signature"], "Clusters failures by cause."),
            ("index_polarity_groups", "Partition sample IDs by polarity/group.", ["samples"], ["polarity_index"], "Maintains positive, negative, boundary, and hard indexes."),
            ("route_site_aware_embedding", "Select the embedding index for a site and group.", ["sample", "default_index"], ["index_route"], "Reduces cross-site retrieval bias."),
            ("monitor_embedding_drift", "Detect embedding outliers against a centroid.", ["embedding", "centroid", "threshold"], ["drift_report"], "Finds samples outside the existing bank distribution."),
        ],
    )
    _register_many(
        registry,
        ResultReviewToolSet.agent_name,
        ResultReviewToolSet,
        [
            ("analyze_metric_delta", "Compare baseline and result metrics.", ["sample"], ["delta_summary"], "Measures whether the new method helped a sample."),
            ("generate_hard_case_delta_report", "Summarize gains on low-Dice hard cases.", ["rows", "thresholds", "quantiles"], ["hard_case_delta_report"], "Reports whether the hard-case bank actually improved."),
            ("classify_failure_case", "Classify the likely failure mode.", ["sample"], ["failure_mode"], "Turns bad results into actionable error classes."),
            ("check_confidence_consistency", "Find high-confidence wrong predictions.", ["sample"], ["confidence_audit"], "Flags dangerous confidence/quality mismatch."),
            ("review_mask_sanity", "Check prediction mask sanity.", ["sample"], ["mask_sanity"], "Catches empty, huge, or fragmented predictions."),
            ("detect_regression", "Detect meaningful regression vs baseline.", ["sample", "min_harm"], ["regression_flag"], "Protects against methods that hurt existing cases."),
            ("audit_exemplar_effect", "Judge whether exemplars helped or hurt.", ["sample"], ["exemplar_effect"], "Evaluates retrieval usefulness per sample."),
            ("update_continual_bank_item", "Prepare accepted results for continual bank update.", ["sample", "review"], ["continual_bank_item"], "Feeds verified cases back into the sample library."),
        ],
    )
    return registry


def explain_sample_library_toolsets() -> list[dict[str, Any]]:
    return create_sample_library_tool_registry().list_tool_specs()


def group_tool_specs_by_agent(specs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(str(spec.get("agent", "")), []).append(spec)
    return grouped
