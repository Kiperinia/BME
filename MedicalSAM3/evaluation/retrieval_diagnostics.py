"""结构化检索诊断导出辅助工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from MedicalSAM3.exemplar_bank.bank import PrototypeBankEntry


def _mean_scalar(value: Optional[torch.Tensor]) -> float:
    """计算张量的均值并返回标量，无效输入返回 0.0。

    参数：
        - value: 输入张量。

    返回：
        - 均值标量值。
    """
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.detach().float().mean().item())


def _mean_abs_scalar(value: Optional[torch.Tensor]) -> float:
    """计算张量的绝对值的均值，无效输入返回 0.0。

    参数：
        - value: 输入张量。

    返回：
        - 绝对值均值标量值。
    """
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.detach().float().abs().mean().item())


def _batch_scalar(value: Any, batch_index: int = 0) -> float:
    """从批次张量中提取指定索引的均值标量。

    参数：
        - value: 批次张量或值。
        - batch_index: 批次索引。

    返回：
        - 均值标量值。
    """
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        if value.dim() == 0:
            return float(value.detach().float().item())
        if batch_index < value.shape[0]:
            return float(value[batch_index].detach().float().mean().item())
    return 0.0


def _score_list(value: Any) -> list[float]:
    """将张量或嵌套列表转换为扁平化的浮点数列表。

    参数：
        - value: 输入张量或列表。

    返回：
        - 浮点数列表。
    """
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, list):
        output: list[float] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                output.extend([float(entry) for entry in item.detach().cpu().flatten().tolist()])
            else:
                output.append(float(item))
        return output
    return []


def _serialize_entries(
    entries: list[PrototypeBankEntry],
    scores: list[float],
    weights: list[float],
) -> list[dict[str, Any]]:
    """将原型银行条目序列化为可 JSON 序列化的字典列表。

    参数：
        - entries: 原型银行条目列表。
        - scores: 相似度分数列表。
        - weights: 检索权重列表。

    返回：
        - 序列化后的字典列表。
    """
    serialized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        serialized.append(
            {
                "prototype_id": entry.prototype_id,
                "crop_path": entry.crop_path,
                "source_dataset": entry.source_dataset,
                "polarity": entry.polarity,
                "similarity_score": float(scores[index]) if index < len(scores) else 0.0,
                "retrieval_weight": float(weights[index]) if index < len(weights) else 0.0,
            }
        )
    return serialized


def _serialize_nested_topk(payload: dict[str, Any], polarity: str, batch_index: int) -> list[dict[str, Any]]:
    """序列化嵌套在 multi_bank_fusion 中的 top-k 条目信息。

    参数：
        - payload: multi_bank_fusion 的 payload 字典。
        - polarity: 极性（"positive" 或 "negative"）。
        - batch_index: 批次索引。

    返回：
        - 序列化后的条目字典列表。
    """
    entries = payload.get(f"{polarity}_entries", [])
    scores = payload.get(f"{polarity}_scores", [])
    weights = payload.get(f"{polarity}_weights")
    current_entries = entries[batch_index] if isinstance(entries, list) and batch_index < len(entries) else []
    current_scores = _score_list(scores[batch_index]) if isinstance(scores, list) and batch_index < len(scores) else []
    current_weights = _score_list(weights[batch_index]) if isinstance(weights, torch.Tensor) and batch_index < weights.shape[0] else []
    return _serialize_entries(current_entries, current_scores, current_weights)


def _stats(values: list[float]) -> dict[str, float]:
    """计算浮点数列表的基本统计量（均值、标准差、最小值、最大值）。

    参数：
        - values: 浮点数列表。

    返回：
        - 包含 mean、std、min、max 的字典。
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _histogram(values: list[float], bins: int = 10) -> dict[str, Any]:
    """计算浮点数列表的直方图分布。

    参数：
        - values: 浮点数列表。
        - bins: 柱数，默认为 10。

    返回：
        - 包含 min、max、edges、counts 的字典。
    """
    if not values:
        return {"min": 0.0, "max": 0.0, "edges": [], "counts": []}
    tensor = torch.tensor(values, dtype=torch.float32)
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if max_value <= min_value:
        max_value = min_value + 1e-6
    counts = torch.histc(tensor, bins=max(bins, 1), min=min_value, max=max_value)
    step = (max_value - min_value) / float(max(bins, 1))
    edges = [min_value + step * index for index in range(max(bins, 1) + 1)]
    return {
        "min": min_value,
        "max": max_value,
        "edges": edges,
        "counts": [int(round(value)) for value in counts.tolist()],
    }


def _activation_curve(scores: list[float], weights: list[float], bins: int = 10) -> dict[str, Any]:
    """计算分数-权重的激活曲线，按分数分箱统计平均权重。

    参数：
        - scores: 相似度分数列表。
        - weights: 权重列表。
        - bins: 柱数，默认为 10。

    返回：
        - 包含 edges、counts、mean_weight 的字典。
    """
    pair_count = min(len(scores), len(weights))
    if pair_count == 0:
        return {"min": 0.0, "max": 0.0, "edges": [], "counts": [], "mean_weight": []}
    score_tensor = torch.tensor(scores[:pair_count], dtype=torch.float32)
    weight_tensor = torch.tensor(weights[:pair_count], dtype=torch.float32)
    min_value = float(score_tensor.min().item())
    max_value = float(score_tensor.max().item())
    if max_value <= min_value:
        max_value = min_value + 1e-6
    bin_count = max(bins, 1)
    step = (max_value - min_value) / float(bin_count)
    edges = [min_value + step * index for index in range(bin_count + 1)]
    counts: list[int] = []
    mean_weight: list[float] = []
    for index in range(bin_count):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bin_count - 1:
            mask = (score_tensor >= lower) & (score_tensor <= upper)
        else:
            mask = (score_tensor >= lower) & (score_tensor < upper)
        counts.append(int(mask.sum().item()))
        mean_weight.append(float(weight_tensor[mask].mean().item()) if torch.any(mask) else 0.0)
    return {
        "min": min_value,
        "max": max_value,
        "edges": edges,
        "counts": counts,
        "mean_weight": mean_weight,
    }


def _collect_nested(rows: list[dict[str, Any]], *keys: str) -> list[float]:
    """从嵌套字典列表中按键路径收集浮点数值。

    参数：
        - rows: 字典列表。
        - *keys: 键路径（可变参数）。

    返回：
        - 收集到的浮点数列表。
    """
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in keys:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        try:
            values.append(float(current))
        except (TypeError, ValueError):
            continue
    return values


def build_retrieval_diagnostics(
    *,
    image_id: str,
    retrieval: dict[str, Any],
    adapter_aux: dict[str, Any],
    outputs: dict[str, Any],
    batch_index: int = 0,
    sample_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建检索诊断字典，包含 top-k 信息、相似度分数、门控诊断、多库融合等。

    参数：
        - image_id: 图像 ID。
        - retrieval: 检索结果字典。
        - adapter_aux: 适配器辅助输出。
        - outputs: 模型输出字典。
        - batch_index: 批次索引。
        - sample_metadata: 样本元数据。

    返回：
        - 完整的检索诊断字典。
    """
    positive_entries = retrieval.get("positive_entries", [[]])
    negative_entries = retrieval.get("negative_entries", [[]])
    positive_scores = retrieval.get("positive_scores", [])
    negative_scores = retrieval.get("negative_scores", [])
    positive_weights = retrieval.get("positive_weights")
    negative_weights = retrieval.get("negative_weights")

    current_positive_entries = positive_entries[batch_index] if batch_index < len(positive_entries) else []
    current_negative_entries = negative_entries[batch_index] if batch_index < len(negative_entries) else []
    current_positive_scores = _score_list(positive_scores[batch_index]) if batch_index < len(positive_scores) else []
    current_negative_scores = _score_list(negative_scores[batch_index]) if batch_index < len(negative_scores) else []
    current_positive_weights = _score_list(positive_weights[batch_index]) if isinstance(positive_weights, torch.Tensor) and batch_index < positive_weights.shape[0] else []
    current_negative_weights = _score_list(negative_weights[batch_index]) if isinstance(negative_weights, torch.Tensor) and batch_index < negative_weights.shape[0] else []
    retrieval_stability = retrieval.get("retrieval_stability", {})
    multi_bank_fusion = retrieval.get("multi_bank_fusion", {}) if isinstance(retrieval.get("multi_bank_fusion"), dict) else {}
    sample_metadata = sample_metadata or {}
    train_topk_payload = multi_bank_fusion.get("train_topk_exemplar", {}) if isinstance(multi_bank_fusion.get("train_topk_exemplar"), dict) else {}
    site_topk_payload = multi_bank_fusion.get("site_topk_exemplar", {}) if isinstance(multi_bank_fusion.get("site_topk_exemplar"), dict) else {}

    positive_contribution = _mean_scalar(adapter_aux.get("positive_context_map"))
    negative_contribution = _mean_scalar(adapter_aux.get("negative_context_map"))
    gate_value = _mean_scalar(adapter_aux.get("fusion_gate_map"))
    influence_strength = abs(positive_contribution) + abs(negative_contribution) + _mean_abs_scalar(adapter_aux.get("fused_delta"))
    positive_mean_score = _mean_scalar(adapter_aux.get("positive_similarity_score"))
    negative_mean_score = _mean_scalar(adapter_aux.get("negative_similarity_score"))
    if positive_mean_score == 0.0 and current_positive_scores:
        positive_mean_score = float(sum(current_positive_scores) / max(len(current_positive_scores), 1))
    if negative_mean_score == 0.0 and current_negative_scores:
        negative_mean_score = float(sum(current_negative_scores) / max(len(current_negative_scores), 1))

    return {
        "image_id": image_id,
        "top_k": {
            "positive": len(current_positive_entries),
            "negative": len(current_negative_entries),
        },
        "top_k_retrieved_exemplars": {
            "positive": _serialize_entries(current_positive_entries, current_positive_scores, current_positive_weights),
            "negative": _serialize_entries(current_negative_entries, current_negative_scores, current_negative_weights),
        },
        "similarity_score": {
            "positive_mean": positive_mean_score,
            "negative_mean": negative_mean_score,
            "margin": positive_mean_score - negative_mean_score,
        },
        "fusion_gate_value": gate_value,
        "gate_diagnostics": {
            "fusion_gate_mean": gate_value,
            "policy_gate_mean": _mean_scalar(adapter_aux.get("policy_gate_map")),
            "retrieval_confidence_gate": _mean_scalar(adapter_aux.get("retrieval_confidence_gate")),
            "positive_confidence_gate": _mean_scalar(adapter_aux.get("positive_confidence_gate")),
            "negative_confidence_gate": _mean_scalar(adapter_aux.get("negative_confidence_gate")),
            "positive_calibrated_weight": _mean_scalar(adapter_aux.get("positive_calibrated_weight")),
            "negative_calibrated_weight": _mean_scalar(adapter_aux.get("negative_calibrated_weight")),
            "positive_similarity_weight": _mean_scalar(adapter_aux.get("positive_similarity_weight")),
            "negative_similarity_weight": _mean_scalar(adapter_aux.get("negative_similarity_weight")),
            "retrieval_similarity_weight": _mean_scalar(adapter_aux.get("retrieval_similarity_weight")),
            "similarity_temperature": _mean_scalar(adapter_aux.get("similarity_temperature")),
            "segmentation_confidence": _mean_scalar(adapter_aux.get("segmentation_confidence")),
            "segmentation_uncertainty": _mean_scalar(adapter_aux.get("segmentation_uncertainty")),
            "segmentation_entropy": _mean_scalar(adapter_aux.get("segmentation_entropy")),
            "uncertainty_gate": _mean_scalar(adapter_aux.get("uncertainty_gate")),
            "retrieval_activation_ratio": _mean_scalar(adapter_aux.get("retrieval_activation_ratio")),
            "retrieval_suppression_ratio": _mean_scalar(adapter_aux.get("retrieval_suppression_ratio")),
            "similarity_activation_ratio": _mean_scalar(adapter_aux.get("similarity_activation_ratio")),
            "residual_strength": _mean_scalar(adapter_aux.get("residual_strength")),
            "high_confidence_region_modification_ratio": _mean_scalar(adapter_aux.get("high_confidence_region_modification_ratio")),
            "used_baseline_uncertainty": _mean_scalar(adapter_aux.get("used_baseline_uncertainty")),
        },
        "positive_contribution": positive_contribution,
        "negative_contribution": negative_contribution,
        "retrieval_influence_strength": influence_strength,
        "retrieval_stability": {
            "positive_similarity_std": _batch_scalar(retrieval_stability.get("positive_similarity_std"), batch_index),
            "negative_similarity_std": _batch_scalar(retrieval_stability.get("negative_similarity_std"), batch_index),
            "positive_weight_entropy": _batch_scalar(retrieval_stability.get("positive_weight_entropy"), batch_index),
            "negative_weight_entropy": _batch_scalar(retrieval_stability.get("negative_weight_entropy"), batch_index),
        },
        "multi_bank_fusion": {
            "site_id": multi_bank_fusion.get("site_id"),
            "parsed_site_id": multi_bank_fusion.get("parsed_site_id", multi_bank_fusion.get("site_id")),
            "expected_site_bank": multi_bank_fusion.get("expected_site_bank"),
            "fallback_reason": multi_bank_fusion.get("fallback_reason"),
            "selected_bank_paths": list(multi_bank_fusion.get("selected_bank_paths", [])),
            "fallback_to_train_bank": bool(multi_bank_fusion.get("fallback_to_train_bank", False)),
            "train_topk": {
                "positive": _serialize_nested_topk(train_topk_payload, "positive", batch_index),
                "negative": _serialize_nested_topk(train_topk_payload, "negative", batch_index),
            },
            "site_topk": {
                "positive": _serialize_nested_topk(site_topk_payload, "positive", batch_index),
                "negative": _serialize_nested_topk(site_topk_payload, "negative", batch_index),
            },
            "train_similarity": {
                "positive_mean": _batch_scalar(multi_bank_fusion.get("train_similarity", {}).get("positive_mean"), batch_index),
                "negative_mean": _batch_scalar(multi_bank_fusion.get("train_similarity", {}).get("negative_mean"), batch_index),
                "margin": _batch_scalar(multi_bank_fusion.get("train_similarity", {}).get("margin"), batch_index),
            },
            "site_similarity": {
                "positive_mean": _batch_scalar(multi_bank_fusion.get("site_similarity", {}).get("positive_mean"), batch_index),
                "negative_mean": _batch_scalar(multi_bank_fusion.get("site_similarity", {}).get("negative_mean"), batch_index),
                "margin": _batch_scalar(multi_bank_fusion.get("site_similarity", {}).get("margin"), batch_index),
            },
            "train_contribution": _batch_scalar(multi_bank_fusion.get("train_contribution"), batch_index),
            "site_contribution": _batch_scalar(multi_bank_fusion.get("site_contribution"), batch_index),
            "final_fusion_weight": _batch_scalar(multi_bank_fusion.get("final_fusion_weight"), batch_index),
        },
        "site_bank_resolution": {
            "sample_id": str(sample_metadata.get("sample_id") or sample_metadata.get("image_id") or image_id),
            "image_path": str(sample_metadata.get("image_path") or ""),
            "parsed_site_id": multi_bank_fusion.get("parsed_site_id", multi_bank_fusion.get("site_id")),
            "expected_site_bank": multi_bank_fusion.get("expected_site_bank"),
            "fallback_reason": multi_bank_fusion.get("fallback_reason"),
        },
        "wrapper_retrieval_summary": outputs.get("intermediate_features", {}).get("retrieval_prior", {}),
    }


def summarize_retrieval_diagnostics(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    """汇总检索诊断结果，计算相似度分布、门控分布、策略分布、校准权重分布等统计量。

    参数：
        - rows: 诊断行列表。
        - bins: 直方图柱数，默认为 10。

    返回：
        - 汇总统计字典。
    """
    positive_similarity = _collect_nested(rows, "similarity_score", "positive_mean")
    negative_similarity = _collect_nested(rows, "similarity_score", "negative_mean")
    similarity_margin = _collect_nested(rows, "similarity_score", "margin")
    fusion_gate = _collect_nested(rows, "gate_diagnostics", "fusion_gate_mean")
    policy_gate = _collect_nested(rows, "gate_diagnostics", "policy_gate_mean")
    retrieval_confidence = _collect_nested(rows, "gate_diagnostics", "retrieval_confidence_gate")
    positive_gate = _collect_nested(rows, "gate_diagnostics", "positive_confidence_gate")
    negative_gate = _collect_nested(rows, "gate_diagnostics", "negative_confidence_gate")
    positive_weight = _collect_nested(rows, "gate_diagnostics", "positive_calibrated_weight")
    negative_weight = _collect_nested(rows, "gate_diagnostics", "negative_calibrated_weight")
    positive_similarity_weight = _collect_nested(rows, "gate_diagnostics", "positive_similarity_weight")
    negative_similarity_weight = _collect_nested(rows, "gate_diagnostics", "negative_similarity_weight")
    retrieval_similarity_weight = _collect_nested(rows, "gate_diagnostics", "retrieval_similarity_weight")
    similarity_temperature = _collect_nested(rows, "gate_diagnostics", "similarity_temperature")
    segmentation_confidence = _collect_nested(rows, "gate_diagnostics", "segmentation_confidence")
    segmentation_uncertainty = _collect_nested(rows, "gate_diagnostics", "segmentation_uncertainty")
    segmentation_entropy = _collect_nested(rows, "gate_diagnostics", "segmentation_entropy")
    uncertainty_gate = _collect_nested(rows, "gate_diagnostics", "uncertainty_gate")
    activation_ratio = _collect_nested(rows, "gate_diagnostics", "retrieval_activation_ratio")
    suppression_ratio = _collect_nested(rows, "gate_diagnostics", "retrieval_suppression_ratio")
    similarity_activation = _collect_nested(rows, "gate_diagnostics", "similarity_activation_ratio")
    residual_strength = _collect_nested(rows, "gate_diagnostics", "residual_strength")
    high_confidence_region_modification_ratio = _collect_nested(rows, "gate_diagnostics", "high_confidence_region_modification_ratio")
    used_baseline_uncertainty = _collect_nested(rows, "gate_diagnostics", "used_baseline_uncertainty")
    influence = _collect_nested(rows, "retrieval_influence_strength")
    train_contribution = _collect_nested(rows, "multi_bank_fusion", "train_contribution")
    site_contribution = _collect_nested(rows, "multi_bank_fusion", "site_contribution")
    final_fusion_weight = _collect_nested(rows, "multi_bank_fusion", "final_fusion_weight")
    positive_similarity_std = _collect_nested(rows, "retrieval_stability", "positive_similarity_std")
    negative_similarity_std = _collect_nested(rows, "retrieval_stability", "negative_similarity_std")
    positive_weight_entropy = _collect_nested(rows, "retrieval_stability", "positive_weight_entropy")
    negative_weight_entropy = _collect_nested(rows, "retrieval_stability", "negative_weight_entropy")
    retrieval_score = [max(positive_similarity[index], negative_similarity[index]) for index in range(min(len(positive_similarity), len(negative_similarity)))]
    return {
        "count": len(rows),
        "similarity_distribution": {
            "positive": _stats(positive_similarity),
            "negative": _stats(negative_similarity),
            "margin": _stats(similarity_margin),
            "positive_histogram": _histogram(positive_similarity, bins=bins),
            "negative_histogram": _histogram(negative_similarity, bins=bins),
            "margin_histogram": _histogram(similarity_margin, bins=bins),
        },
        "gate_distribution": {
            "fusion_gate": _stats(fusion_gate),
            "policy_gate": _stats(policy_gate),
            "retrieval_confidence_gate": _stats(retrieval_confidence),
            "positive_confidence_gate": _stats(positive_gate),
            "negative_confidence_gate": _stats(negative_gate),
            "fusion_gate_histogram": _histogram(fusion_gate, bins=bins),
            "policy_gate_histogram": _histogram(policy_gate, bins=bins),
            "retrieval_confidence_histogram": _histogram(retrieval_confidence, bins=bins),
        },
        "policy_distribution": {
            "segmentation_confidence": _stats(segmentation_confidence),
            "segmentation_uncertainty": _stats(segmentation_uncertainty),
            "segmentation_entropy": _stats(segmentation_entropy),
            "uncertainty_gate": _stats(uncertainty_gate),
            "retrieval_activation_ratio": _stats(activation_ratio),
            "retrieval_suppression_ratio": _stats(suppression_ratio),
            "similarity_activation_ratio": _stats(similarity_activation),
            "residual_strength": _stats(residual_strength),
            "high_confidence_region_modification_ratio": _stats(high_confidence_region_modification_ratio),
            "used_baseline_uncertainty": _stats(used_baseline_uncertainty),
            "segmentation_confidence_histogram": _histogram(segmentation_confidence, bins=bins),
            "segmentation_uncertainty_histogram": _histogram(segmentation_uncertainty, bins=bins),
            "segmentation_entropy_histogram": _histogram(segmentation_entropy, bins=bins),
            "uncertainty_gate_histogram": _histogram(uncertainty_gate, bins=bins),
            "retrieval_activation_histogram": _histogram(activation_ratio, bins=bins),
            "retrieval_suppression_histogram": _histogram(suppression_ratio, bins=bins),
        },
        "multi_bank_fusion": {
            "train_contribution": _stats(train_contribution),
            "site_contribution": _stats(site_contribution),
            "final_fusion_weight": _stats(final_fusion_weight),
        },
        "calibrated_weight_distribution": {
            "positive": _stats(positive_weight),
            "negative": _stats(negative_weight),
            "positive_histogram": _histogram(positive_weight, bins=bins),
            "negative_histogram": _histogram(negative_weight, bins=bins),
        },
        "similarity_weight_distribution": {
            "positive": _stats(positive_similarity_weight),
            "negative": _stats(negative_similarity_weight),
            "retrieval": _stats(retrieval_similarity_weight),
            "temperature": _stats(similarity_temperature),
            "positive_histogram": _histogram(positive_similarity_weight, bins=bins),
            "negative_histogram": _histogram(negative_similarity_weight, bins=bins),
            "retrieval_histogram": _histogram(retrieval_similarity_weight, bins=bins),
        },
        "activation_curve": {
            "positive": _activation_curve(positive_similarity, positive_similarity_weight, bins=bins),
            "negative": _activation_curve(negative_similarity, negative_similarity_weight, bins=bins),
            "retrieval": _activation_curve(retrieval_score, retrieval_similarity_weight, bins=bins),
        },
        "retrieval_stability": {
            "positive_similarity_std": _stats(positive_similarity_std),
            "negative_similarity_std": _stats(negative_similarity_std),
            "positive_weight_entropy": _stats(positive_weight_entropy),
            "negative_weight_entropy": _stats(negative_weight_entropy),
        },
        "influence_strength": {
            **_stats(influence),
            "histogram": _histogram(influence, bins=bins),
        },
    }


def write_retrieval_diagnostics(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """将检索诊断行写入文件（JSONL 或 JSON 格式）。

    参数：
        - path: 输出文件路径。
        - rows: 诊断行列表。

    返回：
        - 写入的文件路径。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".jsonl":
        destination.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    else:
        destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return destination


def write_retrieval_diagnostics_summary(path: str | Path, rows: list[dict[str, Any]], bins: int = 10) -> Path:
    """生成并保存检索诊断汇总统计 JSON 文件。

    参数：
        - path: 输出文件路径。
        - rows: 诊断行列表。
        - bins: 直方图柱数。

    返回：
        - 写入的文件路径。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summarize_retrieval_diagnostics(rows, bins=bins), indent=2), encoding="utf-8")
    return destination
