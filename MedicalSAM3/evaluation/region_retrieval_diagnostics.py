"""结构化区域感知检索诊断。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _resize_map(value: Any, size: tuple[int, int], *, mode: str = "bilinear") -> torch.Tensor | None:
    """调整特征图到目标空间尺寸，支持多种插值模式。

    参数：
        - value: 输入值（张量或其他类型）。
        - size: 目标尺寸 (高, 宽)。
        - mode: 插值模式，默认为 "bilinear"。

    返回：
        - 调整后的张量，无效输入返回 None。
    """
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    tensor = value.detach().float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[-2:] == size:
        return tensor
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    if align_corners is None:
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)


def _mean_in_mask(values: torch.Tensor | None, mask: torch.Tensor | None) -> float:
    """计算掩码区域内的张量均值。

    参数：
        - values: 值张量。
        - mask: 二值掩码张量。

    返回：
        - 掩码区域内的均值。
    """
    if values is None or mask is None:
        return 0.0
    value_tensor = values.detach().float()
    mask_tensor = mask.detach().float()
    if value_tensor.shape != mask_tensor.shape:
        raise ValueError("values and mask must have the same shape")
    denom = mask_tensor.sum().clamp_min(1e-6)
    return float((value_tensor * mask_tensor).sum().item() / denom.item())


def _scalar_tensor(value: Any, default: float = 0.0) -> float:
    """将张量或标量值统一转换为浮点数。

    参数：
        - value: 输入值。
        - default: 默认值，转换失败时返回。

    返回：
        - 浮点数结果。
    """
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _region_type_statistics(adapter_aux: dict[str, Any]) -> dict[str, float]:
    """从适配器辅助输出中提取区域类型统计信息。

    参数：
        - adapter_aux: 适配器辅助输出字典。

    返回：
        - 包含各区域类型均值的字典。
    """
    payload = adapter_aux.get("region_type_statistics")
    if not isinstance(payload, dict):
        return {
            "boundary": 0.0,
            "low_confidence_lesion": 0.0,
            "high_confidence_foreground": 0.0,
            "high_confidence_background": 0.0,
        }
    return {
        key: _scalar_tensor(payload.get(key))
        for key in ["boundary", "low_confidence_lesion", "high_confidence_foreground", "high_confidence_background"]
    }


def build_region_retrieval_diagnostics(
    *,
    image_id: str,
    retrieval: dict[str, Any],
    adapter_aux: dict[str, Any],
    baseline_mask_logits: torch.Tensor | None,
    corrected_mask_logits: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
    sample_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建区域感知的检索诊断，分析检索区域激活、熵、置信度、病变增益等。

    参数：
        - image_id: 图像 ID。
        - retrieval: 检索结果字典。
        - adapter_aux: 适配器辅助输出。
        - baseline_mask_logits: 基线掩码 logits。
        - corrected_mask_logits: 修正后掩码 logits。
        - gt_mask: 真实掩码。
        - sample_metadata: 样本元数据。

    返回：
        - 包含各项区域诊断指标的字典。
    """
    size = tuple(int(item) for item in corrected_mask_logits.shape[-2:])
    baseline_logits = baseline_mask_logits.detach().float() if isinstance(baseline_mask_logits, torch.Tensor) and baseline_mask_logits.numel() > 0 else torch.zeros_like(corrected_mask_logits.detach().float())
    corrected_logits = corrected_mask_logits.detach().float()
    delta_logits = corrected_logits - baseline_logits

    retrieval_region_mask = _resize_map(adapter_aux.get("retrieval_region_mask"), size, mode="bilinear")
    if retrieval_region_mask is None:
        retrieval_region_mask = torch.zeros_like(corrected_logits)
    retrieval_region_binary = (retrieval_region_mask > 0.05).float()
    entropy_map = _resize_map(adapter_aux.get("segmentation_entropy_map"), size, mode="bilinear")
    confidence_map = _resize_map(adapter_aux.get("segmentation_confidence_map"), size, mode="bilinear")
    boundary_map = _resize_map(adapter_aux.get("boundary_uncertainty_map"), size, mode="bilinear")
    preserve_map = _resize_map(adapter_aux.get("high_confidence_preserve_mask"), size, mode="nearest")
    if preserve_map is None:
        preserve_map = torch.zeros_like(corrected_logits)

    resized_gt = None
    if isinstance(gt_mask, torch.Tensor) and gt_mask.numel() > 0:
        resized_gt = _resize_map(gt_mask, size, mode="nearest")
        if resized_gt is not None:
            resized_gt = (resized_gt > 0.5).float()

    if resized_gt is not None:
        lesion_mask = resized_gt
        background_mask = 1.0 - resized_gt
    else:
        lesion_mask = (torch.sigmoid(baseline_logits) >= 0.5).float()
        background_mask = 1.0 - lesion_mask

    boundary_mask = (boundary_map > 0.25).float() if boundary_map is not None else retrieval_region_binary
    multi_bank_fusion = retrieval.get("multi_bank_fusion", {}) if isinstance(retrieval.get("multi_bank_fusion"), dict) else {}
    sample_metadata = sample_metadata or {}

    return {
        "image_id": image_id,
        "sample_id": str(sample_metadata.get("sample_id") or sample_metadata.get("image_id") or image_id),
        "image_path": str(sample_metadata.get("image_path") or ""),
        "parsed_site_id": multi_bank_fusion.get("parsed_site_id", multi_bank_fusion.get("site_id")),
        "selected_site_id": multi_bank_fusion.get("site_id"),
        "expected_site_bank": multi_bank_fusion.get("expected_site_bank"),
        "fallback_reason": multi_bank_fusion.get("fallback_reason"),
        "selected_bank_paths": list(multi_bank_fusion.get("selected_bank_paths", [])),
        "retrieval_region_activation_ratio": float(retrieval_region_binary.flatten(1).mean(dim=1)[0].item()),
        "mean_entropy_in_region": _mean_in_mask(entropy_map, retrieval_region_binary),
        "mean_confidence_in_region": _mean_in_mask(confidence_map, retrieval_region_binary),
        "lesion_region_logit_gain": _mean_in_mask(delta_logits, lesion_mask * retrieval_region_binary),
        "background_logit_suppression": _mean_in_mask(-delta_logits, background_mask * retrieval_region_binary),
        "boundary_region_improvement_proxy": _mean_in_mask(delta_logits.abs(), boundary_mask * retrieval_region_binary),
        "high_confidence_region_modification_ratio": _scalar_tensor(adapter_aux.get("high_confidence_region_modification_ratio")),
        "protected_region_ratio": float(preserve_map.flatten(1).mean(dim=1)[0].item()),
        "train_bank_contribution": _scalar_tensor(multi_bank_fusion.get("train_contribution")),
        "site_bank_contribution": _scalar_tensor(multi_bank_fusion.get("site_contribution")),
        "region_type_statistics": _region_type_statistics(adapter_aux),
    }


def summarize_region_retrieval_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总区域检索诊断结果，计算各数值指标的均值、最小值和最大值。

    参数：
        - rows: 诊断行列表。

    返回：
        - 汇总统计字典。
    """
    if not rows:
        return {}
    numeric_keys = [
        "retrieval_region_activation_ratio",
        "mean_entropy_in_region",
        "mean_confidence_in_region",
        "lesion_region_logit_gain",
        "background_logit_suppression",
        "boundary_region_improvement_proxy",
        "high_confidence_region_modification_ratio",
        "protected_region_ratio",
        "train_bank_contribution",
        "site_bank_contribution",
    ]
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[key] = {
            "mean": float(tensor.mean().item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }
    summary["region_type_statistics"] = {}
    for key in ["boundary", "low_confidence_lesion", "high_confidence_foreground", "high_confidence_background"]:
        values = [float(row.get("region_type_statistics", {}).get(key, 0.0)) for row in rows]
        tensor = torch.tensor(values, dtype=torch.float32)
        summary["region_type_statistics"][key] = {
            "mean": float(tensor.mean().item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }
    return summary


def write_region_retrieval_diagnostics(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """将区域检索诊断行写入 JSONL 文件。

    参数：
        - path: 输出文件路径。
        - rows: 诊断行列表。

    返回：
        - 写入的文件路径。
    """
    target = Path(path)
    target.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return target


__all__ = [
    "build_region_retrieval_diagnostics",
    "summarize_region_retrieval_diagnostics",
    "write_region_retrieval_diagnostics",
]
