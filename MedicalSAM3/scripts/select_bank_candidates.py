"""从评估产物中选择平衡的持续库候选。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.scripts.common import ensure_dir
from MedicalSAM3.utils.polypgen_site import resolve_polypgen_site


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CandidateRecord:
    """表示一个候选示例的记录，包含基本元数据、评分和检索信息。

    参数：
        - 无

    返回：
        - 提供用于策展流程的候选记录实例
    """
    image_id: str
    image_path: str
    mask_path: str
    dataset_name: str
    site_id: str
    patient_id: str
    polarity: str
    score: float
    reason_flags: list[str]
    metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    retrieval_vs_baseline: dict[str, float]
    retrieval_sensitivity: dict[str, Any]
    prompt_sensitivity_score: float
    retrieval_influence_strength: float
    lesion_area: float
    prediction_area: float
    selected_positive: list[str]
    selected_negative: list[str]
    perceptual_hash: Optional[int]
    feature_vector: Optional[np.ndarray]
    raw_row: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionState:
    """追踪选择过程中的状态（已用图像、患者、哈希、特征），避免重复。

    参数：
        - 无

    返回：
        - 提供用于去重选择的状态实例
    """
    used_image_ids: set[str] = field(default_factory=set)
    used_patients: set[str] = field(default_factory=set)
    selected_hashes: list[int] = field(default_factory=list)
    selected_features: list[np.ndarray] = field(default_factory=list)


def _read_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """从 JSON 或 JSONL 文件读取行数据。

    参数：
        - path: 文件路径

    返回：
        - 字典列表
    """
    if path is None:
        return []
    target = Path(path)
    if not target.exists():
        return []
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _image_id_from_row(row: dict[str, Any]) -> str:
    """从行数据中提取图像 ID。

    参数：
        - row: 行数据字典

    返回：
        - 图像 ID 字符串
    """
    explicit = str(row.get("image_id") or "").strip()
    if explicit:
        return explicit
    image_path = str(row.get("image_path") or "").strip()
    if image_path:
        return Path(image_path).stem
    return ""


def _merge_missing_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    """将 source 中缺失的字段合并到 target。

    参数：
        - target: 目标字典
        - source: 源字典

    返回：
        - 无返回值，仅更新 target 的副作用
    """
    for key, value in source.items():
        current = target.get(key)
        if key not in target or current is None or current == "" or current == [] or current == {}:
            target[key] = value


def _merge_artifacts(
    *,
    metrics_rows: list[dict[str, Any]],
    diagnostics_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并多个来源的行数据（指标、诊断、区域、提示敏感性）为统一的记录。

    参数：
        - metrics_rows: 指标行列表
        - diagnostics_rows: 检索诊断行列表
        - region_rows: 区域检索行列表
        - prompt_rows: 提示敏感性行列表

    返回：
        - 按 image_id 去重合并后的行列表
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        image_id = _image_id_from_row(row)
        if not image_id:
            continue
        merged[image_id] = dict(row)
        merged[image_id]["image_id"] = image_id
    for row in diagnostics_rows:
        image_id = _image_id_from_row(row)
        if not image_id:
            continue
        target = merged.setdefault(image_id, {"image_id": image_id})
        _merge_missing_fields(target, row)
        target["retrieval_diagnostics"] = row
    for row in region_rows:
        image_id = _image_id_from_row(row)
        if not image_id:
            continue
        target = merged.setdefault(image_id, {"image_id": image_id})
        _merge_missing_fields(target, row)
        target["region_retrieval_row"] = row
    for row in prompt_rows:
        image_id = _image_id_from_row(row)
        if not image_id:
            continue
        target = merged.setdefault(image_id, {"image_id": image_id})
        _merge_missing_fields(target, row)
        target["prompt_sensitivity_row"] = row
    return list(merged.values())


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全地将值转换为浮点数。

    参数：
        - value: 输入值
        - default: 转换失败时的默认值

    返回：
        - 浮点数值
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    """安全地获取字典值。

    参数：
        - value: 输入值

    返回：
        - 字典或空字典
    """
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    """安全地获取列表值。

    参数：
        - value: 输入值

    返回：
        - 列表或空列表
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalize_vector(value: np.ndarray | list[float] | tuple[float, ...] | None) -> Optional[np.ndarray]:
    """对向量进行 L2 归一化。

    参数：
        - value: 输入向量

    返回：
        - 归一化后的向量或 None
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return None
    return array / norm


def _vector_from_value(value: Any) -> Optional[np.ndarray]:
    """从各种格式的值中提取特征向量。

    参数：
        - value: 输入值（列表、元组或 JSON 字符串）

    返回：
        - 归一化向量或 None
    """
    if isinstance(value, (list, tuple)):
        return _normalize_vector(list(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list):
            return _normalize_vector(payload)
    return None


def _image_feature_vector(image_path: str | Path | None, size: int = 16) -> Optional[np.ndarray]:
    """从图像文件计算感知特征向量（灰度缩略图 L2 归一化）。

    参数：
        - image_path: 图像路径
        - size: 缩略图尺寸

    返回：
        - 归一化特征向量或 None
    """
    if not image_path:
        return None
    target = Path(image_path)
    if not target.exists() or target.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None
    image = Image.open(target).convert("L").resize((size, size), resample=Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32).reshape(-1)
    pixels = pixels - float(pixels.mean())
    return _normalize_vector(pixels)


def _extract_feature_vector(row: dict[str, Any]) -> Optional[np.ndarray]:
    """从行数据的多个可能字段中提取特征向量。

    参数：
        - row: 行数据字典

    返回：
        - 特征向量或 None
    """
    for key in ("feature_vector", "candidate_feature", "image_feature", "query_feature_vector"):
        vector = _vector_from_value(row.get(key))
        if vector is not None:
            return vector
    for nested_key in ("retrieval_diagnostics", "prompt_sensitivity_row"):
        nested = _safe_dict(row.get(nested_key))
        for key in ("feature_vector", "candidate_feature", "image_feature", "query_feature_vector"):
            vector = _vector_from_value(nested.get(key))
            if vector is not None:
                return vector
    return _image_feature_vector(row.get("image_path"))


def _average_hash(path: str | Path, hash_size: int = 8) -> Optional[int]:
    """计算图像的平均哈希值用于快速去重。

    参数：
        - path: 图像路径
        - hash_size: 哈希尺寸

    返回：
        - 哈希整数值或 None
    """
    target = Path(path)
    if not target.exists() or target.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None
    image = Image.open(target).convert("L").resize((hash_size, hash_size), resample=Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = (pixels >= threshold).astype(np.uint8).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(left: int, right: int) -> int:
    """计算两个哈希值之间的汉明距离。

    参数：
        - left: 左哈希值
        - right: 右哈希值

    返回：
        - 汉明距离
    """
    return int((left ^ right).bit_count())


def _patient_id_from_value(value: str) -> str:
    """从字符串值中解析患者 ID。

    参数：
        - value: 输入字符串

    返回：
        - 患者 ID 字符串
    """
    normalized = value.strip()
    if not normalized:
        return ""
    stem = Path(normalized).stem
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    parts = [part for part in stem.split("_") if part]
    if len(parts) >= 2 and parts[0].upper() in {f"C{index}" for index in range(1, 7)}:
        return f"{parts[0].upper()}_{parts[1]}"
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def _infer_site_id(row: dict[str, Any]) -> str:
    """从行数据中推断站点 ID。

    参数：
        - row: 行数据字典

    返回：
        - 站点 ID 字符串
    """
    site_id = resolve_polypgen_site(
        image_path=row.get("image_path"),
        metadata=row,
        sample_id=row.get("image_id"),
        dataset_name=row.get("dataset_name"),
        warn=False,
    )
    if site_id:
        return site_id
    dataset_name = str(row.get("dataset_name") or "").strip()
    return dataset_name or "unknown"


def _extract_selected_ids(row: dict[str, Any], polarity: str) -> list[str]:
    """从行数据中提取检索时选中的原型 ID。

    参数：
        - row: 行数据字典
        - polarity: 极性（positive/negative）

    返回：
        - 原型 ID 列表
    """
    direct_key = f"selected_{polarity}"
    direct = [str(item) for item in _safe_list(row.get(direct_key)) if str(item).strip()]
    if direct:
        return direct
    diagnostics = _safe_dict(row.get("retrieval_diagnostics"))
    entries = _safe_list(_safe_dict(diagnostics.get("top_k_retrieved_exemplars")).get(polarity))
    identifiers = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prototype_id = str(entry.get("prototype_id") or "").strip()
        if prototype_id:
            identifiers.append(prototype_id)
    return identifiers


def _retrieval_delta(row: dict[str, Any]) -> dict[str, float]:
    """计算检索相对于基线的各项指标增量。

    参数：
        - row: 行数据字典

    返回：
        - 各指标增量字典
    """
    delta = _safe_dict(row.get("retrieval_vs_baseline"))
    metrics = _safe_dict(row.get("metrics"))
    baseline = _safe_dict(row.get("baseline_metrics"))
    return {
        "Dice Delta": _safe_float(delta.get("Dice Delta"), _safe_float(metrics.get("Dice")) - _safe_float(baseline.get("Dice"))),
        "Boundary F1 Delta": _safe_float(delta.get("Boundary F1 Delta"), _safe_float(metrics.get("Boundary F1")) - _safe_float(baseline.get("Boundary F1"))),
        "FNR Delta": _safe_float(delta.get("FNR Delta"), _safe_float(metrics.get("False Negative Rate")) - _safe_float(baseline.get("False Negative Rate"))),
        "FPR Delta": _safe_float(delta.get("FPR Delta"), _safe_float(metrics.get("False Positive Rate")) - _safe_float(baseline.get("False Positive Rate"))),
        "HD95 Delta": _safe_float(delta.get("HD95 Delta"), _safe_float(metrics.get("HD95")) - _safe_float(baseline.get("HD95"))),
        "ASSD Delta": _safe_float(delta.get("ASSD Delta"), _safe_float(metrics.get("ASSD")) - _safe_float(baseline.get("ASSD"))),
    }


def _prompt_sensitivity_score(row: dict[str, Any]) -> float:
    """从行数据中提取提示敏感性评分。

    参数：
        - row: 行数据字典

    返回：
        - 提示敏感性评分
    """
    prompt_row = _safe_dict(row.get("prompt_sensitivity_row"))
    prompt = _safe_dict(prompt_row.get("prompt_sensitivity"))
    return _safe_float(prompt.get("prompt_sensitivity_score"))


def _retrieval_influence_strength(row: dict[str, Any]) -> float:
    """从行数据中提取检索影响强度。

    参数：
        - row: 行数据字典

    返回：
        - 检索影响强度值
    """
    diagnostics = _safe_dict(row.get("retrieval_diagnostics"))
    return _safe_float(diagnostics.get("retrieval_influence_strength"), _safe_float(row.get("retrieval_influence_strength")))


def _positive_score(row: dict[str, Any]) -> float:
    """计算正例候选的综合评分。

    参数：
        - row: 行数据字典

    返回：
        - 正例评分（值越高越适合作为正例）
    """
    metrics = _safe_dict(row.get("metrics"))
    delta = _retrieval_delta(row)
    prompt_score = _prompt_sensitivity_score(row)
    influence = _retrieval_influence_strength(row)
    lesion_area = _safe_float(row.get("lesion_area"))
    prediction_area = _safe_float(row.get("prediction_area"))

    score = 0.0
    score += 0.55 * max(delta["Dice Delta"], 0.0)
    score += 0.20 * max(delta["Boundary F1 Delta"], 0.0)
    score += 0.10 * max(-delta["FNR Delta"], 0.0)
    score += 0.05 * max(-delta["ASSD Delta"], 0.0)
    score += 0.12 * _safe_float(metrics.get("Dice"))
    score += 0.06 * _safe_float(metrics.get("Boundary F1"))
    score += 0.08 * min(max(influence, 0.0), 1.0)
    score += 0.06 * max(0.0, 1.0 - min(prompt_score * 4.0, 1.0))
    score -= 0.10 * max(delta["FPR Delta"], 0.0)
    if lesion_area <= 0.0:
        score -= 1.0
    if lesion_area > 0.0 and prediction_area <= 0.0:
        score -= 0.2
    return score


def _negative_score(row: dict[str, Any]) -> float:
    """计算负例候选的综合评分。

    参数：
        - row: 行数据字典

    返回：
        - 负例评分（值越高越适合作为负例）
    """
    metrics = _safe_dict(row.get("metrics"))
    delta = _retrieval_delta(row)
    prompt_score = _prompt_sensitivity_score(row)
    influence = _retrieval_influence_strength(row)
    lesion_area = _safe_float(row.get("lesion_area"))
    prediction_area = _safe_float(row.get("prediction_area"))
    over_prediction = max(prediction_area - lesion_area, 0.0) / max(lesion_area, 1.0)

    score = 0.0
    score += 0.35 * max(delta["FPR Delta"], 0.0)
    score += 0.25 * max(-delta["Dice Delta"], 0.0)
    score += 0.15 * max(-delta["Boundary F1 Delta"], 0.0)
    score += 0.10 * _safe_float(metrics.get("False Positive Rate"))
    score += 0.10 * min(over_prediction, 1.0)
    score += 0.10 * prompt_score
    score += 0.05 * min(max(influence, 0.0), 1.0)
    return score


def _is_positive_candidate(row: dict[str, Any]) -> bool:
    """判断行数据是否适合作为正例候选。

    参数：
        - row: 行数据字典

    返回：
        - 是否为合格的正例候选
    """
    delta = _retrieval_delta(row)
    metrics = _safe_dict(row.get("metrics"))
    lesion_area = _safe_float(row.get("lesion_area"))
    if lesion_area <= 0.0:
        return False
    return any(
        [
            delta["Dice Delta"] > 0.0,
            delta["Boundary F1 Delta"] > 0.0,
            delta["FNR Delta"] < 0.0,
            _safe_float(metrics.get("Dice")) >= 0.85 and delta["FPR Delta"] <= 0.0,
        ]
    )


def _is_negative_candidate(row: dict[str, Any]) -> bool:
    """判断行数据是否适合作为负例候选。

    参数：
        - row: 行数据字典

    返回：
        - 是否为合格的负例候选
    """
    delta = _retrieval_delta(row)
    lesion_area = _safe_float(row.get("lesion_area"))
    prediction_area = _safe_float(row.get("prediction_area"))
    return any(
        [
            delta["FPR Delta"] > 0.0,
            delta["Dice Delta"] < 0.0,
            delta["Boundary F1 Delta"] < 0.0,
            prediction_area > lesion_area * 1.1,
        ]
    )


def _reason_flags(row: dict[str, Any], polarity: str) -> list[str]:
    """生成候选的标记原因列表。

    参数：
        - row: 行数据字典
        - polarity: 极性（positive/negative）

    返回：
        - 原因标签列表
    """
    delta = _retrieval_delta(row)
    flags: list[str] = []
    if polarity == "positive":
        if delta["Dice Delta"] > 0.0:
            flags.append("retrieval_dice_gain")
        if delta["Boundary F1 Delta"] > 0.0:
            flags.append("retrieval_boundary_gain")
        if delta["FNR Delta"] < 0.0:
            flags.append("retrieval_fnr_reduction")
        if _prompt_sensitivity_score(row) <= 0.05:
            flags.append("prompt_stable")
    else:
        if delta["FPR Delta"] > 0.0:
            flags.append("retrieval_fp_increase")
        if delta["Dice Delta"] < 0.0:
            flags.append("retrieval_dice_drop")
        if _prompt_sensitivity_score(row) > 0.05:
            flags.append("prompt_unstable")
        if _safe_float(row.get("prediction_area")) > _safe_float(row.get("lesion_area")):
            flags.append("over_prediction")
    return flags


def _build_candidate(row: dict[str, Any], polarity: str) -> CandidateRecord:
    """从行数据构建候选记录。

    参数：
        - row: 行数据字典
        - polarity: 极性

    返回：
        - 候选记录实例
    """
    image_id = _image_id_from_row(row)
    image_path = str(row.get("image_path") or "")
    mask_path = str(row.get("mask_path") or "")
    dataset_name = str(row.get("dataset_name") or "")
    site_id = _infer_site_id(row)
    patient_id = _patient_id_from_value(image_id or image_path)
    metrics = _safe_dict(row.get("metrics"))
    baseline_metrics = _safe_dict(row.get("baseline_metrics"))
    retrieval_vs_baseline = _retrieval_delta(row)
    retrieval_sensitivity = _safe_dict(row.get("retrieval_sensitivity"))
    score = _positive_score(row) if polarity == "positive" else _negative_score(row)
    return CandidateRecord(
        image_id=image_id,
        image_path=image_path,
        mask_path=mask_path,
        dataset_name=dataset_name,
        site_id=site_id,
        patient_id=patient_id,
        polarity=polarity,
        score=score,
        reason_flags=_reason_flags(row, polarity),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        retrieval_vs_baseline=retrieval_vs_baseline,
        retrieval_sensitivity=retrieval_sensitivity,
        prompt_sensitivity_score=_prompt_sensitivity_score(row),
        retrieval_influence_strength=_retrieval_influence_strength(row),
        lesion_area=_safe_float(row.get("lesion_area")),
        prediction_area=_safe_float(row.get("prediction_area")),
        selected_positive=_extract_selected_ids(row, "positive"),
        selected_negative=_extract_selected_ids(row, "negative"),
        perceptual_hash=_average_hash(image_path),
        feature_vector=_extract_feature_vector(row),
        raw_row=row,
    )


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个向量的余弦相似度。

    参数：
        - left: 左向量
        - right: 右向量

    返回：
        - 余弦相似度（-1~1）
    """
    if left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right))


def _dedup_reason(
    candidate: CandidateRecord,
    state: SelectionState,
    *,
    max_hash_distance: int,
    feature_similarity_threshold: float,
) -> Optional[str]:
    """检查候选是否与已选中的重复，若重复则返回原因。

    参数：
        - candidate: 候选记录
        - state: 选择状态
        - max_hash_distance: 哈希最大允许距离
        - feature_similarity_threshold: 特征相似度阈值

    返回：
        - 重复原因字符串或 None（不重复）
    """
    if candidate.image_id in state.used_image_ids:
        return "duplicate_image_id"
    if candidate.patient_id and candidate.patient_id in state.used_patients:
        return "duplicate_patient"
    if candidate.perceptual_hash is not None:
        for existing_hash in state.selected_hashes:
            if _hamming_distance(candidate.perceptual_hash, existing_hash) <= max_hash_distance:
                return "perceptual_hash_duplicate"
    if candidate.feature_vector is not None:
        for existing_feature in state.selected_features:
            if _cosine_similarity(candidate.feature_vector, existing_feature) >= feature_similarity_threshold:
                return "feature_similarity_duplicate"
    return None


def _register_candidate(candidate: CandidateRecord, state: SelectionState) -> None:
    """将选中的候选注册到状态中。

    参数：
        - candidate: 候选记录
        - state: 选择状态

    返回：
        - 无返回值，仅更新 state 的副作用
    """
    state.used_image_ids.add(candidate.image_id)
    if candidate.patient_id:
        state.used_patients.add(candidate.patient_id)
    if candidate.perceptual_hash is not None:
        state.selected_hashes.append(candidate.perceptual_hash)
    if candidate.feature_vector is not None:
        state.selected_features.append(candidate.feature_vector)


def _group_candidates(candidates: list[CandidateRecord]) -> dict[str, deque[CandidateRecord]]:
    """按站点 ID 对候选进行分组。

    参数：
        - candidates: 候选记录列表

    返回：
        - 站点 ID 到候选队列的字典
    """
    grouped: dict[str, deque[CandidateRecord]] = defaultdict(deque)
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.image_id)):
        grouped[candidate.site_id].append(candidate)
    return dict(grouped)


def _pick_candidate(
    grouped: dict[str, deque[CandidateRecord]],
    *,
    state: SelectionState,
    site_counts: Counter[str],
    per_site_limit: int,
    max_hash_distance: int,
    feature_similarity_threshold: float,
    skipped: Counter[str],
) -> Optional[CandidateRecord]:
    """从分组中按轮询方式选取一个候选，兼顾站点平衡和去重。

    参数：
        - grouped: 分组后的候选队列字典
        - state: 选择状态
        - site_counts: 各站点已选计数
        - per_site_limit: 每站点最大选择数
        - max_hash_distance: 哈希最大允许距离
        - feature_similarity_threshold: 特征相似度阈值
        - skipped: 跳过原因计数

    返回：
        - 选中的候选或 None（无可用候选）
    """
    site_order = sorted(
        (site for site, queue in grouped.items() if queue and site_counts[site] < per_site_limit),
        key=lambda site: (site_counts[site], -grouped[site][0].score, site),
    )
    for site in site_order:
        queue = grouped[site]
        while queue:
            candidate = queue[0]
            reason = _dedup_reason(
                candidate,
                state,
                max_hash_distance=max_hash_distance,
                feature_similarity_threshold=feature_similarity_threshold,
            )
            if reason is None:
                queue.popleft()
                site_counts[site] += 1
                _register_candidate(candidate, state)
                return candidate
            queue.popleft()
            skipped[reason] += 1
    return None


def _copy_file_if_exists(source: str | Path | None, destination: Path) -> Optional[str]:
    """若源文件存在则复制到目标路径。

    参数：
        - source: 源路径
        - destination: 目标路径

    返回：
        - 目标路径字符串或 None
    """
    if not source:
        return None
    target = Path(source)
    if not target.exists() or not target.is_file():
        return None
    shutil.copy2(target, destination)
    return str(destination)


def _copy_dir_if_exists(source: str | Path | None, destination: Path) -> Optional[str]:
    """若源目录存在则递归复制到目标路径。

    参数：
        - source: 源目录路径
        - destination: 目标目录路径

    返回：
        - 目标路径字符串或 None
    """
    if not source:
        return None
    target = Path(source)
    if not target.exists() or not target.is_dir():
        return None
    shutil.copytree(target, destination, dirs_exist_ok=True)
    return str(destination)


def _target_bank_path(candidate: CandidateRecord, *, bank_root: Path, train_bank_root: Path) -> str:
    """确定候选应添加到的目标库路径。

    参数：
        - candidate: 候选记录
        - bank_root: 持续库根目录
        - train_bank_root: 训练库根目录

    返回：
        - 目标库路径字符串
    """
    site_id = str(candidate.site_id or "").upper()
    if site_id in {f"C{index}" for index in range(1, 7)}:
        return str(bank_root / site_id / candidate.polarity / "images")
    return str(train_bank_root / candidate.polarity / "images")


def _selection_reason(candidate: CandidateRecord) -> str:
    """生成候选的选择原因字符串。

    参数：
        - candidate: 候选记录

    返回：
        - 原因描述字符串
    """
    return ", ".join(candidate.reason_flags) if candidate.reason_flags else "score_ranked"


def _recommended_priority(candidate: CandidateRecord) -> str:
    """根据评分推荐优先级。

    参数：
        - candidate: 候选记录

    返回：
        - 优先级标签（high/medium/low）
    """
    if candidate.polarity == "positive":
        if candidate.score >= 0.25:
            return "high"
        if candidate.score >= 0.12:
            return "medium"
        return "low"
    if candidate.score >= 0.18:
        return "high"
    if candidate.score >= 0.10:
        return "medium"
    return "low"


def _selection_confidence(candidate: CandidateRecord) -> float:
    """计算候选的选择置信度。

    参数：
        - candidate: 候选记录

    返回：
        - 置信度值（0~1）
    """
    diagnostics = _safe_dict(candidate.raw_row.get("retrieval_diagnostics"))
    gate_diagnostics = _safe_dict(diagnostics.get("gate_diagnostics"))
    region_row = _safe_dict(candidate.raw_row.get("region_retrieval_row"))
    if candidate.polarity == "positive":
        confidence = max(
            _safe_float(gate_diagnostics.get("segmentation_confidence")),
            _safe_float(region_row.get("mean_confidence_in_region")),
            1.0 - min(max(candidate.prompt_sensitivity_score, 0.0), 1.0),
        )
        return float(min(max(confidence, 0.0), 1.0))
    return float(
        max(
            _safe_float(gate_diagnostics.get("segmentation_uncertainty")),
            _safe_float(region_row.get("mean_entropy_in_region")),
            _safe_float(candidate.metrics.get("False Positive Rate")),
        )
    )


def _load_binary_mask(path: str | Path | None) -> Optional[np.ndarray]:
    """从文件加载二值掩码。

    参数：
        - path: 掩码文件路径

    返回：
        - 二值掩码数组或 None
    """
    if not path:
        return None
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    image = Image.open(target).convert("L")
    array = np.asarray(image, dtype=np.float32)
    threshold = 0.0 if array.max() <= 1.0 else 127.0
    return array > threshold


def _false_positive_bbox(gt_mask_path: str | Path | None, pred_mask_path: str | Path | None) -> Optional[list[int]]:
    """计算假阳性区域的边界框。

    参数：
        - gt_mask_path: 真值掩码路径
        - pred_mask_path: 预测掩码路径

    返回：
        - [x1,y1,x2,y2] 边界框或 None
    """
    gt_mask = _load_binary_mask(gt_mask_path)
    pred_mask = _load_binary_mask(pred_mask_path)
    if gt_mask is None or pred_mask is None:
        return None
    if gt_mask.shape != pred_mask.shape:
        pred_mask = np.asarray(
            Image.fromarray(pred_mask.astype(np.uint8) * 255).resize((gt_mask.shape[1], gt_mask.shape[0]), resample=Image.NEAREST)
        ) > 127
    false_positive = pred_mask & ~gt_mask
    if not np.any(false_positive):
        return None
    ys, xs = np.where(false_positive)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _save_crop(image_path: str | Path | None, bbox: Optional[list[int]], destination: Path) -> Optional[str]:
    """根据边界框裁剪图像并保存。

    参数：
        - image_path: 图像路径
        - bbox: [x1,y1,x2,y2] 边界框
        - destination: 输出路径

    返回：
        - 输出路径字符串或 None
    """
    if not image_path or bbox is None:
        return None
    target = Path(image_path)
    if not target.exists() or not target.is_file():
        return None
    image = Image.open(target).convert("RGB")
    crop = image.crop(tuple(bbox))
    crop.save(destination)
    return str(destination)


def _stage_candidate_assets(candidate: CandidateRecord, *, candidate_dir: Path, visualization_root: Optional[Path]) -> dict[str, str]:
    """将候选相关的图像、掩码和可视化文件复制到审核目录。

    参数：
        - candidate: 候选记录
        - candidate_dir: 候选审核子目录
        - visualization_root: 可视化文件根目录

    返回：
        - 资产类型到路径的字典
    """
    assets: dict[str, str] = {}
    image_path = Path(candidate.image_path) if candidate.image_path else None
    mask_path = Path(candidate.mask_path) if candidate.mask_path else None
    if image_path is not None:
        copied = _copy_file_if_exists(image_path, candidate_dir / f"image{image_path.suffix or '.png'}")
        if copied:
            assets["image"] = copied
    if mask_path is not None:
        copied = _copy_file_if_exists(mask_path, candidate_dir / f"mask{mask_path.suffix or '.png'}")
        if copied:
            assets["mask"] = copied
    if visualization_root is not None:
        for suffix in ("query", "gt", "pred", "pred_alt", "positive_heatmap", "negative_heatmap", "final_spatial_prior"):
            source = visualization_root / f"{candidate.image_id}_{suffix}.png"
            copied = _copy_file_if_exists(source, candidate_dir / source.name)
            if copied:
                assets[suffix] = copied
        topk_dir = visualization_root / f"{candidate.image_id}_topk"
        copied_dir = _copy_dir_if_exists(topk_dir, candidate_dir / "topk_retrieval")
        if copied_dir:
            assets["topk_retrieval"] = copied_dir
    prompt_row = _safe_dict(candidate.raw_row.get("prompt_sensitivity_row"))
    prompt_dir = prompt_row.get("visualization_dir")
    copied_prompt_dir = _copy_dir_if_exists(prompt_dir, candidate_dir / "prompt_sensitivity")
    if copied_prompt_dir:
        assets["prompt_sensitivity"] = copied_prompt_dir
    if candidate.polarity == "negative":
        bbox = _false_positive_bbox(assets.get("mask") or candidate.mask_path, assets.get("pred"))
        crop_path = _save_crop(candidate.image_path, bbox, candidate_dir / "false_positive_crop.png")
        if crop_path:
            assets["false_positive_crop"] = crop_path
    return assets


def _candidate_payload(
    candidate: CandidateRecord,
    *,
    review_dir: Path,
    staged_assets: dict[str, str],
    bank_root: Path,
    train_bank_root: Path,
) -> dict[str, Any]:
    """生成候选的输出载荷字典。

    参数：
        - candidate: 候选记录
        - review_dir: 审核目录
        - staged_assets: 已复制资产字典
        - bank_root: 持续库根目录
        - train_bank_root: 训练库根目录

    返回：
        - 载荷字典
    """
    false_positive_bbox = _false_positive_bbox(staged_assets.get("mask") or candidate.mask_path, staged_assets.get("pred"))
    target_bank_path = _target_bank_path(candidate, bank_root=bank_root, train_bank_root=train_bank_root)
    selection_reason = _selection_reason(candidate)
    payload = {
        "image_id": candidate.image_id,
        "image_path": candidate.image_path,
        "mask_path": candidate.mask_path,
        "dataset_name": candidate.dataset_name,
        "site_id": candidate.site_id,
        "patient_id": candidate.patient_id,
        "candidate_type": candidate.polarity,
        "candidate_score": candidate.score,
        "selection_reasons": candidate.reason_flags,
        "selection_reason": selection_reason,
        "metrics": candidate.metrics,
        "baseline_metrics": candidate.baseline_metrics,
        "retrieval_vs_baseline": candidate.retrieval_vs_baseline,
        "retrieval_sensitivity": candidate.retrieval_sensitivity,
        "prompt_sensitivity_score": candidate.prompt_sensitivity_score,
        "retrieval_influence_strength": candidate.retrieval_influence_strength,
        "lesion_area": candidate.lesion_area,
        "prediction_area": candidate.prediction_area,
        "selected_positive": candidate.selected_positive,
        "selected_negative": candidate.selected_negative,
        "target_bank_path": target_bank_path,
        "recommended_priority": _recommended_priority(candidate),
        "confidence": _selection_confidence(candidate),
        "Dice": _safe_float(candidate.metrics.get("Dice")),
        "Boundary F1": _safe_float(candidate.metrics.get("Boundary F1")),
        "HD95": _safe_float(candidate.metrics.get("HD95")),
        "ASSD": _safe_float(candidate.metrics.get("ASSD")),
        "FPR": _safe_float(candidate.metrics.get("False Positive Rate")),
        "FNR": _safe_float(candidate.metrics.get("False Negative Rate")),
        "mask_area": candidate.lesion_area,
        "region_retrieval": _safe_dict(candidate.raw_row.get("region_retrieval_row")),
        "review_dir": str(review_dir),
        "staged_assets": staged_assets,
    }
    if candidate.polarity == "negative":
        payload.update(
            {
                "gt_mask_path": candidate.mask_path,
                "pred_mask_path": staged_assets.get("pred", ""),
                "false_positive_bbox": false_positive_bbox,
                "crop_path": staged_assets.get("false_positive_crop", ""),
                "artifact_type": "false_positive_bbox" if false_positive_bbox else "over_prediction",
                "Precision": _safe_float(candidate.metrics.get("Precision"), max(0.0, 1.0 - _safe_float(candidate.metrics.get("False Positive Rate")))),
            }
        )
    return payload


def _write_candidate_outputs(
    *,
    candidates: list[CandidateRecord],
    polarity: str,
    output_dir: Path,
    visualization_root: Optional[Path],
    bank_root: Path,
    train_bank_root: Path,
) -> list[dict[str, Any]]:
    """将选中的候选写入输出目录并复制相关资产。

    参数：
        - candidates: 候选记录列表
        - polarity: 极性
        - output_dir: 输出目录
        - visualization_root: 可视化根目录
        - bank_root: 持续库根目录
        - train_bank_root: 训练库根目录

    返回：
        - 输出载荷字典列表
    """
    polarity_dir = ensure_dir(output_dir / polarity)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        safe_image_id = candidate.image_id.replace("/", "_")
        review_dir = ensure_dir(polarity_dir / f"{index:03d}_{safe_image_id}")
        staged_assets = _stage_candidate_assets(candidate, candidate_dir=review_dir, visualization_root=visualization_root)
        payload = _candidate_payload(
            candidate,
            review_dir=review_dir,
            staged_assets=staged_assets,
            bank_root=bank_root,
            train_bank_root=train_bank_root,
        )
        (review_dir / "candidate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(payload)
    target_path = output_dir / f"{polarity}_candidates.jsonl"
    target_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return rows


def _write_copy_commands(
    *,
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    bank_root: Path,
    output_dir: Path,
) -> Path:
    """生成已注释的复制命令脚本，供用户审核后执行。

    参数：
        - positive_rows: 正例行列表
        - negative_rows: 负例行列表
        - bank_root: 持续库根目录
        - output_dir: 输出目录

    返回：
        - 脚本文件路径
    """
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'# CONTINUAL_BANK_ROOT="${{CONTINUAL_BANK_ROOT:-{bank_root}}}"',
        "# Review each command, then uncomment the ones you approve.",
        "",
    ]
    for row in positive_rows:
        image_path = row.get("image_path")
        mask_path = row.get("mask_path")
        if image_path:
            image_suffix = Path(str(image_path)).suffix or ".png"
            target_root = Path(str(row.get("target_bank_path") or bank_root / row.get("site_id", "") / "positive" / "images"))
            image_target = target_root / f"{row['image_id']}{image_suffix}"
            lines.append(f"# positive {row['image_id']} score={row['candidate_score']:.4f} site={row['site_id']} priority={row.get('recommended_priority')}")
            lines.append(f"# mkdir -p {json.dumps(str(target_root))} {json.dumps(str(target_root.parent / 'masks'))}")
            lines.append(f"# cp {json.dumps(str(image_path))} {json.dumps(str(image_target))}")
            if mask_path:
                mask_suffix = Path(str(mask_path)).suffix or ".png"
                mask_target = target_root.parent / "masks" / f"{row['image_id']}{mask_suffix}"
                lines.append(f"# cp {json.dumps(str(mask_path))} {json.dumps(str(mask_target))}")
            lines.append("")
    for row in negative_rows:
        image_path = row.get("image_path")
        if not image_path:
            continue
        image_suffix = Path(str(image_path)).suffix or ".png"
        target_root = Path(str(row.get("target_bank_path") or bank_root / row.get("site_id", "") / "negative" / "images"))
        image_target = target_root / f"{row['image_id']}{image_suffix}"
        lines.append(f"# negative {row['image_id']} score={row['candidate_score']:.4f} site={row['site_id']} priority={row.get('recommended_priority')}")
        lines.append("# negative masks are intentionally not copied; curate background masks manually if needed")
        lines.append(f"# mkdir -p {json.dumps(str(target_root))}")
        lines.append(f"# cp {json.dumps(str(image_path))} {json.dumps(str(image_target))}")
        lines.append("")
    target = output_dir / "approved_copy_commands.sh"
    target.write_text("\n".join(lines), encoding="utf-8")
    target.chmod(0o755)
    return target


def select_bank_candidates(
    *,
    per_image_metrics_path: str | Path,
    retrieval_diagnostics_path: str | Path | None = None,
    region_retrieval_diagnostics_path: str | Path | None = None,
    prompt_sensitivity_path: str | Path | None = None,
    output_dir: str | Path = "MedicalSAM3/outputs/bank_candidate_review",
    bank_root: str | Path = "MedicalSAM3/banks/continual_bank",
    train_bank_root: str | Path = "MedicalSAM3/banks/train_bank",
    positive_limit: int = 24,
    negative_limit: int = 24,
    per_site_limit: int = 4,
    min_positive_score: float = 0.0,
    min_negative_score: float = 0.01,
    max_hash_distance: int = 3,
    feature_similarity_threshold: float = 0.995,
    visualization_root: str | Path | None = None,
) -> dict[str, Any]:
    """从评估产物中选择平衡的正负例持续库候选。

    参数：
        - per_image_metrics_path: 逐图像指标文件路径
        - retrieval_diagnostics_path: 检索诊断文件路径
        - region_retrieval_diagnostics_path: 区域检索诊断文件路径
        - prompt_sensitivity_path: 提示敏感性文件路径
        - output_dir: 输出目录
        - bank_root: 持续库根目录
        - train_bank_root: 训练库根目录
        - positive_limit: 正例选择上限
        - negative_limit: 负例选择上限
        - per_site_limit: 每站点最大选择数
        - min_positive_score: 正例最低评分
        - min_negative_score: 负例最低评分
        - max_hash_distance: 感知哈希最大允许距离
        - feature_similarity_threshold: 特征相似度阈值
        - visualization_root: 可视化文件根目录

    返回：
        - 选择结果汇总字典
    """
    metrics_rows = _read_rows(per_image_metrics_path)
    diagnostics_rows = _read_rows(retrieval_diagnostics_path)
    region_rows = _read_rows(region_retrieval_diagnostics_path)
    prompt_rows = _read_rows(prompt_sensitivity_path)
    merged_rows = _merge_artifacts(
        metrics_rows=metrics_rows,
        diagnostics_rows=diagnostics_rows,
        region_rows=region_rows,
        prompt_rows=prompt_rows,
    )

    positive_pool = [
        candidate
        for row in merged_rows
        for candidate in [_build_candidate(row, "positive")]
        if _is_positive_candidate(row) and candidate.score >= min_positive_score
    ]
    negative_pool = [
        candidate
        for row in merged_rows
        for candidate in [_build_candidate(row, "negative")]
        if _is_negative_candidate(row) and candidate.score >= min_negative_score
    ]

    positive_groups = _group_candidates(positive_pool)
    negative_groups = _group_candidates(negative_pool)
    positive_selected: list[CandidateRecord] = []
    negative_selected: list[CandidateRecord] = []
    per_site_limit = max(int(per_site_limit), 1)
    target_each = min(max(int(positive_limit), 0), max(int(negative_limit), 0))
    state = SelectionState()
    positive_site_counts: Counter[str] = Counter()
    negative_site_counts: Counter[str] = Counter()
    skipped = {"positive": Counter(), "negative": Counter()}

    while len(positive_selected) < target_each or len(negative_selected) < target_each:
        progressed = False
        if len(positive_selected) < target_each:
            candidate = _pick_candidate(
                positive_groups,
                state=state,
                site_counts=positive_site_counts,
                per_site_limit=per_site_limit,
                max_hash_distance=max_hash_distance,
                feature_similarity_threshold=feature_similarity_threshold,
                skipped=skipped["positive"],
            )
            if candidate is not None:
                positive_selected.append(candidate)
                progressed = True
        if len(negative_selected) < target_each:
            candidate = _pick_candidate(
                negative_groups,
                state=state,
                site_counts=negative_site_counts,
                per_site_limit=per_site_limit,
                max_hash_distance=max_hash_distance,
                feature_similarity_threshold=feature_similarity_threshold,
                skipped=skipped["negative"],
            )
            if candidate is not None:
                negative_selected.append(candidate)
                progressed = True
        if not progressed:
            break

    balanced_target = min(len(positive_selected), len(negative_selected))
    positive_selected = positive_selected[:balanced_target]
    negative_selected = negative_selected[:balanced_target]

    output_root = ensure_dir(output_dir)
    resolved_visualization_root = Path(visualization_root) if visualization_root else None
    if resolved_visualization_root is None:
        default_visualizations = Path(per_image_metrics_path).resolve().parent / "visualizations"
        if default_visualizations.exists():
            resolved_visualization_root = default_visualizations

    positive_rows = _write_candidate_outputs(
        candidates=positive_selected,
        polarity="positive",
        output_dir=output_root,
        visualization_root=resolved_visualization_root,
        bank_root=Path(bank_root),
        train_bank_root=Path(train_bank_root),
    )
    negative_rows = _write_candidate_outputs(
        candidates=negative_selected,
        polarity="negative",
        output_dir=output_root,
        visualization_root=resolved_visualization_root,
        bank_root=Path(bank_root),
        train_bank_root=Path(train_bank_root),
    )
    copy_commands_path = _write_copy_commands(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        bank_root=Path(bank_root),
        output_dir=output_root,
    )

    summary = {
        "artifacts": {
            "positive_candidates": str(output_root / "positive_candidates.jsonl"),
            "negative_candidates": str(output_root / "negative_candidates.jsonl"),
            "review_dir": str(output_root),
            "approved_copy_commands": str(copy_commands_path),
        },
        "pool_sizes": {
            "positive": len(positive_pool),
            "negative": len(negative_pool),
        },
        "selected_counts": {
            "positive": len(positive_rows),
            "negative": len(negative_rows),
        },
        "balanced_target": balanced_target,
        "site_distribution": {
            "positive": dict(Counter(row["site_id"] for row in positive_rows)),
            "negative": dict(Counter(row["site_id"] for row in negative_rows)),
        },
        "skipped": {
            "positive": dict(skipped["positive"]),
            "negative": dict(skipped["negative"]),
        },
    }
    (output_root / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """脚本命令行入口，从评估产物中选择平衡的持续库候选。

    参数：
        - 无

    返回：
        - 进程退出码，0 表示成功
    """
    parser = argparse.ArgumentParser(description="Select balanced continual-bank candidates from evaluation artifacts.")
    parser.add_argument("--per-image-metrics", required=True)
    parser.add_argument("--retrieval-diagnostics", default=None)
    parser.add_argument("--region-retrieval-diagnostics", default=None)
    parser.add_argument("--prompt-sensitivity", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/bank_candidate_review")
    parser.add_argument("--bank-root", default="MedicalSAM3/banks/continual_bank")
    parser.add_argument("--train-bank-root", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--positive-limit", type=int, default=24)
    parser.add_argument("--max-positive", dest="positive_limit", type=int)
    parser.add_argument("--negative-limit", type=int, default=24)
    parser.add_argument("--max-negative", dest="negative_limit", type=int)
    parser.add_argument("--per-site-limit", type=int, default=4)
    parser.add_argument("--min-positive-score", type=float, default=0.0)
    parser.add_argument("--min-negative-score", type=float, default=0.01)
    parser.add_argument("--max-hash-distance", type=int, default=3)
    parser.add_argument("--feature-similarity-threshold", type=float, default=0.995)
    parser.add_argument("--visualization-root", default=None)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    positive_limit = args.positive_limit if args.positive_limit is not None else (args.max_positive if args.max_positive is not None else 24)
    negative_limit = args.negative_limit if args.negative_limit is not None else (args.max_negative if args.max_negative is not None else 24)
    summary = select_bank_candidates(
        per_image_metrics_path=args.per_image_metrics,
        retrieval_diagnostics_path=args.retrieval_diagnostics,
        region_retrieval_diagnostics_path=args.region_retrieval_diagnostics,
        prompt_sensitivity_path=args.prompt_sensitivity,
        output_dir=args.output_dir,
        bank_root=args.bank_root,
        train_bank_root=args.train_bank_root,
        positive_limit=positive_limit,
        negative_limit=negative_limit,
        per_site_limit=args.per_site_limit,
        min_positive_score=args.min_positive_score,
        min_negative_score=args.min_negative_score,
        max_hash_distance=args.max_hash_distance,
        feature_similarity_threshold=args.feature_similarity_threshold,
        visualization_root=args.visualization_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
