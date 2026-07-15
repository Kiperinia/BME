"""生成 RSS-DA 行为及域差距的轻量数值报告。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar_bank import PrototypeBankEntry, RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import load_retrieval_bank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    ensure_dir,
    infer_source_domain,
    read_records,
    resolve_feature_map,
)


def _resolve_hidden_dim(model: torch.nn.Module) -> int:
    """从模型中解析隐藏层维度。

    参数：
        - model: PyTorch 模型

    返回：
        - 隐藏层维度整数值
    """
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def _resolve_runtime_device(requested_device: str) -> str:
    """解析运行时设备（cpu/cuda）。

    参数：
        - requested_device: 请求的设备字符串（auto/cpu/cuda）

    返回：
        - 解析后的设备名
    """
    normalized = requested_device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported --device value: {requested_device}")
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


def _mask_area_ratio(mask_logits: torch.Tensor) -> float:
    """计算预测掩码中前景区域的占比。

    参数：
        - mask_logits: 预测掩码 logits 张量

    返回：
        - 前景区域比例（0~1）
    """
    prob = torch.sigmoid(mask_logits)
    return float((prob > 0.5).float().mean().item())


def _mean_confidence(outputs: dict[str, Any]) -> float:
    """计算输出中的平均置信度。

    参数：
        - outputs: 模型输出字典

    返回：
        - 平均置信度浮点值
    """
    scores = outputs.get("scores")
    if isinstance(scores, torch.Tensor):
        return float(scores.mean().item())
    masks = outputs.get("masks")
    if isinstance(masks, torch.Tensor):
        return float(masks.mean().item())
    return 0.0


def _load_checkpoint_payload(path: Path, device: str) -> object:
    """从磁盘加载检查点文件。

    参数：
        - path: 检查点文件路径
        - device: 加载目标设备

    返回：
        - 检查点数据对象
    """
    return torch.load(path, map_location=device, weights_only=False)


def _maybe_load_rssda_bundle(
    path: Path,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> bool:
    """尝试从检查点加载 RSS-DA 组件（adapter/retriever/similarity_builder）的状态字典。

    参数：
        - path: 检查点路径
        - device: 加载目标设备
        - adapter: RSS-DA 适配器
        - retriever: 原型检索器
        - similarity_builder: 相似度热图构建器

    返回：
        - 是否成功加载了任何组件
    """
    payload = _load_checkpoint_payload(path, device)
    if not isinstance(payload, dict):
        return False
    loaded = False
    adapter_state = payload.get("adapter")
    retriever_state = payload.get("retriever")
    similarity_state = payload.get("similarity_builder")
    if isinstance(adapter_state, dict):
        adapter.load_state_dict(adapter_state, strict=False)
        loaded = True
    if isinstance(retriever_state, dict):
        retriever.load_state_dict(retriever_state, strict=False)
        loaded = True
    if isinstance(similarity_state, dict):
        similarity_builder.load_state_dict(similarity_state, strict=False)
        loaded = True
    return loaded


def _apply_retrieval_mode(retrieval: dict[str, object], mode: str) -> dict[str, object]:
    """根据指定的检索模式对检索结果进行过滤（positive-only 模式清除负例信息）。

    参数：
        - retrieval: 原始检索结果字典
        - mode: 检索模式

    返回：
        - 应用模式后的检索结果
    """
    if mode in {"joint", "semantic", "spatial", "positive-negative"}:
        return retrieval
    if mode != "positive-only":
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    updated = dict(retrieval)
    updated["negative_features"] = torch.zeros_like(retrieval["negative_features"])
    updated["negative_weights"] = torch.zeros_like(retrieval["negative_weights"])
    updated["negative_prototype"] = torch.zeros_like(retrieval["positive_prototype"])
    updated["negative_entries"] = [[] for _ in retrieval["positive_entries"]]
    updated["negative_scores"] = [torch.zeros_like(score) for score in retrieval["positive_scores"]]
    return updated


def _dummy_records(prefix: str, dataset_name: str, count: int) -> list[dict[str, str]]:
    """生成用于测试的虚拟数据记录。

    参数：
        - prefix: 图像 ID 前缀
        - dataset_name: 数据集名称
        - count: 生成数量

    返回：
        - 虚拟记录列表
    """
    return [
        {
            "image_path": "",
            "mask_path": "",
            "dataset_name": dataset_name,
            "image_id": f"{prefix}_{index:03d}",
        }
        for index in range(count)
    ]


def _ensure_records(split_file: str | Path, dummy: bool, prefix: str, dataset_name: str, count: int) -> list[dict[str, Any]]:
    """从拆分文件读取记录，若失败且启用 dummy 则返回虚拟记录。

    参数：
        - split_file: 拆分文件路径
        - dummy: 是否允许使用虚拟数据
        - prefix: 虚拟记录 ID 前缀
        - dataset_name: 数据集名称
        - count: 虚拟记录生成数量

    返回：
        - 有效记录列表
    """
    records = read_records(split_file)
    if dummy and not records:
        return _dummy_records(prefix, dataset_name, count)
    return records


def _create_dummy_bank(bank_dir: Path, hidden_dim: int, seed: int) -> RSSDABank:
    """创建包含随机正负示例的虚拟 RSS-DA 库。

    参数：
        - bank_dir: 库存储目录
        - hidden_dim: 特征维度
        - seed: 随机种子

    返回：
        - 虚拟 RSSDABank 实例
    """
    generator = torch.Generator().manual_seed(seed)
    bank = RSSDABank()
    entries = [
        ("kvasir_positive_a", "positive", "Kvasir"),
        ("kvasir_positive_b", "positive", "Kvasir"),
        ("cvc_positive_a", "positive", "CVC"),
        ("cvc_positive_b", "positive", "CVC"),
        ("negative_a", "negative", "Kvasir"),
        ("negative_b", "negative", "CVC"),
    ]
    for index, (prototype_id, polarity, source_dataset) in enumerate(entries):
        feature_dir = ensure_dir(bank_dir / ("positive_bank" if polarity == "positive" else "negative_bank"))
        feature_path = feature_dir / f"{prototype_id}.pt"
        base = torch.randn(hidden_dim, generator=generator)
        base[index % min(hidden_dim, 8)] += 3.0
        torch.save({"prototype": torch.nn.functional.normalize(base.float(), dim=0)}, feature_path)
        bank.add_entry(
            PrototypeBankEntry(
                prototype_id=prototype_id,
                feature_path=str(feature_path),
                polarity=polarity,
                source_dataset=source_dataset,
                polyp_type="polyp" if polarity == "positive" else "background",
                boundary_quality=0.8,
                confidence=0.9,
                image_id=f"{source_dataset.lower()}_{index:03d}",
                device_metadata={"runtime_device": "dummy"},
                extra_metadata={"source_group": source_dataset},
            )
        )
    bank.save(bank_dir)
    return bank


def _load_or_create_bank(
    path: str | Path,
    hidden_dim: int,
    dummy: bool,
    seed: int,
    *,
    image_size: int = 128,
    precision: str = "fp32",
    checkpoint: Optional[str] = None,
    device: str = "auto",
) -> RSSDABank:
    """从路径加载 RSS-DA 库，若不存在且启用 dummy 则创建虚拟库。

    参数：
        - path: 库路径
        - hidden_dim: 特征维度
        - dummy: 是否允许创建虚拟库
        - seed: 随机种子
        - image_size: 图像尺寸
        - precision: 精度设置
        - checkpoint: 检查点路径
        - device: 设备

    返回：
        - RSSDABank 实例
    """
    bank_path = Path(path)
    if bank_path.exists():
        bank_context = load_retrieval_bank(
            bank_path,
            purpose="validation",
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=dummy,
        )
        if bank_context.bank.entries:
            return bank_context.bank
    if not dummy:
        raise FileNotFoundError(f"RSS-DA bank not found or empty: {bank_path}")
    return _create_dummy_bank(bank_path, hidden_dim=hidden_dim, seed=seed)


def _entry_source_counts(bank: RSSDABank) -> dict[str, dict[str, int]]:
    """统计库中每个数据源的正负示例条目数。

    参数：
        - bank: RSS-DA 库

    返回：
        - 数据源到 {positive: int, negative: int} 的字典
    """
    counts: dict[str, dict[str, int]] = {}
    for entry in bank.entries:
        polarity_counts = counts.setdefault(entry.source_dataset, {"positive": 0, "negative": 0})
        polarity_counts[entry.polarity] = polarity_counts.get(entry.polarity, 0) + 1
    return counts


def _metadata_readiness(bank: RSSDABank) -> dict[str, Any]:
    """检查库条目的元数据完备性（设备信息、医院信息等）。

    参数：
        - bank: RSS-DA 库

    返回：
        - 元数据完备性指标字典
    """
    device_keys: set[str] = set()
    extra_keys: set[str] = set()
    has_hospital = False
    has_device = False
    for entry in bank.entries:
        device_keys.update(str(key) for key in entry.device_metadata.keys())
        extra_keys.update(str(key) for key in entry.extra_metadata.keys())
        has_hospital = has_hospital or ("hospital" in entry.device_metadata) or ("hospital" in entry.extra_metadata)
        has_device = has_device or ("device" in entry.device_metadata) or ("device" in entry.extra_metadata)
    return {
        "source_dataset_count": len({entry.source_dataset for entry in bank.entries}),
        "device_metadata_keys": sorted(device_keys),
        "extra_metadata_keys": sorted(extra_keys),
        "has_hospital_metadata": has_hospital,
        "has_device_metadata": has_device,
        "ready_for_hybrid_domain_score": has_hospital or has_device,
    }


def _selection_from_entries(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    """从库中选择指定索引的条目，计算加权原型和权重。

    参数：
        - bank: RSS-DA 库
        - query_vector: 查询向量
        - entries: 候选条目列表
        - indices: 选中条目的索引列表

    返回：
        - (特征张量, 权重张量, 条目列表, 原型张量, 原始分数) 的元组
    """
    dim = int(query_vector.shape[-1])
    if not indices:
        return (
            torch.zeros(1, 0, dim, device=query_vector.device),
            torch.zeros(1, 0, device=query_vector.device),
            [],
            torch.zeros(1, dim, device=query_vector.device),
            torch.zeros(0, device=query_vector.device),
        )
    selected_entries = [entries[index] for index in indices]
    features = bank.stack_features(selected_entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    raw_scores = torch.matmul(features, query_vector)
    weights = torch.softmax(raw_scores, dim=0)
    prototype = torch.nn.functional.normalize((weights.unsqueeze(-1) * features).sum(dim=0, keepdim=True), dim=-1)
    return features.unsqueeze(0), weights.unsqueeze(0), selected_entries, prototype, raw_scores


def _rank_entry_indices(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    top_k: int,
    strategy: str,
    rng: random.Random,
) -> list[int]:
    """根据指定策略对库条目进行排序并返回前 top_k 个索引。

    参数：
        - bank: RSS-DA 库
        - query_vector: 查询向量
        - entries: 候选条目列表
        - top_k: 选取数量
        - strategy: 排序策略（best/worst/random）
        - rng: 随机数生成器

    返回：
        - 选中条目的索引列表
    """
    if not entries or top_k <= 0:
        return []
    features = bank.stack_features(entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    scores = torch.matmul(features, query_vector)
    count = min(top_k, len(entries))
    if strategy == "best":
        return torch.topk(scores, k=count, largest=True).indices.detach().cpu().tolist()
    if strategy == "worst":
        return torch.topk(scores, k=count, largest=False).indices.detach().cpu().tolist()
    if strategy == "random":
        shuffled = list(range(len(entries)))
        rng.shuffle(shuffled)
        return shuffled[:count]
    raise ValueError(f"Unsupported ranking strategy: {strategy}")


def _override_retrieval(
    base_retrieval: dict[str, Any],
    *,
    positive_override: Optional[tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]] = None,
    negative_override: Optional[tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]] = None,
) -> dict[str, Any]:
    """用指定的正例/负例覆盖检索结果，用于消融实验。

    参数：
        - base_retrieval: 原始检索结果
        - positive_override: 替换正例的元组
        - negative_override: 替换负例的元组

    返回：
        - 覆盖后的检索结果字典
    """
    retrieval = dict(base_retrieval)
    if positive_override is not None:
        retrieval["positive_features"] = positive_override[0]
        retrieval["positive_weights"] = positive_override[1]
        retrieval["positive_entries"] = [positive_override[2]]
        retrieval["positive_prototype"] = positive_override[3]
        retrieval["positive_scores"] = [positive_override[4]]
    if negative_override is not None:
        retrieval["negative_features"] = negative_override[0]
        retrieval["negative_weights"] = negative_override[1]
        retrieval["negative_entries"] = [negative_override[2]]
        retrieval["negative_prototype"] = negative_override[3]
        retrieval["negative_scores"] = [negative_override[4]]
    return retrieval


def _empty_negative_like(retrieval: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    """生成与检索结果维度匹配的空负例占位符。

    参数：
        - retrieval: 检索结果字典

    返回：
        - (空特征, 空权重, 空条目, 零原型, 空分数) 的元组
    """
    dim = int(retrieval["positive_prototype"].shape[-1])
    device = retrieval["positive_prototype"].device
    return (
        torch.zeros(1, 0, dim, device=device),
        torch.zeros(1, 0, device=device),
        [],
        torch.zeros(1, dim, device=device),
        torch.zeros(0, device=device),
    )


def _build_variant_retrievals(
    bank: RSSDABank,
    retriever: PrototypeRetriever,
    query_feature: torch.Tensor,
    query_source: str,
    top_k_positive: int,
    rng: random.Random,
    prefer_cross_domain_positive: bool,
    retrieval_mode: str,
) -> dict[str, Optional[dict[str, Any]]]:
    """构建多种检索变体（正确正例、错误示例、负例示例、随机示例、无检索）。

    参数：
        - bank: RSS-DA 库
        - retriever: 原型检索器
        - query_feature: 查询特征
        - query_source: 查询来源域
        - top_k_positive: 选取正例数量
        - rng: 随机数生成器
        - prefer_cross_domain_positive: 是否优先选用跨域正例
        - retrieval_mode: 检索模式

    返回：
        - 变体名称到检索结果（或 None）的字典
    """
    base_retrieval = _apply_retrieval_mode(
        retriever(
            query_feature,
            query_source_datasets=[query_source],
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        ),
        retrieval_mode,
    )
    query_vector = base_retrieval["projected_query"][0]
    positive_entries = bank.get_entries(polarity="positive", human_verified=True)
    negative_entries = bank.get_entries(polarity="negative", human_verified=True)
    wrong_positive = _selection_from_entries(
        bank,
        query_vector,
        positive_entries,
        _rank_entry_indices(bank, query_vector, positive_entries, top_k_positive, "worst", rng),
    )
    random_positive = _selection_from_entries(
        bank,
        query_vector,
        positive_entries,
        _rank_entry_indices(bank, query_vector, positive_entries, top_k_positive, "random", rng),
    )
    negative_as_positive = _selection_from_entries(
        bank,
        query_vector,
        negative_entries,
        _rank_entry_indices(bank, query_vector, negative_entries, top_k_positive, "best", rng),
    )
    return {
        "correct_positive": base_retrieval,
        "wrong_exemplar": _override_retrieval(base_retrieval, positive_override=wrong_positive),
        "negative_exemplar": _override_retrieval(
            base_retrieval,
            positive_override=negative_as_positive,
            negative_override=_empty_negative_like(base_retrieval),
        ),
        "random_exemplar": _override_retrieval(base_retrieval, positive_override=random_positive),
        "no_retrieval": None,
    }


def _safe_entropy(values: torch.Tensor) -> float:
    """安全计算张量的归一化熵（防止除零）。

    参数：
        - values: 输入张量

    返回：
        - 归一化熵值（0~1）
    """
    flat = values.float().flatten()
    if flat.numel() == 0:
        return 0.0
    flat = flat - flat.min()
    if float(flat.sum().item()) <= 1e-6:
        return 0.0
    probs = flat / flat.sum().clamp_min(1e-6)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum()
    max_entropy = math.log(float(probs.numel())) if probs.numel() > 1 else 1.0
    return float((entropy / max(max_entropy, 1e-6)).item())


def summarize_heatmap(tensor: Optional[torch.Tensor], gt_mask: torch.Tensor, top_percent: float) -> dict[str, float]:
    """汇总热图的统计信息，包括与真值掩码的重叠比。

    参数：
        - tensor: 热图张量（可选）
        - gt_mask: 真值掩码
        - top_percent: 前百分之几视为热点

    返回：
        - 包含 max/mean/entropy/top_percent_activation/hotspot_overlap_ratio 的字典
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return {
            "max": 0.0,
            "mean": 0.0,
            "entropy": 0.0,
            "top_percent_activation": 0.0,
            "hotspot_overlap_ratio": 0.0,
        }
    values = tensor.detach().float().squeeze()
    if values.dim() != 2:
        values = values.reshape(values.shape[-2], values.shape[-1])
    flat = values.flatten()
    top_count = max(1, int(flat.numel() * top_percent))
    top_values = torch.topk(flat, k=top_count).values
    threshold = float(top_values.min().item())
    hotspot = (values >= threshold).float()
    target = gt_mask.detach().float().squeeze()
    if target.shape != values.shape:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(0).unsqueeze(0),
            size=values.shape,
            mode="nearest",
        ).squeeze(0).squeeze(0)
    hotspot_area = hotspot.sum().clamp_min(1.0)
    overlap_ratio = float(((hotspot > 0.5) * (target > 0.5)).float().sum().item() / hotspot_area.item())
    return {
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "entropy": _safe_entropy(values),
        "top_percent_activation": float(top_values.mean().item()),
        "hotspot_overlap_ratio": overlap_ratio,
    }


def _variant_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    """计算当前变体相对于基线的各项指标变化量。

    参数：
        - current: 当前变体结果
        - baseline: 基线结果

    返回：
        - 各指标变化量的字典
    """
    return {
        "Dice change": float(current["metrics"]["Dice"] - baseline["metrics"]["Dice"]),
        "Mask area ratio change": float(current["mask_area_ratio"] - baseline["mask_area_ratio"]),
        "Boundary F1 change": float(current["metrics"]["Boundary F1"] - baseline["metrics"]["Boundary F1"]),
        "Confidence change": float(current["mean_confidence"] - baseline["mean_confidence"]),
    }


def _sensitivity_spread(variants: dict[str, dict[str, Any]]) -> dict[str, float]:
    """计算所有变体在各指标上的极差（最大值-最小值），衡量检索敏感性。

    参数：
        - variants: 变体名称到结果的字典

    返回：
        - 指标到极差值的字典
    """
    keys = {
        "dice_range": [variant["metrics"]["Dice"] for variant in variants.values()],
        "mask_area_ratio_range": [variant["mask_area_ratio"] for variant in variants.values()],
        "boundary_f1_range": [variant["metrics"]["Boundary F1"] for variant in variants.values()],
        "confidence_range": [variant["mean_confidence"] for variant in variants.values()],
    }
    return {name: float(max(values) - min(values)) for name, values in keys.items()}


def _to_score_list(scores: list[torch.Tensor] | list[object]) -> list[float]:
    """将分数张量列表转换为 Python 浮点数列表。

    参数：
        - scores: 分数张量列表

    返回：
        - Python 浮点数列表
    """
    if not scores:
        return []
    first = scores[0]
    if isinstance(first, torch.Tensor):
        return [float(value) for value in first.detach().cpu().tolist()]
    return []


def _run_variant(
    adapter: RetrievalSpatialSemanticAdapter,
    wrapper: Sam3TensorForwardWrapper,
    similarity_builder: SimilarityHeatmapBuilder,
    images: torch.Tensor,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    baseline_outputs: dict[str, Any],
    retrieval: Optional[dict[str, Any]],
    retrieval_mode: str,
    top_percent: float,
) -> dict[str, Any]:
    """运行单一检索变体的完整前向推理并计算指标与热图统计。

    参数：
        - adapter: RSS-DA 适配器
        - wrapper: SAM3 张量前向封装
        - similarity_builder: 相似度热图构建器
        - images: 输入图像张量
        - masks: 真值掩码
        - boxes: 边界框
        - text_prompt: 文本提示
        - query_feature: 查询特征
        - baseline_outputs: 基线输出
        - retrieval: 检索结果（None 时使用基线）
        - retrieval_mode: 检索模式
        - top_percent: 热点前百分之几

    返回：
        - 包含 metrics、heatmap_stats 等结果的字典
    """
    if retrieval is None:
        outputs = baseline_outputs
        metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
        return {
            "metrics": metrics,
            "mask_area_ratio": _mask_area_ratio(outputs["mask_logits"]),
            "mean_confidence": _mean_confidence(outputs),
            "selected_positive_ids": [],
            "selected_positive_scores": [],
            "selected_negative_ids": [],
            "selected_negative_scores": [],
            "heatmap_stats": {},
            "retrieval_summary": {},
        }

    similarity = similarity_builder(
        query_feature,
        retrieval["positive_features"],
        retrieval["negative_features"],
        retrieval["positive_weights"],
        retrieval["negative_weights"],
    )
    _, retrieval_prior, _ = adapter(
        feature_map=query_feature,
        similarity_map=similarity["fused_similarity"],
        positive_prototype=retrieval["positive_prototype"],
        negative_prototype=retrieval["negative_prototype"],
        positive_tokens=retrieval["positive_features"],
        negative_tokens=retrieval["negative_features"],
        positive_similarity=similarity["positive_similarity"],
        negative_similarity=similarity["negative_similarity"],
        positive_weights=retrieval["positive_weights"],
        negative_weights=retrieval["negative_weights"],
        positive_scores=retrieval.get("positive_score_tensor"),
        negative_scores=retrieval.get("negative_score_tensor"),
        baseline_mask_logits=baseline_outputs.get("mask_logits"),
        positive_heatmap=similarity["positive_heatmap"],
        negative_heatmap=similarity["negative_heatmap"],
        mode=retrieval_mode,
    )
    outputs = wrapper(images=images, boxes=boxes, text_prompt=text_prompt, retrieval_prior=retrieval_prior)
    metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
    return {
        "metrics": metrics,
        "mask_area_ratio": _mask_area_ratio(outputs["mask_logits"]),
        "mean_confidence": _mean_confidence(outputs),
        "selected_positive_ids": [entry.prototype_id for entry in retrieval["positive_entries"][0]],
        "selected_positive_scores": _to_score_list(retrieval["positive_scores"]),
        "selected_negative_ids": [entry.prototype_id for entry in retrieval["negative_entries"][0]],
        "selected_negative_scores": _to_score_list(retrieval["negative_scores"]),
        "heatmap_stats": {
            "positive_heatmap": summarize_heatmap(similarity["positive_heatmap"][0, 0], masks[0, 0], top_percent),
            "negative_heatmap": summarize_heatmap(similarity["negative_heatmap"][0, 0], masks[0, 0], top_percent),
            "fused_similarity": summarize_heatmap(similarity["fused_similarity"][0, 0], masks[0, 0], top_percent),
            "spatial_bias_map": summarize_heatmap(retrieval_prior.get("spatial_bias_map", None), masks[0, 0], top_percent),
        },
        "retrieval_summary": outputs.get("intermediate_features", {}).get("retrieval_prior", {}),
    }


def _accumulate_variant(
    target: dict[str, dict[str, float]],
    variant_name: str,
    payload: dict[str, float],
) -> None:
    """累加变体的各项指标值到目标字典。

    参数：
        - target: 累加目标字典
        - variant_name: 变体名称
        - payload: 变体指标字典

    返回：
        - 无返回值，仅更新 target 字典的副作用
    """
    summary = target.setdefault(variant_name, {})
    for key, value in payload.items():
        summary[key] = summary.get(key, 0.0) + float(value)


def _accumulate_heatmaps(
    target: dict[str, dict[str, dict[str, float]]],
    variant_name: str,
    heatmaps: dict[str, dict[str, float]],
) -> None:
    """累加变体的热图统计信息到目标字典。

    参数：
        - target: 累加目标字典
        - variant_name: 变体名称
        - heatmaps: 热图统计字典

    返回：
        - 无返回值，仅更新 target 字典的副作用
    """
    variant_target = target.setdefault(variant_name, {})
    for heatmap_name, stats in heatmaps.items():
        heatmap_target = variant_target.setdefault(heatmap_name, {})
        for key, value in stats.items():
            heatmap_target[key] = heatmap_target.get(key, 0.0) + float(value)


def _average_nested(values: dict[str, Any], count: int) -> dict[str, Any]:
    """递归对嵌套字典中的数值除以 count 取均值。

    参数：
        - values: 嵌套字典（叶节点为数值）
        - count: 除数

    返回：
        - 取均值后的字典结构
    """
    averaged: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            averaged[key] = _average_nested(value, count)
        else:
            averaged[key] = float(value) / max(count, 1)
    return averaged


def _report_gap(internal: dict[str, Any], external: dict[str, Any]) -> dict[str, dict[str, float]]:
    """计算内部验证集与外部测试集之间的域差距报告。

    参数：
        - internal: 内部集汇总
        - external: 外部集汇总

    返回：
        - 变体名称到域差距指标（dice_gap、precision_gap、fpr_gap）的字典
    """
    gap: dict[str, dict[str, float]] = {}
    internal_variants = internal.get("variant_metrics", {})
    external_variants = external.get("variant_metrics", {})
    for variant_name, metrics in internal_variants.items():
        if variant_name not in external_variants:
            continue
        gap[variant_name] = {
            "internal_dice": float(metrics.get("Dice", 0.0)),
            "external_dice": float(external_variants[variant_name].get("Dice", 0.0)),
            "dice_gap": float(metrics.get("Dice", 0.0) - external_variants[variant_name].get("Dice", 0.0)),
            "internal_precision": float(metrics.get("Precision", 0.0)),
            "external_precision": float(external_variants[variant_name].get("Precision", 0.0)),
            "precision_gap": float(metrics.get("Precision", 0.0) - external_variants[variant_name].get("Precision", 0.0)),
            "internal_fpr": float(metrics.get("False Positive Rate", 0.0)),
            "external_fpr": float(external_variants[variant_name].get("False Positive Rate", 0.0)),
            "fpr_gap": float(metrics.get("False Positive Rate", 0.0) - external_variants[variant_name].get("False Positive Rate", 0.0)),
        }
    return gap


def _evaluate_split(
    split_name: str,
    records: list[dict[str, Any]],
    *,
    adapter: RetrievalSpatialSemanticAdapter,
    wrapper: Sam3TensorForwardWrapper,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
    bank: RSSDABank,
    image_size: int,
    retrieval_mode: str,
    top_k_positive: int,
    top_percent: float,
    prefer_cross_domain_positive: bool,
    seed: int,
    device: str,
    max_samples: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在指定拆分（内/外部验证集）上评估 RSS-DA 行为并收集逐行指标。

    参数：
        - split_name: 拆分名称
        - records: 数据记录列表
        - adapter: RSS-DA 适配器
        - wrapper: SAM3 张量前向封装
        - retriever: 原型检索器
        - similarity_builder: 相似度热图构建器
        - bank: RSS-DA 库
        - image_size: 图像尺寸
        - retrieval_mode: 检索模式
        - top_k_positive: 选取正例数量
        - top_percent: 热点前百分之几
        - prefer_cross_domain_positive: 是否优先选用跨域正例
        - seed: 随机种子
        - device: 设备
        - max_samples: 最大样本数限制

    返回：
        - (逐行结果列表, 汇总字典) 的元组
    """
    if max_samples is not None:
        records = records[: max(0, max_samples)]
    loader = DataLoader(SplitSegmentationDataset(records, image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)
    rows: list[dict[str, Any]] = []
    variant_metrics_sum: dict[str, dict[str, float]] = {}
    sensitivity_sum: dict[str, dict[str, float]] = {}
    heatmap_sum: dict[str, dict[str, dict[str, float]]] = {}
    spread_sum: dict[str, float] = {}
    rng = random.Random(seed)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            record = batch["records"][0]
            source_domain = infer_source_domain(
                dataset_name=str(record.get("dataset_name", "")),
                image_id=str(record.get("image_id", "")),
                image_path=str(record.get("image_path", "")),
                mask_path=str(record.get("mask_path", "")),
            )
            baseline = wrapper(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
            query_feature = resolve_feature_map(baseline["image_embeddings"], images)
            variant_retrievals = _build_variant_retrievals(
                bank,
                retriever,
                query_feature,
                query_source=source_domain,
                top_k_positive=top_k_positive,
                rng=random.Random(seed + batch_index),
                prefer_cross_domain_positive=prefer_cross_domain_positive,
                retrieval_mode=retrieval_mode,
            )

            variants: dict[str, dict[str, Any]] = {}
            for variant_name, retrieval in variant_retrievals.items():
                variants[variant_name] = _run_variant(
                    adapter,
                    wrapper,
                    similarity_builder,
                    images,
                    masks,
                    boxes,
                    batch["text_prompt"],
                    query_feature,
                    baseline,
                    retrieval,
                    retrieval_mode,
                    top_percent,
                )

            baseline_variant = variants["no_retrieval"]
            sensitivity = {
                variant_name: _variant_delta(variant_payload, baseline_variant)
                for variant_name, variant_payload in variants.items()
                if variant_name != "no_retrieval"
            }
            spread = _sensitivity_spread(variants)
            rows.append(
                {
                    "split": split_name,
                    "image_id": str(record.get("image_id", "")),
                    "source_domain": source_domain,
                    "variants": variants,
                    "sensitivity_vs_no_retrieval": sensitivity,
                    "sensitivity_spread": spread,
                }
            )

            for variant_name, variant_payload in variants.items():
                metrics_payload = dict(variant_payload["metrics"])
                metrics_payload["mask_area_ratio"] = float(variant_payload["mask_area_ratio"])
                metrics_payload["mean_confidence"] = float(variant_payload["mean_confidence"])
                _accumulate_variant(variant_metrics_sum, variant_name, metrics_payload)
                _accumulate_heatmaps(heatmap_sum, variant_name, variant_payload["heatmap_stats"])
            for variant_name, delta_payload in sensitivity.items():
                _accumulate_variant(sensitivity_sum, variant_name, delta_payload)
            for key, value in spread.items():
                spread_sum[key] = spread_sum.get(key, 0.0) + float(value)

    count = len(rows)
    split_summary = {
        "sample_count": count,
        "variant_metrics": _average_nested(variant_metrics_sum, count),
        "sensitivity_vs_no_retrieval": _average_nested(sensitivity_sum, count),
        "heatmap_stats": _average_nested(heatmap_sum, count),
        "sensitivity_spread": _average_nested(spread_sum, count),
    }
    return rows, split_summary


def main() -> int:
    """脚本命令行入口，生成 RSS-DA 行为数值报告及域差距分析。

    参数：
        - 无

    返回：
        - 进程退出码，0 表示成功
    """
    parser = argparse.ArgumentParser(description="Generate numerical RSS-DA behavior reports.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--internal-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt")
    parser.add_argument("--external-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/rssda_behavior_report")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--retriever-checkpoint", default=None)
    parser.add_argument("--similarity-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--top-k-positive", type=int, default=3)
    parser.add_argument("--top-k-negative", type=int, default=3)
    parser.add_argument("--negative-lambda", type=float, default=0.35)
    parser.add_argument("--retrieval-mode", choices=["joint", "semantic", "spatial", "positive-only", "positive-negative"], default="joint")
    parser.add_argument("--prefer-cross-domain-positive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hotspot-top-percent", type=float, default=0.05)
    parser.add_argument("--dummy-samples-per-split", type=int, default=3)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    internal_records = _ensure_records(args.internal_split_file, args.dummy, "internal", "Kvasir", args.dummy_samples_per_split)
    external_records = _ensure_records(args.external_split_file, args.dummy, "external", "PolypGen", args.dummy_samples_per_split)
    if not internal_records:
        raise FileNotFoundError(f"No internal validation records found in {args.internal_split_file}")
    if not external_records:
        raise FileNotFoundError(f"No external validation records found in {args.external_split_file}")

    device = _resolve_runtime_device(args.device)
    base_model = build_official_sam3_image_model(
        args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
        allow_dummy_fallback=args.dummy,
    )
    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        apply_lora_to_model(base_model, LoRAConfig(stage=args.lora_stage, min_replaced_modules=0))
        load_lora_weights(base_model, args.lora_checkpoint, strict=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    hidden_dim = _resolve_hidden_dim(base_model)
    bank = _load_or_create_bank(
        args.memory_bank,
        hidden_dim=hidden_dim,
        dummy=args.dummy,
        seed=args.seed,
        image_size=args.image_size,
        precision=args.precision,
        checkpoint=args.checkpoint,
        device=device,
    )
    retriever = PrototypeRetriever(bank=bank, feature_dim=hidden_dim, top_k_positive=args.top_k_positive, top_k_negative=args.top_k_negative).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=args.negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(dim=hidden_dim).to(device)
    if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
        loaded_bundle = _maybe_load_rssda_bundle(Path(args.adapter_checkpoint), device, adapter, retriever, similarity_builder)
        if not loaded_bundle:
            adapter.load_state_dict(_load_checkpoint_payload(Path(args.adapter_checkpoint), device), strict=False)
    if args.retriever_checkpoint and Path(args.retriever_checkpoint).exists():
        retriever.load_state_dict(torch.load(args.retriever_checkpoint, map_location=device, weights_only=False), strict=False)
    if args.similarity_checkpoint and Path(args.similarity_checkpoint).exists():
        similarity_builder.load_state_dict(torch.load(args.similarity_checkpoint, map_location=device, weights_only=False), strict=False)
    adapter.eval()
    retriever.eval()
    similarity_builder.eval()

    internal_rows, internal_summary = _evaluate_split(
        "internal",
        internal_records,
        adapter=adapter,
        wrapper=wrapper,
        retriever=retriever,
        similarity_builder=similarity_builder,
        bank=bank,
        image_size=args.image_size,
        retrieval_mode=args.retrieval_mode,
        top_k_positive=args.top_k_positive,
        top_percent=args.hotspot_top_percent,
        prefer_cross_domain_positive=args.prefer_cross_domain_positive,
        seed=args.seed,
        device=device,
        max_samples=args.max_samples_per_split,
    )
    external_rows, external_summary = _evaluate_split(
        "external",
        external_records,
        adapter=adapter,
        wrapper=wrapper,
        retriever=retriever,
        similarity_builder=similarity_builder,
        bank=bank,
        image_size=args.image_size,
        retrieval_mode=args.retrieval_mode,
        top_k_positive=args.top_k_positive,
        top_percent=args.hotspot_top_percent,
        prefer_cross_domain_positive=args.prefer_cross_domain_positive,
        seed=args.seed + 1000,
        device=device,
        max_samples=args.max_samples_per_split,
    )

    report = {
        "config": vars(args),
        "bank_source_counts": _entry_source_counts(bank),
        "metadata_readiness": _metadata_readiness(bank),
        "internal": internal_summary,
        "external": external_summary,
        "gap_report": _report_gap(internal_summary, external_summary),
    }
    (output_dir / "per_image_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in internal_rows + external_rows),
        encoding="utf-8",
    )
    (output_dir / "summary_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
