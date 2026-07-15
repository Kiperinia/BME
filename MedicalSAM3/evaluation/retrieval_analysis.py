"""分析检索是否显著改变分割输出。"""

from __future__ import annotations

import argparse
import itertools
import json
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
    resolve_runtime_device,
    resolve_feature_map,
    seed_everything,
)


def _resolve_hidden_dim(model: torch.nn.Module) -> int:
    """解析模型的隐藏层维度。

    参数：
        - model: PyTorch 模型。

    返回：
        - 隐藏层维度整数值。
    """
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def _resolve_runtime_device(requested_device: str) -> str:
    """解析运行时的计算设备。

    参数：
        - requested_device: 请求的设备名称。

    返回：
        - 解析后的设备名称。
    """
    return resolve_runtime_device(requested_device)


def _load_checkpoint_payload(path: Path, device: str) -> object:
    """加载检查点文件的内容。

    参数：
        - path: 检查点文件路径。
        - device: 加载目标设备。

    返回：
        - 检查点内容对象。
    """
    return torch.load(path, map_location=device, weights_only=False)


def _maybe_load_rssda_bundle(
    path: Path,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> bool:
    """尝试从检查点加载 RSSDA 模型包（adapter、retriever、similarity_builder 的状态字典）。

    参数：
        - path: 检查点文件路径。
        - device: 加载目标设备。
        - adapter: 检索空间语义适配器。
        - retriever: 原型检索器。
        - similarity_builder: 相似度热力图构建器。

    返回：
        - 是否成功加载了至少一个组件。
    """
    payload = _load_checkpoint_payload(path, device)
    if not isinstance(payload, dict):
        return False
    loaded = False
    if isinstance(payload.get("adapter"), dict):
        adapter.load_state_dict(payload["adapter"], strict=False)
        loaded = True
    if isinstance(payload.get("retriever"), dict):
        retriever.load_state_dict(payload["retriever"], strict=False)
        loaded = True
    if isinstance(payload.get("similarity_builder"), dict):
        similarity_builder.load_state_dict(payload["similarity_builder"], strict=False)
        loaded = True
    return loaded


def _dummy_records(prefix: str, dataset_name: str, count: int) -> list[dict[str, str]]:
    """创建虚拟记录列表用于测试。

    参数：
        - prefix: 图像 ID 前缀。
        - dataset_name: 数据集名称。
        - count: 记录数量。

    返回：
        - 包含空路径的虚拟记录列表。
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
    """确保记录列表可用，必要时回退到虚拟记录。

    参数：
        - split_file: 分割文件路径。
        - dummy: 是否允许虚拟回退。
        - prefix: 图像 ID 前缀。
        - dataset_name: 数据集名称。
        - count: 虚拟记录数量。

    返回：
        - 记录列表。
    """
    records = read_records(split_file)
    if dummy and not records:
        return _dummy_records(prefix, dataset_name, count)
    return records


def _create_dummy_bank(bank_dir: Path, hidden_dim: int, seed: int) -> RSSDABank:
    """创建用于测试的虚拟银行库，包含正/负样本条目。

    参数：
        - bank_dir: 银行库输出目录。
        - hidden_dim: 特征维度。
        - seed: 随机种子。

    返回：
        - 创建的 RSSDABank 实例。
    """
    generator = torch.Generator().manual_seed(seed)
    bank = RSSDABank()
    fixtures = [
        ("kvasir_positive_a", "positive", "Kvasir", "polyp"),
        ("cvc_positive_a", "positive", "CVC", "polyp"),
        ("polypgen_positive_a", "positive", "PolypGen", "polyp"),
        ("specular_negative", "negative", "PolypGen", "specular_highlight"),
        ("mucosa_negative", "negative", "Kvasir", "normal_mucosa"),
        ("bubble_negative", "negative", "CVC", "bubble"),
        ("blur_negative", "negative", "PolypGen", "blur_region"),
        ("instrument_negative", "negative", "PolypGen", "instrument_artifact"),
    ]
    for index, (prototype_id, polarity, source_dataset, polyp_type) in enumerate(fixtures):
        feature_dir = ensure_dir(bank_dir / ("positive_bank" if polarity == "positive" else "negative_bank"))
        feature_path = feature_dir / f"{prototype_id}.pt"
        feature = torch.randn(hidden_dim, generator=generator)
        feature[index % max(1, min(hidden_dim, 16))] += 3.0
        torch.save({"prototype": torch.nn.functional.normalize(feature.float(), dim=0)}, feature_path)
        bank.add_entry(
            PrototypeBankEntry(
                prototype_id=prototype_id,
                feature_path=str(feature_path),
                polarity=polarity,
                source_dataset=source_dataset,
                polyp_type=polyp_type,
                boundary_quality=0.8,
                confidence=0.9,
                image_id=f"{source_dataset.lower()}_{index:03d}",
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
    """加载现有银行库，若不存在且允许虚拟回退则创建虚拟银行库。

    参数：
        - path: 银行库路径。
        - hidden_dim: 特征维度。
        - dummy: 是否允许创建虚拟银行库。
        - seed: 随机种子。
        - image_size: 图像尺寸。
        - precision: 精度。
        - checkpoint: 检查点路径。
        - device: 计算设备。

    返回：
        - 加载或创建的 RSSDABank 实例。
    """
    bank_path = Path(path)
    if bank_path.exists():
        bank_context = load_retrieval_bank(
            bank_path,
            purpose="external-eval",
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


def _apply_retrieval_mode(retrieval: dict[str, Any], mode: str) -> dict[str, Any]:
    """应用检索模式，支持 "joint"、"positive-only" 等，通过清零负样本实现。

    参数：
        - retrieval: 检索结果字典。
        - mode: 检索模式名称。

    返回：
        - 应用模式后的检索结果字典。
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


def _selection_from_entries(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    """根据索引从条目列表中选取条目，聚合特征和权重，生成原型。

    参数：
        - bank: RSSDA 银行库。
        - query_vector: 查询向量。
        - entries: 条目列表。
        - indices: 选定条目的索引列表。

    返回：
        - (features, weights, selected_entries, prototype, raw_scores) 的五元组。
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
    """根据策略对条目进行排序并返回 top-k 索引（支持 "best" 和 "random" 策略）。

    参数：
        - bank: RSSDA 银行库。
        - query_vector: 查询向量。
        - entries: 条目列表。
        - top_k: top-k 数量。
        - strategy: 排序策略（"best" 或 "random"）。
        - rng: 随机数生成器。

    返回：
        - 排序后的索引列表。
    """
    if not entries or top_k <= 0:
        return []
    features = bank.stack_features(entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    scores = torch.matmul(features, query_vector)
    count = min(top_k, len(entries))
    if strategy == "best":
        return torch.topk(scores, k=count, largest=True).indices.detach().cpu().tolist()
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
    """用指定的正/负样本覆盖结果替换检索结果中的对应字段。

    参数：
        - base_retrieval: 基础检索结果字典。
        - positive_override: 正样本覆盖的五元组。
        - negative_override: 负样本覆盖的五元组。

    返回：
        - 更新后的检索结果字典。
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
    """创建空的负样本结构（零特征、零权重、空条目），用于模拟无负样本的情况。

    参数：
        - retrieval: 检索结果字典（用于参考维度信息）。

    返回：
        - (zero_features, zero_weights, empty_entries, zero_prototype, zero_scores) 五元组。
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


def _build_prompt_variants(
    bank: RSSDABank,
    retriever: PrototypeRetriever,
    query_feature: torch.Tensor,
    query_source: str,
    top_k_positive: int,
    top_k_negative: int,
    rng: random.Random,
    prefer_cross_domain_positive: bool,
    retrieval_mode: str,
) -> dict[str, Optional[dict[str, Any]]]:
    """构建多个提示变体的检索结果（正例、反例、随机、空），用于敏感性分析。

    参数：
        - bank: RSSDA 银行库。
        - retriever: 原型检索器。
        - query_feature: 查询特征图。
        - query_source: 查询来源数据集。
        - top_k_positive: 正样本 top-k 数量。
        - top_k_negative: 负样本 top-k 数量。
        - rng: 随机数生成器。
        - prefer_cross_domain_positive: 是否优先跨域正样本。
        - retrieval_mode: 检索模式。

    返回：
        - 变体名称到检索结果的映射字典。
    """
    base_retrieval = _apply_retrieval_mode(
        retriever(
            query_feature,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            query_source_datasets=[query_source],
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        ),
        retrieval_mode,
    )
    query_vector = base_retrieval["projected_query"][0]
    positive_entries = bank.get_entries(polarity="positive", human_verified=True)
    negative_entries = bank.get_entries(polarity="negative", human_verified=True)
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
        _rank_entry_indices(bank, query_vector, negative_entries, top_k_negative, "best", rng),
    )
    return {
        "positive_exemplar": base_retrieval,
        "negative_exemplar": _override_retrieval(
            base_retrieval,
            positive_override=negative_as_positive,
            negative_override=_empty_negative_like(base_retrieval),
        ),
        "random_exemplar": _override_retrieval(base_retrieval, positive_override=random_positive),
        "empty_exemplar": None,
    }


def _binary_mask(mask_logits: torch.Tensor) -> torch.Tensor:
    """将 logits 转换为二值掩码（sigmoid > 0.5）。

    参数：
        - mask_logits: 模型输出的原始 logits。

    返回：
        - 二值掩码张量。
    """
    return (torch.sigmoid(mask_logits) > 0.5).float()


def _mask_difference_ratio(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """计算两个掩码之间的差异比率（不同像素占比）。

    参数：
        - mask_a: 第一个掩码 logits。
        - mask_b: 第二个掩码 logits。

    返回：
        - 差异比率浮点数。
    """
    pred_a = _binary_mask(mask_a)
    pred_b = _binary_mask(mask_b)
    difference = (pred_a != pred_b).float().mean()
    return float(difference.item())


def _logit_difference(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """计算两个掩码 logits 之间的平均绝对差异。

    参数：
        - mask_a: 第一个掩码 logits。
        - mask_b: 第二个掩码 logits。

    返回：
        - 平均绝对差异浮点数。
    """
    return float((mask_a.detach().float() - mask_b.detach().float()).abs().mean().item())


def _entry_logs(
    entries: list[PrototypeBankEntry],
    scores: list[float],
    weights: list[float],
    token_response: list[float],
) -> list[dict[str, Any]]:
    """将检索条目转换为可序列化的日志字典列表。

    参数：
        - entries: 原型银行条目列表。
        - scores: 相似度分数列表。
        - weights: 检索权重列表。
        - token_response: Token 响应值列表。

    返回：
        - 序列化后的条目日志列表。
    """
    payload: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        payload.append(
            {
                "prototype_id": entry.prototype_id,
                "polarity": entry.polarity,
                "source_dataset": entry.source_dataset,
                "polyp_type": entry.polyp_type,
                "confidence": float(entry.confidence),
                "similarity_score": float(scores[index]) if index < len(scores) else 0.0,
                "retrieval_weight": float(weights[index]) if index < len(weights) else 0.0,
                "token_response": float(token_response[index]) if index < len(token_response) else 0.0,
            }
        )
    return payload


def _tensor_list(values: torch.Tensor, count: int) -> list[float]:
    """将张量转换为 Python 浮点数列表，取前 count 个元素。

    参数：
        - values: 输入张量。
        - count: 要提取的元素数量。

    返回：
        - 浮点数列表。
    """
    if values.numel() == 0 or count <= 0:
        return []
    return [float(item) for item in values[:count].detach().cpu().tolist()]


def _run_variant(
    *,
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
) -> dict[str, Any]:
    """运行单个检索变体的前向传播，计算分割指标和注意力日志。

    参数：
        - adapter: 检索空间语义适配器。
        - wrapper: SAM3 张量前向包装器。
        - similarity_builder: 相似度热力图构建器。
        - images: 输入图像张量。
        - masks: 真实掩码张量。
        - boxes: 边界框张量。
        - text_prompt: 文本提示列表。
        - query_feature: 查询特征图。
        - baseline_outputs: 基线模型输出。
        - retrieval: 检索结果（None 表示不使用检索）。
        - retrieval_mode: 检索模式。

    返回：
        - 包含指标、掩码 logits 和注意力日志的字典。
    """
    if retrieval is None:
        outputs = baseline_outputs
        metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
        return {
            "metrics": metrics,
            "mask_logits": outputs["mask_logits"].detach(),
            "attention_log": {
                "positive_prototypes": [],
                "negative_prototypes": [],
                "fusion_alpha": 0.0,
                "negative_lambda": 0.0,
                "gate_mean": 0.0,
            },
        }

    similarity = similarity_builder(
        query_feature,
        retrieval["positive_features"],
        retrieval["negative_features"],
        retrieval["positive_weights"],
        retrieval["negative_weights"],
    )
    _, retrieval_prior, adapter_aux = adapter(
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
    positive_entries = retrieval["positive_entries"][0]
    negative_entries = retrieval["negative_entries"][0]
    attention_log = {
        "positive_prototypes": _entry_logs(
            positive_entries,
            _tensor_list(retrieval["positive_scores"][0], len(positive_entries)),
            _tensor_list(retrieval["positive_weights"][0], len(positive_entries)),
            _tensor_list(adapter_aux["positive_token_response"][0], len(positive_entries)),
        ),
        "negative_prototypes": _entry_logs(
            negative_entries,
            _tensor_list(retrieval["negative_scores"][0], len(negative_entries)),
            _tensor_list(retrieval["negative_weights"][0], len(negative_entries)),
            _tensor_list(adapter_aux["negative_token_response"][0], len(negative_entries)),
        ),
        "fusion_alpha": float(adapter_aux["fusion_alpha"].detach().float().mean().item()),
        "negative_lambda": float(adapter_aux["negative_lambda"].detach().float().mean().item()),
        "gate_mean": float(adapter_aux["fusion_gate_map"].detach().float().mean().item()),
        "similarity_temperature": float(similarity["temperature"].detach().float().mean().item()),
        "similarity_fusion_weight": [float(item) for item in similarity["fusion_weight"].detach().float().flatten().cpu().tolist()],
        "wrapper_retrieval_summary": outputs.get("intermediate_features", {}).get("retrieval_prior", {}),
    }
    return {
        "metrics": metrics,
        "mask_logits": outputs["mask_logits"].detach(),
        "attention_log": attention_log,
    }


def _variance(values: list[float]) -> float:
    """计算浮点数列表的方差（无偏校正）。

    参数：
        - values: 浮点数列表。

    返回：
        - 方差值。
    """
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.var(tensor, unbiased=False).item())


def _prompt_sensitivity(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """计算提示敏感性指标，包括掩码差异、logits差异、Dice/IoU 方差和综合评分。

    参数：
        - variants: 变体名称到输出的映射字典。

    返回：
        - 包含各敏感性指标的字典。
    """
    variant_names = ["positive_exemplar", "negative_exemplar", "random_exemplar", "empty_exemplar"]
    pairwise: dict[str, float] = {}
    pairwise_logit: dict[str, float] = {}
    pairwise_values: list[float] = []
    logit_values: list[float] = []
    for left_name, right_name in itertools.combinations(variant_names, 2):
        ratio = _mask_difference_ratio(variants[left_name]["mask_logits"], variants[right_name]["mask_logits"])
        logit_shift = _logit_difference(variants[left_name]["mask_logits"], variants[right_name]["mask_logits"])
        pairwise[f"{left_name}__vs__{right_name}"] = ratio
        pairwise_logit[f"{left_name}__vs__{right_name}"] = logit_shift
        pairwise_values.append(ratio)
        logit_values.append(logit_shift)
    dice_values = [float(variants[name]["metrics"]["Dice"]) for name in variant_names]
    iou_values = [float(variants[name]["metrics"]["IoU"]) for name in variant_names]
    mean_mask_difference = float(sum(pairwise_values) / max(len(pairwise_values), 1))
    mean_logit_difference = float(sum(logit_values) / max(len(logit_values), 1))
    score = 0.4 * mean_mask_difference + 0.2 * _variance(dice_values) + 0.2 * _variance(iou_values) + 0.2 * mean_logit_difference
    return {
        "mask_difference_ratio": pairwise,
        "logit_difference": pairwise_logit,
        "mean_mask_difference_ratio": mean_mask_difference,
        "mean_logit_difference": mean_logit_difference,
        "dice_variance": _variance(dice_values),
        "iou_variance": _variance(iou_values),
        "prompt_sensitivity_score": float(score),
    }


def _average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """对多个字典行按 key 取平均。

    参数：
        - rows: 字典列表，各字典有相同的 key。

    返回：
        - 各 key 的平均值字典。
    """
    if not rows:
        return {}
    summary: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            summary[key] = summary.get(key, 0.0) + float(value)
    return {key: value / len(rows) for key, value in summary.items()}


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """按域分组并平均各组的指标。

    参数：
        - rows: 包含 "domain" 和 "metrics" 字段的字典列表。

    返回：
        - 域名称到平均指标字典的映射。
    """
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["domain"], []).append(row["metrics"])
    return {domain: _average_rows(domain_rows) for domain, domain_rows in grouped.items()}


def _delta_metrics(current: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """计算当前指标相对于基线的差值（current - baseline）。

    参数：
        - current: 当前指标字典。
        - baseline: 基线指标字典。

    返回：
        - 各域的指标差值字典。
    """
    output: dict[str, dict[str, float]] = {}
    for domain, metrics in current.items():
        baseline_metrics = baseline.get(domain, {})
        output[domain] = {key: float(metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)) for key in metrics.keys()}
    return output


def main() -> int:
    """命令行入口：分析检索对 MedEx-SAM3 分割的影响，生成提示敏感性报告。

    参数：
        - 无。

    返回：
        - 退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(description="Analyze retrieval influence on MedEx-SAM3 segmentation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--internal-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt")
    parser.add_argument("--external-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/retrieval_analysis")
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
    parser.add_argument("--prefer-cross-domain-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--dummy-samples-per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
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

    split_rows = []
    prompt_rows = []
    attention_rows = []
    baseline_metrics_rows = []
    retrieval_metrics_rows = []
    split_inputs = [
        ("internal", internal_records),
        ("external", external_records),
    ]

    with torch.no_grad():
        for split_name, split_records in split_inputs:
            if args.max_samples_per_split is not None:
                split_records = split_records[: max(0, args.max_samples_per_split)]
            loader = DataLoader(SplitSegmentationDataset(split_records, args.image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)
            for batch_index, batch in enumerate(loader):
                images = batch["images"].to(device)
                masks = batch["masks"].to(device)
                boxes = batch["boxes"].to(device)
                record = batch["records"][0]
                image_id = str(record.get("image_id", f"{split_name}_{batch_index:03d}"))
                source_domain = infer_source_domain(
                    dataset_name=str(record.get("dataset_name", "")),
                    image_id=str(record.get("image_id", "")),
                    image_path=str(record.get("image_path", "")),
                    mask_path=str(record.get("mask_path", "")),
                )

                baseline_outputs = wrapper(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
                query_feature = resolve_feature_map(baseline_outputs["image_embeddings"], images)
                variant_retrievals = _build_prompt_variants(
                    bank,
                    retriever,
                    query_feature,
                    query_source=source_domain,
                    top_k_positive=args.top_k_positive,
                    top_k_negative=args.top_k_negative,
                    rng=random.Random(args.seed + batch_index),
                    prefer_cross_domain_positive=args.prefer_cross_domain_positive,
                    retrieval_mode=args.retrieval_mode,
                )
                variants = {
                    name: _run_variant(
                        adapter=adapter,
                        wrapper=wrapper,
                        similarity_builder=similarity_builder,
                        images=images,
                        masks=masks,
                        boxes=boxes,
                        text_prompt=batch["text_prompt"],
                        query_feature=query_feature,
                        baseline_outputs=baseline_outputs,
                        retrieval=retrieval,
                        retrieval_mode=args.retrieval_mode,
                    )
                    for name, retrieval in variant_retrievals.items()
                }

                prompt_sensitivity = _prompt_sensitivity(variants)
                prompt_rows.append({
                    "split": split_name,
                    "image_id": image_id,
                    "source_domain": source_domain,
                    **prompt_sensitivity,
                })
                split_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        "variants": {name: {"metrics": payload["metrics"]} for name, payload in variants.items()},
                        "prompt_sensitivity": prompt_sensitivity,
                    }
                )
                attention_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        "variants": {name: payload["attention_log"] for name, payload in variants.items()},
                    }
                )
                baseline_metrics_rows.append(
                    {
                        "split": split_name,
                        "domain": source_domain,
                        "metrics": variants["empty_exemplar"]["metrics"],
                    }
                )
                retrieval_metrics_rows.append(
                    {
                        "split": split_name,
                        "domain": source_domain,
                        "metrics": variants["positive_exemplar"]["metrics"],
                    }
                )

    prompt_by_domain: dict[str, list[dict[str, float]]] = {}
    for row in prompt_rows:
        prompt_by_domain.setdefault(row["source_domain"], []).append(
            {
                "mean_mask_difference_ratio": row["mean_mask_difference_ratio"],
                "mean_logit_difference": row["mean_logit_difference"],
                "dice_variance": row["dice_variance"],
                "iou_variance": row["iou_variance"],
                "prompt_sensitivity_score": row["prompt_sensitivity_score"],
            }
        )

    prompt_summary = {
        "overall": _average_rows(
            [
                {
                    "mean_mask_difference_ratio": row["mean_mask_difference_ratio"],
                    "mean_logit_difference": row["mean_logit_difference"],
                    "dice_variance": row["dice_variance"],
                    "iou_variance": row["iou_variance"],
                    "prompt_sensitivity_score": row["prompt_sensitivity_score"],
                }
                for row in prompt_rows
            ]
        ),
        "by_domain": {domain: _average_rows(domain_rows) for domain, domain_rows in prompt_by_domain.items()},
    }

    lora_only_summary = _group_metrics(baseline_metrics_rows)
    retrieval_summary = _group_metrics(retrieval_metrics_rows)
    ablation_summary = {
        "lora_only": lora_only_summary,
        "lora_plus_retrieval": retrieval_summary,
        "delta": _delta_metrics(retrieval_summary, lora_only_summary),
    }

    (output_dir / "prompt_sensitivity.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prompt_rows),
        encoding="utf-8",
    )
    (output_dir / "prototype_attention_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in attention_rows),
        encoding="utf-8",
    )
    (output_dir / "per_image_analysis.jsonl").write_text(
        "\n".join(json.dumps(row) for row in split_rows),
        encoding="utf-8",
    )
    summary = {
        "config": vars(args),
        "prompt_sensitivity": prompt_summary,
        "retrieval_ablation": ablation_summary,
        "prototype_attention_log": {
            "path": str(output_dir / "prototype_attention_log.jsonl"),
            "sample_count": len(attention_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
