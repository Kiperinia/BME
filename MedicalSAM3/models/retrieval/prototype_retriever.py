"""用于检索条件化域适应的原型检索模块。"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from MedicalSAM3.exemplar_bank.bank import PrototypeBankEntry, RSSDABank


def _weighted_average(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """按权重对特征做加权和并做 L2 归一化。

    参数：
        - features: 形状为 [B, K, C] 的特征张量。
        - weights: 形状为 [B, K] 的权重张量。

    返回：
        - 形状为 [B, C] 的归一化加权平均特征张量。
    """
    if features.numel() == 0:
        return torch.empty(features.shape[0], 0, device=features.device)
    return F.normalize((weights.unsqueeze(-1) * features).sum(dim=1), dim=-1)


def _entropy(weights: torch.Tensor) -> torch.Tensor:
    """计算归一化权重的香农熵。

    参数：
        - weights: 权重张量。

    返回：
        - 权重分布的熵标量。
    """
    if weights.numel() == 0:
        return weights.new_tensor(0.0)
    normalized = weights / weights.sum().clamp_min(1e-6)
    return -(normalized * (normalized.clamp_min(1e-6).log())).sum()


def _match_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    """通过截断或零填充将特征最后一维对齐到目标维度。

    参数：
        - features: 输入特征张量。
        - target_dim: 目标特征维度。

    返回：
        - 最后一维等于 target_dim 的特征张量。
    """
    if features.shape[-1] == target_dim:
        return features
    if features.shape[-1] > target_dim:
        return features[..., :target_dim]
    pad_width = target_dim - features.shape[-1]
    return F.pad(features, (0, pad_width))


class PrototypeRetriever(nn.Module):
    """原型检索器，从原型库中检索最相关的正负原型用于条件化分割。

    参数：
        - bank: 原型库实例。
        - feature_dim: 特征维度。
        - top_k_positive: 正原型检索数量。
        - top_k_negative: 负原型检索数量。

    返回：
        - 构建可用的原型检索器模块实例。
    """
    def __init__(
        self,
        bank: RSSDABank,
        feature_dim: int,
        top_k_positive: int = 1,
        top_k_negative: int = 1,
    ) -> None:
        """初始化原型库引用、检索数量与查询投影子模块。

        参数：
            - bank: 原型库实例。
            - feature_dim: 特征维度。
            - top_k_positive: 正原型检索数量。
            - top_k_negative: 负原型检索数量。

        返回：
            - 无返回值，完成子模块与属性的构建。
        """
        super().__init__()
        self.bank = bank
        self.feature_dim = feature_dim
        self.top_k_positive = top_k_positive
        self.top_k_negative = top_k_negative
        self.query_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    @staticmethod
    def global_average_pool(query_feature: torch.Tensor) -> torch.Tensor:
        """对查询特征做全局平均池化并 L2 归一化。

        参数：
            - query_feature: 形状为 [B, C, H, W] 的查询特征张量。

        返回：
            - 形状为 [B, C] 的归一化全局查询向量。
        """
        if query_feature.dim() != 4:
            raise ValueError("query_feature must have shape [B, C, H, W]")
        return F.normalize(F.adaptive_avg_pool2d(query_feature, 1).flatten(1), dim=1)

    def _retrieve_single(
        self,
        query_global: torch.Tensor,
        entries: list[PrototypeBankEntry],
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
        """对单个查询向量从候选条目中检索 top-k 最相似原型。

        参数：
            - query_global: 单个归一化查询向量。
            - entries: 候选原型库条目列表。
            - top_k: 检索数量。

        返回：
            - 由选中特征、softmax 权重、选中条目、加权原型、相似度组成的元组。
        """
        if not entries or top_k <= 0:
            dim = int(query_global.shape[-1])
            return (
                torch.empty(0, dim, device=query_global.device),
                torch.empty(0, device=query_global.device),
                [],
                torch.empty(dim, device=query_global.device),
                torch.empty(0, device=query_global.device),
            )
        bank_features = self.bank.stack_features(entries, device=query_global.device)
        bank_features = _match_feature_dim(bank_features, int(query_global.shape[-1]))
        bank_features = F.normalize(bank_features, dim=-1)
        similarities = torch.matmul(bank_features, query_global)
        values, indices = torch.topk(similarities, k=min(top_k, similarities.shape[0]))
        selected_features = bank_features[indices]
        selected_entries = [entries[int(index)] for index in indices.detach().cpu().tolist()]
        weights = torch.softmax(values, dim=0)
        prototype = _weighted_average(selected_features.unsqueeze(0), weights.unsqueeze(0))[0]
        return selected_features, weights, selected_entries, prototype, values

    def forward(
        self,
        query_feature: torch.Tensor,
        top_k_positive: Optional[int] = None,
        top_k_negative: Optional[int] = None,
        query_source_datasets: Optional[list[str]] = None,
        prefer_cross_domain_positive: bool = False,
    ) -> dict[str, Any]:
        """对批量查询特征检索正负原型并汇总为结果字典。

        参数：
            - query_feature: 形状为 [B, C, H, W] 的查询特征张量。
            - top_k_positive: 可选正原型检索数量覆盖。
            - top_k_negative: 可选负原型检索数量覆盖。
            - query_source_datasets: 可选各样本来源数据集名称列表。
            - prefer_cross_domain_positive: 是否优先检索跨域正原型。

        返回：
            - 包含正负原型特征、权重、原型向量及检索稳定性统计的字典。
        """
        query_global = self.global_average_pool(query_feature)
        projected_query = F.normalize(self.query_projection(query_global), dim=1)
        positive_entries = self.bank.get_entries(polarity="positive", human_verified=True)
        negative_entries = self.bank.get_entries(polarity="negative", human_verified=True)

        positive_features_batch = []
        positive_weights_batch = []
        positive_prototypes = []
        positive_selected_batch = []
        positive_scores_batch = []
        negative_features_batch = []
        negative_weights_batch = []
        negative_prototypes = []
        negative_selected_batch = []
        negative_scores_batch = []

        for query_index in range(query_global.shape[0]):
            query_source = None
            if query_source_datasets is not None and query_index < len(query_source_datasets):
                query_source = str(query_source_datasets[query_index])
            query_positive_entries = positive_entries
            if prefer_cross_domain_positive and query_source:
                cross_domain_entries = [entry for entry in positive_entries if str(entry.source_dataset) != query_source]
                if cross_domain_entries:
                    query_positive_entries = cross_domain_entries
            pos_features, pos_weights, pos_selected, pos_proto, pos_scores = self._retrieve_single(
                projected_query[query_index],
                query_positive_entries,
                top_k_positive or self.top_k_positive,
            )
            neg_features, neg_weights, neg_selected, neg_proto, neg_scores = self._retrieve_single(
                projected_query[query_index],
                negative_entries,
                top_k_negative or self.top_k_negative,
            )
            positive_features_batch.append(pos_features)
            positive_weights_batch.append(pos_weights)
            positive_prototypes.append(pos_proto)
            positive_selected_batch.append(pos_selected)
            positive_scores_batch.append(pos_scores)
            negative_features_batch.append(neg_features)
            negative_weights_batch.append(neg_weights)
            negative_prototypes.append(neg_proto)
            negative_selected_batch.append(neg_selected)
            negative_scores_batch.append(neg_scores)

        dim = int(query_global.shape[-1])
        max_pos = max((item.shape[0] for item in positive_features_batch), default=0)
        max_neg = max((item.shape[0] for item in negative_features_batch), default=0)
        positive_features = torch.zeros(query_global.shape[0], max_pos, dim, device=query_global.device)
        positive_weights = torch.zeros(query_global.shape[0], max_pos, device=query_global.device)
        negative_features = torch.zeros(query_global.shape[0], max_neg, dim, device=query_global.device)
        negative_weights = torch.zeros(query_global.shape[0], max_neg, device=query_global.device)
        positive_score_tensor = torch.zeros(query_global.shape[0], max_pos, device=query_global.device)
        negative_score_tensor = torch.zeros(query_global.shape[0], max_neg, device=query_global.device)

        for batch_index, item in enumerate(positive_features_batch):
            if item.numel() == 0:
                continue
            positive_features[batch_index, : item.shape[0]] = item
            positive_weights[batch_index, : item.shape[0]] = positive_weights_batch[batch_index]
            positive_score_tensor[batch_index, : item.shape[0]] = positive_scores_batch[batch_index]
        for batch_index, item in enumerate(negative_features_batch):
            if item.numel() == 0:
                continue
            negative_features[batch_index, : item.shape[0]] = item
            negative_weights[batch_index, : item.shape[0]] = negative_weights_batch[batch_index]
            negative_score_tensor[batch_index, : item.shape[0]] = negative_scores_batch[batch_index]

        positive_prototype = torch.stack([
            proto if proto.numel() else torch.zeros(dim, device=query_global.device) for proto in positive_prototypes
        ], dim=0)
        negative_prototype = torch.stack([
            proto if proto.numel() else torch.zeros(dim, device=query_global.device) for proto in negative_prototypes
        ], dim=0)
        positive_similarity_mean = torch.stack([
            scores.mean() if scores.numel() > 0 else query_global.new_tensor(0.0) for scores in positive_scores_batch
        ], dim=0)
        negative_similarity_mean = torch.stack([
            scores.mean() if scores.numel() > 0 else query_global.new_tensor(0.0) for scores in negative_scores_batch
        ], dim=0)
        positive_similarity_std = torch.stack([
            scores.std(unbiased=False) if scores.numel() > 0 else query_global.new_tensor(0.0) for scores in positive_scores_batch
        ], dim=0)
        negative_similarity_std = torch.stack([
            scores.std(unbiased=False) if scores.numel() > 0 else query_global.new_tensor(0.0) for scores in negative_scores_batch
        ], dim=0)
        positive_weight_entropy = torch.stack([
            _entropy(weights) if weights.numel() > 0 else query_global.new_tensor(0.0) for weights in positive_weights_batch
        ], dim=0)
        negative_weight_entropy = torch.stack([
            _entropy(weights) if weights.numel() > 0 else query_global.new_tensor(0.0) for weights in negative_weights_batch
        ], dim=0)

        return {
            "query_global": query_global,
            "projected_query": projected_query,
            "query_source_datasets": query_source_datasets or [],
            "positive_features": positive_features,
            "positive_weights": positive_weights,
            "positive_score_tensor": positive_score_tensor,
            "positive_entries": positive_selected_batch,
            "positive_prototype": positive_prototype,
            "positive_scores": positive_scores_batch,
            "negative_features": negative_features,
            "negative_weights": negative_weights,
            "negative_score_tensor": negative_score_tensor,
            "negative_entries": negative_selected_batch,
            "negative_prototype": negative_prototype,
            "negative_scores": negative_scores_batch,
            "retrieval_stability": {
                "positive_similarity_mean": positive_similarity_mean,
                "negative_similarity_mean": negative_similarity_mean,
                "margin": positive_similarity_mean - negative_similarity_mean,
                "positive_similarity_std": positive_similarity_std,
                "negative_similarity_std": negative_similarity_std,
                "positive_weight_entropy": positive_weight_entropy,
                "negative_weight_entropy": negative_weight_entropy,
            },
        }
