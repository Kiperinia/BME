"""MedEx-SAM3 的原型构建与方差感知融合模块。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .memory_bank import ExemplarItem, ExemplarMemoryBank

logger = logging.getLogger(__name__)


class PrototypeBuilder:
    """示例原型构建器。

    提供均值、加权、注意力融合、聚类子原型等多种原型构建策略，
    并支持基于方差的拒绝判断与正/负/边界三类原型的端到端构建。

    参数：
        - variance_threshold: 原型方差拒绝阈值，超过该值则触发聚类降级。
    """
    def __init__(self, variance_threshold: float = 0.4) -> None:
        """初始化原型构建器。

        参数：
            - variance_threshold: 原型方差拒绝阈值。

        返回：
            - 无返回值，仅完成对象初始化。
        """
        self.variance_threshold = variance_threshold

    @staticmethod
    def build_mean_prototype(embeddings: torch.Tensor) -> torch.Tensor:
        """构建均值原型。

        参数：
            - embeddings: 示例嵌入张量，形状为 (N, C)。

        返回：
            - L2 归一化后的均值原型向量，形状为 (C,)。
        """
        return F.normalize(embeddings.mean(dim=0), dim=0)

    @staticmethod
    def build_weighted_prototype(embeddings: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """构建分数加权原型。

        对评分做 softmax 得到权重，再对嵌入加权求和。

        参数：
            - embeddings: 示例嵌入张量，形状为 (N, C)。
            - scores: 每个示例的评分张量，形状为 (N,)。

        返回：
            - L2 归一化后的加权原型向量 (C,) 以及对应权重向量 (N,)。
        """
        scores = scores.to(embeddings.device)
        weights = torch.softmax(scores.float(), dim=0)
        prototype = torch.sum(weights.unsqueeze(-1) * embeddings, dim=0)
        return F.normalize(prototype, dim=0), weights

    @staticmethod
    def build_attention_fused_prototype(query_embedding: torch.Tensor, exemplar_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """构建基于注意力融合的原型。

        以查询嵌入与各示例嵌入的相似度作为注意力权重进行加权融合。

        参数：
            - query_embedding: 查询嵌入向量，形状为 (C,)。
            - exemplar_embeddings: 示例嵌入张量，形状为 (N, C)。

        返回：
            - L2 归一化后的融合原型向量 (C,) 以及对应注意力权重 (N,)。
        """
        exemplars = F.normalize(exemplar_embeddings, dim=-1)
        query = F.normalize(query_embedding.to(exemplars.device), dim=-1)
        weights = torch.softmax(exemplars @ query, dim=0)
        prototype = torch.sum(weights.unsqueeze(-1) * exemplars, dim=0)
        return F.normalize(prototype, dim=0), weights

    @staticmethod
    def build_clustered_subprototypes(embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """通过 k-means 风格聚类构建多个子原型。

        参数：
            - embeddings: 示例嵌入张量，形状为 (N, C)。
            - n_clusters: 期望的聚类簇数，会被限制在 [1, N] 范围内。

        返回：
            - L2 归一化后的子原型张量，形状为 (n_clusters, C)。
        """
        n_clusters = max(1, min(n_clusters, embeddings.shape[0]))
        centers = embeddings[:n_clusters].clone()
        for _ in range(6):
            distances = torch.cdist(embeddings, centers)
            assignments = distances.argmin(dim=1)
            for cluster_index in range(n_clusters):
                mask = assignments == cluster_index
                if mask.any():
                    centers[cluster_index] = embeddings[mask].mean(dim=0)
        return F.normalize(centers, dim=-1)

    @staticmethod
    def compute_prototype_variance(embeddings: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """计算示例嵌入到原型的方差。

        参数：
            - embeddings: 示例嵌入张量，形状为 (N, C)。
            - prototype: 原型嵌入，形状为 (C,) 或 (M, C)；为多原型时取最近距离。

        返回：
            - 距离平方均值标量张量。
        """
        if prototype.dim() == 2:
            distances = torch.cdist(embeddings, prototype).min(dim=1).values
            return distances.pow(2).mean()
        distances = (embeddings - prototype.unsqueeze(0)).pow(2).sum(dim=-1)
        return distances.mean()

    def reject_if_high_variance(self, variance: torch.Tensor, threshold: float) -> bool:
        """判断原型方差是否过高并决定是否拒绝。

        参数：
            - variance: 待判定的方差标量张量。
            - threshold: 拒绝阈值。

        返回：
            - 若方差超过阈值返回 True，否则返回 False。
        """
        is_high = bool(variance.item() > threshold)
        if is_high:
            logger.warning("Prototype variance %.4f exceeded threshold %.4f", variance.item(), threshold)
        return is_high

    @staticmethod
    def _load_embedding(item: ExemplarItem) -> torch.Tensor:
        """从磁盘加载示例条目对应的嵌入张量。

        支持直接张量或字典形式（自动识别常见的嵌入键名）。

        参数：
            - item: 示例条目，需包含有效的 embedding_path。

        返回：
            - 加载并压缩到一维的嵌入张量；若格式不支持则抛出 TypeError。
        """
        if item.embedding_path is None:
            raise ValueError(f"Missing embedding_path for exemplar {item.item_id}")
        embedding = torch.load(Path(item.embedding_path), map_location="cpu", weights_only=False)
        if isinstance(embedding, dict):
            for key in ["foreground_embedding", "global_embedding", "boundary_embedding", "context_embedding", "embedding"]:
                value = embedding.get(key)
                if isinstance(value, torch.Tensor):
                    return value.squeeze(0) if value.dim() > 1 else value
        if isinstance(embedding, torch.Tensor):
            return embedding.squeeze(0) if embedding.dim() > 1 else embedding
        raise TypeError(f"Unsupported embedding payload for {item.item_id}")

    @staticmethod
    def _item_score(query: torch.Tensor, embedding: torch.Tensor, item: ExemplarItem) -> float:
        """计算单个示例条目相对查询的综合评分。

        综合余弦相似度与条目上的各项质量/难度/风险分项加权得到。

        参数：
            - query: 查询嵌入向量。
            - embedding: 示例嵌入向量。
            - item: 示例条目，提供各项分值。

        返回：
            - 综合评分浮点值。
        """
        embedding = embedding.to(query.device)
        similarity = F.cosine_similarity(query.unsqueeze(0), embedding.unsqueeze(0)).item()
        return (
            0.30 * similarity
            + 0.20 * item.quality_score
            + 0.15 * item.boundary_score
            + 0.15 * item.diversity_score
            + 0.10 * item.difficulty_score
            - 0.10 * item.uncertainty_score
            - 0.20 * item.false_positive_risk
        )

    def _build_single_type(
        self,
        query: torch.Tensor,
        items: list[ExemplarItem],
        top_k: int,
    ) -> dict[str, Any]:
        """为单一类型（positive/negative/boundary）构建原型。

        流程：加载并评分示例 -> 取 top_k -> 加权原型 -> 方差校验，
        若方差过高则退化为聚类子原型。

        参数：
            - query: 查询嵌入向量，形状为 (C,)。
            - items: 候选示例条目列表。
            - top_k: 选取的最优示例数量。

        返回：
            - 包含 prototype、selected_item_ids、weights、variance 的结果字典。
        """
        if top_k <= 0 or not items:
            return {"prototype": None, "selected_item_ids": [], "weights": [], "variance": None}

        scored = []
        for item in items:
            embedding = self._load_embedding(item).float().to(query.device)
            score = self._item_score(query, embedding, item)
            scored.append((score, item, F.normalize(embedding, dim=0)))
        scored.sort(key=lambda entry: entry[0], reverse=True)
        selected = scored[: min(top_k, len(scored))]
        scores_tensor = torch.tensor([entry[0] for entry in selected], dtype=torch.float32, device=query.device)
        embeddings = torch.stack([entry[2] for entry in selected], dim=0)
        prototype, weights = self.build_weighted_prototype(embeddings, scores_tensor)
        variance = self.compute_prototype_variance(embeddings, prototype)
        if self.reject_if_high_variance(variance, self.variance_threshold) and embeddings.shape[0] > 1:
            prototype = self.build_clustered_subprototypes(embeddings, n_clusters=min(2, embeddings.shape[0]))
        return {
            "prototype": prototype,
            "selected_item_ids": [entry[1].item_id for entry in selected],
            "weights": weights.tolist(),
            "variance": float(variance.item()),
        }

    def build_positive_negative_boundary_prototypes(
        self,
        query: torch.Tensor,
        memory_bank: ExemplarMemoryBank,
        top_k: int,
    ) -> dict[str, Any]:
        """为正例、负例、边界三类示例分别构建原型。

        参数：
            - query: 查询嵌入向量，形状为 (C,) 或 (1, C)。
            - memory_bank: 示例记忆库实例。
            - top_k: 每类选取的最优示例数量。

        返回：
            - 包含 positive、negative、boundary 三类原型构建结果的字典。
        """
        if query.dim() == 2:
            if query.shape[0] != 1:
                raise ValueError("PrototypeBuilder currently expects a single query embedding")
            query = query[0]
        query = F.normalize(query.float(), dim=0)
        return {
            "positive": self._build_single_type(query, memory_bank.get_items(type="positive", human_verified=True), top_k),
            "negative": self._build_single_type(query, memory_bank.get_items(type="negative", human_verified=True), top_k),
            "boundary": self._build_single_type(query, memory_bank.get_items(type="boundary", human_verified=True), top_k),
        }
