"""Prototype construction and variance-aware fusion for MedEx-SAM3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .memory_bank import ExemplarItem, ExemplarMemoryBank

logger = logging.getLogger(__name__)


class PrototypeBuilder:
    def __init__(self, variance_threshold: float = 0.4) -> None:
        self.variance_threshold = variance_threshold

    @staticmethod
    def build_mean_prototype(embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings.mean(dim=0), dim=0)

    @staticmethod
    def build_weighted_prototype(embeddings: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = scores.to(embeddings.device)
        weights = torch.softmax(scores.float(), dim=0)
        prototype = torch.sum(weights.unsqueeze(-1) * embeddings, dim=0)
        return F.normalize(prototype, dim=0), weights

    @staticmethod
    def build_attention_fused_prototype(query_embedding: torch.Tensor, exemplar_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        exemplars = F.normalize(exemplar_embeddings, dim=-1)
        query = F.normalize(query_embedding.to(exemplars.device), dim=-1)
        weights = torch.softmax(exemplars @ query, dim=0)
        prototype = torch.sum(weights.unsqueeze(-1) * exemplars, dim=0)
        return F.normalize(prototype, dim=0), weights

    @staticmethod
    def build_clustered_subprototypes(embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
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
        if prototype.dim() == 2:
            distances = torch.cdist(embeddings, prototype).min(dim=1).values
            return distances.pow(2).mean()
        distances = (embeddings - prototype.unsqueeze(0)).pow(2).sum(dim=-1)
        return distances.mean()

    def reject_if_high_variance(self, variance: torch.Tensor, threshold: float) -> bool:
        is_high = bool(variance.item() > threshold)
        if is_high:
            logger.warning("Prototype variance %.4f exceeded threshold %.4f", variance.item(), threshold)
        return is_high

    @staticmethod
    def _load_embedding(item: ExemplarItem) -> torch.Tensor:
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
