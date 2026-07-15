from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .exemplar_bank_schemas import QueryFeatureBatch, RetrievalCandidate, RetrievalPackage


@dataclass(slots=True)
class RetrievedFeatureSet:
    """brief:
        Represent RetrievedFeatureSet state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    positive_tokens: torch.Tensor
    negative_tokens: torch.Tensor
    boundary_tokens: torch.Tensor
    positive_map: torch.Tensor
    negative_map: torch.Tensor
    boundary_map: torch.Tensor
    candidate_metadata: list[RetrievalCandidate]


class SemanticProjector(nn.Module):
    """brief:
        Represent SemanticProjector state and behavior.

    parameter:
        - dim: Input value for dim.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """brief:
            Handle forward.

        parameter:
            - embedding: Input value for embedding.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return F.normalize(self.proj(embedding), dim=-1)


class BoundaryEmbeddingHead(nn.Module):
    """brief:
        Represent BoundaryEmbeddingHead state and behavior.

    parameter:
        - dim: Input value for dim.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """brief:
            Handle forward.

        parameter:
            - embedding: Input value for embedding.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        return F.normalize(self.proj(embedding), dim=-1)


class MultiScaleFeatureAligner(nn.Module):
    """brief:
        Represent MultiScaleFeatureAligner state and behavior.

    parameter:
        - dim: Input value for dim.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.scale_mix = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, spatial_feature: torch.Tensor) -> torch.Tensor:
        """brief:
            Handle forward.

        parameter:
            - spatial_feature: Input value for spatial_feature.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        low = F.avg_pool2d(spatial_feature, kernel_size=2, stride=2)
        mid = spatial_feature
        high = F.interpolate(spatial_feature, scale_factor=2.0, mode="bilinear", align_corners=False)
        low_up = F.interpolate(low, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        high_down = F.interpolate(high, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        return self.scale_mix(torch.cat([low_up, mid, high_down], dim=1))


class CrossAttentionReranker(nn.Module):
    """brief:
        Represent CrossAttentionReranker state and behavior.

    parameter:
        - dim: Input value for dim.
        - num_heads: Input value for num_heads.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.
            - num_heads: Input value for num_heads.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.score_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """brief:
            Handle forward.

        parameter:
            - query_tokens: Input value for query_tokens.
            - memory_tokens: Input value for memory_tokens.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        attended, weights = self.attn(query_tokens, memory_tokens, memory_tokens, need_weights=True)
        pooled = attended.mean(dim=1)
        return pooled, self.score_head(pooled).squeeze(-1)


class GatedRetrievalFusion(nn.Module):
    """brief:
        Represent GatedRetrievalFusion state and behavior.

    parameter:
        - dim: Input value for dim.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.alpha_head = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.neg_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.prompt_proj = nn.Linear(dim * 3, dim)
        self.decoder_proj = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.logit_proj = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(
        self,
        query_semantic: torch.Tensor,
        positive_proto: torch.Tensor,
        negative_proto: torch.Tensor,
        boundary_proto: torch.Tensor,
        spatial_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """brief:
            Handle forward.

        parameter:
            - query_semantic: Input value for query_semantic.
            - positive_proto: Input value for positive_proto.
            - negative_proto: Input value for negative_proto.
            - boundary_proto: Input value for boundary_proto.
            - spatial_map: Input value for spatial_map.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        fused_token = torch.cat([positive_proto, negative_proto, boundary_proto], dim=-1)
        alpha = self.alpha_head(fused_token)
        negative_lambda = self.neg_head(negative_proto)
        prompt_tokens = self.prompt_proj(fused_token).unsqueeze(1)

        positive_map = positive_proto.unsqueeze(-1).unsqueeze(-1).expand_as(spatial_map)
        negative_map = negative_proto.unsqueeze(-1).unsqueeze(-1).expand_as(spatial_map)
        boundary_map = boundary_proto.unsqueeze(-1).unsqueeze(-1).expand_as(spatial_map)
        decoder_feature_bias_map = self.decoder_proj(torch.cat([positive_map, negative_map, boundary_map], dim=1))
        semantic_prototype_map = alpha.unsqueeze(-1) * positive_map
        spatial_bias_map = torch.norm(boundary_map - negative_map, dim=1, keepdim=True)
        mask_logit_bias_map = self.logit_proj(decoder_feature_bias_map) - negative_lambda.unsqueeze(-1).unsqueeze(-1) * self.logit_proj(negative_map)

        retrieval_prior = {
            "semantic_prototype": alpha * positive_proto + (1.0 - alpha) * query_semantic,
            "semantic_prototype_map": semantic_prototype_map,
            "spatial_bias_map": spatial_bias_map,
            "decoder_feature_bias_map": decoder_feature_bias_map,
            "mask_logit_bias_map": mask_logit_bias_map,
            "fusion_alpha": alpha,
            "negative_lambda": negative_lambda,
        }
        return prompt_tokens, retrieval_prior


class RetrievalConfidenceEstimator(nn.Module):
    """brief:
        Represent RetrievalConfidenceEstimator state and behavior.

    parameter:
        - dim: Input value for dim.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.confidence_head = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, positive_proto: torch.Tensor, negative_proto: torch.Tensor, boundary_proto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """brief:
            Handle forward.

        parameter:
            - positive_proto: Input value for positive_proto.
            - negative_proto: Input value for negative_proto.
            - boundary_proto: Input value for boundary_proto.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        fused = torch.cat([positive_proto, negative_proto, boundary_proto], dim=-1)
        confidence = self.confidence_head(fused)
        uncertainty = self.uncertainty_head(fused)
        return confidence, uncertainty


class ExemplarRetrievalPipeline(nn.Module):
    """brief:
        Represent ExemplarRetrievalPipeline state and behavior.

    parameter:
        - dim: Input value for dim.
        - num_heads: Input value for num_heads.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    def __init__(self, dim: int = 256, num_heads: int = 8) -> None:
        """brief:
            Initialize this object.

        parameter:
            - dim: Input value for dim.
            - num_heads: Input value for num_heads.

        retrival:
            - Returns None; performs side effects described in the brief section.
        """
        super().__init__()
        self.semantic_projector = SemanticProjector(dim)
        self.boundary_projector = BoundaryEmbeddingHead(dim)
        self.multi_scale = MultiScaleFeatureAligner(dim)
        self.positive_reranker = CrossAttentionReranker(dim, num_heads=num_heads)
        self.negative_reranker = CrossAttentionReranker(dim, num_heads=num_heads)
        self.boundary_reranker = CrossAttentionReranker(dim, num_heads=num_heads)
        self.fusion = GatedRetrievalFusion(dim)
        self.confidence = RetrievalConfidenceEstimator(dim)

    def forward(self, query: QueryFeatureBatch, retrieved: RetrievedFeatureSet) -> RetrievalPackage:
        """brief:
            Handle forward.

        parameter:
            - query: Input value for query.
            - retrieved: Input value for retrieved.

        retrival:
            - Returns the computed value for the caller or workflow.
        """
        query_semantic = self.semantic_projector(query.semantic_embedding)
        query_boundary = self.boundary_projector(query.boundary_embedding)
        aligned_spatial = self.multi_scale(query.spatial_embedding)
        query_tokens = query.hidden_states if query.hidden_states is not None else query_semantic.unsqueeze(1)

        positive_proto, positive_score = self.positive_reranker(query_tokens, retrieved.positive_tokens)
        negative_proto, negative_score = self.negative_reranker(query_tokens, retrieved.negative_tokens)
        boundary_proto, boundary_score = self.boundary_reranker(query_tokens, retrieved.boundary_tokens)

        confidence, uncertainty = self.confidence(positive_proto, negative_proto, boundary_proto)
        prompt_tokens, retrieval_prior = self.fusion(
            query_semantic=query_semantic,
            positive_proto=positive_proto,
            negative_proto=negative_proto,
            boundary_proto=boundary_proto + query_boundary,
            spatial_map=aligned_spatial,
        )
        diagnostics = {
            "positive_score": positive_score.detach().cpu().tolist(),
            "negative_score": negative_score.detach().cpu().tolist(),
            "boundary_score": boundary_score.detach().cpu().tolist(),
        }
        return RetrievalPackage(
            prompt_tokens=prompt_tokens,
            retrieval_prior=retrieval_prior,
            confidence=confidence,
            uncertainty=uncertainty,
            selected_candidates=retrieved.candidate_metadata,
            diagnostics=diagnostics,
        )
