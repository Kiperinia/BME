"""Prototype-to-prompt token projection for MedEx-SAM3."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _reduce_proto(proto: torch.Tensor) -> torch.Tensor:
    if proto.dim() == 2:
        return proto
    if proto.dim() == 3:
        return proto.mean(dim=1)
    raise ValueError("Prototype tensor must be [B, C] or [B, K, C]")


def _project_tokens(proto: torch.Tensor, projector: "_TokenProjector") -> torch.Tensor:
    if proto.dim() == 2:
        return projector(proto)
    if proto.dim() == 3:
        batch_size, groups, dim = proto.shape
        flat_tokens = projector(proto.reshape(batch_size * groups, dim))
        return flat_tokens.reshape(batch_size, groups * projector.num_tokens, dim)
    raise ValueError("Prototype tensor must be [B, C] or [B, K, C]")


class _TokenProjector(nn.Module):
    def __init__(self, dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim = x.shape
        return self.proj(x).reshape(batch_size, self.num_tokens, dim)


class ExemplarPromptAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_pos_tokens: int = 2,
        num_neg_tokens: int = 1,
        num_boundary_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.num_pos_tokens = num_pos_tokens
        self.num_neg_tokens = num_neg_tokens
        self.num_boundary_tokens = num_boundary_tokens
        self.positive_proj = _TokenProjector(dim, num_pos_tokens)
        self.negative_proj = _TokenProjector(dim, num_neg_tokens)
        self.boundary_proj = _TokenProjector(dim, num_boundary_tokens)
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.token_norm = nn.LayerNorm(dim)

    def forward(
        self,
        positive_proto: torch.Tensor,
        negative_proto: Optional[torch.Tensor] = None,
        boundary_proto: Optional[torch.Tensor] = None,
        query_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positive_summary = _reduce_proto(positive_proto)
        batch_size, dim = positive_summary.shape
        query_summary = query_feat if query_feat is not None else positive_summary
        negative_summary = _reduce_proto(negative_proto) if negative_proto is not None else torch.zeros_like(positive_summary)
        boundary_summary = _reduce_proto(boundary_proto) if boundary_proto is not None else torch.zeros_like(positive_summary)

        gates = torch.sigmoid(
            self.fusion_gate(
                torch.cat([query_summary, positive_summary, negative_summary, boundary_summary], dim=-1)
            )
        )
        positive_tokens = _project_tokens(positive_proto, self.positive_proj) * gates[:, 0:1, None]
        negative_tokens = _project_tokens(negative_proto, self.negative_proj) * gates[:, 1:2, None] if negative_proto is not None else self.negative_proj(negative_summary) * gates[:, 1:2, None]
        boundary_tokens = _project_tokens(boundary_proto, self.boundary_proj) * gates[:, 2:3, None] if boundary_proto is not None else self.boundary_proj(boundary_summary) * gates[:, 2:3, None]
        suppression_gate = gates[:, 3:4]

        prompt_tokens = torch.cat([positive_tokens, boundary_tokens], dim=1)
        prompt_tokens = self.token_norm(prompt_tokens)
        aux = {
            "positive_tokens": self.token_norm(positive_tokens),
            "negative_tokens": self.token_norm(negative_tokens),
            "boundary_tokens": self.token_norm(boundary_tokens),
            "fusion_weights": gates,
            "suppression_gate": suppression_gate,
        }
        return prompt_tokens, aux
