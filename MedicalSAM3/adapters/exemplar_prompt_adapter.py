"""MedEx-SAM3 的原型到提示 token 投影模块。

将正/负/边界示例原型投影为可学习的提示 token，并通过门控融合
生成供分割解码器使用的提示序列。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _reduce_proto(proto: torch.Tensor) -> torch.Tensor:
    """将原型张量压缩为每个样本单一向量表示。

    参数：
        - proto: 原型张量，支持 [B, C] 或 [B, K, C]

    返回：
        - 形如 [B, C] 的压缩后原型向量
    """
    if proto.dim() == 2:
        return proto
    if proto.dim() == 3:
        return proto.mean(dim=1)
    raise ValueError("Prototype tensor must be [B, C] or [B, K, C]")


def _project_tokens(proto: torch.Tensor, projector: "_TokenProjector") -> torch.Tensor:
    """将原型张量投影为多 token 提示序列。

    参数：
        - proto: 原型张量，支持 [B, C] 或 [B, K, C]
        - projector: token 投影器实例

    返回：
        - 投影后的提示 token 张量
    """
    if proto.dim() == 2:
        return projector(proto)
    if proto.dim() == 3:
        batch_size, groups, dim = proto.shape
        flat_tokens = projector(proto.reshape(batch_size * groups, dim))
        return flat_tokens.reshape(batch_size, groups * projector.num_tokens, dim)
    raise ValueError("Prototype tensor must be [B, C] or [B, K, C]")


class _TokenProjector(nn.Module):
    """将单一向量投影为指定数量的 token 序列。

    参数：
        - dim: 特征维度
        - num_tokens: 每个输入向量投影出的 token 数量
    """

    def __init__(self, dim: int, num_tokens: int) -> None:
        """初始化两层 MLP 投影器。

        参数：
            - dim: 特征维度
            - num_tokens: 每个输入向量投影出的 token 数量

        返回：
            - 无返回值，完成投影器构建
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算：投影并重塑为 token 序列。

        参数：
            - x: 形如 [B, C] 的输入向量

        返回：
            - 形如 [B, num_tokens, C] 的 token 张量
        """
        batch_size, dim = x.shape
        return self.proj(x).reshape(batch_size, self.num_tokens, dim)


class ExemplarPromptAdapter(nn.Module):
    """示例提示适配器，将正/负/边界原型转为门控融合的提示 token。

    参数：
        - dim: 特征维度
        - num_pos_tokens: 正例 token 数量
        - num_neg_tokens: 负例 token 数量
        - num_boundary_tokens: 边界 token 数量
    """

    def __init__(
        self,
        dim: int,
        num_pos_tokens: int = 4,
        num_neg_tokens: int = 2,
        num_boundary_tokens: int = 2,
    ) -> None:
        """初始化三类原型投影器、融合门与 token 归一化层。

        参数：
            - dim: 特征维度
            - num_pos_tokens: 正例 token 数量
            - num_neg_tokens: 负例 token 数量
            - num_boundary_tokens: 边界 token 数量

        返回：
            - 无返回值，完成各子层构建
        """
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
        """前向计算：门控融合三类原型 token 并输出提示序列与辅助信息。

        参数：
            - positive_proto: 正例原型张量
            - negative_proto: 负例原型张量，可选
            - boundary_proto: 边界原型张量，可选
            - query_feat: 查询特征，用于驱动门控，可选

        返回：
            - 二元组：(融合后的提示 token, 辅助信息字典)
        """
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
        positive_tokens = self.token_norm(_project_tokens(positive_proto, self.positive_proj)) * gates[:, 0:1, None]
        if negative_proto is not None:
            negative_tokens = self.token_norm(_project_tokens(negative_proto, self.negative_proj)) * gates[:, 1:2, None]
        else:
            negative_tokens = torch.zeros(
                batch_size,
                self.num_neg_tokens,
                dim,
                device=positive_summary.device,
                dtype=positive_summary.dtype,
            )
        if boundary_proto is not None:
            boundary_tokens = self.token_norm(_project_tokens(boundary_proto, self.boundary_proj)) * gates[:, 2:3, None]
        else:
            boundary_tokens = torch.zeros(
                batch_size,
                self.num_boundary_tokens,
                dim,
                device=positive_summary.device,
                dtype=positive_summary.dtype,
            )
        suppression_gate = gates[:, 3:4]

        prompt_tokens = torch.cat([positive_tokens, boundary_tokens, negative_tokens], dim=1)
        aux = {
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "boundary_tokens": boundary_tokens,
            "fusion_weights": gates,
            "suppression_gate": suppression_gate,
        }
        return prompt_tokens, aux
