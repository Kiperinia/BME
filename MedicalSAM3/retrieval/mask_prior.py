"""检索掩码先验聚合，用于局部检索引导。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


def _load_mask(mask_path: str | None, spatial_size: tuple[int, int]) -> torch.Tensor | None:
    """从文件路径加载掩码图像并调整到指定空间尺寸。

    参数：
        - mask_path: 掩码文件路径。
        - spatial_size: 目标空间尺寸 (高, 宽)。

    返回：
        - 加载并二值化的掩码张量，若路径无效则返回 None。
    """
    if not mask_path:
        return None
    target = Path(mask_path)
    if not target.exists():
        return None
    image = Image.open(target).convert("L").resize((spatial_size[1], spatial_size[0]), resample=Image.NEAREST)
    array = np.asarray(image).astype("float32")
    threshold = 0.0 if array.max() <= 1.0 else 127.0
    return torch.from_numpy((array > threshold).astype("float32")).unsqueeze(0)


def _weighted_mask_prior(entries: list[list[Any]], weights: torch.Tensor | None, spatial_size: tuple[int, int]) -> torch.Tensor | None:
    """根据权重对检索条目的掩码进行加权聚合，生成掩码先验。

    参数：
        - entries: 批次的检索条目列表。
        - weights: 每个条目对应的权重张量。
        - spatial_size: 目标空间尺寸 (高, 宽)。

    返回：
        - 加权聚合后的掩码先验张量，若无有效掩码则返回 None。
    """
    if weights is None or not isinstance(weights, torch.Tensor) or not entries:
        return None
    priors = []
    for batch_index, batch_entries in enumerate(entries):
        masks = []
        mask_weights = []
        for entry_index, entry in enumerate(batch_entries):
            if batch_index >= weights.shape[0] or entry_index >= weights.shape[1]:
                continue
            weight = float(weights[batch_index, entry_index].detach().cpu().item())
            if weight <= 0.0:
                continue
            mask = _load_mask(getattr(entry, "mask_path", None), spatial_size)
            if mask is None:
                continue
            masks.append(mask)
            mask_weights.append(weight)
        if not masks:
            priors.append(torch.zeros(1, spatial_size[0], spatial_size[1], dtype=torch.float32))
            continue
        stacked = torch.stack(masks, dim=0)
        weight_tensor = torch.tensor(mask_weights, dtype=torch.float32).view(-1, 1, 1, 1)
        prior = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6)
        priors.append(prior)
    return torch.stack(priors, dim=0)


def attach_retrieved_mask_priors(retrieval: dict[str, Any], spatial_size: tuple[int, int]) -> dict[str, Any]:
    """将检索到的正/负样本掩码先验附加到检索结果字典中。

    参数：
        - retrieval: 原始检索结果字典。
        - spatial_size: 目标空间尺寸 (高, 宽)。

    返回：
        - 更新后的检索结果字典，包含 mask_prior 相关字段。
    """
    updated = dict(retrieval)
    positive_prior = _weighted_mask_prior(
        retrieval.get("positive_entries", []),
        retrieval.get("positive_weights"),
        spatial_size,
    )
    negative_prior = _weighted_mask_prior(
        retrieval.get("negative_entries", []),
        retrieval.get("negative_weights"),
        spatial_size,
    )
    if positive_prior is not None:
        updated["positive_mask_prior"] = positive_prior.to(device=retrieval["positive_features"].device, dtype=retrieval["positive_features"].dtype)
    if negative_prior is not None:
        updated["negative_mask_prior"] = negative_prior.to(device=retrieval["negative_features"].device, dtype=retrieval["negative_features"].dtype)
    updated["mask_prior_available"] = bool(positive_prior is not None or negative_prior is not None)
    return updated


__all__ = ["attach_retrieved_mask_priors"]
