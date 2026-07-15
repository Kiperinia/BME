"""Prototype extraction for RSS-DA memory banks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .bank import PrototypeBankEntry


def masked_average_pool(feature_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对特征图执行掩码加权平均池化并 L2 归一化。

    根据掩码区域对特征图进行空间池化，输出归一化的特征向量。

    参数：
        - feature_map: 特征图张量，形状 (B, C, H, W)
        - mask: 掩码张量，形状 (B, 1, H, W) 或 (B, H, W)
        - eps: 防止除零的小常数（默认 1e-6）

    返回：
        - 归一化后的特征向量张量，形状 (B, C)
    """
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError("mask must have shape [B, 1, H, W]")
    resized_mask = F.interpolate(mask.float(), size=feature_map.shape[-2:], mode="nearest")
    weights = resized_mask.flatten(2)
    features = feature_map.flatten(2)
    denom = weights.sum(dim=-1).clamp_min(eps)
    pooled = (features * weights).sum(dim=-1) / denom
    return F.normalize(pooled, dim=1)


def _resolve_feature_map(feature: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """将不同格式的特征表示解析为 4D 特征图。

    支持 4D 张量直接返回、3D 序列特征重塑为空间特征图、或降维平铺。

    参数：
        - feature: 特征张量
        - images: 原始图像张量，用于推断空间尺寸

    返回：
        - 形状为 (B, C, H, W) 的特征图张量
    """
    if feature.dim() == 4:
        return feature
    if feature.dim() == 3:
        batch_size, tokens, channels = feature.shape
        side = int(tokens ** 0.5)
        if side * side == tokens:
            return feature.transpose(1, 2).reshape(batch_size, channels, side, side)
        reduced = feature.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        return reduced.repeat(1, 1, max(images.shape[-2] // 16, 1), max(images.shape[-1] // 16, 1))
    raise ValueError("Unsupported feature tensor shape")


class PrototypeExtractor:
    """基于 SAM3 模型的原型提取器。

    支持从特征图、模型输出或原始图像中提取原型特征向量。
    """
    def __init__(self, wrapper: Optional[torch.nn.Module] = None) -> None:
        """初始化原型提取器。

        参数：
            - wrapper: SAM3 模型包装器（可选）

        返回：
            - None
        """
        self.wrapper = wrapper

    def extract_from_feature_map(self, feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """直接从特征图和掩码提取原型。

        参数：
            - feature_map: 特征图张量
            - mask: 掩码张量

        返回：
            - 原型特征向量张量
        """
        return masked_average_pool(feature_map, mask)

    def extract_from_outputs(self, outputs: dict[str, object], images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """从 SAM3 模型输出字典中提取原型。

        参数：
            - outputs: 模型输出字典，需包含 "image_embeddings"
            - images: 原始图像张量
            - mask: 掩码张量

        返回：
            - 原型特征向量张量
        """
        feature = outputs.get("image_embeddings")
        if not isinstance(feature, torch.Tensor):
            raise ValueError("SAM3 outputs did not contain image_embeddings for prototype extraction")
        feature_map = _resolve_feature_map(feature, images)
        return self.extract_from_feature_map(feature_map, mask)

    def extract_from_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        text_prompt: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """从原始图像执行完整的前向推理并提取原型。

        需要初始化时提供 wrapper。

        参数：
            - images: 图像张量
            - masks: 掩码张量
            - boxes: 可选的边界框提示
            - text_prompt: 可选的文本提示

        返回：
            - (原型特征向量, 模型输出字典) 元组
        """
        if self.wrapper is None:
            raise RuntimeError("PrototypeExtractor requires a SAM3 wrapper to extract from raw images")
        outputs = self.wrapper(images=images, boxes=boxes, text_prompt=text_prompt)
        return self.extract_from_outputs(outputs, images, masks), outputs

    def save_prototype(
        self,
        root: str | Path,
        prototype: torch.Tensor,
        entry: PrototypeBankEntry,
    ) -> PrototypeBankEntry:
        """将原型特征和元数据保存到磁盘。

        按极性组织到 positive_bank 或 negative_bank 子目录。

        参数：
            - root: 保存根目录
            - prototype: 原型特征张量
            - entry: 原型条目对象（含元数据）

        返回：
            - 更新了特征路径的 PrototypeBankEntry 对象
        """
        destination = Path(root)
        bank_dir = destination / ("positive_bank" if entry.polarity == "positive" else "negative_bank")
        bank_dir.mkdir(parents=True, exist_ok=True)
        feature_path = bank_dir / f"{entry.prototype_id}.pt"
        metadata_path = bank_dir / f"{entry.prototype_id}.json"
        stored_entry = replace(entry, feature_path=str(feature_path))
        torch.save({"prototype": prototype.detach().cpu()}, feature_path)
        metadata_path.write_text(json.dumps(stored_entry.__dict__, indent=2), encoding="utf-8")
        return stored_entry
