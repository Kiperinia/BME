"""基于文件系统的正/负样本检索库加载器。"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MedicalSAM3.exemplar_bank import PrototypeBankEntry, RSSDABank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import full_image_box, infer_source_domain, resolve_feature_map, resolve_runtime_device

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STRICT_PROTOCOL_PURPOSES = {"train", "validation", "external-eval"}


def _match_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    """调整特征张量的最后一维以匹配目标维度（截断或填充）。

    参数：
        - features: 输入特征张量。
        - target_dim: 目标维度。

    返回：
        - 调整后的特征张量。
    """
    if features.shape[-1] == target_dim:
        return features
    if features.shape[-1] > target_dim:
        return features[..., :target_dim]
    return F.pad(features, (0, target_dim - features.shape[-1]))


@dataclass
class LoadedBankContext:
    """已加载银行库的上下文数据类，包含银行对象、解析路径、来源和统计信息。

    参数：
        - 无。

    返回：
        - 用于下游工作流的已加载银行上下文实例。
    """
    bank: RSSDABank
    resolved_path: Path
    source: str
    cache_root: Optional[Path] = None
    stats: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _resolve_runtime_device(requested_device: str) -> str:
    """解析运行时的计算设备（CPU/CUDA）。

    参数：
        - requested_device: 请求的设备名称。

    返回：
        - 解析后的设备名称。
    """
    return resolve_runtime_device(requested_device)


def _looks_like_directory_bank(path: Path) -> bool:
    """检查路径是否看起来是目录结构的银行库（包含 positive/negative 子目录）。

    参数：
        - path: 待检查的路径。

    返回：
        - 是否为目录格式的银行库。
    """
    return (path / "positive").is_dir() and (path / "negative").is_dir()


def _looks_like_metadata_bank(path: Path) -> bool:
    """检查路径是否看起来是元数据格式的银行库（JSON 文件或包含 metadata.json）。

    参数：
        - path: 待检查的路径。

    返回：
        - 是否为元数据格式的银行库。
    """
    if path.is_file():
        return path.suffix == ".json"
    return (path / "metadata.json").exists() or (path / "positive_bank").is_dir() or (path / "negative_bank").is_dir()


def resolve_protocol_bank_path(bank_path: str | Path, purpose: str = "train") -> Path:
    """根据协议目的解析银行路径，校验 strict 协议下的路径合法性。

    参数：
        - bank_path: 银行路径。
        - purpose: 使用目的，如 "train"、"validation"、"external-eval"。

    返回：
        - 解析后的银行路径。
    """
    resolved = Path(bank_path)
    normalized_purpose = purpose.strip().lower()
    if (resolved / "train_bank").is_dir() and (resolved / "continual_bank").is_dir() and normalized_purpose in STRICT_PROTOCOL_PURPOSES:
        resolved = resolved / "train_bank"
    if normalized_purpose in STRICT_PROTOCOL_PURPOSES and resolved.name == "continual_bank":
        raise ValueError("Strict retrieval protocol only allows train_bank for train/validation/external evaluation.")
    if normalized_purpose in STRICT_PROTOCOL_PURPOSES and "continual_bank" in resolved.parts:
        continual_index = resolved.parts.index("continual_bank")
        if continual_index == len(resolved.parts) - 1:
            raise ValueError("continual_bank root is not allowed in strict external retrieval protocol.")
    return resolved


def load_retrieval_bank(
    bank_path: str | Path,
    *,
    purpose: str = "train",
    checkpoint: Optional[str] = None,
    device: str = "auto",
    precision: str = "fp16",
    image_size: int = 128,
    cache_dir: Optional[str | Path] = None,
    allow_dummy_fallback: bool = False,
) -> LoadedBankContext:
    """加载检索银行库，支持元数据格式和目录格式，自动解析路径并返回上下文。

    参数：
        - bank_path: 银行路径。
        - purpose: 使用目的。
        - checkpoint: 模型检查点路径。
        - device: 计算设备。
        - precision: 精度（如 "fp16"、"fp32"）。
        - image_size: 图像尺寸。
        - cache_dir: 缓存目录。
        - allow_dummy_fallback: 是否允许使用虚拟数据回退。

    返回：
        - LoadedBankContext 实例，包含银行对象和加载信息。
    """
    resolved_path = resolve_protocol_bank_path(bank_path, purpose=purpose)
    if _looks_like_metadata_bank(resolved_path):
        bank = RSSDABank.load(resolved_path)
        return LoadedBankContext(bank=bank, resolved_path=resolved_path, source="rssda_bank")
    if _looks_like_directory_bank(resolved_path):
        loader = DirectoryBankLoader(
            bank_root=resolved_path,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            cache_dir=cache_dir,
            allow_dummy_fallback=allow_dummy_fallback,
        )
        return loader.build_context()
    bank = RSSDABank.load(resolved_path)
    if bank.entries:
        return LoadedBankContext(bank=bank, resolved_path=resolved_path, source="rssda_bank")
    raise FileNotFoundError(f"Retrieval bank not found or empty: {resolved_path}")


class DirectoryBankLoader:
    """目录结构银行库加载器，从 positive/negative 图像目录加载并缓存特征。

    参数：
        - bank_root: 银行库根目录。
        - checkpoint: 模型检查点路径。
        - device: 计算设备。
        - precision: 精度。
        - image_size: 图像尺寸。
        - cache_dir: 缓存目录。
        - allow_dummy_fallback: 是否允许虚拟数据回退。
        - default_top_k: 默认 top-k 数量。

    返回：
        - 用于构建和检索银行库的加载器实例。
    """
    def __init__(
        self,
        bank_root: str | Path,
        *,
        checkpoint: Optional[str] = None,
        device: str = "auto",
        precision: str = "fp16",
        image_size: int = 128,
        cache_dir: Optional[str | Path] = None,
        allow_dummy_fallback: bool = False,
        default_top_k: int = 1,
    ) -> None:
        """初始化 DirectoryBankLoader，设置银行根目录和模型参数。

        参数：
            - bank_root: 银行库根目录。
            - checkpoint: 模型检查点路径。
            - device: 计算设备。
            - precision: 精度。
            - image_size: 图像尺寸。
            - cache_dir: 缓存目录。
            - allow_dummy_fallback: 是否允许虚拟数据回退。
            - default_top_k: 默认 top-k 数量。
        """
        self.bank_root = Path(bank_root)
        self.checkpoint = checkpoint
        self.device = _resolve_runtime_device(device)
        self.precision = precision
        self.image_size = image_size
        self.allow_dummy_fallback = allow_dummy_fallback
        self.default_top_k = default_top_k
        self.cache_root = Path(cache_dir) if cache_dir is not None else self.bank_root / ".cache"
        self._wrapper: Optional[Sam3TensorForwardWrapper] = None
        self._bank: Optional[RSSDABank] = None
        self._feature_dim: Optional[int] = None
        self._stats: dict[str, int] = {}
        self._warnings: list[str] = []

    @property
    def last_stats(self) -> dict[str, int]:
        """返回最近一次构建的统计信息。

        参数：
            - 无。

        返回：
            - 包含正/负样本数、缓存命中/未命中次数的字典。
        """
        return dict(self._stats)

    @property
    def last_warnings(self) -> list[str]:
        """返回最近一次构建的警告信息列表。

        参数：
            - 无。

        返回：
            - 警告字符串列表。
        """
        return list(self._warnings)

    def build_context(self) -> LoadedBankContext:
        """构建完整的加载银行上下文，包含银行对象和统计信息。

        参数：
            - 无。

        返回：
            - LoadedBankContext 实例。
        """
        bank = self.build_bank()
        return LoadedBankContext(
            bank=bank,
            resolved_path=self.bank_root,
            source="directory_bank",
            cache_root=self.cache_root,
            stats=self.last_stats,
            warnings=self.last_warnings,
        )

    def build_bank(self) -> RSSDABank:
        """构建 RSSDA 银行库，扫描图像目录并提取特征，利用缓存加速。

        参数：
            - 无。

        返回：
            - 构建完成的 RSSDABank 实例。
        """
        positive_paths = self._scan_images("positive")
        negative_paths = self._scan_images("negative")
        cache_hits = 0
        cache_misses = 0
        bank = RSSDABank(version="directory_bank_v1")
        work_items: list[tuple[str, Path, Path, PrototypeBankEntry]] = []

        for polarity, image_paths in (("positive", positive_paths), ("negative", negative_paths)):
            for image_path in image_paths:
                cache_path = self._cache_path(image_path, polarity)
                entry = self._build_entry(image_path, cache_path, polarity)
                if self._is_cache_valid(image_path, cache_path):
                    cache_hits += 1
                else:
                    cache_misses += 1
                    work_items.append((polarity, image_path, cache_path, entry))
                bank.add_entry(entry)

        if work_items:
            self._ensure_wrapper()
            for polarity, image_path, cache_path, _ in work_items:
                prototype = self._encode_image(image_path)
                self._write_cache(image_path, cache_path, prototype, polarity)

        self._bank = bank
        self._stats = {
            "positive_count": len(positive_paths),
            "negative_count": len(negative_paths),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "warning_count": len(self._warnings),
        }
        if self._feature_dim is None and bank.entries:
            self._feature_dim = int(RSSDABank.load_feature(bank.entries[0]).shape[-1])
        return bank

    def retrieve(
        self,
        query_feature: torch.Tensor,
        *,
        top_k: Optional[int] = None,
        top_k_positive: Optional[int] = None,
        top_k_negative: Optional[int] = None,
        query_source_datasets: Optional[list[str]] = None,
        prefer_cross_domain_positive: bool = False,
    ) -> dict[str, Any]:
        """根据查询特征从银行库中检索 top-k 正/负样本原型。

        参数：
            - query_feature: 查询特征图。
            - top_k: 全局 top-k 数量。
            - top_k_positive: 正样本 top-k 数量。
            - top_k_negative: 负样本 top-k 数量。
            - query_source_datasets: 查询来源数据集列表，用于跨域偏好。
            - prefer_cross_domain_positive: 是否优先选择跨域正样本。

        返回：
            - 包含检索结果（特征、权重、原型、条目、分数等）的字典。
        """
        bank = self._bank or self.build_bank()
        query_global = F.normalize(F.adaptive_avg_pool2d(query_feature, 1).flatten(1), dim=1)
        positive_entries = bank.get_entries(polarity="positive", human_verified=True)
        negative_entries = bank.get_entries(polarity="negative", human_verified=True)
        k_positive = max(1, top_k_positive or top_k or self.default_top_k)
        k_negative = max(1, top_k_negative or top_k or self.default_top_k)

        positive_prototypes = []
        negative_prototypes = []
        positive_selected = []
        negative_selected = []
        positive_scores = []
        negative_scores = []
        positive_feature_list = []
        negative_feature_list = []
        positive_weight_list = []
        negative_weight_list = []
        score_positive_mean = []
        score_negative_mean = []
        score_margin = []

        for index in range(query_global.shape[0]):
            query_source = None
            if query_source_datasets is not None and index < len(query_source_datasets):
                query_source = str(query_source_datasets[index])
            current_positive_entries = positive_entries
            if prefer_cross_domain_positive and query_source:
                cross_domain = [entry for entry in positive_entries if str(entry.source_dataset) != query_source]
                if cross_domain:
                    current_positive_entries = cross_domain
            pos_feature, pos_entries, pos_values, pos_selected_features, pos_weights = self._retrieve_single(query_global[index], current_positive_entries, k_positive)
            neg_feature, neg_entries, neg_values, neg_selected_features, neg_weights = self._retrieve_single(query_global[index], negative_entries, k_negative)
            positive_prototypes.append(pos_feature)
            negative_prototypes.append(neg_feature)
            positive_selected.append(pos_entries)
            negative_selected.append(neg_entries)
            positive_scores.append(pos_values)
            negative_scores.append(neg_values)
            positive_feature_list.append(pos_selected_features)
            negative_feature_list.append(neg_selected_features)
            positive_weight_list.append(pos_weights)
            negative_weight_list.append(neg_weights)
            pos_mean = pos_values.mean() if pos_values.numel() > 0 else query_global.new_tensor(0.0)
            neg_mean = neg_values.mean() if neg_values.numel() > 0 else query_global.new_tensor(0.0)
            score_positive_mean.append(pos_mean)
            score_negative_mean.append(neg_mean)
            score_margin.append(pos_mean - neg_mean)

        dim = int(query_global.shape[-1])
        max_pos = max((item.shape[0] for item in positive_feature_list), default=0)
        max_neg = max((item.shape[0] for item in negative_feature_list), default=0)
        positive_features = torch.zeros(query_global.shape[0], max_pos, dim, device=query_global.device)
        negative_features = torch.zeros(query_global.shape[0], max_neg, dim, device=query_global.device)
        positive_weights = torch.zeros(query_global.shape[0], max_pos, device=query_global.device)
        negative_weights = torch.zeros(query_global.shape[0], max_neg, device=query_global.device)
        positive_score_tensor = torch.zeros(query_global.shape[0], max_pos, device=query_global.device)
        negative_score_tensor = torch.zeros(query_global.shape[0], max_neg, device=query_global.device)

        for batch_index, item in enumerate(positive_feature_list):
            if item.numel() == 0:
                continue
            positive_features[batch_index, : item.shape[0]] = item
            positive_weights[batch_index, : item.shape[0]] = positive_weight_list[batch_index]
            positive_score_tensor[batch_index, : item.shape[0]] = positive_scores[batch_index]
        for batch_index, item in enumerate(negative_feature_list):
            if item.numel() == 0:
                continue
            negative_features[batch_index, : item.shape[0]] = item
            negative_weights[batch_index, : item.shape[0]] = negative_weight_list[batch_index]
            negative_score_tensor[batch_index, : item.shape[0]] = negative_scores[batch_index]

        positive_similarity_std = torch.stack([
            values.std(unbiased=False) if values.numel() > 0 else query_global.new_tensor(0.0) for values in positive_scores
        ], dim=0)
        negative_similarity_std = torch.stack([
            values.std(unbiased=False) if values.numel() > 0 else query_global.new_tensor(0.0) for values in negative_scores
        ], dim=0)
        positive_weight_entropy = torch.stack([
            -(weights * (weights.clamp_min(1e-6).log())).sum() if weights.numel() > 0 else query_global.new_tensor(0.0) for weights in positive_weight_list
        ], dim=0)
        negative_weight_entropy = torch.stack([
            -(weights * (weights.clamp_min(1e-6).log())).sum() if weights.numel() > 0 else query_global.new_tensor(0.0) for weights in negative_weight_list
        ], dim=0)

        return {
            "query_global": query_global,
            "projected_query": query_global,
            "query_source_datasets": query_source_datasets or [],
            "positive_features": positive_features,
            "negative_features": negative_features,
            "positive_weights": positive_weights,
            "negative_weights": negative_weights,
            "positive_score_tensor": positive_score_tensor,
            "negative_score_tensor": negative_score_tensor,
            "positive_prototype": torch.stack(positive_prototypes, dim=0),
            "negative_prototype": torch.stack(negative_prototypes, dim=0),
            "positive_prototype_feature": torch.stack(positive_prototypes, dim=0),
            "negative_prototype_feature": torch.stack(negative_prototypes, dim=0),
            "similarity_score": {
                "positive_topk_mean": torch.stack(score_positive_mean, dim=0),
                "negative_topk_mean": torch.stack(score_negative_mean, dim=0),
                "margin": torch.stack(score_margin, dim=0),
            },
            "retrieval_stability": {
                "positive_similarity_mean": torch.stack(score_positive_mean, dim=0),
                "negative_similarity_mean": torch.stack(score_negative_mean, dim=0),
                "margin": torch.stack(score_margin, dim=0),
                "positive_similarity_std": positive_similarity_std,
                "negative_similarity_std": negative_similarity_std,
                "positive_weight_entropy": positive_weight_entropy,
                "negative_weight_entropy": negative_weight_entropy,
            },
            "top_k_positive": k_positive,
            "top_k_negative": k_negative,
            "positive_entries": positive_selected,
            "negative_entries": negative_selected,
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
        }

    def _retrieve_single(
        self,
        query_global: torch.Tensor,
        entries: list[PrototypeBankEntry],
        top_k: int,
    ) -> tuple[torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor, torch.Tensor]:
        """单条查询的检索逻辑，计算相似度并返回 top-k 原型、条目和权重。

        参数：
            - query_global: 单个查询的全局特征向量。
            - entries: 待检索的条目列表。
            - top_k: 返回的 top-k 数量。

        返回：
            - (prototype, selected_entries, values, selected_features, weights) 五元组。
        """
        dim = int(query_global.shape[-1])
        if not entries:
            return (
                torch.zeros(dim, device=query_global.device),
                [],
                torch.zeros(0, device=query_global.device),
                torch.zeros(0, dim, device=query_global.device),
                torch.zeros(0, device=query_global.device),
            )
        bank_features = self._bank_features(entries, device=query_global.device)
        bank_features = _match_feature_dim(bank_features, int(query_global.shape[-1]))
        similarities = torch.matmul(bank_features, query_global)
        values, indices = torch.topk(similarities, k=min(top_k, similarities.shape[0]))
        selected_features = bank_features[indices]
        weights = torch.softmax(values, dim=0)
        prototype = F.normalize((weights.unsqueeze(-1) * selected_features).sum(dim=0), dim=0)
        selected_entries = [entries[int(index)] for index in indices.detach().cpu().tolist()]
        return prototype, selected_entries, values, selected_features, weights

    def _bank_features(self, entries: list[PrototypeBankEntry], device: str | torch.device) -> torch.Tensor:
        """加载条目列表的特征，对齐维度后堆叠并归一化。

        参数：
            - entries: 原型银行条目列表。
            - device: 目标计算设备。

        返回：
            - 归一化后的特征堆叠张量。
        """
        features = [RSSDABank.load_feature(entry, device=device) for entry in entries]
        target_dim = max((int(feature.shape[-1]) for feature in features), default=0)
        aligned = [_match_feature_dim(feature, target_dim) for feature in features]
        return F.normalize(torch.stack(aligned, dim=0), dim=-1)

    def _scan_images(self, polarity: str) -> list[Path]:
        """扫描指定极性的图像目录，支持 legacy 和 structured 两种布局。

        参数：
            - polarity: 极性（"positive" 或 "negative"）。

        返回：
            - 图像文件路径列表。
        """
        root = self.bank_root / polarity
        if not root.exists():
            return []
        structured_root = root / "images"
        legacy_images = sorted(
            path for path in root.glob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        structured_images = (
            sorted(
                path for path in structured_root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            if structured_root.exists()
            else []
        )

        if structured_images:
            if legacy_images:
                self._warn(
                    f"Detected both legacy and structured bank layout under {root}; preferring {structured_root}."
                )
            self._validate_structured_pairing(polarity=polarity, image_paths=structured_images)
            return structured_images

        if structured_root.exists() and legacy_images:
            self._warn(
                f"Structured image directory {structured_root} is empty; falling back to legacy bank files under {root}."
            )
        if legacy_images:
            self._warn(
                f"Using legacy bank layout under {root}. Upgrade to {structured_root} + {root / 'masks'} for explicit image/mask pairing."
            )
        return legacy_images

    def _build_entry(self, image_path: Path, cache_path: Path, polarity: str) -> PrototypeBankEntry:
        """根据图像路径构建 PrototypeBankEntry 对象。

        参数：
            - image_path: 图像文件路径。
            - cache_path: 特征缓存路径。
            - polarity: 极性（"positive" 或 "negative"）。

        返回：
            - 构建的 PrototypeBankEntry 实例。
        """
        dataset_name = infer_source_domain(
            dataset_name=image_path.parent.name,
            image_id=image_path.stem,
            image_path=str(image_path),
        )
        prototype_id = self._prototype_id(image_path, polarity)
        mask_path = self._resolve_mask_path(image_path, polarity)
        return PrototypeBankEntry(
            prototype_id=prototype_id,
            feature_path=str(cache_path),
            polarity=polarity,
            source_dataset=dataset_name,
            polyp_type=polarity,
            boundary_quality=1.0,
            confidence=1.0,
            image_id=image_path.stem,
            crop_path=str(image_path),
            mask_path=None if mask_path is None else str(mask_path),
            device_metadata={
                "device": self.device,
                "precision": self.precision,
                "image_size": self.image_size,
                "bank_root": str(self.bank_root),
            },
            human_verified=True,
            extra_metadata={
                "bank_split": self.bank_root.name,
                "relative_path": self._relative_bank_path(image_path),
            },
        )

    def _prototype_id(self, image_path: Path, polarity: str) -> str:
        """根据图像路径和极性生成唯一原型 ID。

        参数：
            - image_path: 图像文件路径。
            - polarity: 极性。

        返回：
            - 原型 ID 字符串。
        """
        relative_path = self._relative_bank_path(image_path)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", relative_path.rsplit(".", 1)[0]).strip("_").lower()
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
        return f"{polarity}_{slug}_{digest}"

    def _cache_path(self, image_path: Path, polarity: str) -> Path:
        """根据图像路径和极性生成缓存文件路径。

        参数：
            - image_path: 图像文件路径。
            - polarity: 极性。

        返回：
            - 缓存文件路径。
        """
        prototype_id = self._prototype_id(image_path, polarity)
        target = self.cache_root / polarity
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{prototype_id}.pt"

    def _relative_bank_path(self, image_path: Path) -> str:
        """计算图像路径相对于银行根目录的 POSIX 风格路径。

        参数：
            - image_path: 图像文件路径。

        返回：
            - 相对路径字符串（POSIX 格式）。
        """
        try:
            return image_path.relative_to(self.bank_root).as_posix()
        except ValueError:
            return image_path.name

    def _resolve_mask_path(self, image_path: Path, polarity: str) -> Path | None:
        """解析与图像对应的掩码文件路径。

        参数：
            - image_path: 图像文件路径。
            - polarity: 极性。

        返回：
            - 掩码文件路径，若不存在则返回 None。
        """
        mask_root = self.bank_root / polarity / "masks"
        image_root = self.bank_root / polarity / "images"
        if image_root in image_path.parents and mask_root.exists():
            relative_path = image_path.relative_to(image_root)
            for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
                candidate = (mask_root / relative_path).with_suffix(suffix)
                if candidate.exists():
                    return candidate
        if mask_root.exists():
            for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
                candidate = mask_root / f"{image_path.stem}{suffix}"
                if candidate.exists():
                    return candidate
        return None

    def _validate_structured_pairing(self, *, polarity: str, image_paths: list[Path]) -> None:
        """验证结构化银行布局中图像与掩码的配对完整性。

        参数：
            - polarity: 极性。
            - image_paths: 图像路径列表。
        """
        mask_root = self.bank_root / polarity / "masks"
        if not mask_root.exists():
            self._warn(
                f"Structured bank layout detected for {self.bank_root / polarity}, but {mask_root} is missing. Masks will be unavailable."
            )
            return
        missing_masks = [path for path in image_paths if self._resolve_mask_path(path, polarity) is None]
        if missing_masks:
            example_names = ", ".join(path.name for path in missing_masks[:3])
            self._warn(
                f"Structured bank layout under {self.bank_root / polarity} has {len(missing_masks)} images without matching masks. Example: {example_names}"
            )

    def _warn(self, message: str) -> None:
        """记录并发出警告消息，避免重复。

        参数：
            - message: 警告消息内容。
        """
        if message in self._warnings:
            return
        self._warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _is_cache_valid(self, image_path: Path, cache_path: Path) -> bool:
        """检查缓存文件是否仍然有效（源文件未修改且参数匹配）。

        参数：
            - image_path: 源图像文件路径。
            - cache_path: 缓存文件路径。

        返回：
            - 缓存是否有效。
        """
        if not cache_path.exists():
            return False
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        source_stat = image_path.stat()
        prototype = payload.get("prototype")
        if not isinstance(prototype, torch.Tensor):
            return False
        return (
            payload.get("source_path") == str(image_path.resolve())
            and int(payload.get("source_size", -1)) == int(source_stat.st_size)
            and int(payload.get("source_mtime_ns", -1)) == int(source_stat.st_mtime_ns)
            and int(payload.get("image_size", -1)) == int(self.image_size)
        )

    def _write_cache(self, image_path: Path, cache_path: Path, prototype: torch.Tensor, polarity: str) -> None:
        """将提取的原型特征写入缓存文件，附带源文件元信息以验证有效性。

        参数：
            - image_path: 源图像文件路径。
            - cache_path: 缓存文件路径。
            - prototype: 原型特征张量。
            - polarity: 极性。
        """
        source_stat = image_path.stat()
        torch.save(
            {
                "prototype": prototype.detach().cpu(),
                "source_path": str(image_path.resolve()),
                "source_size": int(source_stat.st_size),
                "source_mtime_ns": int(source_stat.st_mtime_ns),
                "image_size": int(self.image_size),
                "polarity": polarity,
            },
            cache_path,
        )

    def _ensure_wrapper(self) -> Sam3TensorForwardWrapper:
        """确保模型包装器已初始化（懒加载），加载 SAM3 模型并冻结。

        参数：
            - 无。

        返回：
            - Sam3TensorForwardWrapper 实例。
        """
        if self._wrapper is None:
            model = build_official_sam3_image_model(
                self.checkpoint,
                device=self.device,
                dtype=self.precision,
                compile_model=False,
                allow_dummy_fallback=self.allow_dummy_fallback,
            )
            freeze_model(model)
            hidden_dim = getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", None)))
            if hidden_dim is not None:
                self._feature_dim = int(hidden_dim)
            self._wrapper = Sam3TensorForwardWrapper(model=model, device=self.device, dtype=self.precision)
        return self._wrapper

    def _encode_image(self, image_path: Path) -> torch.Tensor:
        """对单张图像进行编码，提取全局原型特征。

        参数：
            - image_path: 图像文件路径。

        返回：
            - 提取的归一化原型特征张量。
        """
        wrapper = self._ensure_wrapper()
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image_array = np.asarray(image).astype("float32") / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        boxes = full_image_box(self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = wrapper(images=image_tensor, boxes=boxes, text_prompt=["polyp"])
            feature_map = resolve_feature_map(outputs.get("image_embeddings"), image_tensor)
            prototype = F.normalize(F.adaptive_avg_pool2d(feature_map, 1).flatten(1), dim=1)[0]
        self._feature_dim = int(prototype.shape[-1])
        return prototype.detach().cpu()
