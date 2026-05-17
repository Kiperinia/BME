"""Filesystem-backed positive/negative retrieval bank loader."""

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
    if features.shape[-1] == target_dim:
        return features
    if features.shape[-1] > target_dim:
        return features[..., :target_dim]
    return F.pad(features, (0, target_dim - features.shape[-1]))


@dataclass
class LoadedBankContext:
    bank: RSSDABank
    resolved_path: Path
    source: str
    cache_root: Optional[Path] = None
    stats: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _resolve_runtime_device(requested_device: str) -> str:
    return resolve_runtime_device(requested_device)


def _looks_like_directory_bank(path: Path) -> bool:
    return (path / "positive").is_dir() and (path / "negative").is_dir()


def _looks_like_metadata_bank(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".json"
    return (path / "metadata.json").exists() or (path / "positive_bank").is_dir() or (path / "negative_bank").is_dir()


def resolve_protocol_bank_path(bank_path: str | Path, purpose: str = "train") -> Path:
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
        return dict(self._stats)

    @property
    def last_warnings(self) -> list[str]:
        return list(self._warnings)

    def build_context(self) -> LoadedBankContext:
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
        features = [RSSDABank.load_feature(entry, device=device) for entry in entries]
        target_dim = max((int(feature.shape[-1]) for feature in features), default=0)
        aligned = [_match_feature_dim(feature, target_dim) for feature in features]
        return F.normalize(torch.stack(aligned, dim=0), dim=-1)

    def _scan_images(self, polarity: str) -> list[Path]:
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
        relative_path = self._relative_bank_path(image_path)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", relative_path.rsplit(".", 1)[0]).strip("_").lower()
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
        return f"{polarity}_{slug}_{digest}"

    def _cache_path(self, image_path: Path, polarity: str) -> Path:
        prototype_id = self._prototype_id(image_path, polarity)
        target = self.cache_root / polarity
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{prototype_id}.pt"

    def _relative_bank_path(self, image_path: Path) -> str:
        try:
            return image_path.relative_to(self.bank_root).as_posix()
        except ValueError:
            return image_path.name

    def _resolve_mask_path(self, image_path: Path, polarity: str) -> Path | None:
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
        if message in self._warnings:
            return
        self._warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _is_cache_valid(self, image_path: Path, cache_path: Path) -> bool:
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