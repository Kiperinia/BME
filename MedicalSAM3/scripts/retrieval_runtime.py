"""Shared runtime helpers for retrieval-conditioned MedEx-SAM3 scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar_bank import RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import (
    DirectoryBankLoader,
    LoadedBankContext,
    SiteBankResolution,
    annotate_single_bank_retrieval,
    fuse_multi_bank_retrieval,
    load_retrieval_bank,
    resolve_site_bank_paths,
)
from MedicalSAM3.retrieval.mask_prior import attach_retrieved_mask_priors
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import resolve_feature_map

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class RetrievalBankBackend:
    name: str
    bank_context: LoadedBankContext
    retriever: PrototypeRetriever
    retrieval_backend: str
    directory_loader: Optional[DirectoryBankLoader] = None


@dataclass
class RetrievalRuntime:
    device: str
    hidden_dim: int
    bank_context: LoadedBankContext
    wrapper: Sam3TensorForwardWrapper
    retriever: PrototypeRetriever
    similarity_builder: SimilarityHeatmapBuilder
    adapter: RetrievalSpatialSemanticAdapter
    retrieval_backend: str
    directory_loader: Optional[DirectoryBankLoader] = None
    primary_backend: Optional[RetrievalBankBackend] = None
    site_bank_mode: str = "train_plus_site"
    continual_bank_root: Optional[Path] = None
    bank_loader_config: dict[str, Any] = field(default_factory=dict)
    bank_backend_cache: dict[str, RetrievalBankBackend] = field(default_factory=dict)


def resolve_hidden_dim(model: torch.nn.Module) -> int:
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def apply_retrieval_mode(retrieval: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode in {"joint", "semantic", "spatial", "positive-negative"}:
        return retrieval
    if mode not in {"positive-only", "negative-only"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    updated = dict(retrieval)
    if mode == "positive-only":
        updated["negative_features"] = torch.zeros_like(retrieval["negative_features"])
        updated["negative_weights"] = torch.zeros_like(retrieval["negative_weights"])
        updated["negative_score_tensor"] = torch.zeros_like(retrieval.get("negative_score_tensor", retrieval["negative_weights"]))
        updated["negative_prototype"] = torch.zeros_like(retrieval["positive_prototype"])
        updated["negative_entries"] = [[] for _ in retrieval["positive_entries"]]
        updated["negative_scores"] = [torch.zeros_like(score) for score in retrieval["positive_scores"]]
        if "negative_mask_prior" in retrieval:
            updated["negative_mask_prior"] = torch.zeros_like(retrieval["negative_mask_prior"])
        return updated
    updated["positive_features"] = torch.zeros_like(retrieval["positive_features"])
    updated["positive_weights"] = torch.zeros_like(retrieval["positive_weights"])
    updated["positive_score_tensor"] = torch.zeros_like(retrieval.get("positive_score_tensor", retrieval["positive_weights"]))
    updated["positive_prototype"] = torch.zeros_like(retrieval["negative_prototype"])
    updated["positive_entries"] = [[] for _ in retrieval["negative_entries"]]
    updated["positive_scores"] = [torch.zeros_like(score) for score in retrieval["negative_scores"]]
    if "positive_mask_prior" in retrieval:
        updated["positive_mask_prior"] = torch.zeros_like(retrieval["positive_mask_prior"])
    return updated


def parse_bbox(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("bbox must be formatted as x1,y1,x2,y2")
    return [float(part) for part in parts]


def load_bbox_mapping(path: str | Path) -> dict[str, list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): [float(item) for item in value] for key, value in payload.items()}
    if isinstance(payload, list):
        mapping: dict[str, list[float]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = str(item.get("image") or item.get("image_id") or item.get("path") or "")
            bbox = item.get("bbox")
            if key and isinstance(bbox, list) and len(bbox) == 4:
                mapping[key] = [float(entry) for entry in bbox]
        return mapping
    raise ValueError(f"Unsupported bbox mapping payload: {path}")


def collect_input_images(input_path: str | Path) -> list[Path]:
    target = Path(input_path)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Input path not found: {target}")
    return sorted(path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def load_image_tensor(image_path: str | Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((image_size, image_size))
    array = np.asarray(resized).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def scale_bbox(bbox: list[float], original_size: tuple[int, int], image_size: int) -> torch.Tensor:
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("original image size must be positive")
    scale_x = image_size / float(width)
    scale_y = image_size / float(height)
    x1, y1, x2, y2 = bbox
    return torch.tensor([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=torch.float32)


def load_rssda_bundle_components(
    path: str | Path,
    *,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> dict[str, bool]:
    payload = torch.load(path, map_location=device, weights_only=False)
    loaded = {"adapter": False, "retriever": False, "similarity_builder": False}
    if not isinstance(payload, dict):
        return loaded
    if isinstance(payload.get("adapter"), dict):
        adapter.load_state_dict(payload["adapter"], strict=False)
        loaded["adapter"] = True
    if isinstance(payload.get("retriever"), dict):
        retriever.load_state_dict(payload["retriever"], strict=False)
        loaded["retriever"] = True
    if isinstance(payload.get("similarity_builder"), dict):
        similarity_builder.load_state_dict(payload["similarity_builder"], strict=False)
        loaded["similarity_builder"] = True
    return loaded


def _build_bank_backend(
    *,
    name: str,
    bank_context: LoadedBankContext,
    feature_dim: int,
    top_k_positive: int,
    top_k_negative: int,
    device: str,
    checkpoint: Optional[str],
    precision: str,
    image_size: int,
    allow_dummy_fallback: bool,
    reference_retriever: PrototypeRetriever,
    fallback_backend: str,
) -> RetrievalBankBackend:
    retriever = PrototypeRetriever(
        bank=bank_context.bank,
        feature_dim=feature_dim,
        top_k_positive=top_k_positive,
        top_k_negative=top_k_negative,
    ).to(device)
    retriever.load_state_dict(reference_retriever.state_dict(), strict=False)
    retriever.eval()
    directory_loader = None
    retrieval_backend = fallback_backend
    if bank_context.source == "directory_bank" and fallback_backend == "directory_loader":
        directory_loader = DirectoryBankLoader(
            bank_context.resolved_path,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=allow_dummy_fallback,
            default_top_k=max(top_k_positive, top_k_negative),
        )
        retrieval_backend = "directory_loader"
    return RetrievalBankBackend(
        name=name,
        bank_context=bank_context,
        retriever=retriever,
        retrieval_backend=retrieval_backend,
        directory_loader=directory_loader,
    )


def _backend_cache_key(path: Path) -> str:
    return str(path.resolve()) if path.exists() else str(path)


def _resolve_site_bank_root(memory_bank: str | Path, explicit_root: str | Path | None) -> Path | None:
    if explicit_root is not None:
        return Path(explicit_root)
    memory_bank_path = Path(memory_bank)
    parent = memory_bank_path.parent if memory_bank_path.name == "train_bank" else memory_bank_path
    candidate = parent / "continual_bank"
    if candidate.exists():
        return candidate
    return None


def _default_site_resolution(runtime: RetrievalRuntime) -> SiteBankResolution:
    return SiteBankResolution(
        mode="train_only",
        site_id=None,
        train_bank_path=runtime.bank_context.resolved_path,
        continual_bank_root=runtime.continual_bank_root or runtime.bank_context.resolved_path.parent,
        site_bank_path=None,
        expected_site_bank=None,
        selected_bank_paths=[runtime.bank_context.resolved_path],
        fallback_to_train_bank=False,
        fallback_reason=None,
        warnings=[],
    )


def _resolve_bank_selection(runtime: RetrievalRuntime, sample_metadata: dict[str, Any] | None) -> SiteBankResolution:
    if runtime.continual_bank_root is None:
        return _default_site_resolution(runtime)
    return resolve_site_bank_paths(
        sample_metadata=sample_metadata,
        train_bank=runtime.bank_context.resolved_path,
        continual_bank_root=runtime.continual_bank_root,
        mode=runtime.site_bank_mode,
    )


def _get_backend(runtime: RetrievalRuntime, bank_path: Path) -> RetrievalBankBackend:
    cache_key = _backend_cache_key(bank_path)
    if runtime.primary_backend is not None and cache_key == _backend_cache_key(runtime.bank_context.resolved_path):
        return runtime.primary_backend
    cached = runtime.bank_backend_cache.get(cache_key)
    if cached is not None:
        return cached

    config = runtime.bank_loader_config
    bank_context = load_retrieval_bank(
        bank_path,
        purpose=str(config.get("bank_purpose", "external-eval")),
        checkpoint=config.get("checkpoint"),
        device=runtime.device,
        precision=str(config.get("precision", "fp32")),
        image_size=int(config.get("image_size", 128)),
        allow_dummy_fallback=bool(config.get("allow_dummy_fallback", False)),
    )
    backend = _build_bank_backend(
        name=bank_path.name,
        bank_context=bank_context,
        feature_dim=runtime.hidden_dim,
        top_k_positive=int(config.get("top_k_positive", 1)),
        top_k_negative=int(config.get("top_k_negative", 1)),
        device=runtime.device,
        checkpoint=config.get("checkpoint"),
        precision=str(config.get("precision", "fp32")),
        image_size=int(config.get("image_size", 128)),
        allow_dummy_fallback=bool(config.get("allow_dummy_fallback", False)),
        reference_retriever=runtime.retriever,
        fallback_backend=runtime.retrieval_backend,
    )
    runtime.bank_backend_cache[cache_key] = backend
    return backend


def _run_backend_retrieval(
    backend: RetrievalBankBackend,
    query_feature: torch.Tensor,
    *,
    top_k: Optional[int],
    top_k_positive: Optional[int],
    top_k_negative: Optional[int],
    query_source: Optional[str],
    prefer_cross_domain_positive: bool,
) -> dict[str, Any]:
    if backend.retrieval_backend == "directory_loader" and backend.directory_loader is not None:
        return backend.directory_loader.retrieve(
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            query_source_datasets=[query_source] if query_source else None,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
    return backend.retriever(
        query_feature,
        top_k_positive=top_k_positive or top_k,
        top_k_negative=top_k_negative or top_k,
        query_source_datasets=[query_source] if query_source else None,
        prefer_cross_domain_positive=prefer_cross_domain_positive,
    )


def resolve_effective_bank(runtime: RetrievalRuntime, *, sample_metadata: dict[str, Any] | None = None) -> RSSDABank:
    resolution = _resolve_bank_selection(runtime, sample_metadata)
    if len(resolution.selected_bank_paths) <= 1:
        selected_path = resolution.selected_bank_paths[0] if resolution.selected_bank_paths else runtime.bank_context.resolved_path
        return _get_backend(runtime, selected_path).bank_context.bank
    entries = []
    seen: set[tuple[str, str]] = set()
    for bank_path in resolution.selected_bank_paths:
        backend = _get_backend(runtime, bank_path)
        for entry in backend.bank_context.bank.entries:
            key = (entry.prototype_id, entry.feature_path)
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return RSSDABank(entries=entries, version="runtime_multi_bank")


def build_retrieval_runtime(
    *,
    memory_bank: str | Path,
    bank_purpose: str,
    checkpoint: Optional[str],
    adapter_checkpoint: Optional[str],
    retriever_checkpoint: Optional[str],
    similarity_checkpoint: Optional[str],
    lora_checkpoint: Optional[str],
    lora_stage: str,
    device: str,
    precision: str,
    image_size: int,
    top_k: Optional[int],
    top_k_positive: Optional[int],
    top_k_negative: Optional[int],
    negative_lambda: float,
    positive_weight: float,
    negative_weight: float,
    similarity_threshold: float,
    confidence_scale: float,
    similarity_weighting: str,
    similarity_temperature: Optional[float],
    retrieval_policy: str,
    uncertainty_threshold: float,
    uncertainty_scale: float,
    policy_activation_threshold: float,
    residual_strength: float,
    allow_dummy_fallback: bool,
    continual_bank_root: Optional[str | Path] = None,
    site_bank_mode: str = "train_plus_site",
) -> RetrievalRuntime:
    resolved_top_k_positive = int(top_k_positive or top_k or 1)
    resolved_top_k_negative = int(top_k_negative or top_k or 1)
    bank_context = load_retrieval_bank(
        memory_bank,
        purpose=bank_purpose,
        checkpoint=checkpoint,
        device=device,
        precision=precision,
        image_size=image_size,
        allow_dummy_fallback=allow_dummy_fallback,
    )
    if bank_purpose == "external-eval" and hasattr(bank_context.bank, "check_no_external_leakage"):
        if not bool(bank_context.bank.check_no_external_leakage(["PolypGen"])):
            raise RuntimeError("PolypGen leakage detected in retrieval bank for external evaluation.")
    base_model = build_official_sam3_image_model(
        checkpoint,
        device=device,
        dtype=precision,
        compile_model=False,
        allow_dummy_fallback=allow_dummy_fallback,
    )
    if lora_checkpoint and Path(lora_checkpoint).exists():
        apply_lora_to_model(base_model, LoRAConfig(stage=lora_stage, min_replaced_modules=0))
        load_lora_weights(base_model, lora_checkpoint, strict=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=precision)
    hidden_dim = resolve_hidden_dim(base_model)
    retriever = PrototypeRetriever(
        bank=bank_context.bank,
        feature_dim=hidden_dim,
        top_k_positive=resolved_top_k_positive,
        top_k_negative=resolved_top_k_negative,
    ).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(
        dim=hidden_dim,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        similarity_threshold=similarity_threshold,
        confidence_scale=confidence_scale,
        similarity_weighting=similarity_weighting,
        similarity_temperature=similarity_temperature,
        retrieval_policy=retrieval_policy,
        uncertainty_threshold=uncertainty_threshold,
        uncertainty_scale=uncertainty_scale,
        policy_activation_threshold=policy_activation_threshold,
        residual_strength=residual_strength,
    ).to(device)

    directory_loader = None
    if bank_context.source == "directory_bank":
        directory_loader = DirectoryBankLoader(
            bank_context.resolved_path,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=allow_dummy_fallback,
            default_top_k=max(resolved_top_k_positive, resolved_top_k_negative),
        )

    loaded_components = {"adapter": False, "retriever": False, "similarity_builder": False}
    if adapter_checkpoint and Path(adapter_checkpoint).exists():
        loaded_components = load_rssda_bundle_components(
            adapter_checkpoint,
            device=device,
            adapter=adapter,
            retriever=retriever,
            similarity_builder=similarity_builder,
        )
        if not loaded_components["adapter"]:
            adapter.load_state_dict(torch.load(adapter_checkpoint, map_location=device, weights_only=False), strict=False)
            loaded_components["adapter"] = True
    if retriever_checkpoint and Path(retriever_checkpoint).exists():
        retriever.load_state_dict(torch.load(retriever_checkpoint, map_location=device, weights_only=False), strict=False)
        loaded_components["retriever"] = True
    if similarity_checkpoint and Path(similarity_checkpoint).exists():
        similarity_builder.load_state_dict(torch.load(similarity_checkpoint, map_location=device, weights_only=False), strict=False)
        loaded_components["similarity_builder"] = True

    adapter.eval()
    retriever.eval()
    similarity_builder.eval()
    retrieval_backend = "directory_loader" if directory_loader is not None and not loaded_components["retriever"] else "trainable_retriever"
    primary_backend = RetrievalBankBackend(
        name="train_bank",
        bank_context=bank_context,
        retriever=retriever,
        retrieval_backend=retrieval_backend,
        directory_loader=directory_loader,
    )
    inferred_continual_root = _resolve_site_bank_root(memory_bank, continual_bank_root)
    runtime = RetrievalRuntime(
        device=device,
        hidden_dim=hidden_dim,
        bank_context=bank_context,
        wrapper=wrapper,
        retriever=retriever,
        similarity_builder=similarity_builder,
        adapter=adapter,
        retrieval_backend=retrieval_backend,
        directory_loader=directory_loader,
        primary_backend=primary_backend,
        site_bank_mode=site_bank_mode,
        continual_bank_root=inferred_continual_root,
        bank_loader_config={
            "bank_purpose": bank_purpose,
            "checkpoint": checkpoint,
            "precision": precision,
            "image_size": image_size,
            "allow_dummy_fallback": allow_dummy_fallback,
            "top_k_positive": resolved_top_k_positive,
            "top_k_negative": resolved_top_k_negative,
        },
        bank_backend_cache={_backend_cache_key(bank_context.resolved_path): primary_backend},
    )
    return runtime


def resolve_retrieval(
    runtime: RetrievalRuntime,
    query_feature: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    top_k_positive: Optional[int] = None,
    top_k_negative: Optional[int] = None,
    retrieval_mode: str,
    query_source: Optional[str] = None,
    prefer_cross_domain_positive: bool = True,
    sample_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolution = _resolve_bank_selection(runtime, sample_metadata)
    backends = [_get_backend(runtime, bank_path) for bank_path in resolution.selected_bank_paths]
    if len(backends) <= 1:
        backend = backends[0] if backends else runtime.primary_backend
        if backend is None:
            raise RuntimeError("No retrieval backend available.")
        retrieval = _run_backend_retrieval(
            backend,
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            query_source=query_source,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
        retrieval = annotate_single_bank_retrieval(
            retrieval,
            resolution=resolution,
            bank_label="train" if backend.bank_context.resolved_path == runtime.bank_context.resolved_path else "site",
            bank_path=str(backend.bank_context.resolved_path),
        )
    else:
        train_backend = backends[0]
        site_backend = backends[1]
        train_retrieval = _run_backend_retrieval(
            train_backend,
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive or top_k,
            top_k_negative=top_k_negative or top_k,
            query_source=query_source,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
        site_retrieval = _run_backend_retrieval(
            site_backend,
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive or top_k,
            top_k_negative=top_k_negative or top_k,
            query_source=query_source,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
        retrieval = fuse_multi_bank_retrieval(
            train_retrieval=train_retrieval,
            site_retrieval=site_retrieval,
            resolution=resolution,
            train_bank_path=str(train_backend.bank_context.resolved_path),
            site_bank_path=str(site_backend.bank_context.resolved_path),
        )
    retrieval = attach_retrieved_mask_priors(retrieval, spatial_size=(int(query_feature.shape[-2]), int(query_feature.shape[-1])))
    return apply_retrieval_mode(retrieval, retrieval_mode)


def run_retrieval_forward(
    runtime: RetrievalRuntime,
    *,
    images: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    retrieval: dict[str, Any],
    retrieval_mode: str,
    baseline_mask_logits: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    similarity = runtime.similarity_builder(
        query_feature,
        retrieval["positive_features"],
        retrieval["negative_features"],
        retrieval["positive_weights"],
        retrieval["negative_weights"],
    )
    positive_heatmap = similarity["positive_heatmap"]
    negative_heatmap = similarity["negative_heatmap"]
    if isinstance(retrieval.get("positive_mask_prior"), torch.Tensor):
        positive_heatmap = torch.clamp(0.7 * positive_heatmap + 0.3 * retrieval["positive_mask_prior"].to(positive_heatmap), 0.0, 1.0)
    if isinstance(retrieval.get("negative_mask_prior"), torch.Tensor):
        negative_heatmap = torch.clamp(0.7 * negative_heatmap + 0.3 * retrieval["negative_mask_prior"].to(negative_heatmap), 0.0, 1.0)
    _, retrieval_prior, adapter_aux = runtime.adapter(
        feature_map=query_feature,
        similarity_map=similarity["fused_similarity"],
        positive_prototype=retrieval["positive_prototype"],
        negative_prototype=retrieval["negative_prototype"],
        positive_tokens=retrieval["positive_features"],
        negative_tokens=retrieval["negative_features"],
        positive_similarity=similarity["positive_similarity"],
        negative_similarity=similarity["negative_similarity"],
        positive_weights=retrieval["positive_weights"],
        negative_weights=retrieval["negative_weights"],
        positive_scores=retrieval.get("positive_score_tensor"),
        negative_scores=retrieval.get("negative_score_tensor"),
        baseline_mask_logits=baseline_mask_logits,
        positive_heatmap=positive_heatmap,
        negative_heatmap=negative_heatmap,
        mode=retrieval_mode,
    )
    outputs = runtime.wrapper(images=images, boxes=boxes, text_prompt=text_prompt, retrieval_prior=retrieval_prior)
    return outputs, retrieval_prior, adapter_aux, similarity


def infer_query_feature(runtime: RetrievalRuntime, images: torch.Tensor, boxes: torch.Tensor, text_prompt: list[str]) -> tuple[dict[str, Any], torch.Tensor]:
    baseline = runtime.wrapper(images=images, boxes=boxes, text_prompt=text_prompt)
    return baseline, resolve_feature_map(baseline["image_embeddings"], images)