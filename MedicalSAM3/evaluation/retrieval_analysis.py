"""Analyze whether retrieval materially changes segmentation outputs."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar_bank import PrototypeBankEntry, RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import load_retrieval_bank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    ensure_dir,
    infer_source_domain,
    read_records,
    resolve_runtime_device,
    resolve_feature_map,
    seed_everything,
)


def _resolve_hidden_dim(model: torch.nn.Module) -> int:
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def _resolve_runtime_device(requested_device: str) -> str:
    return resolve_runtime_device(requested_device)


def _load_checkpoint_payload(path: Path, device: str) -> object:
    return torch.load(path, map_location=device, weights_only=False)


def _maybe_load_rssda_bundle(
    path: Path,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> bool:
    payload = _load_checkpoint_payload(path, device)
    if not isinstance(payload, dict):
        return False
    loaded = False
    if isinstance(payload.get("adapter"), dict):
        adapter.load_state_dict(payload["adapter"], strict=False)
        loaded = True
    if isinstance(payload.get("retriever"), dict):
        retriever.load_state_dict(payload["retriever"], strict=False)
        loaded = True
    if isinstance(payload.get("similarity_builder"), dict):
        similarity_builder.load_state_dict(payload["similarity_builder"], strict=False)
        loaded = True
    return loaded


def _dummy_records(prefix: str, dataset_name: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "image_path": "",
            "mask_path": "",
            "dataset_name": dataset_name,
            "image_id": f"{prefix}_{index:03d}",
        }
        for index in range(count)
    ]


def _ensure_records(split_file: str | Path, dummy: bool, prefix: str, dataset_name: str, count: int) -> list[dict[str, Any]]:
    records = read_records(split_file)
    if dummy and not records:
        return _dummy_records(prefix, dataset_name, count)
    return records


def _create_dummy_bank(bank_dir: Path, hidden_dim: int, seed: int) -> RSSDABank:
    generator = torch.Generator().manual_seed(seed)
    bank = RSSDABank()
    fixtures = [
        ("kvasir_positive_a", "positive", "Kvasir", "polyp"),
        ("cvc_positive_a", "positive", "CVC", "polyp"),
        ("polypgen_positive_a", "positive", "PolypGen", "polyp"),
        ("specular_negative", "negative", "PolypGen", "specular_highlight"),
        ("mucosa_negative", "negative", "Kvasir", "normal_mucosa"),
        ("bubble_negative", "negative", "CVC", "bubble"),
        ("blur_negative", "negative", "PolypGen", "blur_region"),
        ("instrument_negative", "negative", "PolypGen", "instrument_artifact"),
    ]
    for index, (prototype_id, polarity, source_dataset, polyp_type) in enumerate(fixtures):
        feature_dir = ensure_dir(bank_dir / ("positive_bank" if polarity == "positive" else "negative_bank"))
        feature_path = feature_dir / f"{prototype_id}.pt"
        feature = torch.randn(hidden_dim, generator=generator)
        feature[index % max(1, min(hidden_dim, 16))] += 3.0
        torch.save({"prototype": torch.nn.functional.normalize(feature.float(), dim=0)}, feature_path)
        bank.add_entry(
            PrototypeBankEntry(
                prototype_id=prototype_id,
                feature_path=str(feature_path),
                polarity=polarity,
                source_dataset=source_dataset,
                polyp_type=polyp_type,
                boundary_quality=0.8,
                confidence=0.9,
                image_id=f"{source_dataset.lower()}_{index:03d}",
            )
        )
    bank.save(bank_dir)
    return bank


def _load_or_create_bank(
    path: str | Path,
    hidden_dim: int,
    dummy: bool,
    seed: int,
    *,
    image_size: int = 128,
    precision: str = "fp32",
    checkpoint: Optional[str] = None,
    device: str = "auto",
) -> RSSDABank:
    bank_path = Path(path)
    if bank_path.exists():
        bank_context = load_retrieval_bank(
            bank_path,
            purpose="external-eval",
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=dummy,
        )
        if bank_context.bank.entries:
            return bank_context.bank
    if not dummy:
        raise FileNotFoundError(f"RSS-DA bank not found or empty: {bank_path}")
    return _create_dummy_bank(bank_path, hidden_dim=hidden_dim, seed=seed)


def _apply_retrieval_mode(retrieval: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode in {"joint", "semantic", "spatial", "positive-negative"}:
        return retrieval
    if mode != "positive-only":
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    updated = dict(retrieval)
    updated["negative_features"] = torch.zeros_like(retrieval["negative_features"])
    updated["negative_weights"] = torch.zeros_like(retrieval["negative_weights"])
    updated["negative_prototype"] = torch.zeros_like(retrieval["positive_prototype"])
    updated["negative_entries"] = [[] for _ in retrieval["positive_entries"]]
    updated["negative_scores"] = [torch.zeros_like(score) for score in retrieval["positive_scores"]]
    return updated


def _selection_from_entries(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    dim = int(query_vector.shape[-1])
    if not indices:
        return (
            torch.zeros(1, 0, dim, device=query_vector.device),
            torch.zeros(1, 0, device=query_vector.device),
            [],
            torch.zeros(1, dim, device=query_vector.device),
            torch.zeros(0, device=query_vector.device),
        )
    selected_entries = [entries[index] for index in indices]
    features = bank.stack_features(selected_entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    raw_scores = torch.matmul(features, query_vector)
    weights = torch.softmax(raw_scores, dim=0)
    prototype = torch.nn.functional.normalize((weights.unsqueeze(-1) * features).sum(dim=0, keepdim=True), dim=-1)
    return features.unsqueeze(0), weights.unsqueeze(0), selected_entries, prototype, raw_scores


def _rank_entry_indices(
    bank: RSSDABank,
    query_vector: torch.Tensor,
    entries: list[PrototypeBankEntry],
    top_k: int,
    strategy: str,
    rng: random.Random,
) -> list[int]:
    if not entries or top_k <= 0:
        return []
    features = bank.stack_features(entries, device=query_vector.device)
    features = torch.nn.functional.normalize(features, dim=-1)
    scores = torch.matmul(features, query_vector)
    count = min(top_k, len(entries))
    if strategy == "best":
        return torch.topk(scores, k=count, largest=True).indices.detach().cpu().tolist()
    if strategy == "random":
        shuffled = list(range(len(entries)))
        rng.shuffle(shuffled)
        return shuffled[:count]
    raise ValueError(f"Unsupported ranking strategy: {strategy}")


def _override_retrieval(
    base_retrieval: dict[str, Any],
    *,
    positive_override: Optional[tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]] = None,
    negative_override: Optional[tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]] = None,
) -> dict[str, Any]:
    retrieval = dict(base_retrieval)
    if positive_override is not None:
        retrieval["positive_features"] = positive_override[0]
        retrieval["positive_weights"] = positive_override[1]
        retrieval["positive_entries"] = [positive_override[2]]
        retrieval["positive_prototype"] = positive_override[3]
        retrieval["positive_scores"] = [positive_override[4]]
    if negative_override is not None:
        retrieval["negative_features"] = negative_override[0]
        retrieval["negative_weights"] = negative_override[1]
        retrieval["negative_entries"] = [negative_override[2]]
        retrieval["negative_prototype"] = negative_override[3]
        retrieval["negative_scores"] = [negative_override[4]]
    return retrieval


def _empty_negative_like(retrieval: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[PrototypeBankEntry], torch.Tensor, torch.Tensor]:
    dim = int(retrieval["positive_prototype"].shape[-1])
    device = retrieval["positive_prototype"].device
    return (
        torch.zeros(1, 0, dim, device=device),
        torch.zeros(1, 0, device=device),
        [],
        torch.zeros(1, dim, device=device),
        torch.zeros(0, device=device),
    )


def _build_prompt_variants(
    bank: RSSDABank,
    retriever: PrototypeRetriever,
    query_feature: torch.Tensor,
    query_source: str,
    top_k_positive: int,
    top_k_negative: int,
    rng: random.Random,
    prefer_cross_domain_positive: bool,
    retrieval_mode: str,
) -> dict[str, Optional[dict[str, Any]]]:
    base_retrieval = _apply_retrieval_mode(
        retriever(
            query_feature,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            query_source_datasets=[query_source],
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        ),
        retrieval_mode,
    )
    query_vector = base_retrieval["projected_query"][0]
    positive_entries = bank.get_entries(polarity="positive", human_verified=True)
    negative_entries = bank.get_entries(polarity="negative", human_verified=True)
    random_positive = _selection_from_entries(
        bank,
        query_vector,
        positive_entries,
        _rank_entry_indices(bank, query_vector, positive_entries, top_k_positive, "random", rng),
    )
    negative_as_positive = _selection_from_entries(
        bank,
        query_vector,
        negative_entries,
        _rank_entry_indices(bank, query_vector, negative_entries, top_k_negative, "best", rng),
    )
    return {
        "positive_exemplar": base_retrieval,
        "negative_exemplar": _override_retrieval(
            base_retrieval,
            positive_override=negative_as_positive,
            negative_override=_empty_negative_like(base_retrieval),
        ),
        "random_exemplar": _override_retrieval(base_retrieval, positive_override=random_positive),
        "empty_exemplar": None,
    }


def _binary_mask(mask_logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(mask_logits) > 0.5).float()


def _mask_difference_ratio(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    pred_a = _binary_mask(mask_a)
    pred_b = _binary_mask(mask_b)
    difference = (pred_a != pred_b).float().mean()
    return float(difference.item())


def _logit_difference(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    return float((mask_a.detach().float() - mask_b.detach().float()).abs().mean().item())


def _entry_logs(
    entries: list[PrototypeBankEntry],
    scores: list[float],
    weights: list[float],
    token_response: list[float],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        payload.append(
            {
                "prototype_id": entry.prototype_id,
                "polarity": entry.polarity,
                "source_dataset": entry.source_dataset,
                "polyp_type": entry.polyp_type,
                "confidence": float(entry.confidence),
                "similarity_score": float(scores[index]) if index < len(scores) else 0.0,
                "retrieval_weight": float(weights[index]) if index < len(weights) else 0.0,
                "token_response": float(token_response[index]) if index < len(token_response) else 0.0,
            }
        )
    return payload


def _tensor_list(values: torch.Tensor, count: int) -> list[float]:
    if values.numel() == 0 or count <= 0:
        return []
    return [float(item) for item in values[:count].detach().cpu().tolist()]


def _run_variant(
    *,
    adapter: RetrievalSpatialSemanticAdapter,
    wrapper: Sam3TensorForwardWrapper,
    similarity_builder: SimilarityHeatmapBuilder,
    images: torch.Tensor,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    baseline_outputs: dict[str, Any],
    retrieval: Optional[dict[str, Any]],
    retrieval_mode: str,
) -> dict[str, Any]:
    if retrieval is None:
        outputs = baseline_outputs
        metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
        return {
            "metrics": metrics,
            "mask_logits": outputs["mask_logits"].detach(),
            "attention_log": {
                "positive_prototypes": [],
                "negative_prototypes": [],
                "fusion_alpha": 0.0,
                "negative_lambda": 0.0,
                "gate_mean": 0.0,
            },
        }

    similarity = similarity_builder(
        query_feature,
        retrieval["positive_features"],
        retrieval["negative_features"],
        retrieval["positive_weights"],
        retrieval["negative_weights"],
    )
    _, retrieval_prior, adapter_aux = adapter(
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
        baseline_mask_logits=baseline_outputs.get("mask_logits"),
        positive_heatmap=similarity["positive_heatmap"],
        negative_heatmap=similarity["negative_heatmap"],
        mode=retrieval_mode,
    )
    outputs = wrapper(images=images, boxes=boxes, text_prompt=text_prompt, retrieval_prior=retrieval_prior)
    metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
    positive_entries = retrieval["positive_entries"][0]
    negative_entries = retrieval["negative_entries"][0]
    attention_log = {
        "positive_prototypes": _entry_logs(
            positive_entries,
            _tensor_list(retrieval["positive_scores"][0], len(positive_entries)),
            _tensor_list(retrieval["positive_weights"][0], len(positive_entries)),
            _tensor_list(adapter_aux["positive_token_response"][0], len(positive_entries)),
        ),
        "negative_prototypes": _entry_logs(
            negative_entries,
            _tensor_list(retrieval["negative_scores"][0], len(negative_entries)),
            _tensor_list(retrieval["negative_weights"][0], len(negative_entries)),
            _tensor_list(adapter_aux["negative_token_response"][0], len(negative_entries)),
        ),
        "fusion_alpha": float(adapter_aux["fusion_alpha"].detach().float().mean().item()),
        "negative_lambda": float(adapter_aux["negative_lambda"].detach().float().mean().item()),
        "gate_mean": float(adapter_aux["fusion_gate_map"].detach().float().mean().item()),
        "similarity_temperature": float(similarity["temperature"].detach().float().mean().item()),
        "similarity_fusion_weight": [float(item) for item in similarity["fusion_weight"].detach().float().flatten().cpu().tolist()],
        "wrapper_retrieval_summary": outputs.get("intermediate_features", {}).get("retrieval_prior", {}),
    }
    return {
        "metrics": metrics,
        "mask_logits": outputs["mask_logits"].detach(),
        "attention_log": attention_log,
    }


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.var(tensor, unbiased=False).item())


def _prompt_sensitivity(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variant_names = ["positive_exemplar", "negative_exemplar", "random_exemplar", "empty_exemplar"]
    pairwise: dict[str, float] = {}
    pairwise_logit: dict[str, float] = {}
    pairwise_values: list[float] = []
    logit_values: list[float] = []
    for left_name, right_name in itertools.combinations(variant_names, 2):
        ratio = _mask_difference_ratio(variants[left_name]["mask_logits"], variants[right_name]["mask_logits"])
        logit_shift = _logit_difference(variants[left_name]["mask_logits"], variants[right_name]["mask_logits"])
        pairwise[f"{left_name}__vs__{right_name}"] = ratio
        pairwise_logit[f"{left_name}__vs__{right_name}"] = logit_shift
        pairwise_values.append(ratio)
        logit_values.append(logit_shift)
    dice_values = [float(variants[name]["metrics"]["Dice"]) for name in variant_names]
    iou_values = [float(variants[name]["metrics"]["IoU"]) for name in variant_names]
    mean_mask_difference = float(sum(pairwise_values) / max(len(pairwise_values), 1))
    mean_logit_difference = float(sum(logit_values) / max(len(logit_values), 1))
    score = 0.4 * mean_mask_difference + 0.2 * _variance(dice_values) + 0.2 * _variance(iou_values) + 0.2 * mean_logit_difference
    return {
        "mask_difference_ratio": pairwise,
        "logit_difference": pairwise_logit,
        "mean_mask_difference_ratio": mean_mask_difference,
        "mean_logit_difference": mean_logit_difference,
        "dice_variance": _variance(dice_values),
        "iou_variance": _variance(iou_values),
        "prompt_sensitivity_score": float(score),
    }


def _average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    summary: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            summary[key] = summary.get(key, 0.0) + float(value)
    return {key: value / len(rows) for key, value in summary.items()}


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["domain"], []).append(row["metrics"])
    return {domain: _average_rows(domain_rows) for domain, domain_rows in grouped.items()}


def _delta_metrics(current: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for domain, metrics in current.items():
        baseline_metrics = baseline.get(domain, {})
        output[domain] = {key: float(metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)) for key in metrics.keys()}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze retrieval influence on MedEx-SAM3 segmentation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--internal-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt")
    parser.add_argument("--external-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/retrieval_analysis")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--retriever-checkpoint", default=None)
    parser.add_argument("--similarity-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--top-k-positive", type=int, default=3)
    parser.add_argument("--top-k-negative", type=int, default=3)
    parser.add_argument("--negative-lambda", type=float, default=0.35)
    parser.add_argument("--retrieval-mode", choices=["joint", "semantic", "spatial", "positive-only", "positive-negative"], default="joint")
    parser.add_argument("--prefer-cross-domain-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--dummy-samples-per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    internal_records = _ensure_records(args.internal_split_file, args.dummy, "internal", "Kvasir", args.dummy_samples_per_split)
    external_records = _ensure_records(args.external_split_file, args.dummy, "external", "PolypGen", args.dummy_samples_per_split)
    if not internal_records:
        raise FileNotFoundError(f"No internal validation records found in {args.internal_split_file}")
    if not external_records:
        raise FileNotFoundError(f"No external validation records found in {args.external_split_file}")

    device = _resolve_runtime_device(args.device)
    base_model = build_official_sam3_image_model(
        args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
        allow_dummy_fallback=args.dummy,
    )
    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        apply_lora_to_model(base_model, LoRAConfig(stage=args.lora_stage, min_replaced_modules=0))
        load_lora_weights(base_model, args.lora_checkpoint, strict=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    hidden_dim = _resolve_hidden_dim(base_model)
    bank = _load_or_create_bank(
        args.memory_bank,
        hidden_dim=hidden_dim,
        dummy=args.dummy,
        seed=args.seed,
        image_size=args.image_size,
        precision=args.precision,
        checkpoint=args.checkpoint,
        device=device,
    )
    retriever = PrototypeRetriever(bank=bank, feature_dim=hidden_dim, top_k_positive=args.top_k_positive, top_k_negative=args.top_k_negative).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=args.negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(dim=hidden_dim).to(device)
    if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
        loaded_bundle = _maybe_load_rssda_bundle(Path(args.adapter_checkpoint), device, adapter, retriever, similarity_builder)
        if not loaded_bundle:
            adapter.load_state_dict(_load_checkpoint_payload(Path(args.adapter_checkpoint), device), strict=False)
    if args.retriever_checkpoint and Path(args.retriever_checkpoint).exists():
        retriever.load_state_dict(torch.load(args.retriever_checkpoint, map_location=device, weights_only=False), strict=False)
    if args.similarity_checkpoint and Path(args.similarity_checkpoint).exists():
        similarity_builder.load_state_dict(torch.load(args.similarity_checkpoint, map_location=device, weights_only=False), strict=False)
    adapter.eval()
    retriever.eval()
    similarity_builder.eval()

    split_rows = []
    prompt_rows = []
    attention_rows = []
    baseline_metrics_rows = []
    retrieval_metrics_rows = []
    split_inputs = [
        ("internal", internal_records),
        ("external", external_records),
    ]

    with torch.no_grad():
        for split_name, split_records in split_inputs:
            if args.max_samples_per_split is not None:
                split_records = split_records[: max(0, args.max_samples_per_split)]
            loader = DataLoader(SplitSegmentationDataset(split_records, args.image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)
            for batch_index, batch in enumerate(loader):
                images = batch["images"].to(device)
                masks = batch["masks"].to(device)
                boxes = batch["boxes"].to(device)
                record = batch["records"][0]
                image_id = str(record.get("image_id", f"{split_name}_{batch_index:03d}"))
                source_domain = infer_source_domain(
                    dataset_name=str(record.get("dataset_name", "")),
                    image_id=str(record.get("image_id", "")),
                    image_path=str(record.get("image_path", "")),
                    mask_path=str(record.get("mask_path", "")),
                )

                baseline_outputs = wrapper(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
                query_feature = resolve_feature_map(baseline_outputs["image_embeddings"], images)
                variant_retrievals = _build_prompt_variants(
                    bank,
                    retriever,
                    query_feature,
                    query_source=source_domain,
                    top_k_positive=args.top_k_positive,
                    top_k_negative=args.top_k_negative,
                    rng=random.Random(args.seed + batch_index),
                    prefer_cross_domain_positive=args.prefer_cross_domain_positive,
                    retrieval_mode=args.retrieval_mode,
                )
                variants = {
                    name: _run_variant(
                        adapter=adapter,
                        wrapper=wrapper,
                        similarity_builder=similarity_builder,
                        images=images,
                        masks=masks,
                        boxes=boxes,
                        text_prompt=batch["text_prompt"],
                        query_feature=query_feature,
                        baseline_outputs=baseline_outputs,
                        retrieval=retrieval,
                        retrieval_mode=args.retrieval_mode,
                    )
                    for name, retrieval in variant_retrievals.items()
                }

                prompt_sensitivity = _prompt_sensitivity(variants)
                prompt_rows.append({
                    "split": split_name,
                    "image_id": image_id,
                    "source_domain": source_domain,
                    **prompt_sensitivity,
                })
                split_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        "variants": {name: {"metrics": payload["metrics"]} for name, payload in variants.items()},
                        "prompt_sensitivity": prompt_sensitivity,
                    }
                )
                attention_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        "variants": {name: payload["attention_log"] for name, payload in variants.items()},
                    }
                )
                baseline_metrics_rows.append(
                    {
                        "split": split_name,
                        "domain": source_domain,
                        "metrics": variants["empty_exemplar"]["metrics"],
                    }
                )
                retrieval_metrics_rows.append(
                    {
                        "split": split_name,
                        "domain": source_domain,
                        "metrics": variants["positive_exemplar"]["metrics"],
                    }
                )

    prompt_by_domain: dict[str, list[dict[str, float]]] = {}
    for row in prompt_rows:
        prompt_by_domain.setdefault(row["source_domain"], []).append(
            {
                "mean_mask_difference_ratio": row["mean_mask_difference_ratio"],
                "mean_logit_difference": row["mean_logit_difference"],
                "dice_variance": row["dice_variance"],
                "iou_variance": row["iou_variance"],
                "prompt_sensitivity_score": row["prompt_sensitivity_score"],
            }
        )

    prompt_summary = {
        "overall": _average_rows(
            [
                {
                    "mean_mask_difference_ratio": row["mean_mask_difference_ratio"],
                    "mean_logit_difference": row["mean_logit_difference"],
                    "dice_variance": row["dice_variance"],
                    "iou_variance": row["iou_variance"],
                    "prompt_sensitivity_score": row["prompt_sensitivity_score"],
                }
                for row in prompt_rows
            ]
        ),
        "by_domain": {domain: _average_rows(domain_rows) for domain, domain_rows in prompt_by_domain.items()},
    }

    lora_only_summary = _group_metrics(baseline_metrics_rows)
    retrieval_summary = _group_metrics(retrieval_metrics_rows)
    ablation_summary = {
        "lora_only": lora_only_summary,
        "lora_plus_retrieval": retrieval_summary,
        "delta": _delta_metrics(retrieval_summary, lora_only_summary),
    }

    (output_dir / "prompt_sensitivity.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prompt_rows),
        encoding="utf-8",
    )
    (output_dir / "prototype_attention_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in attention_rows),
        encoding="utf-8",
    )
    (output_dir / "per_image_analysis.jsonl").write_text(
        "\n".join(json.dumps(row) for row in split_rows),
        encoding="utf-8",
    )
    summary = {
        "config": vars(args),
        "prompt_sensitivity": prompt_summary,
        "retrieval_ablation": ablation_summary,
        "prototype_attention_log": {
            "path": str(output_dir / "prototype_attention_log.jsonl"),
            "sample_count": len(attention_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())