"""Quantify whether retrieval changes segmentation behavior and external robustness."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.exemplar_prompt_adapter import ExemplarPromptAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.evaluation.retrieval_analysis import (
    _apply_retrieval_mode,
    _build_prompt_variants,
    _ensure_records,
    _load_checkpoint_payload,
    _load_or_create_bank,
    _maybe_load_rssda_bundle,
    _resolve_hidden_dim,
    _resolve_runtime_device,
)
from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank
from MedicalSAM3.exemplar.prototype_builder import PrototypeBuilder
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    ensure_dir,
    infer_source_domain,
    resolve_feature_map,
    seed_everything,
)
from MedicalSAM3.visualization.retrieval_vis import (
    save_false_positive_overlay,
    save_mask_difference_visualization,
    save_retrieved_prototype_panel,
    save_similarity_heatmap_overlay,
)


def _binary_mask(mask_logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(mask_logits.detach().float()) > 0.5).float()


def _mask_difference_ratio(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    pred_a = _binary_mask(mask_a)
    pred_b = _binary_mask(mask_b)
    xor = ((pred_a > 0.5) ^ (pred_b > 0.5)).float().sum()
    union = ((pred_a > 0.5) | (pred_b > 0.5)).float().sum()
    if float(union.item()) <= 1e-6:
        return 0.0
    return float((xor / (union + 1e-6)).item())


def _mean_logit_difference(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    return float((mask_a.detach().float() - mask_b.detach().float()).abs().mean().item())


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _tensor_stats(values: Optional[torch.Tensor]) -> dict[str, float | None]:
    if values is None or not isinstance(values, torch.Tensor) or values.numel() == 0:
        return {
            "mean": None,
            "abs_mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    tensor = values.detach().float()
    return {
        "mean": float(tensor.mean().item()),
        "abs_mean": float(tensor.abs().mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _create_dummy_exemplar_memory_bank(bank_dir: Path, hidden_dim: int, seed: int) -> ExemplarMemoryBank:
    generator = torch.Generator().manual_seed(seed)
    ensure_dir(bank_dir)
    bank = ExemplarMemoryBank()
    fixtures = [
        ("pos_kvasir_a", "positive", "Kvasir"),
        ("pos_cvc_a", "positive", "CVC"),
        ("neg_mucosa_a", "negative", "Kvasir"),
        ("neg_bubble_a", "negative", "CVC"),
        ("boundary_kvasir_a", "boundary", "Kvasir"),
        ("boundary_cvc_a", "boundary", "CVC"),
    ]
    for index, (item_id, item_type, source_dataset) in enumerate(fixtures):
        feature_path = bank_dir / f"{item_id}.pt"
        embedding = torch.randn(hidden_dim, generator=generator)
        embedding[index % max(1, min(hidden_dim, 16))] += 3.0
        torch.save(torch.nn.functional.normalize(embedding.float(), dim=0), feature_path)
        bank.add_item(
            ExemplarItem(
                item_id=item_id,
                image_id=f"{source_dataset.lower()}_{index:03d}",
                crop_path="",
                mask_path=None,
                bbox=[0.0, 0.0, 64.0, 64.0],
                embedding_path=str(feature_path),
                type=item_type,
                source_dataset=source_dataset,
                fold_id=0,
                human_verified=True,
                quality_score=0.9,
                boundary_score=0.8,
                diversity_score=0.7,
                difficulty_score=0.6,
                uncertainty_score=0.1,
                false_positive_risk=0.1,
                created_at="2026-05-11T00:00:00Z",
                version="v0",
                notes="",
            )
        )
    bank.save(bank_dir)
    return bank


def _load_or_create_exemplar_memory_bank(
    path: Optional[str | Path],
    hidden_dim: int,
    dummy: bool,
    seed: int,
    fallback_dir: Path,
) -> Optional[ExemplarMemoryBank]:
    if path is None:
        if not dummy:
            return None
        target = fallback_dir / "dummy_exemplar_memory_bank"
        return _create_dummy_exemplar_memory_bank(target, hidden_dim=hidden_dim, seed=seed)
    target = Path(path)
    bank = ExemplarMemoryBank.load(target)
    if bank.items:
        return bank
    if not dummy:
        return None
    return _create_dummy_exemplar_memory_bank(target, hidden_dim=hidden_dim, seed=seed)


def _prompt_adapter_diagnostics(
    query_feature: torch.Tensor,
    bank: Optional[ExemplarMemoryBank],
    prompt_adapter: Optional[ExemplarPromptAdapter],
) -> dict[str, Any]:
    if bank is None or prompt_adapter is None or not bank.trainable_items:
        return {
            "positive_fusion_strength": None,
            "negative_fusion_strength": None,
            "boundary_fusion_strength": None,
            "suppression_gate": None,
            "selected_positive_ids": [],
            "selected_negative_ids": [],
            "selected_boundary_ids": [],
        }
    builder = PrototypeBuilder()
    query_embedding = F.normalize(F.adaptive_avg_pool2d(query_feature, 1).flatten(1), dim=1)[0]
    result = builder.build_positive_negative_boundary_prototypes(query_embedding, bank, top_k=3)
    positive_proto = result["positive"]["prototype"]
    if positive_proto is None:
        return {
            "positive_fusion_strength": None,
            "negative_fusion_strength": None,
            "boundary_fusion_strength": None,
            "suppression_gate": None,
            "selected_positive_ids": [],
            "selected_negative_ids": [],
            "selected_boundary_ids": [],
        }

    def _batchify(proto: Any) -> Optional[torch.Tensor]:
        if proto is None:
            return None
        tensor = proto if isinstance(proto, torch.Tensor) else torch.tensor(proto)
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor.unsqueeze(0)

    _, aux = prompt_adapter(
        positive_proto=_batchify(result["positive"]["prototype"]).to(query_feature.device),
        negative_proto=_batchify(result["negative"]["prototype"]).to(query_feature.device) if result["negative"]["prototype"] is not None else None,
        boundary_proto=_batchify(result["boundary"]["prototype"]).to(query_feature.device) if result["boundary"]["prototype"] is not None else None,
        query_feat=F.normalize(F.adaptive_avg_pool2d(query_feature, 1).flatten(1), dim=1),
    )
    fusion_weights = aux["fusion_weights"].detach().float()
    return {
        "positive_fusion_strength": float(fusion_weights[:, 0].mean().item()),
        "negative_fusion_strength": float(fusion_weights[:, 1].mean().item()),
        "boundary_fusion_strength": float(fusion_weights[:, 2].mean().item()),
        "suppression_gate": float(aux["suppression_gate"].detach().float().mean().item()),
        "selected_positive_ids": result["positive"]["selected_item_ids"],
        "selected_negative_ids": result["negative"]["selected_item_ids"],
        "selected_boundary_ids": result["boundary"]["selected_item_ids"],
    }


def _summarize_prototype_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized = []
    for entry in entries:
        summarized.append(
            {
                "prototype_id": entry.get("prototype_id", ""),
                "similarity_score": float(entry.get("similarity_score", 0.0)),
                "retrieval_weight": float(entry.get("retrieval_weight", 0.0)),
                "polarity": entry.get("polarity", ""),
                "source_dataset": entry.get("source_dataset", ""),
                "polyp_type": entry.get("polyp_type", ""),
                "crop_path": entry.get("crop_path"),
            }
        )
    return summarized


def _variant_retrieval_entries(retrieval: Optional[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if retrieval is None:
        return [], []
    positive = []
    negative = []
    for entry, score_tensor, weight_tensor in zip(
        retrieval["positive_entries"][0],
        retrieval["positive_scores"][0].detach().cpu().tolist() if retrieval["positive_scores"] else [],
        retrieval["positive_weights"][0].detach().cpu().tolist() if retrieval["positive_weights"].numel() > 0 else [],
    ):
        positive.append(
            {
                "prototype_id": entry.prototype_id,
                "similarity_score": float(score_tensor),
                "retrieval_weight": float(weight_tensor),
                "polarity": entry.polarity,
                "source_dataset": entry.source_dataset,
                "polyp_type": entry.polyp_type,
                "crop_path": entry.crop_path,
            }
        )
    for entry, score_tensor, weight_tensor in zip(
        retrieval["negative_entries"][0],
        retrieval["negative_scores"][0].detach().cpu().tolist() if retrieval["negative_scores"] else [],
        retrieval["negative_weights"][0].detach().cpu().tolist() if retrieval["negative_weights"].numel() > 0 else [],
    ):
        negative.append(
            {
                "prototype_id": entry.prototype_id,
                "similarity_score": float(score_tensor),
                "retrieval_weight": float(weight_tensor),
                "polarity": entry.polarity,
                "source_dataset": entry.source_dataset,
                "polyp_type": entry.polyp_type,
                "crop_path": entry.crop_path,
            }
        )
    return positive, negative


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
    prompt_bank: Optional[ExemplarMemoryBank],
    prompt_adapter: Optional[ExemplarPromptAdapter],
) -> dict[str, Any]:
    prompt_diag = _prompt_adapter_diagnostics(query_feature, prompt_bank, prompt_adapter)
    if retrieval is None:
        outputs = baseline_outputs
        metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
        return {
            "outputs": outputs,
            "metrics": metrics,
            "mask_logits": outputs["mask_logits"].detach(),
            "similarity": None,
            "retrieval_prior": {},
            "fusion_diagnostics": {
                "gate_value": {"mean": 0.0, "abs_mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "positive_fusion_strength": {"mean": 0.0, "abs_mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "negative_fusion_strength": {"mean": 0.0, "abs_mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "boundary_fusion_strength": prompt_diag.get("boundary_fusion_strength"),
                "prompt_adapter": prompt_diag,
            },
            "positive_entries": [],
            "negative_entries": [],
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
    positive_entries, negative_entries = _variant_retrieval_entries(retrieval)
    fusion_diagnostics = {
        "gate_value": _tensor_stats(adapter_aux.get("fusion_gate_map")),
        "positive_fusion_strength": _tensor_stats(adapter_aux.get("positive_context_map")),
        "negative_fusion_strength": _tensor_stats(adapter_aux.get("negative_context_map")),
        "boundary_fusion_strength": prompt_diag.get("boundary_fusion_strength"),
        "fusion_alpha": float(adapter_aux.get("fusion_alpha", torch.tensor(0.0)).detach().float().mean().item()),
        "negative_lambda": float(adapter_aux.get("negative_lambda", torch.tensor(0.0)).detach().float().mean().item()),
        "mask_logit_scale": float(adapter_aux.get("mask_logit_scale", torch.tensor(0.0)).detach().float().mean().item()),
        "prompt_adapter": prompt_diag,
        "wrapper_retrieval_summary": outputs.get("intermediate_features", {}).get("retrieval_prior", {}),
    }
    return {
        "outputs": outputs,
        "metrics": metrics,
        "mask_logits": outputs["mask_logits"].detach(),
        "similarity": similarity,
        "retrieval_prior": retrieval_prior,
        "fusion_diagnostics": fusion_diagnostics,
        "positive_entries": _summarize_prototype_entries(positive_entries),
        "negative_entries": _summarize_prototype_entries(negative_entries),
    }


def _pairwise_mask_differences(variants: dict[str, dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for left_name, right_name in itertools.combinations(["positive_exemplar", "negative_exemplar", "random_exemplar", "empty_exemplar"], 2):
        output[f"{left_name}__vs__{right_name}"] = _mask_difference_ratio(
            variants[left_name]["mask_logits"],
            variants[right_name]["mask_logits"],
        )
    return output


def _prompt_sensitivity_from_pairs(pairwise: dict[str, float]) -> dict[str, Any]:
    values = list(pairwise.values())
    return {
        "pairwise_mask_difference_ratio": pairwise,
        "prompt_sensitivity_score": _safe_mean(values),
    }


def _average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    summary: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            summary[key] = summary.get(key, 0.0) + float(value)
    return {key: value / max(len(rows), 1) for key, value in summary.items()}


def _group_metric_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["domain"], []).append(row["metrics"])
    return {domain: _average_metrics(items) for domain, items in grouped.items()}


def _delta_metrics(current: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    delta: dict[str, dict[str, float]] = {}
    for domain, metrics in current.items():
        baseline_metrics = baseline.get(domain, {})
        delta[domain] = {key: float(metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)) for key in metrics.keys()}
    return delta


def _visualize_image_case(
    vis_dir: Path,
    image_id: str,
    image_tensor: torch.Tensor,
    gt_mask: torch.Tensor,
    variants: dict[str, dict[str, Any]],
    negative_ablation: dict[str, dict[str, Any]],
) -> None:
    case_dir = ensure_dir(vis_dir / image_id)
    positive_variant = variants["positive_exemplar"]
    save_retrieved_prototype_panel(
        image_tensor,
        positive_variant["positive_entries"],
        positive_variant["negative_entries"],
        case_dir / "retrieved_prototypes.png",
    )
    if positive_variant.get("similarity") is not None:
        similarity = positive_variant["similarity"]
        save_similarity_heatmap_overlay(image_tensor, similarity["positive_heatmap"], case_dir / "positive_similarity_overlay.png", title="positive similarity")
        save_similarity_heatmap_overlay(image_tensor, similarity["negative_heatmap"], case_dir / "negative_similarity_overlay.png", title="negative similarity")
        save_similarity_heatmap_overlay(image_tensor, similarity["fused_similarity"], case_dir / "fused_similarity_overlay.png", title="fused similarity")
    save_mask_difference_visualization(
        image_tensor,
        variants["positive_exemplar"]["mask_logits"],
        variants["empty_exemplar"]["mask_logits"],
        case_dir / "mask_diff_positive_vs_empty.png",
        "positive exemplar",
        "empty exemplar",
    )
    save_mask_difference_visualization(
        image_tensor,
        variants["positive_exemplar"]["mask_logits"],
        variants["negative_exemplar"]["mask_logits"],
        case_dir / "mask_diff_positive_vs_negative.png",
        "positive exemplar",
        "negative exemplar",
    )
    save_mask_difference_visualization(
        image_tensor,
        negative_ablation["positive_only"]["mask_logits"],
        negative_ablation["positive_negative"]["mask_logits"],
        case_dir / "mask_diff_positive_only_vs_positive_negative.png",
        "positive-only",
        "positive+negative",
    )
    save_false_positive_overlay(
        image_tensor,
        negative_ablation["positive_only"]["mask_logits"],
        gt_mask,
        case_dir / "false_positive_positive_only.png",
        title="positive-only retrieval",
    )
    save_false_positive_overlay(
        image_tensor,
        negative_ablation["positive_negative"]["mask_logits"],
        gt_mask,
        case_dir / "false_positive_positive_negative.png",
        title="positive+negative retrieval",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify retrieval influence on MedEx-SAM3 segmentation behavior.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--internal-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt")
    parser.add_argument("--external-split-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--exemplar-memory-bank", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/retrieval_influence")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--retriever-checkpoint", default=None)
    parser.add_argument("--similarity-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--prompt-checkpoint", default=None)
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
    parser.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    vis_dir = ensure_dir(output_dir / "visualizations")
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
    exemplar_bank = _load_or_create_exemplar_memory_bank(args.exemplar_memory_bank, hidden_dim, args.dummy, args.seed + 77, output_dir)
    retriever = PrototypeRetriever(bank=bank, feature_dim=hidden_dim, top_k_positive=args.top_k_positive, top_k_negative=args.top_k_negative).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=args.negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(dim=hidden_dim).to(device)
    prompt_adapter = ExemplarPromptAdapter(hidden_dim).to(device) if exemplar_bank is not None else None
    if prompt_adapter is not None and args.prompt_checkpoint and Path(args.prompt_checkpoint).exists():
        prompt_adapter.load_state_dict(torch.load(args.prompt_checkpoint, map_location=device, weights_only=False), strict=False)
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
    if prompt_adapter is not None:
        prompt_adapter.eval()

    per_image_rows = []
    prompt_rows = []
    fusion_rows = []
    baseline_metrics_rows = []
    retrieval_metrics_rows = []
    negative_positive_only_rows = []
    negative_positive_negative_rows = []
    split_inputs = [("internal", internal_records), ("external", external_records)]

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
                variants_retrieval = _build_prompt_variants(
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
                same_image_variants = {
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
                        retrieval=retrieval_variant,
                        retrieval_mode=args.retrieval_mode,
                        prompt_bank=exemplar_bank,
                        prompt_adapter=prompt_adapter,
                    )
                    for name, retrieval_variant in variants_retrieval.items()
                }
                pairwise = _pairwise_mask_differences(same_image_variants)
                prompt_sensitivity = _prompt_sensitivity_from_pairs(pairwise)
                prompt_sensitivity["mean_logit_difference"] = _safe_mean(
                    [
                        _mean_logit_difference(same_image_variants[left]["mask_logits"], same_image_variants[right]["mask_logits"])
                        for left, right in itertools.combinations(["positive_exemplar", "negative_exemplar", "random_exemplar", "empty_exemplar"], 2)
                    ]
                )

                base_retrieval = retriever(
                    query_feature,
                    top_k_positive=args.top_k_positive,
                    top_k_negative=args.top_k_negative,
                    query_source_datasets=[source_domain],
                    prefer_cross_domain_positive=args.prefer_cross_domain_positive,
                )
                negative_ablation = {
                    "positive_only": _run_variant(
                        adapter=adapter,
                        wrapper=wrapper,
                        similarity_builder=similarity_builder,
                        images=images,
                        masks=masks,
                        boxes=boxes,
                        text_prompt=batch["text_prompt"],
                        query_feature=query_feature,
                        baseline_outputs=baseline_outputs,
                        retrieval=_apply_retrieval_mode(base_retrieval, "positive-only"),
                        retrieval_mode="positive-only",
                        prompt_bank=exemplar_bank,
                        prompt_adapter=prompt_adapter,
                    ),
                    "positive_negative": _run_variant(
                        adapter=adapter,
                        wrapper=wrapper,
                        similarity_builder=similarity_builder,
                        images=images,
                        masks=masks,
                        boxes=boxes,
                        text_prompt=batch["text_prompt"],
                        query_feature=query_feature,
                        baseline_outputs=baseline_outputs,
                        retrieval=_apply_retrieval_mode(base_retrieval, "positive-negative"),
                        retrieval_mode="positive-negative",
                        prompt_bank=exemplar_bank,
                        prompt_adapter=prompt_adapter,
                    ),
                }

                per_image_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        "same_image_different_exemplar": {
                            variant_name: {
                                "metrics": payload["metrics"],
                                "positive_entries": payload["positive_entries"],
                                "negative_entries": payload["negative_entries"],
                            }
                            for variant_name, payload in same_image_variants.items()
                        },
                        "prompt_sensitivity": prompt_sensitivity,
                        "negative_retrieval_ablation": {
                            variant_name: payload["metrics"] for variant_name, payload in negative_ablation.items()
                        },
                    }
                )
                prompt_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        **prompt_sensitivity,
                    }
                )
                fusion_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "source_domain": source_domain,
                        "variants": {variant_name: payload["fusion_diagnostics"] for variant_name, payload in same_image_variants.items()},
                        "negative_retrieval": {variant_name: payload["fusion_diagnostics"] for variant_name, payload in negative_ablation.items()},
                    }
                )
                baseline_metrics_rows.append({"domain": source_domain, "metrics": same_image_variants["empty_exemplar"]["metrics"]})
                retrieval_metrics_rows.append({"domain": source_domain, "metrics": same_image_variants["positive_exemplar"]["metrics"]})
                negative_positive_only_rows.append({"domain": source_domain, "metrics": negative_ablation["positive_only"]["metrics"]})
                negative_positive_negative_rows.append({"domain": source_domain, "metrics": negative_ablation["positive_negative"]["metrics"]})

                if args.save_visualizations:
                    _visualize_image_case(
                        vis_dir=vis_dir,
                        image_id=image_id,
                        image_tensor=images[0],
                        gt_mask=masks[0],
                        variants=same_image_variants,
                        negative_ablation=negative_ablation,
                    )

    overall_prompt_sensitivity = _average_metrics(
        [
            {
                "prompt_sensitivity_score": row["prompt_sensitivity_score"],
                "mean_logit_difference": row["mean_logit_difference"],
            }
            for row in prompt_rows
        ]
    )
    prompt_by_domain: dict[str, list[dict[str, float]]] = {}
    for row in prompt_rows:
        prompt_by_domain.setdefault(row["source_domain"], []).append(
            {
                "prompt_sensitivity_score": row["prompt_sensitivity_score"],
                "mean_logit_difference": row["mean_logit_difference"],
            }
        )
    retrieval_ablation = {
        "lora_only": _group_metric_rows(baseline_metrics_rows),
        "lora_plus_retrieval": _group_metric_rows(retrieval_metrics_rows),
    }
    retrieval_ablation["delta"] = _delta_metrics(retrieval_ablation["lora_plus_retrieval"], retrieval_ablation["lora_only"])
    negative_ablation_summary = {
        "positive_only": _group_metric_rows(negative_positive_only_rows),
        "positive_negative": _group_metric_rows(negative_positive_negative_rows),
    }
    negative_ablation_summary["delta"] = _delta_metrics(negative_ablation_summary["positive_negative"], negative_ablation_summary["positive_only"])

    (output_dir / "same_image_different_exemplar.jsonl").write_text(
        "\n".join(json.dumps(row) for row in per_image_rows),
        encoding="utf-8",
    )
    (output_dir / "prompt_sensitivity.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prompt_rows),
        encoding="utf-8",
    )
    (output_dir / "fusion_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in fusion_rows),
        encoding="utf-8",
    )
    summary = {
        "config": vars(args),
        "prompt_sensitivity": {
            "overall_mean_sensitivity": overall_prompt_sensitivity.get("prompt_sensitivity_score", 0.0),
            "overall_mean_logit_difference": overall_prompt_sensitivity.get("mean_logit_difference", 0.0),
            "by_domain": {domain: _average_metrics(rows) for domain, rows in prompt_by_domain.items()},
        },
        "retrieval_ablation": retrieval_ablation,
        "negative_retrieval_ablation": negative_ablation_summary,
        "artifacts": {
            "per_image": str(output_dir / "same_image_different_exemplar.jsonl"),
            "prompt_sensitivity": str(output_dir / "prompt_sensitivity.jsonl"),
            "fusion_diagnostics": str(output_dir / "fusion_diagnostics.jsonl"),
            "visualizations": str(vis_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())