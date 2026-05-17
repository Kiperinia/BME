"""Validate RSS-DA with retrieval sensitivity and spatial-semantic visualizations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.evaluation.region_retrieval_diagnostics import (
    build_region_retrieval_diagnostics,
    summarize_region_retrieval_diagnostics,
    write_region_retrieval_diagnostics,
)
from MedicalSAM3.evaluation.retrieval_calibration import write_retrieval_calibration_report
from MedicalSAM3.evaluation.retrieval_diagnostics import build_retrieval_diagnostics, summarize_retrieval_diagnostics, write_retrieval_diagnostics
from MedicalSAM3.exemplar_bank import RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import load_retrieval_bank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    SplitSegmentationDataset,
    apply_config_overrides,
    collate_batch,
    compute_segmentation_metrics,
    ensure_dir,
    infer_source_domain,
    load_config,
    log_runtime_environment,
    read_records,
    resolve_runtime_device,
    resolve_feature_map,
)
from MedicalSAM3.scripts.retrieval_runtime import build_retrieval_runtime, infer_query_feature, resolve_retrieval, run_retrieval_forward
from MedicalSAM3.visualization.region_retrieval_vis import save_region_retrieval_panel


def _resolve_hidden_dim(model: torch.nn.Module) -> int:
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def _save_gray(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().float().squeeze().numpy()
    if array.max() <= 1.0:
        array = array * 255.0
    Image.fromarray(array.clip(0, 255).astype(np.uint8)).save(path)


def _save_heatmap(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = array - array.min()
    denom = max(float(array.max()), 1e-6)
    norm = array / denom
    rgb = np.stack([norm, np.square(norm), 1.0 - norm], axis=-1)
    Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8)).save(path)


def _mask_area(mask_logits: torch.Tensor) -> float:
    return float((torch.sigmoid(mask_logits) > 0.5).float().sum().item())


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
    adapter_state = payload.get("adapter")
    retriever_state = payload.get("retriever")
    similarity_state = payload.get("similarity_builder")
    if isinstance(adapter_state, dict):
        adapter.load_state_dict(adapter_state, strict=False)
        loaded = True
    if isinstance(retriever_state, dict):
        retriever.load_state_dict(retriever_state, strict=False)
        loaded = True
    if isinstance(similarity_state, dict):
        similarity_builder.load_state_dict(similarity_state, strict=False)
        loaded = True
    return loaded


def _apply_retrieval_mode(retrieval: dict[str, object], mode: str) -> dict[str, object]:
    if mode == "baseline":
        return retrieval
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
        return updated
    updated["positive_features"] = torch.zeros_like(retrieval["positive_features"])
    updated["positive_weights"] = torch.zeros_like(retrieval["positive_weights"])
    updated["positive_score_tensor"] = torch.zeros_like(retrieval.get("positive_score_tensor", retrieval["positive_weights"]))
    updated["positive_prototype"] = torch.zeros_like(retrieval["negative_prototype"])
    updated["positive_entries"] = [[] for _ in retrieval["negative_entries"]]
    updated["positive_scores"] = [torch.zeros_like(score) for score in retrieval["negative_scores"]]
    return updated


def _contains_polypgen_records(records: list[dict[str, object]]) -> bool:
    for record in records:
        searchable = " ".join(
            str(record.get(key, "")).lower()
            for key in ["dataset_name", "image_id", "image_path", "mask_path"]
        )
        if "polypgen" in searchable:
            return True
    return False


def _bank_has_polypgen_leakage(bank_root: Path) -> bool:
    if not bank_root.exists():
        return False
    for path in bank_root.rglob("*"):
        if "polypgen" in path.as_posix().lower():
            return True
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".txt", ".yaml", ".yml", ".csv"}:
            continue
        try:
            if "polypgen" in path.read_text(encoding="utf-8", errors="ignore").lower():
                return True
        except OSError:
            continue
    return False


def _run_adapter_forward(
    adapter: RetrievalSpatialSemanticAdapter,
    wrapper: Sam3TensorForwardWrapper,
    images: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    retrieval: dict[str, object],
    similarity: dict[str, torch.Tensor],
    retrieval_mode: str,
    baseline_mask_logits: Optional[torch.Tensor] = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if retrieval_mode == "baseline":
        return wrapper(images=images, boxes=boxes, text_prompt=text_prompt), {}, {}
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
        baseline_mask_logits=baseline_mask_logits,
        positive_heatmap=similarity["positive_heatmap"],
        negative_heatmap=similarity["negative_heatmap"],
        mode=retrieval_mode,
    )
    outputs = wrapper(images=images, boxes=boxes, text_prompt=text_prompt, retrieval_prior=retrieval_prior)
    return outputs, retrieval_prior, adapter_aux


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RSS-DA.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--continual-bank-root", default=None)
    parser.add_argument("--bank-purpose", default="validation", choices=["train", "validation", "external-eval", "continual-adaptation"])
    parser.add_argument("--site-bank-mode", default="train_plus_site", choices=["train_only", "site_only", "train_plus_site"])
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/rssda_eval")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--retriever-checkpoint", default=None)
    parser.add_argument("--similarity-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--top-k-positive", type=int, default=1)
    parser.add_argument("--top-k-negative", type=int, default=1)
    parser.add_argument("--negative-lambda", type=float, default=0.35)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-scale", type=float, default=8.0)
    parser.add_argument("--similarity-weighting", choices=["hard", "soft"], default="soft")
    parser.add_argument("--similarity-temperature", type=float, default=0.125)
    parser.add_argument("--retrieval-policy", choices=["always-on", "similarity-threshold", "uncertainty-aware", "region-aware", "residual"], default="uncertainty-aware")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.35)
    parser.add_argument("--uncertainty-scale", type=float, default=10.0)
    parser.add_argument("--policy-activation-threshold", type=float, default=0.05)
    parser.add_argument("--residual-strength", type=float, default=0.5)
    parser.add_argument("--retrieval-mode", choices=["baseline", "joint", "semantic", "spatial", "positive-only", "negative-only", "positive-negative"], default="joint")
    parser.add_argument("--no-visualizations", action="store_false", dest="save_visualizations")
    parser.set_defaults(save_visualizations=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_config_overrides(
        args,
        config,
        {
            "top_k_positive": 1,
            "top_k_negative": 1,
            "negative_lambda": 0.35,
            "positive_weight": 1.0,
            "negative_weight": 0.25,
            "similarity_threshold": 0.5,
            "confidence_scale": 8.0,
            "similarity_weighting": "soft",
            "similarity_temperature": 0.125,
            "retrieval_policy": "uncertainty-aware",
            "uncertainty_threshold": 0.35,
            "uncertainty_scale": 10.0,
            "policy_activation_threshold": 0.05,
            "residual_strength": 0.5,
            "retrieval_mode": "joint",
        },
    )

    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"rssda_eval_{index}"} for index in range(3)]
    if not records:
        raise FileNotFoundError("No validation records found for RSS-DA validation.")

    device = resolve_runtime_device(args.device)
    log_runtime_environment(
        "validate_rssda",
        requested_device=args.device,
        resolved_device=device,
        extra={"dummy": bool(args.dummy), "image_size": int(args.image_size)},
    )
    runtime = build_retrieval_runtime(
        memory_bank=args.memory_bank,
        bank_purpose=args.bank_purpose,
        checkpoint=args.checkpoint,
        adapter_checkpoint=args.adapter_checkpoint,
        retriever_checkpoint=args.retriever_checkpoint,
        similarity_checkpoint=args.similarity_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        lora_stage=args.lora_stage,
        device=device,
        precision=args.precision,
        image_size=args.image_size,
        top_k=None,
        top_k_positive=args.top_k_positive,
        top_k_negative=args.top_k_negative,
        negative_lambda=args.negative_lambda,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        similarity_threshold=args.similarity_threshold,
        confidence_scale=args.confidence_scale,
        similarity_weighting=args.similarity_weighting,
        similarity_temperature=args.similarity_temperature,
        retrieval_policy=args.retrieval_policy,
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_scale=args.uncertainty_scale,
        policy_activation_threshold=args.policy_activation_threshold,
        residual_strength=args.residual_strength,
        allow_dummy_fallback=args.dummy,
        continual_bank_root=args.continual_bank_root,
        site_bank_mode=args.site_bank_mode,
    )
    bank_context = runtime.bank_context
    if _contains_polypgen_records(records) and _bank_has_polypgen_leakage(bank_context.resolved_path):
        raise RuntimeError("PolypGen leakage detected in retrieval bank for external evaluation.")

    output_dir = ensure_dir(args.output_dir)
    vis_dir = ensure_dir(output_dir / "visualizations") if args.save_visualizations else None
    run_manifest_path = output_dir / "run_manifest.json"
    validation_log_path = output_dir / "validation_log.jsonl"
    run_manifest_path.write_text(
        json.dumps(
            {
                **vars(args),
                "bank_context": {
                    "resolved_path": str(bank_context.resolved_path),
                    "source": bank_context.source,
                    "cache_root": None if bank_context.cache_root is None else str(bank_context.cache_root),
                    "stats": bank_context.stats,
                    "warnings": list(bank_context.warnings),
                },
                "site_bank_mode": args.site_bank_mode,
                "continual_bank_root": None if runtime.continual_bank_root is None else str(runtime.continual_bank_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    validation_log_path.write_text("", encoding="utf-8")
    loader = DataLoader(SplitSegmentationDataset(records, args.image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)

    rows = []
    diagnostics_rows = []
    region_diagnostics_rows = []
    summary = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if args.dummy or batch_index == 1 or batch_index % 10 == 0:
                print(json.dumps({"progress": {"script": "validate_rssda", "step": batch_index, "steps": len(loader), "device": device, "dummy": bool(args.dummy)}}, ensure_ascii=True), flush=True)
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            record = dict(batch["records"][0])
            image_id = str(record["image_id"])

            baseline, query_feature = infer_query_feature(runtime, images=images, boxes=boxes, text_prompt=batch["text_prompt"])
            query_source = infer_source_domain(
                dataset_name=str(record.get("dataset_name", "")),
                image_id=str(record.get("image_id", "")),
                image_path=str(record.get("image_path", "")),
                mask_path=str(record.get("mask_path", "")),
            )
            retrieval = resolve_retrieval(
                runtime,
                query_feature,
                top_k_positive=args.top_k_positive,
                top_k_negative=args.top_k_negative,
                retrieval_mode=args.retrieval_mode,
                query_source=query_source,
                prefer_cross_domain_positive=True,
                sample_metadata=record,
            )
            adapted, retrieval_prior, adapter_aux, similarity = run_retrieval_forward(
                runtime,
                images=images,
                boxes=boxes,
                text_prompt=batch["text_prompt"],
                query_feature=query_feature,
                retrieval=retrieval,
                retrieval_mode=args.retrieval_mode,
                baseline_mask_logits=baseline.get("mask_logits"),
            )

            alt_positive = retrieval["positive_features"][:, 1:, :] if retrieval["positive_features"].shape[1] > 1 else retrieval["positive_features"]
            alt_weights = retrieval["positive_weights"][:, 1:] if retrieval["positive_weights"].shape[1] > 1 else retrieval["positive_weights"]
            alt_retrieval = dict(retrieval)
            alt_retrieval["positive_features"] = alt_positive
            alt_retrieval["positive_weights"] = alt_weights
            if "positive_score_tensor" in retrieval:
                alt_retrieval["positive_score_tensor"] = retrieval["positive_score_tensor"][:, 1:] if retrieval["positive_score_tensor"].shape[1] > 1 else retrieval["positive_score_tensor"]
            adapted_alt, alt_prior, _, _ = run_retrieval_forward(
                runtime,
                images=images,
                boxes=boxes,
                text_prompt=batch["text_prompt"],
                query_feature=query_feature,
                retrieval=alt_retrieval,
                retrieval_mode=args.retrieval_mode,
                baseline_mask_logits=baseline.get("mask_logits"),
            )

            metrics = compute_segmentation_metrics(adapted["mask_logits"], masks)
            baseline_metrics = compute_segmentation_metrics(baseline["mask_logits"], masks)
            alt_metrics = compute_segmentation_metrics(adapted_alt["mask_logits"], masks)
            sensitivity = {
                "Dice Delta": abs(metrics["Dice"] - alt_metrics["Dice"]),
                "Mask Area Delta": abs(_mask_area(adapted["mask_logits"]) - _mask_area(adapted_alt["mask_logits"])),
                "Boundary F1 Delta": abs(metrics["Boundary F1"] - alt_metrics["Boundary F1"]),
                "Attention Proxy Delta": float((retrieval_prior.get("spatial_bias_map", torch.zeros(1, 1, 1, 1, device=device)) - alt_prior.get("spatial_bias_map", torch.zeros(1, 1, 1, 1, device=device))).abs().mean().item()),
                "Baseline Dice Delta": metrics["Dice"] - baseline_metrics["Dice"],
            }
            multi_bank_fusion = retrieval.get("multi_bank_fusion", {}) if isinstance(retrieval.get("multi_bank_fusion"), dict) else {}
            row = {
                "image_id": image_id,
                "image_path": str(record.get("image_path", "")),
                "mask_path": str(record.get("mask_path", "")),
                "dataset_name": str(record.get("dataset_name", "")),
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "retrieval_vs_baseline": {
                    "Dice Delta": metrics["Dice"] - baseline_metrics["Dice"],
                    "Boundary F1 Delta": metrics["Boundary F1"] - baseline_metrics["Boundary F1"],
                    "FNR Delta": metrics["False Negative Rate"] - baseline_metrics["False Negative Rate"],
                    "HD95 Delta": metrics["HD95"] - baseline_metrics["HD95"],
                    "ASSD Delta": metrics["ASSD"] - baseline_metrics["ASSD"],
                },
                "retrieval_sensitivity": sensitivity,
                "retrieval_mode": args.retrieval_mode,
                "lesion_area": float(masks.sum().item()),
                "prediction_area": _mask_area(adapted["mask_logits"]),
                "selected_positive": [entry.prototype_id for entry in retrieval["positive_entries"][0]],
                "selected_negative": [entry.prototype_id for entry in retrieval["negative_entries"][0]],
                "multi_bank_fusion": {
                    "site_id": multi_bank_fusion.get("site_id"),
                    "parsed_site_id": multi_bank_fusion.get("parsed_site_id", multi_bank_fusion.get("site_id")),
                    "expected_site_bank": multi_bank_fusion.get("expected_site_bank"),
                    "fallback_reason": multi_bank_fusion.get("fallback_reason"),
                    "selected_bank_paths": list(multi_bank_fusion.get("selected_bank_paths", [])),
                    "train_contribution": float(multi_bank_fusion.get("train_contribution", torch.tensor([1.0]))[0].item()) if isinstance(multi_bank_fusion.get("train_contribution"), torch.Tensor) else float(multi_bank_fusion.get("train_contribution", 1.0)),
                    "site_contribution": float(multi_bank_fusion.get("site_contribution", torch.tensor([0.0]))[0].item()) if isinstance(multi_bank_fusion.get("site_contribution"), torch.Tensor) else float(multi_bank_fusion.get("site_contribution", 0.0)),
                    "final_fusion_weight": float(multi_bank_fusion.get("final_fusion_weight", torch.tensor([0.0]))[0].item()) if isinstance(multi_bank_fusion.get("final_fusion_weight"), torch.Tensor) else float(multi_bank_fusion.get("final_fusion_weight", 0.0)),
                },
            }
            diagnostics_rows.append(
                build_retrieval_diagnostics(
                    image_id=image_id,
                    retrieval=retrieval,
                    adapter_aux=adapter_aux,
                    outputs=adapted,
                    batch_index=0,
                    sample_metadata=record,
                )
            )
            region_diagnostics_rows.append(
                build_region_retrieval_diagnostics(
                    image_id=image_id,
                    retrieval=retrieval,
                    adapter_aux=adapter_aux,
                    baseline_mask_logits=baseline.get("mask_logits"),
                    corrected_mask_logits=adapted["mask_logits"],
                    gt_mask=masks,
                    sample_metadata=record,
                )
            )
            rows.append(row)
            with validation_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            for key, value in metrics.items():
                summary[key] = summary.get(key, 0.0) + value
            for key, value in sensitivity.items():
                summary[key] = summary.get(key, 0.0) + value

            if vis_dir is not None:
                Image.fromarray((images[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(vis_dir / f"{image_id}_query.png")
                _save_gray(vis_dir / f"{image_id}_gt.png", masks[0, 0])
                _save_gray(vis_dir / f"{image_id}_pred.png", torch.sigmoid(adapted["mask_logits"][0, 0]))
                _save_heatmap(vis_dir / f"{image_id}_positive_heatmap.png", similarity["positive_heatmap"][0, 0])
                _save_heatmap(vis_dir / f"{image_id}_negative_heatmap.png", similarity["negative_heatmap"][0, 0])
                if "spatial_bias_map" in retrieval_prior:
                    _save_heatmap(vis_dir / f"{image_id}_final_spatial_prior.png", retrieval_prior["spatial_bias_map"][0, 0])
                _save_gray(vis_dir / f"{image_id}_pred_alt.png", torch.sigmoid(adapted_alt["mask_logits"][0, 0]))
                save_region_retrieval_panel(
                    query_image=images,
                    baseline_mask_logits=baseline["mask_logits"],
                    corrected_mask_logits=adapted["mask_logits"],
                    adapter_aux=adapter_aux,
                    retrieval=retrieval,
                    gt_mask=masks,
                    output_path=vis_dir / f"{image_id}_region_retrieval_panel.png",
                )

                exemplar_dir = ensure_dir(vis_dir / f"{image_id}_topk")
                for exemplar_index, entry in enumerate(retrieval["positive_entries"][0]):
                    if entry.crop_path and Path(entry.crop_path).exists():
                        shutil.copyfile(entry.crop_path, exemplar_dir / f"positive_{exemplar_index}_{Path(entry.crop_path).name}")
                for exemplar_index, entry in enumerate(retrieval["negative_entries"][0]):
                    if entry.crop_path and Path(entry.crop_path).exists():
                        shutil.copyfile(entry.crop_path, exemplar_dir / f"negative_{exemplar_index}_{Path(entry.crop_path).name}")

    summary = {key: value / max(len(rows), 1) for key, value in summary.items()}
    diagnostics_summary = summarize_retrieval_diagnostics(diagnostics_rows)
    region_diagnostics_summary = summarize_region_retrieval_diagnostics(region_diagnostics_rows)
    summary["retrieval_mode"] = args.retrieval_mode
    summary["top_k_positive"] = args.top_k_positive
    summary["top_k_negative"] = args.top_k_negative
    summary["positive_weight"] = args.positive_weight
    summary["negative_weight"] = args.negative_weight
    summary["similarity_threshold"] = args.similarity_threshold
    summary["confidence_scale"] = args.confidence_scale
    summary["similarity_weighting"] = args.similarity_weighting
    summary["similarity_temperature"] = args.similarity_temperature
    summary["retrieval_policy"] = args.retrieval_policy
    summary["uncertainty_threshold"] = args.uncertainty_threshold
    summary["uncertainty_scale"] = args.uncertainty_scale
    summary["policy_activation_threshold"] = args.policy_activation_threshold
    summary["residual_strength"] = args.residual_strength
    summary["site_bank_mode"] = args.site_bank_mode
    summary["continual_bank_root"] = None if runtime.continual_bank_root is None else str(runtime.continual_bank_root)
    summary["retrieval_diagnostics_summary"] = diagnostics_summary
    summary["region_retrieval_diagnostics_summary"] = region_diagnostics_summary
    summary["bank_context"] = {
        "resolved_path": str(bank_context.resolved_path),
        "source": bank_context.source,
        "cache_root": None if bank_context.cache_root is None else str(bank_context.cache_root),
        "stats": bank_context.stats,
        "warnings": list(bank_context.warnings),
    }
    summary["artifacts"] = {
        "per_image_metrics": str(output_dir / "per_image_metrics.jsonl"),
        "retrieval_diagnostics": str(output_dir / "retrieval_diagnostics.jsonl"),
        "region_retrieval_diagnostics": str(output_dir / "region_retrieval_diagnostics.jsonl"),
        "visualizations": None if vis_dir is None else str(vis_dir),
    }
    (output_dir / "per_image_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    write_retrieval_diagnostics(output_dir / "retrieval_diagnostics.jsonl", diagnostics_rows)
    write_region_retrieval_diagnostics(output_dir / "region_retrieval_diagnostics.jsonl", region_diagnostics_rows)
    calibration_report_path = write_retrieval_calibration_report(output_dir, diagnostics_rows, rows)
    summary["artifacts"]["retrieval_calibration"] = str(calibration_report_path)
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())