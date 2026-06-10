"""Validate MedEx-SAM3 on fold validation or PolypGen external test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MedicalSAM3.adapters.exemplar_prompt_adapter import ExemplarPromptAdapter
from MedicalSAM3.adapters.lora import load_lora_weights
from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank
from MedicalSAM3.exemplar.prototype_builder import PrototypeBuilder
from MedicalSAM3.retrieval.region_uncertainty import entropy_from_logits
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    MedExSam3SegmentationModel,
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    ensure_dir,
    read_records,
)
from MedicalSAM3.yolo_adapter.cli import add_yolo_bbox_args, build_box_provider_from_args


def _overlay_boundary(image: torch.Tensor, pred_mask: torch.Tensor) -> Image.Image:
    array = (image.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    array[mask > 0, 1] = 255
    return Image.fromarray(array)


def _resolve_hidden_dim(base_model: torch.nn.Module) -> int:
    return int(getattr(base_model, "hidden_dim", getattr(base_model, "_medex_hidden_dim", getattr(base_model, "embed_dim", 128))))


def _mask_area_ratio(mask_logits: torch.Tensor) -> float:
    mask = (torch.sigmoid(mask_logits) > 0.5).float()
    return float(mask.flatten(1).mean(dim=1).mean().item())


def _hard_case_gate_decision(
    warmup_outputs: dict[str, object],
    *,
    enabled: bool,
    max_baseline_confidence: float,
    min_entropy: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, object]:
    mask_logits = warmup_outputs["mask_logits"]
    if not isinstance(mask_logits, torch.Tensor):
        raise TypeError("warmup_outputs['mask_logits'] must be a tensor")
    score = warmup_outputs.get("scores")
    baseline_confidence = float(score.detach().float().mean().item()) if isinstance(score, torch.Tensor) else 0.0
    mean_entropy = float(entropy_from_logits(mask_logits).detach().float().mean().item())
    area_ratio = _mask_area_ratio(mask_logits)

    reasons: list[str] = []
    if baseline_confidence <= max_baseline_confidence:
        reasons.append("low_baseline_confidence")
    if mean_entropy >= min_entropy:
        reasons.append("high_entropy")
    if area_ratio <= min_area_ratio:
        reasons.append("tiny_mask")
    if area_ratio >= max_area_ratio:
        reasons.append("large_mask")

    use_exemplar = (not enabled) or bool(reasons)
    if not enabled:
        reasons = ["gate_disabled"]
    elif not reasons:
        reasons = ["baseline_preserved"]
    return {
        "enabled": enabled,
        "use_exemplar": use_exemplar,
        "reasons": reasons,
        "baseline_confidence": baseline_confidence,
        "mean_entropy": mean_entropy,
        "area_ratio": area_ratio,
        "thresholds": {
            "max_baseline_confidence": max_baseline_confidence,
            "min_entropy": min_entropy,
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
        },
    }


def _prompt_tokens_from_bank(
    bank: ExemplarMemoryBank | None,
    builder: PrototypeBuilder,
    prompt_adapter: ExemplarPromptAdapter,
    warmup_outputs: dict[str, object],
    *,
    top_k_positive: int,
    top_k_negative: int,
    top_k_boundary: int,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    if bank is None or not bank.trainable_items:
        return None, {}

    query = warmup_outputs["query_embedding"][0]
    positive = builder._build_single_type(query, bank.get_items(type="positive", human_verified=True), top_k_positive)  # noqa: SLF001
    negative = builder._build_single_type(query, bank.get_items(type="negative", human_verified=True), top_k_negative)  # noqa: SLF001
    boundary = builder._build_single_type(query, bank.get_items(type="boundary", human_verified=True), top_k_boundary)  # noqa: SLF001
    if positive["prototype"] is None:
        return None, {}

    positive_proto = positive["prototype"].unsqueeze(0) if positive["prototype"].dim() == 1 else positive["prototype"].unsqueeze(0)
    negative_proto = None if negative["prototype"] is None else negative["prototype"].unsqueeze(0) if negative["prototype"].dim() == 1 else negative["prototype"].unsqueeze(0)
    boundary_proto = None if boundary["prototype"] is None else boundary["prototype"].unsqueeze(0) if boundary["prototype"].dim() == 1 else boundary["prototype"].unsqueeze(0)
    prompt_tokens, aux = prompt_adapter(
        positive_proto=positive_proto,
        negative_proto=negative_proto,
        boundary_proto=boundary_proto,
        query_feat=warmup_outputs["query_embedding"],
    )
    return prompt_tokens, {
        "positive_ids": positive["selected_item_ids"],
        "negative_ids": negative["selected_item_ids"],
        "boundary_ids": boundary["selected_item_ids"],
        "fusion_weights": aux["fusion_weights"].detach().cpu().tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MedEx-SAM3.")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--external-test", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--memory-bank", default=None)
    parser.add_argument("--prompt-checkpoint", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/eval")
    parser.add_argument("--no-medical-adapter", action="store_true")
    parser.add_argument("--no-boundary-adapter", action="store_true")
    parser.add_argument("--top-k-positive", type=int, default=2)
    parser.add_argument("--top-k-negative", type=int, default=1)
    parser.add_argument("--top-k-boundary", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--hard-case-gate", action="store_true")
    parser.add_argument("--gate-max-baseline-confidence", type=float, default=0.955)
    parser.add_argument("--gate-min-entropy", type=float, default=0.35)
    parser.add_argument("--gate-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--gate-max-area-ratio", type=float, default=0.35)
    parser.add_argument("--dummy", action="store_true")
    add_yolo_bbox_args(parser)
    args = parser.parse_args()

    split_file = Path(args.split_file) if args.split_file else Path(
        "MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt"
        if args.external_test
        else "MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt"
    )
    records = read_records(split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"val_{i}"} for i in range(3)]
    if not records:
        raise FileNotFoundError(f"No validation records found in {split_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = build_official_sam3_image_model(
        args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
        allow_dummy_fallback=args.dummy,
    )
    freeze_model(base_model)
    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        load_lora_weights(base_model, args.lora_checkpoint, strict=False)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    model = MedExSam3SegmentationModel(
        wrapper=wrapper,
        enable_medical_adapter=not args.no_medical_adapter,
        enable_boundary_adapter=not args.no_boundary_adapter,
        embed_dim=_resolve_hidden_dim(base_model),
    ).to(device)
    if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
        adapter_state = torch.load(args.adapter_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(adapter_state, strict=False)
    bank = ExemplarMemoryBank.load(args.memory_bank) if args.memory_bank else None
    prompt_adapter = ExemplarPromptAdapter(_resolve_hidden_dim(base_model)).to(device)
    if args.prompt_checkpoint and Path(args.prompt_checkpoint).exists():
        prompt_adapter.load_state_dict(torch.load(args.prompt_checkpoint, map_location=device, weights_only=False), strict=False)
    builder = PrototypeBuilder()
    box_provider = build_box_provider_from_args(args, default_cache_name="validate_medex_sam3.json")
    loader = DataLoader(
        SplitSegmentationDataset(records, args.image_size, box_provider=box_provider),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
    )
    output_dir = ensure_dir(args.output_dir)
    vis_dir = None if args.no_visualizations else ensure_dir(output_dir / "visualizations")
    per_image_path = output_dir / "per_image_metrics.jsonl"
    per_image_json_path = output_dir / "per_image_metrics.json"
    summary_path = output_dir / "summary_metrics.json"
    failure_path = output_dir / "failure_cases.json"
    delta_distribution_path = output_dir / "delta_dice_distribution.json"
    per_image_path.write_text("", encoding="utf-8")

    metrics_sum = {}
    baseline_metrics_sum = {}
    delta_dice_values: list[float] = []
    gate_used_count = 0
    gate_preserved_count = 0
    failure_rows = []
    saved_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            warmup = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
            baseline_metrics = compute_segmentation_metrics(warmup["mask_logits"], masks)
            gate_decision = _hard_case_gate_decision(
                warmup,
                enabled=args.hard_case_gate,
                max_baseline_confidence=args.gate_max_baseline_confidence,
                min_entropy=args.gate_min_entropy,
                min_area_ratio=args.gate_min_area_ratio,
                max_area_ratio=args.gate_max_area_ratio,
            )
            prompt_tokens = None
            selection: dict[str, object] = {}
            if bool(gate_decision["use_exemplar"]):
                gate_used_count += 1
                prompt_tokens, selection = _prompt_tokens_from_bank(
                    bank,
                    builder,
                    prompt_adapter,
                    warmup,
                    top_k_positive=args.top_k_positive,
                    top_k_negative=args.top_k_negative,
                    top_k_boundary=args.top_k_boundary,
                )
                outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], exemplar_prompt_tokens=prompt_tokens)
            else:
                gate_preserved_count += 1
                outputs = warmup
            metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
            delta_dice = float(metrics.get("Dice", 0.0) - baseline_metrics.get("Dice", 0.0))
            delta_dice_values.append(delta_dice)
            mean_confidence = float(outputs["scores"].mean().item())
            metrics["mean confidence"] = mean_confidence
            metrics["Prompt Sensitivity"] = float(torch.var(outputs["masks"]).item()) if prompt_tokens is not None else 0.0
            row = {
                "image_id": batch["records"][0]["image_id"],
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "delta_dice": delta_dice,
                "mode": "external" if args.external_test else "fold",
                "selected_exemplars": selection,
                "hard_case_gate": gate_decision,
            }
            saved_rows.append(row)
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            for key, value in baseline_metrics.items():
                baseline_metrics_sum[key] = baseline_metrics_sum.get(key, 0.0) + value

            image_id = batch["records"][0]["image_id"]
            if vis_dir is not None:
                pred_mask = F.interpolate(outputs["masks"], size=images.shape[-2:], mode="bilinear", align_corners=False)
                Image.fromarray((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(vis_dir / f"{image_id}_image.png")
                Image.fromarray((masks[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(vis_dir / f"{image_id}_gt.png")
                Image.fromarray((pred_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(vis_dir / f"{image_id}_pred.png")
                _overlay_boundary(images[0].cpu(), pred_mask[0, 0].cpu()).save(vis_dir / f"{image_id}_boundary_overlay.png")
            if metrics.get("Dice", 0.0) < 0.5:
                failure_rows.append(row)

    summary = {key: value / max(len(saved_rows), 1) for key, value in metrics_sum.items()}
    baseline_summary = {key: value / max(len(saved_rows), 1) for key, value in baseline_metrics_sum.items()}
    delta_bins = {
        "<=-0.10": 0,
        "(-0.10,-0.03]": 0,
        "(-0.03,0]": 0,
        "(0,0.03]": 0,
        "(0.03,0.10]": 0,
        ">0.10": 0,
    }
    for value in delta_dice_values:
        if value <= -0.10:
            delta_bins["<=-0.10"] += 1
        elif value <= -0.03:
            delta_bins["(-0.10,-0.03]"] += 1
        elif value <= 0.0:
            delta_bins["(-0.03,0]"] += 1
        elif value <= 0.03:
            delta_bins["(0,0.03]"] += 1
        elif value <= 0.10:
            delta_bins["(0.03,0.10]"] += 1
        else:
            delta_bins[">0.10"] += 1
    delta_distribution = {
        "count": len(delta_dice_values),
        "mean_delta_dice": float(sum(delta_dice_values) / max(len(delta_dice_values), 1)),
        "min_delta_dice": float(min(delta_dice_values)) if delta_dice_values else 0.0,
        "max_delta_dice": float(max(delta_dice_values)) if delta_dice_values else 0.0,
        "positive_delta_count": sum(1 for value in delta_dice_values if value > 0.0),
        "negative_delta_count": sum(1 for value in delta_dice_values if value < 0.0),
        "bins": delta_bins,
        "target_mean_delta_positive": float(sum(delta_dice_values) / max(len(delta_dice_values), 1)) > 0.0,
    }
    summary["baseline_Dice"] = baseline_summary.get("Dice", 0.0)
    summary["delta_Dice"] = delta_distribution["mean_delta_dice"]
    summary["hard_case_gate_enabled"] = bool(args.hard_case_gate)
    summary["hard_case_gate_used_count"] = gate_used_count
    summary["hard_case_gate_preserved_count"] = gate_preserved_count
    summary["hard_case_gate_used_ratio"] = gate_used_count / max(len(saved_rows), 1)
    with per_image_path.open("w", encoding="utf-8") as handle:
        for row in saved_rows:
            handle.write(json.dumps(row) + "\n")
    per_image_json_path.write_text(json.dumps(saved_rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failure_path.write_text(json.dumps(failure_rows, indent=2), encoding="utf-8")
    delta_distribution_path.write_text(json.dumps(delta_distribution, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
