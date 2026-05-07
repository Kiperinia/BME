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
from MedicalSAM3.agents.quality_evaluator import QualityEvaluator
from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank
from MedicalSAM3.exemplar.prototype_builder import PrototypeBuilder
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


def _overlay_boundary(image: torch.Tensor, pred_mask: torch.Tensor) -> Image.Image:
    array = (image.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    array[mask > 0, 1] = 255
    return Image.fromarray(array)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MedEx-SAM3.")
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt")
    parser.add_argument("--external-file", default="MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--memory-bank", default=None)
    parser.add_argument("--prompt-checkpoint", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/validation")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--mode", choices=["fold", "external"], default="fold")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    records = read_records(args.split_file if args.mode == "fold" else args.external_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"val_{i}"} for i in range(3)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = build_official_sam3_image_model(args.checkpoint, device=device, dtype=args.precision, compile_model=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    model = MedExSam3SegmentationModel(wrapper=wrapper, enable_medical_adapter=True, enable_boundary_adapter=True, embed_dim=int(getattr(base_model, "embed_dim", 128))).to(device)
    bank = ExemplarMemoryBank.load(args.memory_bank) if args.memory_bank else None
    prompt_adapter = ExemplarPromptAdapter(int(getattr(base_model, "embed_dim", 128))).to(device)
    if args.prompt_checkpoint and Path(args.prompt_checkpoint).exists():
        prompt_adapter.load_state_dict(torch.load(args.prompt_checkpoint, map_location=device, weights_only=False), strict=False)
    builder = PrototypeBuilder()
    evaluator = QualityEvaluator()
    loader = DataLoader(SplitSegmentationDataset(records, args.image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)
    output_dir = ensure_dir(args.output_dir)
    vis_dir = ensure_dir(output_dir / "visualizations")
    metrics_path = output_dir / ("external_polypgen_metrics.json" if args.mode == "external" else "per_image_metrics.jsonl")
    failure_path = output_dir / "failure_cases.jsonl"
    metrics_path.write_text("" if metrics_path.suffix == ".jsonl" else "{}", encoding="utf-8")
    failure_path.write_text("", encoding="utf-8")

    metrics_sum = {}
    saved_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            prompt_tokens = None
            selection = {}
            if bank and bank.trainable_items:
                warmup = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], gt_mask=masks)
                query = warmup["query_embedding"][0]
                positive = builder._build_single_type(query, bank.get_items(type="positive", human_verified=True), 3)  # noqa: SLF001
                negative = builder._build_single_type(query, bank.get_items(type="negative", human_verified=True), 1)  # noqa: SLF001
                boundary = builder._build_single_type(query, bank.get_items(type="boundary", human_verified=True), 1)  # noqa: SLF001
                if positive["prototype"] is not None:
                    prompt_tokens, aux = prompt_adapter(
                        positive_proto=positive["prototype"].unsqueeze(0) if positive["prototype"].dim() == 1 else positive["prototype"].unsqueeze(0),
                        negative_proto=None if negative["prototype"] is None else negative["prototype"].unsqueeze(0),
                        boundary_proto=None if boundary["prototype"] is None else boundary["prototype"].unsqueeze(0),
                        query_feat=warmup["query_embedding"],
                    )
                    selection = {
                        "positive_ids": positive["selected_item_ids"],
                        "negative_ids": negative["selected_item_ids"],
                        "boundary_ids": boundary["selected_item_ids"],
                        "fusion_weights": aux["fusion_weights"].detach().cpu().tolist(),
                    }
            outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], exemplar_prompt_tokens=prompt_tokens, gt_mask=masks)
            metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
            quality = evaluator.evaluate(outputs["mask_logits"], outputs["masks"], outputs["scores"], gt_mask=masks)
            metrics["Prompt Sensitivity"] = float(torch.var(outputs["masks"]).item()) if prompt_tokens is not None else 0.0
            row = {
                "image_id": batch["records"][0]["image_id"],
                "metrics": metrics,
                "quality": quality,
                "selected_exemplars": selection,
            }
            saved_rows.append(row)
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

            image_id = batch["records"][0]["image_id"]
            pred_mask = F.interpolate(outputs["masks"], size=images.shape[-2:], mode="bilinear", align_corners=False)
            Image.fromarray((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(vis_dir / f"{image_id}_image.png")
            Image.fromarray((masks[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(vis_dir / f"{image_id}_gt.png")
            Image.fromarray((pred_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(vis_dir / f"{image_id}_pred.png")
            _overlay_boundary(images[0].cpu(), pred_mask[0, 0].cpu()).save(vis_dir / f"{image_id}_boundary_overlay.png")
            if quality["failure_type"] != "uncertain":
                with failure_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

    summary = {key: value / max(len(saved_rows), 1) for key, value in metrics_sum.items()}
    if args.mode == "external":
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        with metrics_path.open("w", encoding="utf-8") as handle:
            for row in saved_rows:
                handle.write(json.dumps(row) + "\n")
        (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
