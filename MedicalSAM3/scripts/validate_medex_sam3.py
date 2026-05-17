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


def _resolve_hidden_dim(base_model: torch.nn.Module) -> int:
    return int(getattr(base_model, "hidden_dim", getattr(base_model, "_medex_hidden_dim", getattr(base_model, "embed_dim", 128))))


def _prompt_tokens_from_bank(
    bank: ExemplarMemoryBank | None,
    builder: PrototypeBuilder,
    prompt_adapter: ExemplarPromptAdapter,
    warmup_outputs: dict[str, object],
) -> tuple[torch.Tensor | None, dict[str, object]]:
    if bank is None or not bank.trainable_items:
        return None, {}

    query = warmup_outputs["query_embedding"][0]
    positive = builder._build_single_type(query, bank.get_items(type="positive", human_verified=True), 3)  # noqa: SLF001
    negative = builder._build_single_type(query, bank.get_items(type="negative", human_verified=True), 1)  # noqa: SLF001
    boundary = builder._build_single_type(query, bank.get_items(type="boundary", human_verified=True), 1)  # noqa: SLF001
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
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--dummy", action="store_true")
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
        enable_medical_adapter=True,
        enable_boundary_adapter=True,
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
    loader = DataLoader(SplitSegmentationDataset(records, args.image_size), batch_size=1, shuffle=False, collate_fn=collate_batch)
    output_dir = ensure_dir(args.output_dir)
    vis_dir = ensure_dir(output_dir / "visualizations")
    per_image_path = output_dir / "per_image_metrics.jsonl"
    summary_path = output_dir / "summary_metrics.json"
    failure_path = output_dir / "failure_cases.json"
    per_image_path.write_text("", encoding="utf-8")

    metrics_sum = {}
    failure_rows = []
    saved_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            warmup = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
            prompt_tokens, selection = _prompt_tokens_from_bank(bank, builder, prompt_adapter, warmup)
            outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], exemplar_prompt_tokens=prompt_tokens)
            metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
            mean_confidence = float(outputs["scores"].mean().item())
            metrics["mean confidence"] = mean_confidence
            metrics["Prompt Sensitivity"] = float(torch.var(outputs["masks"]).item()) if prompt_tokens is not None else 0.0
            row = {
                "image_id": batch["records"][0]["image_id"],
                "metrics": metrics,
                "mode": "external" if args.external_test else "fold",
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
            if metrics.get("Dice", 0.0) < 0.5:
                failure_rows.append(row)

    summary = {key: value / max(len(saved_rows), 1) for key, value in metrics_sum.items()}
    with per_image_path.open("w", encoding="utf-8") as handle:
        for row in saved_rows:
            handle.write(json.dumps(row) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failure_path.write_text(json.dumps(failure_rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
