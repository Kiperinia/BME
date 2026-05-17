"""Build retrieval-conditioned prototype banks for RSS-DA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar_bank import PrototypeBankEntry, PrototypeExtractor, RSSDABank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    boundary_band,
    ensure_dir,
    infer_source_domain,
    load_record_tensors,
    log_runtime_environment,
    mask_to_box,
    read_records,
    resolve_runtime_device,
    resolve_feature_map,
    seed_everything,
)


def _infer_polyp_type(image: torch.Tensor, mask: torch.Tensor) -> str:
    fg = image * mask
    bg = image * (1.0 - mask)
    fg_mean = fg.sum(dim=(-2, -1)) / mask.sum().clamp_min(1.0)
    bg_mean = bg.sum(dim=(-2, -1)) / (1.0 - mask).sum().clamp_min(1.0)
    contrast = (fg_mean - bg_mean).abs().mean().item()
    red_bias = (fg_mean[0] - fg_mean[1:].mean()).item()
    box = mask_to_box(mask)
    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    aspect_ratio = min(width, height) / max(width, height)
    if red_bias > 0.12:
        return "bloody polyp"
    if contrast < 0.08:
        return "low contrast polyp"
    if aspect_ratio < 0.55:
        return "flat polyp"
    return "polyp"


def _select_negative_region(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, str]:
    background = (1.0 - mask).clamp(0.0, 1.0)
    gray = image.mean(dim=0, keepdim=True)
    bright = ((image > 0.85).all(dim=0, keepdim=True).float() * background)
    if bright.sum().item() >= 4:
        return bright, "specular highlight"
    grad_x = F.pad((gray[:, :, 1:] - gray[:, :, :-1]).abs(), (1, 0, 0, 0))
    grad_y = F.pad((gray[:, 1:, :] - gray[:, :-1, :]).abs(), (0, 0, 1, 0))
    folds = ((grad_x + grad_y) > 0.18).float() * background
    if folds.sum().item() >= 8:
        return folds, "folds"
    bubble_like = ((gray > 0.7).float() * background)
    if 4 <= bubble_like.sum().item() <= max(background.sum().item() * 0.08, 4.0):
        return bubble_like, "bubbles"
    if float(gray.var().item()) < 0.01:
        return background, "blur"
    return background, "mucosa"


def _boundary_quality(mask: torch.Tensor) -> float:
    band = boundary_band(mask.unsqueeze(0)).squeeze(0)
    return float((band.sum() / mask.sum().clamp_min(1.0)).clamp(0.0, 1.0).item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RSS-DA prototype bank.")
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/rssda_bank")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"bank_{index}"} for index in range(6)]
    if args.max_items is not None:
        records = records[: max(0, args.max_items)]
    if not records:
        raise FileNotFoundError("No records found for RSS-DA bank building.")

    output_dir = ensure_dir(args.output_dir)
    run_manifest_path = output_dir / "run_manifest.json"
    build_log_path = output_dir / "bank_build_log.jsonl"
    run_manifest_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    build_log_path.write_text("", encoding="utf-8")
    device = resolve_runtime_device(args.device)
    log_runtime_environment(
        "build_rssda_bank",
        requested_device=args.device,
        resolved_device=device,
        extra={"dummy": bool(args.dummy), "image_size": int(args.image_size)},
    )
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
    extractor = PrototypeExtractor(wrapper=wrapper)
    bank = RSSDABank()

    for index, record in enumerate(records):
        if args.dummy or index == 0 or (index + 1) % 50 == 0:
            print(json.dumps({"progress": {"script": "build_rssda_bank", "index": index + 1, "total": len(records), "device": device, "dummy": bool(args.dummy)}}, ensure_ascii=True), flush=True)
        source_domain = infer_source_domain(
            dataset_name=str(record.get("dataset_name", "")),
            image_id=str(record.get("image_id", "")),
            image_path=str(record.get("image_path", "")),
            mask_path=str(record.get("mask_path", "")),
        )
        if source_domain == "PolypGen":
            with build_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"index": index, "image_id": record.get("image_id"), "status": "skipped_external"}) + "\n")
            continue
        image, mask = load_record_tensors(record, args.image_size, fallback_index=index)
        images = image.unsqueeze(0).to(device)
        masks = mask.unsqueeze(0).to(device)
        boxes = mask_to_box(mask).unsqueeze(0).to(device)
        positive_proto, outputs = extractor.extract_from_images(images=images, masks=masks, boxes=boxes, text_prompt=["polyp"])
        positive_label = _infer_polyp_type(image, mask)
        negative_mask, negative_label = _select_negative_region(image, mask)
        negative_feature_map = resolve_feature_map(outputs["image_embeddings"], images)
        negative_proto = extractor.extract_from_feature_map(negative_feature_map, negative_mask.unsqueeze(0).to(device))
        confidence = float(outputs["scores"].mean().item())
        boundary_quality = _boundary_quality(mask)
        device_metadata = {
            "device": device,
            "precision": args.precision,
            "used_dummy_fallback": bool(outputs.get("used_dummy_fallback", False)),
        }

        for polarity, prototype, label in [
            ("positive", positive_proto[0], positive_label),
            ("negative", negative_proto[0], negative_label),
        ]:
            entry = PrototypeBankEntry(
                prototype_id=f"{record['image_id']}_{polarity}",
                feature_path="",
                polarity=polarity,
                source_dataset=source_domain,
                polyp_type=label,
                boundary_quality=boundary_quality,
                confidence=confidence,
                image_id=str(record.get("image_id", "")),
                crop_path=str(record.get("image_path", "")) or None,
                mask_path=str(record.get("mask_path", "")) or None,
                device_metadata=device_metadata,
                human_verified=True,
                extra_metadata={
                    "box": boxes[0].detach().cpu().tolist(),
                    "source_group": str(record.get("source_group", record.get("dataset_name", source_domain))),
                },
            )
            stored = extractor.save_prototype(output_dir, prototype, entry)
            bank.add_entry(stored)
            with build_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "image_id": record.get("image_id"),
                            "dataset_name": record.get("dataset_name"),
                            "prototype_id": stored.prototype_id,
                            "polarity": stored.polarity,
                            "polyp_type": stored.polyp_type,
                            "boundary_quality": stored.boundary_quality,
                            "confidence": stored.confidence,
                            "feature_path": stored.feature_path,
                        }
                    )
                    + "\n"
                )

    metadata_path = bank.save(output_dir)
    summary = {
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "positive_count": len(bank.get_entries(polarity="positive")),
        "negative_count": len(bank.get_entries(polarity="negative")),
        "datasets": sorted({entry.source_dataset for entry in bank.entries}),
    }
    (output_dir / "bank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())