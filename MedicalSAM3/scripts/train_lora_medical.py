"""Train LoRA and medical adapters on MedEx-SAM3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, mark_only_lora_as_trainable, save_lora_weights
from MedicalSAM3.exemplar.losses import MedExLossComposer
from MedicalSAM3.sam3_official.build_model import (
    build_official_sam3_image_model,
    count_trainable_parameters,
    freeze_model,
    print_trainable_parameters,
)
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    MedExSam3SegmentationModel,
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    dump_config,
    ensure_dir,
    load_config,
    read_records,
    seed_everything,
)


def _device_from_args(precision: str) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return device, dtype_map.get(precision, torch.float32)


def _scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())

    return LambdaLR(optimizer, lr_lambda)


def _load_fold_records(split_dir: Path, fold: int, dummy: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_records = read_records(split_dir / f"fold_{fold}" / "train_ids.txt")
    val_records = read_records(split_dir / f"fold_{fold}" / "val_ids.txt")
    if dummy and not train_records:
        train_records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"train_{i}"} for i in range(8)]
        val_records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"val_{i}"} for i in range(2)]
    return train_records, val_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LoRA and medical adapters for MedEx-SAM3.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root", default="MedicalSAM3/data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--enable-vision-lora", action="store_true")
    parser.add_argument("--enable-detector-lora", action="store_true")
    parser.add_argument("--enable-mask-decoder-lora", action="store_true")
    parser.add_argument("--enable-boundary-adapter", action="store_true")
    parser.add_argument("--enable-msfa-adapter", action="store_true")
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--split-dir", default="MedicalSAM3/outputs/medex_sam3/splits")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output_dir = ensure_dir(Path(args.output_dir) / f"fold_{args.fold}")
    split_dir = Path(config.get("split_dir", args.split_dir))
    train_records, val_records = _load_fold_records(split_dir, args.fold, args.dummy)
    if not train_records:
        raise FileNotFoundError("No training records found. Run prepare_5fold_polyp.py first or use --dummy.")

    device, autocast_dtype = _device_from_args(args.precision)
    base_model = build_official_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
    )
    freeze_model(base_model)
    target_scopes = []
    if args.enable_vision_lora or not any([args.enable_vision_lora, args.enable_detector_lora, args.enable_mask_decoder_lora]):
        target_scopes.append("vision_encoder")
    if args.enable_detector_lora:
        target_scopes.append("detector_decoder")
    if args.enable_mask_decoder_lora or not any([args.enable_vision_lora, args.enable_detector_lora, args.enable_mask_decoder_lora]):
        target_scopes.append("mask_decoder")
    lora_config = LoRAConfig(target_scopes=target_scopes)
    apply_lora_to_model(base_model, lora_config)
    mark_only_lora_as_trainable(base_model)
    if args.freeze_text_encoder:
        for name, parameter in base_model.named_parameters():
            if "text_encoder" in name:
                parameter.requires_grad = False

    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    embed_dim = int(getattr(base_model, "embed_dim", 128))
    model = MedExSam3SegmentationModel(
        wrapper=wrapper,
        enable_medical_adapter=bool(config.get("enable_medical_adapter", False)),
        enable_msfa_adapter=args.enable_msfa_adapter,
        enable_boundary_adapter=args.enable_boundary_adapter,
        embed_dim=embed_dim,
    ).to(device)
    print_trainable_parameters(model)
    trainable, total, ratio = count_trainable_parameters(model)

    train_loader = DataLoader(
        SplitSegmentationDataset(train_records, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        SplitSegmentationDataset(val_records, args.image_size),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = _scheduler(optimizer, total_steps=total_steps, warmup_steps=max(total_steps // 10, 1))
    criterion = MedExLossComposer(w_contrast=0.0, w_neg=0.0, w_consistency=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda" and args.precision in {"fp16", "bf16"})

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_dice = float(state.get("best_dice", -1.0))

    log_path = output_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda" and args.precision in {"fp16", "bf16"}):
                outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], gt_mask=masks)
                loss, aux = criterion(outputs["mask_logits"], masks)
                if outputs["adapter_aux"].get("boundary_loss") is not None:
                    loss = loss + 0.1 * outputs["adapter_aux"]["boundary_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            log_path.write_text("", encoding="utf-8") if not log_path.exists() else None
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "loss": float(loss.item()), "lr": scheduler.get_last_lr()[0]}) + "\n")

        model.eval()
        metrics_sum: dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                masks = batch["masks"].to(device)
                boxes = batch["boxes"].to(device)
                outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], gt_mask=masks)
                metrics = compute_segmentation_metrics(outputs["mask_logits"], masks)
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
        val_metrics = {key: value / max(len(val_loader), 1) for key, value in metrics_sum.items()}
        val_metrics["trainable_ratio"] = ratio
        val_metrics["trainable_parameters"] = trainable
        val_metrics["total_parameters"] = total
        (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

        checkpoint = {
            "epoch": epoch,
            "best_dice": best_dice,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics.get("Dice", 0.0) >= best_dice:
            best_dice = val_metrics.get("Dice", 0.0)
            save_lora_weights(model, output_dir / "best_lora.pt")
            torch.save(
                {
                    key: value
                    for key, value in model.state_dict().items()
                    if "medical_adapter" in key or "msfa_adapter" in key or "boundary_adapter" in key or "refine_head" in key
                },
                output_dir / "best_adapter.pt",
            )
        dump_config(output_dir / "config_used.yaml", {
            **config,
            "fold": args.fold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "precision": args.precision,
            "dummy": args.dummy,
        })

    print(json.dumps({"output_dir": str(output_dir), "best_dice": best_dice, "trainable_ratio": ratio}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
