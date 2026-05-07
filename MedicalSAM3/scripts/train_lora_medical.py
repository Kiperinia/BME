"""Train LoRA and medical adapters on MedEx-SAM3 with strict preflight gates."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return device, dtype_map.get(precision, torch.float32)


def _autocast_enabled(device: str, precision: str) -> bool:
    return device == "cuda" and precision in {"fp16", "bf16"}


def _scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())

    return LambdaLR(optimizer, lr_lambda)


def _default_target_scopes(stage: str) -> list[str]:
    normalized = stage.lower()
    if normalized == "stage_a":
        return ["vision_encoder", "mask_decoder"]
    if normalized == "stage_b":
        return ["detector_decoder", "prompt_encoder", "exemplar_projection"]
    if normalized == "stage_c":
        return ["detector_encoder", "detector_decoder"]
    raise ValueError(f"Unsupported stage: {stage}")


def _resolve_target_scopes(args: argparse.Namespace) -> list[str]:
    scopes = set(_default_target_scopes(args.stage))
    if args.enable_vision_lora:
        scopes.add("vision_encoder")
    if args.enable_detector_lora:
        scopes.update(["detector_encoder", "detector_decoder"])
    if args.enable_mask_decoder_lora:
        scopes.add("mask_decoder")
    return sorted(scopes)


def _contains_polypgen(records: list[dict[str, Any]]) -> bool:
    return any("polypgen" in str(record.get("dataset_name", "")).lower() for record in records)


def _read_split_records(split_dir: Path, fold: int) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    fold_dir = split_dir / f"fold_{fold}"
    train_file = fold_dir / "train_ids.txt"
    val_file = fold_dir / "val_ids.txt"
    train_records = read_records(train_file)
    val_records = read_records(val_file)
    return train_file, val_file, train_records, val_records


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _save_adapter_weights(model: torch.nn.Module, path: Path) -> None:
    adapter_state = {
        key: value
        for key, value in model.state_dict().items()
        if "medical_adapter" in key or "msfa_adapter" in key or "boundary_adapter" in key or "refine_head" in key
    }
    torch.save(adapter_state, path)


def _iter_limited(loader: DataLoader, max_steps: Optional[int]) -> Iterable[tuple[int, dict[str, Any]]]:
    for step_index, batch in enumerate(loader, start=1):
        if max_steps is not None and step_index > max_steps:
            break
        yield step_index, batch


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "images": batch["images"].to(device),
        "masks": batch["masks"].to(device),
        "boxes": batch["boxes"].to(device),
        "text_prompt": batch["text_prompt"],
        "records": batch["records"],
    }


def _build_training_stack(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: str,
) -> tuple[torch.nn.Module, Sam3TensorForwardWrapper, MedExSam3SegmentationModel, list[str]]:
    base_model = build_official_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
        allow_dummy_fallback=args.allow_dummy,
    )
    freeze_model(base_model)

    lora_config = LoRAConfig(
        stage=args.stage,
        target_scopes=_resolve_target_scopes(args),
        min_replaced_modules=args.min_lora_modules,
    )
    replaced = apply_lora_to_model(base_model, lora_config)
    mark_only_lora_as_trainable(base_model)

    if args.freeze_text_encoder:
        for name, parameter in base_model.named_parameters():
            if "text_encoder" in name or "language_backbone" in name:
                parameter.requires_grad = False

    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    embed_dim = wrapper.hidden_dim or getattr(base_model, "embed_dim", None) or 128
    model = MedExSam3SegmentationModel(
        wrapper=wrapper,
        enable_medical_adapter=bool(config.get("enable_medical_adapter", False)),
        enable_msfa_adapter=args.enable_msfa_adapter,
        enable_boundary_adapter=args.enable_boundary_adapter,
        embed_dim=int(embed_dim),
    ).to(device)
    return base_model, wrapper, model, replaced


def _run_preflight(
    args: argparse.Namespace,
    config: dict[str, Any],
    report_path: Path,
    split_dir: Path,
    device: str,
    autocast_dtype: torch.dtype,
) -> dict[str, Any]:
    train_file, val_file, train_records, val_records = _read_split_records(split_dir, args.fold)
    split_exists = train_file.exists() and val_file.exists()
    blocking_issues: list[str] = []
    warnings_list: list[str] = []

    report: dict[str, Any] = {
        "fold": args.fold,
        "official_sam3_build_success": False,
        "used_dummy_fallback": False,
        "split_exists": split_exists,
        "train_records_count": len(train_records),
        "val_records_count": len(val_records),
        "polypgen_leakage_passed": False,
        "lora_replaced_module_count": 0,
        "lora_replaced_modules": [],
        "trainable_parameter_ratio": 0.0,
        "forward_success": False,
        "backward_success": False,
        "ready_for_training": False,
        "blocking_issues": blocking_issues,
        "warnings": warnings_list,
    }

    if not split_exists:
        blocking_issues.append("split files missing")
    if not train_records:
        blocking_issues.append("train split is empty")
    if not val_records:
        blocking_issues.append("val split is empty")

    polypgen_leakage_passed = not _contains_polypgen(train_records) and not _contains_polypgen(val_records)
    report["polypgen_leakage_passed"] = polypgen_leakage_passed
    if not polypgen_leakage_passed:
        blocking_issues.append("PolypGen leakage detected in train/val records")

    try:
        base_model, _wrapper, model, replaced = _build_training_stack(args, config, device=device)
        report["official_sam3_build_success"] = bool(getattr(base_model, "_medex_used_official_sam3", False))
        report["used_dummy_fallback"] = bool(getattr(base_model, "_medex_used_dummy_fallback", False))
        report["lora_replaced_module_count"] = len(replaced)
        report["lora_replaced_modules"] = replaced

        if args.require_official_sam3 and not report["official_sam3_build_success"]:
            blocking_issues.append("official SAM3 build required but unavailable")
        if report["used_dummy_fallback"] and not args.allow_dummy:
            blocking_issues.append("dummy fallback used without --allow-dummy")

        trainable, _, ratio = count_trainable_parameters(model)
        report["trainable_parameter_ratio"] = ratio
        if trainable <= 0 or ratio <= 0.0:
            blocking_issues.append("no trainable parameters after LoRA injection")
        elif ratio > 0.2:
            blocking_issues.append(f"trainable parameter ratio too high: {ratio:.6f}")
        elif ratio > 0.05:
            warnings_list.append(f"trainable parameter ratio is high: {ratio:.6f}")

        if split_exists and train_records:
            criterion = MedExLossComposer(w_contrast=0.0, w_neg=0.0, w_consistency=0.0)
            loader = DataLoader(
                SplitSegmentationDataset(train_records, args.image_size),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )
            first_batch = next(iter(loader))
            runtime_batch = _move_batch(first_batch, device)
            model.train()
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
                enabled=_autocast_enabled(device, args.precision),
            ):
                outputs = model(
                    images=runtime_batch["images"],
                    boxes=runtime_batch["boxes"],
                    text_prompt=runtime_batch["text_prompt"],
                    gt_mask=runtime_batch["masks"],
                )
                if "mask_logits" not in outputs:
                    raise RuntimeError("forward output missing mask_logits")
                loss, _ = criterion(outputs["mask_logits"], runtime_batch["masks"])
            report["forward_success"] = True
            loss.backward()
            report["backward_success"] = any(
                parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
            )
            if not report["backward_success"]:
                blocking_issues.append("backward produced no gradients on trainable parameters")
    except Exception as exc:
        blocking_issues.append(str(exc))

    report["ready_for_training"] = not blocking_issues
    _write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LoRA and medical adapters for MedEx-SAM3.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root", default="MedicalSAM3/data")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3")
    parser.add_argument("--report-dir", default=None)
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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-dummy", action="store_true")
    parser.add_argument("--require-official-sam3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-lora-modules", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--stage", choices=["stage_a", "stage_b", "stage_c"], default="stage_a")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output_dir = ensure_dir(Path(args.output_dir) / f"fold_{args.fold}")
    report_dir = ensure_dir(Path(args.report_dir) if args.report_dir else output_dir)
    preflight_report_path = report_dir / "preflight_report.json"
    split_dir = Path(config.get("split_dir", args.split_dir))

    device, autocast_dtype = _device_from_args(args.precision)
    dump_config(
        output_dir / "config_used.yaml",
        {
            **config,
            "fold": args.fold,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "precision": args.precision,
            "dummy": args.dummy,
            "allow_dummy": args.allow_dummy,
            "require_official_sam3": args.require_official_sam3,
            "min_lora_modules": args.min_lora_modules,
            "max_train_steps": args.max_train_steps,
            "max_val_steps": args.max_val_steps,
            "stage": args.stage,
        },
    )

    preflight_report = _run_preflight(
        args=args,
        config=config,
        report_path=preflight_report_path,
        split_dir=split_dir,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2))
        return 0

    if not preflight_report["ready_for_training"]:
        raise RuntimeError("Preflight failed; see preflight_report.json for blocking issues.")

    train_file, val_file, train_records, val_records = _read_split_records(split_dir, args.fold)
    if args.dummy and (args.max_train_steps or 0) > 2:
        warnings.warn("Dummy local smoke should keep --max-train-steps <= 2.", stacklevel=2)
    if args.dummy and (args.max_val_steps or 0) > 2:
        warnings.warn("Dummy local smoke should keep --max-val-steps <= 2.", stacklevel=2)

    _base_model, _wrapper, model, replaced = _build_training_stack(args, config, device=device)
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

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    effective_train_steps = min(len(train_loader), args.max_train_steps) if args.max_train_steps else len(train_loader)
    total_steps = max(effective_train_steps * args.epochs, 1)
    scheduler = _scheduler(optimizer, total_steps=total_steps, warmup_steps=max(total_steps // 10, 1))
    criterion = MedExLossComposer(w_contrast=0.0, w_neg=0.0, w_consistency=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=_autocast_enabled(device, args.precision))

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
    val_metrics_path = output_dir / "val_metrics.json"
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step_index, batch in _iter_limited(train_loader, args.max_train_steps):
            runtime_batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
                enabled=_autocast_enabled(device, args.precision),
            ):
                outputs = model(
                    images=runtime_batch["images"],
                    boxes=runtime_batch["boxes"],
                    text_prompt=runtime_batch["text_prompt"],
                    gt_mask=runtime_batch["masks"],
                )
                loss, _ = criterion(outputs["mask_logits"], runtime_batch["masks"])
                if outputs["adapter_aux"].get("boundary_loss") is not None:
                    loss = loss + 0.1 * outputs["adapter_aux"]["boundary_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step_index,
                            "global_step": global_step,
                            "loss": float(loss.item()),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    + "\n"
                )

        model.eval()
        metrics_sum: dict[str, float] = {}
        val_steps = 0
        with torch.no_grad():
            for _, batch in _iter_limited(val_loader, args.max_val_steps):
                runtime_batch = _move_batch(batch, device)
                outputs = model(
                    images=runtime_batch["images"],
                    boxes=runtime_batch["boxes"],
                    text_prompt=runtime_batch["text_prompt"],
                    gt_mask=runtime_batch["masks"],
                )
                metrics = compute_segmentation_metrics(outputs["mask_logits"], runtime_batch["masks"])
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                val_steps += 1

        val_metrics = {key: value / max(val_steps, 1) for key, value in metrics_sum.items()}
        val_metrics["epoch"] = epoch
        val_metrics["trainable_ratio"] = ratio
        val_metrics["trainable_parameters"] = trainable
        val_metrics["total_parameters"] = total
        val_metrics["lora_replaced_module_count"] = len(replaced)
        val_metrics_path.write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

        current_dice = float(val_metrics.get("Dice", 0.0))
        if current_dice >= best_dice:
            best_dice = current_dice
            save_lora_weights(model, output_dir / "best_lora.pt")
            _save_adapter_weights(model, output_dir / "best_adapter.pt")

        checkpoint = {
            "epoch": epoch,
            "best_dice": best_dice,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "lora_replaced_modules": replaced,
        }
        torch.save(checkpoint, output_dir / "last.pt")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_dice": best_dice,
                "trainable_ratio": ratio,
                "train_file": str(train_file),
                "val_file": str(val_file),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
