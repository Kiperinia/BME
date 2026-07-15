"""在 MedEx-SAM3 上训练 LoRA 与医学适配器，包含严格的预飞行检查。"""

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
from MedicalSAM3.yolo_adapter.cli import add_yolo_bbox_args, build_box_provider_from_args


def _device_from_args(requested_device: str, precision: str) -> tuple[str, torch.dtype]:
    """根据命令行参数解析设备和精度类型。

    参数：
        - requested_device: 请求的设备类型，可选 "auto"/"cuda"/"cpu"
        - precision: 精度字符串，可选 "fp32"/"fp16"/"bf16"

    返回：
        - 由 (设备字符串, 自动混合精度 dtype) 组成的元组
    """
    normalized_device = str(requested_device).strip().lower()
    if normalized_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")
        device = "cuda"
    elif normalized_device == "cpu":
        device = "cpu"
    else:
        raise ValueError(f"Unsupported --device value: {requested_device}")

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    autocast_dtype = dtype_map.get(precision, torch.float32)
    if device == "cpu":
        autocast_dtype = torch.float32
    return device, autocast_dtype


def _autocast_enabled(device: str, precision: str) -> bool:
    """判断是否启用自动混合精度。

    参数：
        - device: 设备字符串
        - precision: 精度字符串

    返回：
        - 是否启用自动混合精度的布尔值
    """
    return device == "cuda" and precision in {"fp16", "bf16"}


def _scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    """构建带预热余弦退火的学习率调度器。

    参数：
        - optimizer: 优化器
        - total_steps: 总训练步数
        - warmup_steps: 预热步数

    返回：
        - LambdaLR 学习率调度器
    """
    def lr_lambda(step: int) -> float:
        """计算指定步数的学习率缩放系数。

        参数：
            - step: 当前训练步数

        返回：
            - 学习率缩放系数
        """
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())

    return LambdaLR(optimizer, lr_lambda)


def _default_target_scopes(stage: str) -> list[str]:
    """根据训练阶段返回默认的 LoRA 目标模块作用域。

    参数：
        - stage: 训练阶段，可选 "stage_a"/"stage_b"/"stage_c"

    返回：
        - 目标模块作用域字符串列表
    """
    normalized = stage.lower()
    if normalized == "stage_a":
        return ["vision_encoder", "mask_decoder"]
    if normalized == "stage_b":
        return ["detector_decoder", "prompt_encoder", "exemplar_projection"]
    if normalized == "stage_c":
        return ["detector_encoder", "detector_decoder"]
    raise ValueError(f"Unsupported stage: {stage}")


def _resolve_target_scopes(args: argparse.Namespace) -> list[str]:
    """根据命令行参数解析 LoRA 目标模块作用域。

    参数：
        - args: 命令行参数对象

    返回：
        - 排序后的目标模块作用域列表
    """
    scopes = set(_default_target_scopes(args.stage))
    if args.enable_vision_lora:
        scopes.add("vision_encoder")
    if args.enable_detector_lora:
        scopes.update(["detector_encoder", "detector_decoder"])
    if args.enable_mask_decoder_lora:
        scopes.add("mask_decoder")
    return sorted(scopes)


def _contains_polypgen(records: list[dict[str, Any]]) -> bool:
    """检查记录列表中是否包含 PolypGen 数据集样本。

    参数：
        - records: 记录字典列表

    返回：
        - 是否包含 PolypGen 样本的布尔值
    """
    return any("polypgen" in str(record.get("dataset_name", "")).lower() for record in records)


def _read_split_records(split_dir: Path, fold: int) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """读取指定折的训练和验证记录文件。

    参数：
        - split_dir: 划分目录路径
        - fold: 折数索引

    返回：
        - 由 (训练文件路径, 验证文件路径, 训练记录列表, 验证记录列表) 组成的元组
    """
    fold_dir = split_dir / f"fold_{fold}"
    train_file = fold_dir / "train_ids.txt"
    val_file = fold_dir / "val_ids.txt"
    train_records = read_records(train_file)
    val_records = read_records(val_file)
    return train_file, val_file, train_records, val_records


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """将字典以 JSON 格式写入文件。

    参数：
        - path: 输出文件路径
        - payload: 要写入的字典

    返回：
        - 写入后的文件 Path 对象
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _save_adapter_weights(model: torch.nn.Module, path: Path) -> None:
    """保存模型中医学适配器、边界适配器和精修头的权重。

    参数：
        - model: 训练模型
        - path: 权重保存路径

    返回：
        - 无返回值，仅执行保存操作
    """
    adapter_state = {
        key: value
        for key, value in model.state_dict().items()
        if "medical_adapter" in key or "boundary_adapter" in key or "refine_head" in key
    }
    torch.save(adapter_state, path)


def _iter_limited(loader: DataLoader, max_steps: Optional[int]) -> Iterable[tuple[int, dict[str, Any]]]:
    """以受限步数迭代数据加载器。

    参数：
        - loader: 数据加载器
        - max_steps: 最大迭代步数，为 None 时不限制

    返回：
        - 产生 (步数索引, 批次数据) 的迭代器
    """
    for step_index, batch in enumerate(loader, start=1):
        if max_steps is not None and step_index > max_steps:
            break
        yield step_index, batch


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """将批次张量移动到指定设备。

    参数：
        - batch: 批次字典
        - device: 目标设备字符串

    返回：
        - 张量已移动到目标设备的批次字典
    """
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
    """构建训练栈，包括 SAM3 基础模型、LoRA 注入和分割模型。

    参数：
        - args: 命令行参数对象
        - config: 配置字典
        - device: 目标设备字符串

    返回：
        - 由 (基础模型, 包装器, 分割模型, 被替换模块名列表) 组成的元组
    """
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
    """运行预飞行检查，验证数据划分、模型构建、前向与反向传播。

    参数：
        - args: 命令行参数对象
        - config: 配置字典
        - report_path: 预飞行报告输出路径
        - split_dir: 数据划分目录
        - device: 目标设备字符串
        - autocast_dtype: 自动混合精度 dtype

    返回：
        - 包含预飞行检查结果的字典
    """
    train_file, val_file, train_records, val_records = _read_split_records(split_dir, args.fold)
    split_exists = train_file.exists() and val_file.exists()
    blocking_issues: list[str] = []
    warnings_list: list[str] = []

    report: dict[str, Any] = {
        "fold": args.fold,
        "device": device,
        "precision": args.precision,
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
            box_provider = build_box_provider_from_args(args, default_cache_name="train_lora_medical_preflight.json")
            loader = DataLoader(
                SplitSegmentationDataset(
                    train_records,
                    args.image_size,
                    box_padding_ratio=args.box_padding_ratio,
                    box_jitter_ratio=args.train_box_jitter_ratio,
                    box_provider=box_provider,
                ),
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
            loss, _ = criterion(outputs["mask_logits"].float(), runtime_batch["masks"].float())
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
    """脚本命令行入口，执行 LoRA 与适配器训练。

    参数：
        - 无

    返回：
        - 进程退出码，0 表示成功
    """
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device: auto, cuda, or cpu.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--enable-vision-lora", action="store_true")
    parser.add_argument("--enable-detector-lora", action="store_true")
    parser.add_argument("--enable-mask-decoder-lora", action="store_true")
    parser.add_argument("--enable-boundary-adapter", action="store_true")
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--split-dir", default="MedicalSAM3/outputs/medex_sam3/splits")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-dummy", action="store_true")
    parser.add_argument("--require-official-sam3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-lora-modules", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--box-padding-ratio", type=float, default=0.05)
    parser.add_argument("--train-box-jitter-ratio", type=float, default=0.05)
    parser.add_argument("--stage", choices=["stage_a", "stage_b", "stage_c"], default="stage_a")
    add_yolo_bbox_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output_dir = ensure_dir(Path(args.output_dir) / f"fold_{args.fold}")
    report_dir = ensure_dir(Path(args.report_dir) if args.report_dir else output_dir)
    preflight_report_path = report_dir / "preflight_report.json"
    split_dir = Path(config.get("split_dir", args.split_dir))

    device, autocast_dtype = _device_from_args(args.device, args.precision)
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
            "requested_device": args.device,
            "device": device,
            "dummy": args.dummy,
            "allow_dummy": args.allow_dummy,
            "require_official_sam3": args.require_official_sam3,
            "min_lora_modules": args.min_lora_modules,
            "max_train_steps": args.max_train_steps,
            "max_val_steps": args.max_val_steps,
            "box_padding_ratio": args.box_padding_ratio,
            "train_box_jitter_ratio": args.train_box_jitter_ratio,
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
    box_provider = build_box_provider_from_args(args, default_cache_name="train_lora_medical.json")

    train_loader = DataLoader(
        SplitSegmentationDataset(
            train_records,
            args.image_size,
            box_padding_ratio=args.box_padding_ratio,
            box_jitter_ratio=args.train_box_jitter_ratio,
            box_provider=box_provider,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        SplitSegmentationDataset(
            val_records,
            args.image_size,
            box_padding_ratio=args.box_padding_ratio,
            box_jitter_ratio=0.0,
            box_provider=box_provider,
        ),
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

    log_path.write_text("", encoding="utf-8")

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
            loss, loss_parts = criterion(outputs["mask_logits"].float(), runtime_batch["masks"].float())
            adapter_boundary = outputs["adapter_aux"].get("boundary_loss")
            if adapter_boundary is not None:
                loss = loss + 0.1 * adapter_boundary.float()
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
                            "bce": float(loss_parts["bce"].item()),
                            "dice": float(loss_parts["dice"].item()),
                            "boundary": float(loss_parts["boundary"].item()),
                            "adapter_boundary": float(adapter_boundary.item()) if adapter_boundary is not None else None,
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
                "device": device,
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
