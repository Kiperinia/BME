"""Preflight checks and readiness checklist generation for MedEx-SAM3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from MedicalSAM3.adapters.lora import DEFAULT_LORA_REPORT, LoRAConfig, apply_lora_to_model
from MedicalSAM3.exemplar.losses import MedExLossComposer
from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank
from MedicalSAM3.exemplar.prototype_builder import PrototypeBuilder
from MedicalSAM3.sam3_official.build_model import (
    build_official_sam3_image_model,
    count_trainable_parameters,
    freeze_model,
)
from MedicalSAM3.sam3_official.module_inspector import list_named_modules, suggest_lora_targets, write_inspection_outputs
from MedicalSAM3.sam3_official.tensor_forward import DEFAULT_TENSOR_FORWARD_REPORT, run_tensor_forward_smoke_test
from MedicalSAM3.scripts.common import ensure_dir, load_config, read_records


CORE_FILES = {
    "MedicalSAM3/sam3_official/build_model.py": {
        "module": "MedicalSAM3.sam3_official.build_model",
        "symbols": [
            "build_official_sam3_image_model",
            "freeze_model",
            "unfreeze_by_keywords",
            "count_trainable_parameters",
            "print_trainable_parameters",
            "DummyOfficialSam3ImageModel",
        ],
        "blocking": True,
    },
    "MedicalSAM3/sam3_official/tensor_forward.py": {
        "module": "MedicalSAM3.sam3_official.tensor_forward",
        "symbols": [
            "Sam3TensorForwardWrapper",
            "_is_official_sam3_model",
            "_call_official_model",
            "_normalize_xyxy_boxes",
            "_normalize_xy_points",
        ],
        "blocking": True,
    },
    "MedicalSAM3/sam3_official/module_inspector.py": {
        "module": "MedicalSAM3.sam3_official.module_inspector",
        "symbols": ["list_named_modules", "find_modules_by_keywords", "suggest_lora_targets"],
        "blocking": False,
    },
    "MedicalSAM3/adapters/lora.py": {
        "module": "MedicalSAM3.adapters.lora",
        "symbols": [
            "LoRAConfig",
            "LoRALinear",
            "apply_lora_to_model",
            "mark_only_lora_as_trainable",
            "save_lora_weights",
            "load_lora_weights",
            "merge_lora_weights",
        ],
        "blocking": True,
    },
    "MedicalSAM3/adapters/medical_adapter.py": {
        "module": "MedicalSAM3.adapters.medical_adapter",
        "symbols": ["MedicalImageAdapter", "MultiScaleMedicalAdapter"],
        "blocking": False,
    },
    "MedicalSAM3/adapters/boundary_adapter.py": {
        "module": "MedicalSAM3.adapters.boundary_adapter",
        "symbols": ["BoundaryAwareAdapter"],
        "blocking": False,
    },
    "MedicalSAM3/adapters/exemplar_prompt_adapter.py": {
        "module": "MedicalSAM3.adapters.exemplar_prompt_adapter",
        "symbols": ["ExemplarPromptAdapter"],
        "blocking": False,
    },
    "MedicalSAM3/exemplar/memory_bank.py": {
        "module": "MedicalSAM3.exemplar.memory_bank",
        "symbols": ["ExemplarItem", "ExemplarMemoryBank"],
        "blocking": False,
    },
    "MedicalSAM3/exemplar/prototype_builder.py": {
        "module": "MedicalSAM3.exemplar.prototype_builder",
        "symbols": ["PrototypeBuilder"],
        "blocking": False,
    },
    "MedicalSAM3/exemplar/exemplar_encoder.py": {
        "module": "MedicalSAM3.exemplar.exemplar_encoder",
        "symbols": ["ExemplarEncoder"],
        "blocking": False,
    },
    "MedicalSAM3/exemplar/losses.py": {
        "module": "MedicalSAM3.exemplar.losses",
        "symbols": [
            "MedExLossComposer",
            "ExemplarInfoNCELoss",
            "NegativeSuppressionLoss",
            "ExemplarConsistencyLoss",
            "BoundaryBandDiceLoss",
        ],
        "blocking": False,
    },
    "MedicalSAM3/scripts/common.py": {
        "module": "MedicalSAM3.scripts.common",
        "symbols": [
            "MedExSam3SegmentationModel",
            "SplitSegmentationDataset",
            "compute_segmentation_metrics",
            "collate_batch",
            "read_records",
            "write_records",
        ],
        "blocking": True,
    },
}

OPTIONAL_SCRIPTS = [
    "MedicalSAM3/scripts/prepare_5fold_polyp.py",
    "MedicalSAM3/scripts/train_lora_medical.py",
    "MedicalSAM3/scripts/train_exemplar_prompt.py",
    "MedicalSAM3/scripts/build_exemplar_bank.py",
    "MedicalSAM3/scripts/update_memory_from_review.py",
    "MedicalSAM3/scripts/validate_medex_sam3.py",
    "MedicalSAM3/scripts/run_ablation.py",
    "MedicalSAM3/scripts/summarize_cv_results.py",
]


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default or {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_code_audit(report_dir: Path) -> dict[str, Any]:
    existing_files: list[str] = []
    missing_files: list[str] = []
    import_errors: dict[str, str] = {}
    missing_symbols: dict[str, list[str]] = {}
    blocking_issues: list[str] = []
    non_blocking_issues: list[str] = []

    def has_symbol(module: Any, symbol: str) -> bool:
        if hasattr(module, symbol):
            return True
        for value in module.__dict__.values():
            if hasattr(value, symbol):
                return True
        return False

    for relative_path, spec in CORE_FILES.items():
        file_path = Path(relative_path)
        if file_path.exists():
            existing_files.append(relative_path)
        else:
            missing_files.append(relative_path)
            issue = f"missing file: {relative_path}"
            if spec["blocking"]:
                blocking_issues.append(issue)
            else:
                non_blocking_issues.append(issue)
            continue

        try:
            module = importlib.import_module(spec["module"])
        except Exception as exc:
            import_errors[relative_path] = str(exc)
            issue = f"import failed: {relative_path}: {exc}"
            if spec["blocking"]:
                blocking_issues.append(issue)
            else:
                non_blocking_issues.append(issue)
            continue

        missing = [symbol for symbol in spec["symbols"] if not has_symbol(module, symbol)]
        if missing:
            missing_symbols[relative_path] = missing
            issue = f"missing symbols: {relative_path}: {', '.join(missing)}"
            if spec["blocking"]:
                blocking_issues.append(issue)
            else:
                non_blocking_issues.append(issue)

    for relative_path in OPTIONAL_SCRIPTS:
        if Path(relative_path).exists():
            existing_files.append(relative_path)
        else:
            missing_files.append(relative_path)
            non_blocking_issues.append(f"missing optional script: {relative_path}")

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "existing_files": sorted(existing_files),
        "missing_files": sorted(missing_files),
        "import_success": not import_errors,
        "import_errors": import_errors,
        "missing_symbols": missing_symbols,
        "blocking_issues": blocking_issues,
        "non_blocking_issues": non_blocking_issues,
        "next_action": "run module inspector and preflight checks" if not blocking_issues else "fix blocking issues before training",
    }
    _write_json(report_dir / "code_audit.json", report)
    return report


def _run_loss_backward_check() -> bool:
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    gt_mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    criterion = MedExLossComposer()
    loss, _ = criterion(logits, gt_mask)
    loss.backward()
    return logits.grad is not None


def _split_status(split_dir: Path, fold: int) -> dict[str, Any]:
    summary = _read_json(split_dir / "split_summary.json", {})
    train_records = read_records(split_dir / f"fold_{fold}" / "train_ids.txt")
    val_records = read_records(split_dir / f"fold_{fold}" / "val_ids.txt")
    return {
        "split_exists": (split_dir / f"fold_{fold}" / "train_ids.txt").exists() and (split_dir / f"fold_{fold}" / "val_ids.txt").exists(),
        "train_records_count": len(train_records),
        "val_records_count": len(val_records),
        "external_polypgen_count": int(summary.get("external_polypgen_count", 0)),
        "polypgen_leakage_passed": bool(summary.get("leakage_check_passed", False)) and not any(
            "polypgen" in str(record.get("dataset_name", "")).lower() for record in train_records + val_records
        ),
        "warnings": list(summary.get("warnings", [])),
    }


def _short_train_status(results_dir: Path, fold: int) -> tuple[bool, list[str]]:
    fold_dir = results_dir / f"fold_{fold}"
    warnings: list[str] = []
    config = load_config(fold_dir / "config_used.yaml")
    is_dummy = bool(config.get("dummy", False))
    success = (fold_dir / "last.pt").exists() and (fold_dir / "best_lora.pt").exists() and not is_dummy
    if is_dummy and (fold_dir / "last.pt").exists():
        warnings.append("existing short-train artifacts are dummy-only and do not unlock full training")
    return success, warnings


def _embedding_dim_from_item(item: Any) -> int | None:
    embedding_path = Path(str(getattr(item, "embedding_path", "") or ""))
    if not embedding_path.exists():
        return None
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ["foreground_embedding", "global_embedding", "boundary_embedding", "context_embedding", "embedding"]:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[-1])
    if isinstance(payload, torch.Tensor):
        return int(payload.shape[-1])
    return None


def _memory_bank_status(memory_dir: Path, expected_dim: int | None) -> dict[str, Any]:
    bank = ExemplarMemoryBank.load(memory_dir)
    positives = bank.get_items(type="positive", human_verified=True)
    embedding_dim = _embedding_dim_from_item(positives[0]) if positives else None
    dim_matches = expected_dim is not None and embedding_dim == expected_dim if embedding_dim is not None else False
    return {
        "exists": memory_dir.exists(),
        "trainable_items": len(bank.trainable_items),
        "positive_human_verified_items": len(positives),
        "no_polypgen_leakage": bank.check_no_external_leakage(["PolypGen"]),
        "embedding_dim": embedding_dim,
        "dim_matches": dim_matches,
    }


def _maybe_run_short_train(args: argparse.Namespace) -> bool:
    if not args.run_short_train:
        return False
    command = [
        sys.executable,
        "MedicalSAM3/scripts/train_lora_medical.py",
        "--fold",
        str(args.fold),
        "--checkpoint",
        str(args.checkpoint),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--image-size",
        str(args.image_size),
        "--precision",
        args.precision,
        "--require-official-sam3",
        "--min-lora-modules",
        str(args.min_lora_modules),
        "--max-train-steps",
        "10",
        "--max-val-steps",
        "5",
    ]
    subprocess.run(command, check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MedEx-SAM3 preflight checks.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--require-official-sam3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-lora-modules", type=int, default=1)
    parser.add_argument("--allow-dummy", action="store_true")
    parser.add_argument("--run-short-train", action="store_true")
    parser.add_argument("--report-dir", default="MedicalSAM3/outputs/medex_sam3/preflight")
    parser.add_argument("--split-dir", default="MedicalSAM3/outputs/medex_sam3/splits")
    parser.add_argument("--results-dir", default="MedicalSAM3/outputs/medex_sam3")
    args = parser.parse_args()

    report_dir = ensure_dir(args.report_dir)
    results_dir = Path(args.results_dir)
    split_dir = Path(args.split_dir)
    warnings_list: list[str] = []
    blocking_issues: list[str] = []

    audit_report = _run_code_audit(report_dir)
    blocking_issues.extend(audit_report.get("blocking_issues", []))

    official_sam3_build_success = False
    dummy_fallback_used = False
    model_hidden_dim = None
    lora_targets_non_empty = False
    lora_replaced_module_count = 0
    trainable_parameter_ratio = 0.0

    try:
        model = build_official_sam3_image_model(
            checkpoint_path=args.checkpoint,
            device="cpu",
            dtype=args.precision,
            compile_model=False,
            allow_dummy_fallback=args.allow_dummy,
            report_path=str(report_dir / "model_build_report.json"),
        )
        official_sam3_build_success = bool(getattr(model, "_medex_used_official_sam3", False))
        dummy_fallback_used = bool(getattr(model, "_medex_used_dummy_fallback", False))
        model_hidden_dim = getattr(model, "_medex_hidden_dim", getattr(model, "hidden_dim", getattr(model, "embed_dim", None)))
        modules = list_named_modules(model)
        lora_targets = suggest_lora_targets(model)
        write_inspection_outputs(modules, lora_targets)
        lora_targets_non_empty = bool(lora_targets)
        freeze_model(model)
        replaced = apply_lora_to_model(model, LoRAConfig(stage="stage_a", min_replaced_modules=args.min_lora_modules))
        lora_replaced_module_count = len(replaced)
        trainable_parameter_ratio = count_trainable_parameters(model)[2]
    except Exception as exc:
        blocking_issues.append(str(exc))

    tensor_report = run_tensor_forward_smoke_test(
        checkpoint_path=args.checkpoint,
        device="cpu",
        dtype=args.precision,
        image_size=args.image_size,
        allow_dummy=args.allow_dummy,
        report_path=str(report_dir / DEFAULT_TENSOR_FORWARD_REPORT.name),
    )
    if not tensor_report.get("forward_success"):
        blocking_issues.append(str(tensor_report.get("error") or "tensor forward smoke failed"))

    split_status = _split_status(split_dir, args.fold)
    warnings_list.extend(split_status.pop("warnings", []))
    if not split_status["split_exists"]:
        blocking_issues.append("split files missing")
    if not split_status["polypgen_leakage_passed"]:
        blocking_issues.append("PolypGen leakage detected in split records")

    loss_backward_success = _run_loss_backward_check()
    if not loss_backward_success:
        blocking_issues.append("loss backward check failed")

    if args.run_short_train:
        _maybe_run_short_train(args)
    single_fold_short_train_success, short_train_warnings = _short_train_status(results_dir, args.fold)
    warnings_list.extend(short_train_warnings)

    memory_status = _memory_bank_status(results_dir / "exemplar_bank", expected_dim=int(model_hidden_dim) if model_hidden_dim is not None else None)

    checklist = {
        "official_sam3_build_success": official_sam3_build_success,
        "dummy_fallback_used": dummy_fallback_used,
        "tensor_forward_success": bool(tensor_report.get("forward_success", False)),
        "tensor_backward_success": tensor_report.get("backward_success"),
        "lora_replaced_module_count": lora_replaced_module_count,
        "lora_targets_non_empty": lora_targets_non_empty,
        "trainable_parameter_ratio": trainable_parameter_ratio,
        **split_status,
        "loss_backward_success": loss_backward_success,
        "single_fold_short_train_success": single_fold_short_train_success,
        "ready_for_full_training": False,
        "ready_for_exemplar_training": False,
        "blocking_issues": blocking_issues,
        "warnings": warnings_list,
    }

    checklist["ready_for_full_training"] = all(
        [
            checklist["official_sam3_build_success"],
            not checklist["dummy_fallback_used"],
            checklist["tensor_forward_success"],
            checklist["lora_replaced_module_count"] >= args.min_lora_modules,
            checklist["trainable_parameter_ratio"] > 0.0,
            checklist["trainable_parameter_ratio"] < 0.05,
            checklist["train_records_count"] > 0,
            checklist["val_records_count"] > 0,
            checklist["polypgen_leakage_passed"],
            checklist["loss_backward_success"],
            checklist["single_fold_short_train_success"],
        ]
    )

    checklist["ready_for_exemplar_training"] = all(
        [
            checklist["ready_for_full_training"],
            memory_status["exists"],
            memory_status["positive_human_verified_items"] >= 1,
            memory_status["no_polypgen_leakage"],
            memory_status["dim_matches"],
        ]
    )

    _write_json(report_dir / "readiness_checklist.json", checklist)
    print(json.dumps(checklist, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())