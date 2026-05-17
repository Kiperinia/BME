"""Mine hard cases and build continual adaptation inputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.scripts.common import ensure_dir, read_records, write_records


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_image_key(value: str | Path) -> str:
    stem = Path(value).stem.lower()
    return re.sub(r"(?:_0+)?$", "", stem)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _records_by_image_id(split_file: str | Path) -> dict[str, dict[str, Any]]:
    return {str(record.get("image_id", "")): record for record in read_records(split_file)}


def _suggest_polarity(record: dict[str, Any]) -> str:
    mask_path = str(record.get("mask_path", "") or "")
    return "positive" if mask_path else "negative"


def _mine_hard_cases(
    *,
    metrics_path: str | Path,
    split_file: str | Path,
    output_dir: str | Path,
    dice_threshold: float,
    fpr_threshold: float,
    max_cases: int | None,
) -> Path:
    rows = _load_jsonl(metrics_path)
    records = _records_by_image_id(split_file)
    destination = ensure_dir(output_dir)
    hard_case_dir = ensure_dir(destination / "hard_cases")
    manifest = []
    for row in rows:
        metrics = row.get("metrics", {})
        image_id = str(row.get("image_id", ""))
        dice = float(metrics.get("Dice", 0.0))
        fpr = float(metrics.get("False Positive Rate", 0.0))
        if dice > dice_threshold and fpr < fpr_threshold:
            continue
        record = records.get(image_id)
        if record is None:
            continue
        image_path = Path(str(record.get("image_path", "")))
        mask_path = Path(str(record.get("mask_path", ""))) if record.get("mask_path") else None
        preview_path = None
        if image_path.exists():
            preview_path = hard_case_dir / image_path.name
            shutil.copyfile(image_path, preview_path)
        if mask_path is not None and mask_path.exists():
            shutil.copyfile(mask_path, hard_case_dir / mask_path.name)
        manifest.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "dataset_name": record.get("dataset_name"),
                "metrics": metrics,
                "preview_path": None if preview_path is None else str(preview_path),
                "suggested_polarity": _suggest_polarity(record),
                "review_polarity": _suggest_polarity(record),
                "accept": False,
                "notes": "",
            }
        )
        if max_cases is not None and len(manifest) >= max_cases:
            break
    manifest_path = destination / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _apply_reviewed_manifest(manifest_path: str | Path, continual_bank: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("review manifest must be a list")
    bank_root = ensure_dir(continual_bank)
    accepted_records = []
    applied = []
    for entry in manifest:
        if not bool(entry.get("accept", False)):
            continue
        polarity = str(entry.get("review_polarity") or entry.get("suggested_polarity") or "positive")
        if polarity not in {"positive", "negative"}:
            raise ValueError(f"Unsupported polarity in reviewed manifest: {polarity}")
        image_path = Path(str(entry.get("image_path", "")))
        if not image_path.exists():
            continue
        target_dir = ensure_dir(bank_root / polarity)
        target_path = target_dir / image_path.name
        shutil.copyfile(image_path, target_path)
        applied.append(
            {
                "image_id": entry.get("image_id"),
                "polarity": polarity,
                "target_path": str(target_path),
            }
        )
        mask_path = str(entry.get("mask_path") or "")
        if mask_path:
            accepted_records.append(
                {
                    "image_path": str(image_path),
                    "mask_path": mask_path,
                    "dataset_name": entry.get("dataset_name", "continual"),
                    "image_id": entry.get("image_id", image_path.stem),
                }
            )
    destination = ensure_dir(output_dir)
    accepted_records_path = destination / "accepted_records.jsonl"
    write_records(accepted_records_path, accepted_records)
    summary = {
        "applied_count": len(applied),
        "accepted_training_records": len(accepted_records),
        "accepted_records_path": str(accepted_records_path),
        "applied": applied,
    }
    (destination / "apply_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _collect_bank_training_records(continual_bank: str | Path, split_file: str | Path, output_dir: str | Path) -> dict[str, Any]:
    bank_root = Path(continual_bank)
    if not bank_root.exists():
        raise FileNotFoundError(bank_root)

    records = read_records(split_file)
    by_name: dict[str, dict[str, Any]] = {}
    by_stem: dict[str, dict[str, Any]] = {}
    by_normalized_stem: dict[str, dict[str, Any]] = {}
    for record in records:
        image_path = Path(str(record.get("image_path", "")))
        if image_path.name:
            by_name.setdefault(image_path.name.lower(), record)
        if image_path.stem:
            by_stem.setdefault(image_path.stem.lower(), record)
            by_normalized_stem.setdefault(_normalize_image_key(image_path), record)

    matched_records: list[dict[str, Any]] = []
    matched_image_paths: set[str] = set()
    unmatched_bank_images: list[str] = []
    for polarity in ("positive", "negative"):
        polarity_dir = bank_root / polarity
        if not polarity_dir.exists():
            continue
        for image_path in sorted(path for path in polarity_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            record = (
                by_name.get(image_path.name.lower())
                or by_stem.get(image_path.stem.lower())
                or by_normalized_stem.get(_normalize_image_key(image_path))
            )
            if record is None:
                unmatched_bank_images.append(str(image_path))
                continue
            record_key = str(record.get("image_path", ""))
            if record_key in matched_image_paths:
                continue
            matched_image_paths.add(record_key)
            matched_records.append(
                {
                    "image_path": str(record.get("image_path", "")),
                    "mask_path": str(record.get("mask_path", "")),
                    "dataset_name": record.get("dataset_name", "continual"),
                    "image_id": record.get("image_id", Path(str(record.get("image_path", ""))).stem),
                }
            )

    destination = ensure_dir(output_dir)
    records_path = destination / "continual_bank_records.jsonl"
    write_records(records_path, matched_records)
    summary = {
        "bank_root": str(bank_root),
        "matched_training_records": len(matched_records),
        "records_path": str(records_path),
        "unmatched_bank_images": unmatched_bank_images,
    }
    (destination / "collect_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare continual adaptation review queues and apply reviewed exemplars.")
    parser.add_argument("--per-image-metrics", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/continual_adaptation")
    parser.add_argument("--continual-bank", default="MedicalSAM3/banks/continual_bank")
    parser.add_argument("--dice-threshold", type=float, default=0.85)
    parser.add_argument("--fpr-threshold", type=float, default=0.02)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--apply-reviewed-manifest", default=None)
    parser.add_argument("--collect-bank-records", action="store_true")
    args = parser.parse_args()

    if args.collect_bank_records:
        if not args.split_file:
            raise ValueError("--split-file is required when collecting bank records.")
        summary = _collect_bank_training_records(args.continual_bank, args.split_file, args.output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.apply_reviewed_manifest:
        summary = _apply_reviewed_manifest(args.apply_reviewed_manifest, args.continual_bank, args.output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if not args.per_image_metrics or not args.split_file:
        raise ValueError("--per-image-metrics and --split-file are required when mining hard cases.")

    manifest_path = _mine_hard_cases(
        metrics_path=args.per_image_metrics,
        split_file=args.split_file,
        output_dir=args.output_dir,
        dice_threshold=args.dice_threshold,
        fpr_threshold=args.fpr_threshold,
        max_cases=args.max_cases,
    )
    print(json.dumps({"review_manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())