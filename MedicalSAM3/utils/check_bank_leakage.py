"""Strict leakage checks between evaluation samples and retrieval banks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SampleSignature:
    image_id: str
    image_path: str
    mask_path: str
    file_name: str
    stem: str
    patient_id: str
    perceptual_hash: Optional[int]
    mask_hash: Optional[int]


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
            continue
        parts = stripped.split("\t")
        if len(parts) >= 4:
            rows.append(
                {
                    "image_path": parts[0],
                    "mask_path": parts[1],
                    "dataset_name": parts[2],
                    "image_id": parts[3],
                }
            )
    return rows


def _patient_id_from_value(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    leaf = Path(normalized).stem
    if "__" in leaf:
        leaf = leaf.split("__", 1)[1]
    parts = [part for part in leaf.split("_") if part]
    if len(parts) >= 2 and parts[0].upper() in {f"C{index}" for index in range(1, 7)}:
        return f"{parts[0].upper()}_{parts[1]}"
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return leaf


def _average_hash(path: str | Path, hash_size: int = 8) -> Optional[int]:
    target = Path(path)
    if not target.exists() or target.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None
    image = Image.open(target).convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(image).astype("float32")
    threshold = float(pixels.mean())
    bits = (pixels >= threshold).astype(np.uint8).flatten().tolist()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def _resolve_bank_mask_path(image_path: Path) -> str:
    if "masks" in image_path.parts:
        return str(image_path)
    if "images" in image_path.parts:
        parts = list(image_path.parts)
        image_index = parts.index("images")
        mask_root = Path(*parts[:image_index], "masks", *parts[image_index + 1 : -1])
        for suffix in SUPPORTED_IMAGE_SUFFIXES:
            candidate = (mask_root / image_path.name).with_suffix(suffix)
            if candidate.exists():
                return str(candidate)
    sibling_mask_root = image_path.parent.parent / "masks" if image_path.parent.name == "images" else image_path.parent / "masks"
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = sibling_mask_root / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return str(candidate)
    return ""


def _build_signature(record: dict[str, Any]) -> SampleSignature:
    image_path = str(record.get("image_path") or record.get("crop_path") or "")
    mask_path = str(record.get("mask_path") or "")
    image_id = str(record.get("image_id") or Path(image_path).stem)
    file_name = Path(image_path).name if image_path else f"{image_id}.png"
    stem = Path(file_name).stem
    patient_id = _patient_id_from_value(image_id or image_path or stem)
    return SampleSignature(
        image_id=image_id,
        image_path=image_path,
        mask_path=mask_path,
        file_name=file_name,
        stem=stem,
        patient_id=patient_id,
        perceptual_hash=_average_hash(image_path),
        mask_hash=_average_hash(mask_path) if mask_path else None,
    )


def _collect_bank_records(bank_root: str | Path) -> list[dict[str, Any]]:
    root = Path(bank_root)
    records: list[dict[str, Any]] = []
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            if isinstance(entry, dict):
                records.append(entry)

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if "masks" in path.parts:
            continue
        records.append(
            {
                "image_id": path.stem,
                "image_path": str(path),
                "mask_path": _resolve_bank_mask_path(path),
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("image_id", "")), str(record.get("image_path", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _pair_key(eval_sample: SampleSignature, bank_sample: SampleSignature) -> tuple[str, str]:
    return (eval_sample.image_id or eval_sample.file_name, bank_sample.image_id or bank_sample.file_name)


def _pair_payload(
    eval_sample: SampleSignature,
    bank_sample: SampleSignature,
    *,
    reasons: list[str],
    hash_distance: Optional[int] = None,
    mask_hash_distance: Optional[int] = None,
) -> dict[str, Any]:
    payload = {
        "eval_image_id": eval_sample.image_id,
        "eval_image_path": eval_sample.image_path,
        "eval_mask_path": eval_sample.mask_path,
        "bank_image_id": bank_sample.image_id,
        "bank_image_path": bank_sample.image_path,
        "bank_mask_path": bank_sample.mask_path,
        "patient_id": eval_sample.patient_id or bank_sample.patient_id,
        "reasons": sorted(set(reasons)),
    }
    if hash_distance is not None:
        payload["hash_distance"] = int(hash_distance)
    if mask_hash_distance is not None:
        payload["mask_hash_distance"] = int(mask_hash_distance)
    return payload


def run_bank_leakage_check(
    *,
    eval_records_path: str | Path,
    bank_root: str | Path,
    max_hash_distance: int = 3,
) -> dict[str, Any]:
    eval_records = _read_records(eval_records_path)
    bank_records = _collect_bank_records(bank_root)
    eval_samples = [_build_signature(record) for record in eval_records]
    bank_samples = [_build_signature(record) for record in bank_records]

    filename_overlap: list[dict[str, Any]] = []
    patient_overlap: list[dict[str, Any]] = []
    hash_overlap: list[dict[str, Any]] = []
    mask_overlap: list[dict[str, Any]] = []
    image_overlap: list[str] = []
    suspicious_pairs: dict[tuple[str, str], dict[str, Any]] = {}

    for eval_sample in eval_samples:
        for bank_sample in bank_samples:
            reasons: list[str] = []
            hash_distance: Optional[int] = None
            mask_hash_distance: Optional[int] = None

            if eval_sample.image_id and (eval_sample.image_id == bank_sample.image_id or eval_sample.stem.lower() == bank_sample.stem.lower()):
                image_overlap.append(bank_sample.image_id)
                reasons.append("image_overlap")
            if eval_sample.file_name and eval_sample.file_name.lower() == bank_sample.file_name.lower():
                reasons.append("filename_overlap")
            if eval_sample.patient_id and eval_sample.patient_id == bank_sample.patient_id:
                reasons.append("patient_overlap")
            if eval_sample.perceptual_hash is not None and bank_sample.perceptual_hash is not None:
                hash_distance = _hamming_distance(int(eval_sample.perceptual_hash), int(bank_sample.perceptual_hash))
                if hash_distance <= max_hash_distance:
                    reasons.append("hash_overlap")
            if eval_sample.mask_hash is not None and bank_sample.mask_hash is not None:
                mask_hash_distance = _hamming_distance(int(eval_sample.mask_hash), int(bank_sample.mask_hash))
                if mask_hash_distance <= max_hash_distance:
                    reasons.append("mask_overlap")

            if not reasons:
                continue

            payload = _pair_payload(
                eval_sample,
                bank_sample,
                reasons=reasons,
                hash_distance=hash_distance if "hash_overlap" in reasons else None,
                mask_hash_distance=mask_hash_distance if "mask_overlap" in reasons else None,
            )
            suspicious_pairs[_pair_key(eval_sample, bank_sample)] = payload
            if "filename_overlap" in reasons:
                filename_overlap.append(payload)
            if "patient_overlap" in reasons:
                patient_overlap.append(payload)
            if "hash_overlap" in reasons:
                hash_overlap.append(payload)
            if "mask_overlap" in reasons:
                mask_overlap.append(payload)

    return {
        "total_eval_samples": len(eval_samples),
        "total_bank_samples": len(bank_samples),
        "eval_count": len(eval_samples),
        "bank_count": len(bank_samples),
        "image_overlap": sorted(set(image_overlap)),
        "filename_overlap": filename_overlap,
        "patient_overlap": patient_overlap,
        "hash_overlap": hash_overlap,
        "mask_overlap": mask_overlap,
        "perceptual_hash_overlap": hash_overlap,
        "suspicious_pairs": sorted(suspicious_pairs.values(), key=lambda item: (item["eval_image_id"], item["bank_image_id"])),
        "leakage_detected": bool(suspicious_pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check retrieval bank leakage against evaluation records.")
    parser.add_argument("--eval-records", required=True)
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--max-hash-distance", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict-leakage-check", action="store_true")
    args = parser.parse_args()

    summary = run_bank_leakage_check(
        eval_records_path=args.eval_records,
        bank_root=args.bank_root,
        max_hash_distance=args.max_hash_distance,
    )
    payload = json.dumps(summary, indent=2)
    output_path = Path(args.output) if args.output else Path(args.bank_root) / "leakage_report.json"
    output_path.write_text(payload, encoding="utf-8")
    print(payload)
    if args.strict_leakage_check and summary["leakage_detected"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())