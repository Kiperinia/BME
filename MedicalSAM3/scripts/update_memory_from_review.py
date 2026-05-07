"""Update exemplar memory bank from reviewed CSV decisions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.exemplar.memory_bank import ExemplarItem
from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank
from MedicalSAM3.scripts.common import ensure_dir


def _write_dummy_review_csv(bank: ExemplarMemoryBank, path: Path) -> Path:
    fieldnames = [
        "item_id",
        "image_id",
        "crop_path",
        "mask_path",
        "type",
        "source_dataset",
        "accept",
        "quality_score",
        "boundary_score",
        "notes",
    ]
    rows = []
    for item in bank.items:
        rows.append(
            {
                "item_id": item.item_id,
                "image_id": item.image_id,
                "crop_path": item.crop_path,
                "mask_path": item.mask_path or "",
                "type": item.type,
                "source_dataset": item.source_dataset,
                "accept": "yes",
                "quality_score": "0.9",
                "boundary_score": "0.8",
                "notes": item.notes,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _bank_stats(bank: ExemplarMemoryBank) -> dict[str, object]:
    return {
        "version": bank.version,
        "total_items": len(bank.items),
        "trainable_items": len(bank.trainable_items),
        "positive_items": len(bank.get_items(type="positive")),
        "negative_items": len(bank.get_items(type="negative")),
        "boundary_items": len(bank.get_items(type="boundary")),
        "human_verified_positive_items": len(bank.get_items(type="positive", human_verified=True)),
        "has_polypgen_leakage": not bank.check_no_external_leakage(["PolypGen"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update memory bank from human review CSV.")
    parser.add_argument("--memory-bank", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank/memory_v0.json")
    parser.add_argument("--review-csv", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank/review_queue.csv")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    memory_bank_path = Path(args.memory_bank)
    output_dir = ensure_dir(args.output_dir)
    bank = ExemplarMemoryBank.load(memory_bank_path)
    review_csv = Path(args.review_csv)
    if args.dummy and not review_csv.exists():
        _write_dummy_review_csv(bank, review_csv)
    if not review_csv.exists():
        raise FileNotFoundError("Review CSV not found. Export and annotate review_queue.csv first, or use --dummy.")

    rows = []
    with review_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    items_by_id = {item.item_id: item for item in bank.items}
    next_items: list[ExemplarItem] = []
    rejected_rows: list[dict[str, object]] = []
    accepted_ids: set[str] = set()

    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        if not item_id or item_id not in items_by_id:
            continue
        item = items_by_id[item_id]
        accepted_ids.add(item_id)
        accept_value = str(row.get("accept", "")).strip().lower()
        revised_type = str(row.get("type", item.type)).strip() or item.type
        if "polypgen" in item.source_dataset.lower():
            bank.reject_item(item.item_id, "external_dataset_leakage")
            rejected_rows.append({"item_id": item.item_id, "reason": "external_dataset_leakage"})
            continue
        if accept_value != "yes":
            bank.reject_item(item.item_id, "review_rejected")
            rejected_rows.append({"item_id": item.item_id, "reason": "review_rejected"})
            continue

        next_items.append(
            ExemplarItem(
                item_id=item.item_id,
                image_id=item.image_id,
                crop_path=item.crop_path,
                mask_path=item.mask_path,
                bbox=item.bbox,
                embedding_path=item.embedding_path,
                type=revised_type,
                source_dataset=item.source_dataset,
                fold_id=item.fold_id,
                human_verified=True,
                quality_score=float(row.get("quality_score") or item.quality_score),
                boundary_score=float(row.get("boundary_score") or item.boundary_score),
                diversity_score=item.diversity_score,
                difficulty_score=item.difficulty_score,
                uncertainty_score=item.uncertainty_score,
                false_positive_risk=item.false_positive_risk,
                created_at=item.created_at,
                version="v1",
                notes=str(row.get("notes") or item.notes),
            )
        )

    for item in bank.items:
        if item.item_id not in accepted_ids:
            rejected_rows.append({"item_id": item.item_id, "reason": "not_reviewed"})

    updated_bank = ExemplarMemoryBank(items=next_items)
    updated_bank.rejected_items = bank.rejected_items + rejected_rows
    updated_bank.changelog = bank.changelog
    version_path = updated_bank.save(output_dir / "memory_v1.json")
    (output_dir / "bank_stats.json").write_text(json.dumps(_bank_stats(updated_bank), indent=2), encoding="utf-8")
    print(json.dumps({"saved_version": str(version_path), "trainable_items": len(updated_bank.trainable_items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
