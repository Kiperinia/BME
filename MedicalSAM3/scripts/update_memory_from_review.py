"""Update exemplar memory bank from reviewed CSV decisions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.agents.human_review_queue import export_review_queue, import_human_review
from MedicalSAM3.agents.memory_version_manager import MemoryVersionManager
from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank
from MedicalSAM3.scripts.common import ensure_dir


def _write_dummy_review_csv(bank: ExemplarMemoryBank, path: Path) -> Path:
    export_review_queue(bank, path)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["accept"] = "yes"
            row["quality_score"] = row.get("quality_score") or "0.9"
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [
            "item_id", "image_id", "type", "source_dataset", "crop_path", "mask_path",
            "quality_score", "boundary_score", "notes", "accept"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Update memory bank from human review CSV.")
    parser.add_argument("--memory-bank-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--review-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    memory_bank_dir = Path(args.memory_bank_dir)
    bank = ExemplarMemoryBank.load(memory_bank_dir)
    review_csv = Path(args.review_csv) if args.review_csv else memory_bank_dir / "review_queue_reviewed.csv"
    if args.dummy and not review_csv.exists():
        _write_dummy_review_csv(bank, review_csv)
    if not review_csv.exists():
        raise FileNotFoundError("Review CSV not found. Export and annotate review_queue.csv first, or use --dummy.")

    updated_bank = import_human_review(review_csv, bank)
    manager = MemoryVersionManager(args.output_dir or memory_bank_dir)
    version_path = manager.save_new_version(updated_bank)
    print(json.dumps({"saved_version": str(version_path), "trainable_items": len(updated_bank.trainable_items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
