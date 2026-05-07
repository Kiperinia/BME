"""CSV and HTML review queue helpers for human-verified exemplar curation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank


def _resolve_bank(memory_bank: Union[ExemplarMemoryBank, str, Path, None], review_csv: Optional[Path] = None) -> ExemplarMemoryBank:
    if isinstance(memory_bank, ExemplarMemoryBank):
        return memory_bank
    if memory_bank is not None:
        return ExemplarMemoryBank.load(memory_bank)
    if review_csv is not None:
        return ExemplarMemoryBank.load(review_csv.parent)
    return ExemplarMemoryBank()


def export_review_queue(memory_bank: ExemplarMemoryBank, output_html_or_csv: str | Path) -> Path:
    destination = Path(output_html_or_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = memory_bank.get_items(human_verified=False)
    rows = [
        {
            "item_id": item.item_id,
            "image_id": item.image_id,
            "type": item.type,
            "source_dataset": item.source_dataset,
            "crop_path": item.crop_path,
            "mask_path": item.mask_path or "",
            "quality_score": item.quality_score,
            "boundary_score": item.boundary_score,
            "notes": item.notes,
            "accept": "",
        }
        for item in candidates
    ]
    if destination.suffix.lower() == ".csv":
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
                "item_id", "image_id", "type", "source_dataset", "crop_path", "mask_path",
                "quality_score", "boundary_score", "notes", "accept"
            ])
            writer.writeheader()
            writer.writerows(rows)
        return destination

    html_lines = [
        "<html><body><table border='1'>",
        "<tr><th>item_id</th><th>image_id</th><th>type</th><th>dataset</th><th>crop</th><th>accept</th></tr>",
    ]
    for row in rows:
        html_lines.append(
            f"<tr><td>{row['item_id']}</td><td>{row['image_id']}</td><td>{row['type']}</td>"
            f"<td>{row['source_dataset']}</td><td>{row['crop_path']}</td><td></td></tr>"
        )
    html_lines.append("</table></body></html>")
    destination.write_text("\n".join(html_lines), encoding="utf-8")
    return destination


def import_human_review(
    review_csv: str | Path,
    memory_bank: Union[ExemplarMemoryBank, str, Path, None] = None,
) -> ExemplarMemoryBank:
    review_path = Path(review_csv)
    bank = _resolve_bank(memory_bank, review_csv=review_path)
    with review_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        decisions = list(reader)

    items_by_id = {item.item_id: item for item in bank.items}
    for row in decisions:
        item_id = row.get("item_id", "")
        item = items_by_id.get(item_id)
        if item is None:
            continue
        accept = row.get("accept", "").strip().lower()
        if accept in {"yes", "y", "1", "true"}:
            item.human_verified = True
            item.type = row.get("type", item.type) or item.type
            item.notes = row.get("notes", item.notes)
            if row.get("quality_score"):
                item.quality_score = float(row["quality_score"])
            if row.get("boundary_score"):
                item.boundary_score = float(row["boundary_score"])
        elif accept in {"no", "n", "0", "false"}:
            bank.reject_item(item_id, row.get("notes", "human_rejected"))
    return bank
