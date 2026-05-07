"""Build candidate exemplar crops and embeddings from the training split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MedicalSAM3.agents.human_review_queue import export_review_queue
from MedicalSAM3.agents.leakage_checker import LeakageChecker
from MedicalSAM3.exemplar.exemplar_encoder import ExemplarEncoder
from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank
from MedicalSAM3.scripts.common import ensure_dir, load_record_tensors, mask_to_box, read_records


def _crop_tensor(image: torch.Tensor, mask: torch.Tensor, margin_ratio: float = 0.15) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    box = mask_to_box(mask)
    x1, y1, x2, y2 = [int(value) for value in box.tolist()]
    width = x2 - x1
    height = y2 - y1
    mx = max(int(width * margin_ratio), 1)
    my = max(int(height * margin_ratio), 1)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(image.shape[-1], x2 + mx)
    y2 = min(image.shape[-2], y2 + my)
    return image[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2], [float(x1), float(y1), float(x2), float(y2)]


def _save_crop(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu()
    if array.dim() == 3 and array.shape[0] == 1:
        image = Image.fromarray((array.squeeze(0).numpy() * 255).astype(np.uint8))
    else:
        image = Image.fromarray((array.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MedEx-SAM3 candidate exemplar bank.")
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"dummy_{i}"} for i in range(6)]
    if not records:
        raise FileNotFoundError("No train records found for exemplar bank construction.")

    output_dir = ensure_dir(args.output_dir)
    crops_dir = ensure_dir(output_dir / "crops")
    masks_dir = ensure_dir(output_dir / "crop_masks")
    embeddings_dir = ensure_dir(output_dir / "embeddings")
    encoder = ExemplarEncoder(embed_dim=128)
    bank = ExemplarMemoryBank()
    checker = LeakageChecker()
    review_queue = []

    for index, record in enumerate(records):
        if "polypgen" in record.get("dataset_name", "").lower():
            continue
        image, mask = load_record_tensors(record, args.image_size, fallback_index=index)
        pos_crop, pos_mask, pos_bbox = _crop_tensor(image, mask)
        neg_crop = image[:, : pos_crop.shape[-2], : pos_crop.shape[-1]]
        neg_mask = torch.zeros_like(pos_mask)
        boundary_mask = (F.max_pool2d(pos_mask.unsqueeze(0), 3, 1, 1) - pos_mask.unsqueeze(0)).clamp(0, 1).squeeze(0)

        for exemplar_type, crop_tensor, mask_tensor in [
            ("positive", pos_crop, pos_mask),
            ("boundary", pos_crop, boundary_mask),
            ("negative", neg_crop, neg_mask),
        ]:
            item_id = f"{record['image_id']}_{exemplar_type}"
            crop_path = crops_dir / f"{item_id}.png"
            crop_mask_path = masks_dir / f"{item_id}.png"
            embedding_path = embeddings_dir / f"{item_id}.pt"
            _save_crop(crop_path, crop_tensor)
            _save_crop(crop_mask_path, mask_tensor)
            with torch.no_grad():
                embeddings = encoder(crop_tensor.unsqueeze(0), mask_tensor.unsqueeze(0) if exemplar_type != "negative" else None)
            torch.save(embeddings, embedding_path)

            item = ExemplarItem(
                item_id=item_id,
                image_id=record["image_id"],
                crop_path=str(crop_path),
                mask_path=str(crop_mask_path),
                bbox=pos_bbox,
                embedding_path=str(embedding_path),
                type=exemplar_type,
                source_dataset=record["dataset_name"],
                fold_id=0,
                human_verified=False,
                quality_score=0.5 if exemplar_type == "negative" else 0.8,
                boundary_score=0.9 if exemplar_type == "boundary" else 0.6,
                diversity_score=0.5,
                difficulty_score=0.5,
                uncertainty_score=0.2,
                false_positive_risk=0.4 if exemplar_type == "negative" else 0.1,
                created_at=datetime.now(timezone.utc).isoformat(),
                version="v0",
                notes="candidate",
            )
            ok, reason = checker.check_item(item)
            if not ok:
                bank.reject_item(item.item_id, reason or "leakage")
                continue
            bank.add_item(item)
            review_queue.append(asdict(item))

    bank_path = bank.save(output_dir)
    (output_dir / "candidate_bank.json").write_text(json.dumps([asdict(item) for item in bank.items], indent=2), encoding="utf-8")
    (output_dir / "review_queue.json").write_text(json.dumps(review_queue, indent=2), encoding="utf-8")
    export_review_queue(bank, output_dir / "review_queue.csv")
    print(json.dumps({"memory_bank": str(bank_path), "candidate_count": len(bank.items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
