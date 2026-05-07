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

from MedicalSAM3.agents.leakage_checker import LeakageChecker
from MedicalSAM3.exemplar.exemplar_encoder import ExemplarEncoder
from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model
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


def _write_review_queue_csv(bank: ExemplarMemoryBank, path: Path) -> Path:
    header = [
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
    rows = [
        {
            "item_id": item.item_id,
            "image_id": item.image_id,
            "crop_path": item.crop_path,
            "mask_path": item.mask_path or "",
            "type": item.type,
            "source_dataset": item.source_dataset,
            "accept": "",
            "quality_score": "",
            "boundary_score": "",
            "notes": item.notes,
        }
        for item in bank.items
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(str(row[column]).replace(",", " ") for column in header)
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _infer_embed_dim(checkpoint_path: str | None, allow_dummy: bool) -> int:
    preflight_report = Path("MedicalSAM3/outputs/medex_sam3/preflight/model_build_report.json")
    if preflight_report.exists():
        try:
            payload = json.loads(preflight_report.read_text(encoding="utf-8"))
            hidden_dim = payload.get("hidden_dim") or payload.get("embed_dim")
            if hidden_dim is not None:
                return int(hidden_dim)
        except Exception:
            pass

    if checkpoint_path is None and not allow_dummy:
        return 128

    try:
        model = build_official_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device="cpu",
            dtype="fp32",
            compile_model=False,
            allow_dummy_fallback=allow_dummy,
        )
        return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))
    except Exception:
        return 128


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MedEx-SAM3 candidate exemplar bank.")
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"dummy_{i}"} for i in range(6)]
    if not records:
        raise FileNotFoundError("No train records found for exemplar bank construction.")
    if args.max_items is not None:
        records = records[: max(args.max_items, 0)]

    output_dir = ensure_dir(args.output_dir)
    crops_dir = ensure_dir(output_dir / "crops")
    masks_dir = ensure_dir(output_dir / "masks")
    embeddings_dir = ensure_dir(output_dir / "embeddings")
    encoder = ExemplarEncoder(embed_dim=_infer_embed_dim(args.checkpoint, allow_dummy=args.dummy))
    bank = ExemplarMemoryBank()
    checker = LeakageChecker()

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

    bank_path = bank.save(output_dir / "memory_v0.json")
    review_queue_path = _write_review_queue_csv(bank, output_dir / "review_queue.csv")
    (output_dir / "bank_stats.json").write_text(json.dumps(_bank_stats(bank), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "memory_bank": str(bank_path),
                "review_queue": str(review_queue_path),
                "candidate_count": len(bank.items),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
