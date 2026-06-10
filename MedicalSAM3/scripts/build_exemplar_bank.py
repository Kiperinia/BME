"""Build candidate exemplar crops and embeddings from the training split."""

from __future__ import annotations

import argparse
import json
import warnings
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
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    MedExSam3SegmentationModel,
    compute_segmentation_metrics,
    ensure_dir,
    load_record_tensors,
    mask_to_box,
    read_records,
    resolve_feature_map,
)
from MedicalSAM3.yolo_adapter.cli import add_yolo_bbox_args, build_box_provider_from_args


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
        "rejected_items": len(bank.rejected_items),
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


def _build_sam3_embedding_stack(
    checkpoint_path: str | None,
    *,
    allow_dummy: bool,
) -> tuple[str, Sam3TensorForwardWrapper | None, MedExSam3SegmentationModel | None]:
    if allow_dummy or checkpoint_path is None:
        return "cpu", None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        base_model = build_official_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype="fp32",
            compile_model=False,
            allow_dummy_fallback=False,
        )
        wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype="fp32")
        embed_dim = int(getattr(base_model, "hidden_dim", getattr(base_model, "_medex_hidden_dim", getattr(base_model, "embed_dim", 128))))
        segmentation_model = MedExSam3SegmentationModel(
            wrapper=wrapper,
            enable_medical_adapter=False,
            enable_boundary_adapter=False,
            embed_dim=embed_dim,
        ).to(device)
        segmentation_model.eval()
        return device, wrapper, segmentation_model
    except Exception as exc:
        warnings.warn(f"SAM3 embedding stack unavailable, falling back to ExemplarEncoder: {exc}", stacklevel=2)
        return "cpu", None, None


def _score_record_with_sam3(
    model: MedExSam3SegmentationModel | None,
    image: torch.Tensor,
    mask: torch.Tensor,
    box: torch.Tensor,
    device: str,
) -> dict[str, float]:
    if model is None:
        return {
            "quality": 0.8,
            "boundary": 0.6,
            "negative_quality": 0.5,
            "difficulty": 0.5,
            "uncertainty": 0.2,
            "false_positive_risk": 0.1,
        }

    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)
        box_batch = box.unsqueeze(0).to(device)
        outputs = model(images=image_batch, boxes=box_batch, text_prompt=["polyp"], gt_mask=mask_batch)
        metrics = compute_segmentation_metrics(outputs["mask_logits"], mask_batch)
        confidence = float(outputs["scores"].detach().float().mean().item()) if isinstance(outputs.get("scores"), torch.Tensor) else 0.8

    dice = float(metrics.get("Dice", 0.0))
    boundary = float(metrics.get("Boundary F1", 0.0))
    false_positive_rate = float(metrics.get("False Positive Rate", 0.0))
    uncertainty = max(0.0, min(1.0, 1.0 - confidence))
    return {
        "quality": max(0.0, min(1.0, dice)),
        "boundary": max(0.0, min(1.0, boundary)),
        "negative_quality": max(0.0, min(1.0, 1.0 - false_positive_rate)),
        "difficulty": max(0.0, min(1.0, 1.0 - dice)),
        "uncertainty": uncertainty,
        "false_positive_risk": max(0.0, min(1.0, false_positive_rate)),
    }


def _save_sam3_embedding(
    wrapper: Sam3TensorForwardWrapper,
    crop_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    embedding_path: Path,
    device: str,
) -> None:
    with torch.no_grad():
        crop_batch = crop_tensor.unsqueeze(0).to(device)
        outputs = wrapper(images=crop_batch, text_prompt=["polyp"])
        feature_map = resolve_feature_map(outputs.get("image_embeddings"), crop_batch).float()
        global_embedding = F.normalize(F.adaptive_avg_pool2d(feature_map, 1).flatten(1), dim=1)
        mask_batch = mask_tensor.unsqueeze(0).to(device)
        resized_mask = F.interpolate(mask_batch.float(), size=feature_map.shape[-2:], mode="nearest")
        mask_sum = resized_mask.sum(dim=(2, 3)).clamp_min(1.0)
        foreground_embedding = (feature_map * resized_mask).sum(dim=(2, 3)) / mask_sum
        if float(resized_mask.sum().item()) <= 0.0:
            foreground_embedding = global_embedding
        payload = {
            "foreground_embedding": F.normalize(foreground_embedding, dim=1).detach().cpu(),
            "global_embedding": global_embedding.detach().cpu(),
            "feature_shape": list(feature_map.shape),
        }
    torch.save(payload, embedding_path)


def _update_diversity_scores(bank: ExemplarMemoryBank) -> None:
    embedding_rows: list[tuple[ExemplarItem, torch.Tensor]] = []
    for item in bank.items:
        if not item.embedding_path:
            continue
        embedding = torch.load(item.embedding_path, map_location="cpu", weights_only=False)
        if isinstance(embedding, dict):
            embedding = embedding.get("foreground_embedding", embedding.get("global_embedding", next(iter(embedding.values()))))
        if not isinstance(embedding, torch.Tensor):
            continue
        vector = embedding.squeeze(0) if embedding.dim() > 1 else embedding
        embedding_rows.append((item, F.normalize(vector.float(), dim=0)))

    if len(embedding_rows) <= 1:
        return
    first_dim = embedding_rows[0][1].shape
    if any(vector.shape != first_dim for _, vector in embedding_rows):
        warnings.warn("Skipping diversity update because exemplar embedding dimensions differ.", stacklevel=2)
        return

    embeddings = torch.stack([vector for _, vector in embedding_rows])
    similarities = embeddings @ embeddings.T
    for index, (item, _) in enumerate(embedding_rows):
        item_similarities = similarities[index].clone()
        item_similarities[index] = -1.0
        item.diversity_score = float(1.0 - item_similarities.max().item())


def _quality_rank(item: ExemplarItem) -> float:
    return (
        item.quality_score
        + 0.5 * item.boundary_score
        + 0.5 * item.diversity_score
        + 0.25 * item.difficulty_score
        - 0.5 * item.uncertainty_score
        - 0.25 * item.false_positive_risk
    )


def _filter_low_value_items(
    bank: ExemplarMemoryBank,
    *,
    min_positive_quality: float,
    min_negative_quality: float,
    min_diversity: float,
    max_uncertainty: float,
    min_negative_false_positive_risk: float,
    min_items_per_type: int,
) -> None:
    protected_ids: set[str] = set()
    for exemplar_type in ["positive", "boundary", "negative"]:
        ranked = sorted(bank.get_items(type=exemplar_type), key=_quality_rank, reverse=True)
        protected_ids.update(item.item_id for item in ranked[: max(min_items_per_type, 0)])

    for item in list(bank.items):
        if item.item_id in protected_ids:
            continue

        reject_reason = ""
        if item.type in {"positive", "boundary"}:
            if item.quality_score < min_positive_quality:
                reject_reason = f"quality_score<{min_positive_quality:.3f}"
            elif item.uncertainty_score > max_uncertainty:
                reject_reason = f"uncertainty_score>{max_uncertainty:.3f}"
            elif item.diversity_score < min_diversity:
                reject_reason = f"diversity_score<{min_diversity:.3f}"
        elif item.type == "negative":
            if item.quality_score < min_negative_quality:
                reject_reason = f"negative_quality_score<{min_negative_quality:.3f}"
            elif item.false_positive_risk < min_negative_false_positive_risk:
                reject_reason = f"false_positive_risk<{min_negative_false_positive_risk:.3f}"
            elif item.diversity_score < min_diversity:
                reject_reason = f"diversity_score<{min_diversity:.3f}"

        if reject_reason:
            bank.reject_item(item.item_id, reject_reason)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MedEx-SAM3 candidate exemplar bank.")
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--min-positive-quality", type=float, default=0.55)
    parser.add_argument("--min-negative-quality", type=float, default=0.60)
    parser.add_argument("--min-diversity", type=float, default=0.02)
    parser.add_argument("--max-uncertainty", type=float, default=0.65)
    parser.add_argument("--min-negative-false-positive-risk", type=float, default=0.30)
    parser.add_argument("--min-items-per-type", type=int, default=8)
    parser.add_argument("--dummy", action="store_true")
    add_yolo_bbox_args(parser)
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
    sam3_device, sam3_wrapper, sam3_model = _build_sam3_embedding_stack(args.checkpoint, allow_dummy=args.dummy)
    box_provider = build_box_provider_from_args(args, default_cache_name="build_exemplar_bank.json")
    bank = ExemplarMemoryBank()
    checker = LeakageChecker()

    for index, record in enumerate(records):
        if "polypgen" in record.get("dataset_name", "").lower():
            continue
        image, mask = load_record_tensors(record, args.image_size, fallback_index=index)
        prompt_box = (
            box_provider.get_box(record, args.image_size, image=image, mask=mask, fallback_index=index)
            if box_provider is not None
            else mask_to_box(mask)
        )
        score_payload = _score_record_with_sam3(sam3_model, image, mask, prompt_box, sam3_device)
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
            if sam3_wrapper is not None:
                _save_sam3_embedding(sam3_wrapper, crop_tensor, mask_tensor, embedding_path, sam3_device)
            else:
                with torch.no_grad():
                    embeddings = encoder(crop_tensor.unsqueeze(0), mask_tensor.unsqueeze(0) if exemplar_type != "negative" else None)
                torch.save(embeddings, embedding_path)

            quality_score = score_payload["quality"]
            boundary_score = score_payload["boundary"]
            false_positive_risk = score_payload["false_positive_risk"]
            if exemplar_type == "negative":
                quality_score = score_payload["negative_quality"]
                boundary_score = 0.3
                false_positive_risk = max(0.4, false_positive_risk)
            elif exemplar_type == "boundary":
                quality_score = score_payload["boundary"]
                boundary_score = score_payload["boundary"]

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
                quality_score=quality_score,
                boundary_score=boundary_score,
                diversity_score=0.5,
                difficulty_score=score_payload["difficulty"],
                uncertainty_score=score_payload["uncertainty"],
                false_positive_risk=false_positive_risk,
                created_at=datetime.now(timezone.utc).isoformat(),
                version="v0",
                notes="candidate",
            )
            ok, reason = checker.check_item(item)
            if not ok:
                bank.reject_item(item.item_id, reason or "leakage")
                continue
            bank.add_item(item)

    _update_diversity_scores(bank)
    _filter_low_value_items(
        bank,
        min_positive_quality=args.min_positive_quality,
        min_negative_quality=args.min_negative_quality,
        min_diversity=args.min_diversity,
        max_uncertainty=args.max_uncertainty,
        min_negative_false_positive_risk=args.min_negative_false_positive_risk,
        min_items_per_type=args.min_items_per_type,
    )
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
