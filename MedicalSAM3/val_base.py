import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.medsam3_base import build_medsam3
from utils.dataset import BUSIDataset, KvasirSEGDataset
from utils.metrics import compute_all_metrics


class ResizeOnlyTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image=None, mask=None, **kwargs):
        result = {}
        if image is not None:
            result["image"] = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        if mask is not None:
            result["mask"] = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
        return result


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    return argparse.ArgumentParser(description="Validate MedSAM3 on Kvasir-SEG or BUSI").parse_args()


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Validate MedSAM3 on Kvasir-SEG or BUSI")
    parser.add_argument(
        "--dataset",
        default="kvasir",
        choices=["kvasir", "busi"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--data-root",
        default=str(script_dir / "data"),
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(script_dir / "checkpoint" / "MedSAM3.pt"),
        help="Path to MedSAM3 checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for validation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Validation image size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask threshold used by metric computation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save predicted masks.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path to save the final summary JSON.",
    )
    return parser


def squeeze_mask_dims(mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() > 4:
        mask = mask.squeeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
    return mask


def build_dataset(args: argparse.Namespace):
    transform = ResizeOnlyTransform(args.image_size)
    if args.dataset == "kvasir":
        return KvasirSEGDataset(
            args.data_root,
            transform=transform,
            image_size=args.image_size,
        )
    return BUSIDataset(
        args.data_root,
        transform=transform,
        image_size=args.image_size,
    )


def save_prediction(save_dir: Path, image_path: str, pred_mask: torch.Tensor) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_np = pred_mask.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    out_name = Path(image_path).stem + ".png"
    cv2.imwrite(str(save_dir / out_name), pred_np)


def run_validation(args: argparse.Namespace) -> Dict[str, float]:
    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {args.dataset} is empty under {args.data_root}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = build_medsam3(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
        load_from_hf=False,
    )
    model.eval()

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mean_confidence": 0.0,
    }
    total_samples = 0
    save_dir = Path(args.save_dir) if args.save_dir else None

    progress = tqdm(dataloader, desc=f"Validate {args.dataset}", unit="batch")
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in progress:
            images = batch["image"].float()
            if images.max() > 1.0:
                images = images / 255.0

            images = images.to(args.device, non_blocking=True)
            bboxes = batch["bbox"].float().to(args.device, non_blocking=True)
            targets = batch["mask"].float().to(args.device, non_blocking=True)

            outputs = model(images, bboxes=bboxes)
            pred_masks = squeeze_mask_dims(outputs["masks"]).float()
            targets = squeeze_mask_dims(targets)

            metrics = compute_all_metrics(pred_masks, targets, threshold=args.threshold)
            batch_size = images.shape[0]
            total_samples += batch_size

            for key in ["dice", "iou", "precision", "recall"]:
                totals[key] += metrics[key] * batch_size

            confidence = outputs["iou_predictions"].detach().float().mean().item()
            totals["mean_confidence"] += confidence * batch_size

            if save_dir is not None:
                image_paths = batch["image_path"]
                for idx in range(batch_size):
                    save_prediction(save_dir, image_paths[idx], pred_masks[idx])

            averaged = {k: v / total_samples for k, v in totals.items()}
            progress.set_postfix(
                dice=f"{averaged['dice']:.4f}",
                iou=f"{averaged['iou']:.4f}",
            )

            if args.max_samples is not None and total_samples >= args.max_samples:
                break

    elapsed = time.perf_counter() - start_time
    summary = {k: v / total_samples for k, v in totals.items()}
    summary.update(
        {
            "dataset": args.dataset,
            "num_samples": total_samples,
            "image_size": args.image_size,
            "device": args.device,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "elapsed_seconds": elapsed,
            "samples_per_second": total_samples / elapsed if elapsed > 0 else 0.0,
        }
    )
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_validation(args)

    print("Validation summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = (
            Path(__file__).resolve().parent
            / "outputs"
            / f"val_{args.dataset}_summary.json"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())