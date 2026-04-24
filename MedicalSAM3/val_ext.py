import argparse
import json
import time
from pathlib import Path
from typing import Dict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.medsam3_base import build_medsam3
from models.medsam3_ext import build_medsam3_extended
from utils.dataset import BUSIDataset, KvasirSEGDataset
from utils.metrics import (
    compute_all_metrics,
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
)


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


class DummyBBoxBaseModel(nn.Module):
    """用于本地烟雾验证的占位基础模型。"""

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = images.shape
        masks = torch.zeros(batch_size, 1, height, width, device=images.device)
        if bboxes is None:
            bboxes = torch.tensor(
                [[width * 0.25, height * 0.25, width * 0.75, height * 0.75]],
                dtype=torch.float32,
                device=images.device,
            ).repeat(batch_size, 1)

        for idx in range(batch_size):
            x1, y1, x2, y2 = bboxes[idx].round().long().tolist()
            x1 = max(0, min(x1, width - 1))
            x2 = max(x1 + 1, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(y1 + 1, min(y2, height))
            masks[idx, :, y1:y2, x1:x2] = 0.85

        scores = torch.full((batch_size, 1), 0.5, dtype=torch.float32, device=images.device)
        return {
            "masks": masks,
            "iou_predictions": scores,
        }


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Validate MedSAM3Extended on Kvasir-SEG or BUSI")
    parser.add_argument("--dataset", default="kvasir", choices=["kvasir", "busi"], help="Dataset name.")
    parser.add_argument("--data-root", default=str(script_dir / "data"), help="Root directory containing dataset folders.")
    parser.add_argument("--checkpoint", default=str(script_dir / "checkpoint" / "MedSAM3.pt"), help="Path to MedSAM3 checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used for validation.")
    parser.add_argument("--image-size", type=int, default=1024, help="Validation image size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold used by metric computation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--save-dir", default=None, help="Optional directory to save predicted masks.")
    parser.add_argument("--summary-path", default=None, help="Optional path to save the final summary JSON.")
    parser.add_argument("--base-model", default="medsam3", choices=["medsam3", "dummy"], help="Choose real MedSAM3 or a dummy backbone for smoke tests.")
    parser.add_argument("--prompt-mode", default="dataset-bbox", choices=["dataset-bbox", "apg-only"], help="Use dataset bbox prompts or let APG generate prompts.")
    parser.add_argument("--disable-msfa", action="store_true", help="Disable MSFA.")
    parser.add_argument("--disable-apg", action="store_true", help="Disable APG.")
    parser.add_argument("--disable-brh", action="store_true", help="Disable BRH.")
    parser.add_argument("--disable-tga", action="store_true", help="Disable TGA.")
    parser.add_argument("--small-polyp-ratio", type=float, default=0.08, help="Mask area ratio threshold for defining small polyps.")
    parser.add_argument("--low-contrast-threshold", type=float, default=0.08, help="Foreground-vs-background contrast threshold for low-contrast samples.")
    parser.add_argument("--blurry-boundary-threshold", type=float, default=0.12, help="Boundary gradient threshold for blurry-boundary samples.")
    return parser


def squeeze_mask_dims(mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() > 4:
        mask = mask.squeeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
    return mask


def to_probability(mask: torch.Tensor) -> torch.Tensor:
    if mask.min() < 0 or mask.max() > 1:
        return mask.sigmoid()
    return mask


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) == 9).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0).float()
    return (dilated - eroded).clamp(0, 1)


def boundary_f1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_b = boundary_band(pred)
    target_b = boundary_band(target)
    tp = (pred_b * target_b).flatten(1).sum(dim=1)
    pred_sum = pred_b.flatten(1).sum(dim=1)
    target_sum = target_b.flatten(1).sum(dim=1)
    precision = (tp + 1e-6) / (pred_sum + 1e-6)
    recall = (tp + 1e-6) / (target_sum + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


def sobel_gradient_magnitude(gray_image: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=gray_image.device,
        dtype=gray_image.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(gray_image, sobel_x, padding=1)
    grad_y = F.conv2d(gray_image, sobel_y, padding=1)
    return (grad_x.square() + grad_y.square()).sqrt()


def estimate_sample_tags(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    small_polyp_ratio: float,
    low_contrast_threshold: float,
    blurry_boundary_threshold: float,
) -> Dict[str, bool]:
    area_ratio = target_mask.mean().item()
    small_polyp = area_ratio <= small_polyp_ratio

    gray = image.mean(dim=0)
    mask2d = target_mask.squeeze(0) > 0.5
    mask4d = mask2d.float().unsqueeze(0).unsqueeze(0)
    ring = (F.max_pool2d(mask4d, kernel_size=31, stride=1, padding=15) - mask4d).clamp(0, 1)
    ring2d = ring.squeeze(0).squeeze(0) > 0.5

    if mask2d.any():
        inside_mean = gray[mask2d].mean().item()
    else:
        inside_mean = 0.0

    if ring2d.any():
        outside_mean = gray[ring2d].mean().item()
    else:
        outside_mean = gray[~mask2d].mean().item() if (~mask2d).any() else 0.0

    contrast_gap = abs(inside_mean - outside_mean)
    low_contrast = contrast_gap <= low_contrast_threshold

    boundary = boundary_band(mask4d).squeeze(0).squeeze(0) > 0.5
    grad_mag = sobel_gradient_magnitude(gray.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    boundary_grad = grad_mag[boundary].mean().item() if boundary.any() else 0.0
    blurry_boundary = boundary_grad <= blurry_boundary_threshold

    return {
        "low_contrast": low_contrast,
        "blurry_boundary": blurry_boundary,
        "small_polyp": small_polyp,
        "hard_case": low_contrast or blurry_boundary or small_polyp,
    }


def init_group_totals() -> Dict[str, float]:
    return {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "boundary_f1": 0.0,
        "count": 0.0,
    }


def summarize_group_totals(group_totals: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for name, totals in group_totals.items():
        count = totals["count"]
        if count == 0:
            report[name] = {
                "count": 0.0,
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "boundary_f1": 0.0,
            }
            continue

        report[name] = {
            "count": count,
            "dice": totals["dice"] / count,
            "iou": totals["iou"] / count,
            "precision": totals["precision"] / count,
            "recall": totals["recall"] / count,
            "boundary_f1": totals["boundary_f1"] / count,
        }
    return report


def build_dataset(args: argparse.Namespace):
    transform = ResizeOnlyTransform(args.image_size)
    if args.dataset == "kvasir":
        return KvasirSEGDataset(args.data_root, transform=transform, image_size=args.image_size)
    return BUSIDataset(args.data_root, transform=transform, image_size=args.image_size)


def save_prediction(save_dir: Path, image_path: str, pred_mask: torch.Tensor, threshold: float) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    pred = pred_mask.detach().cpu()
    if pred.min() < 0 or pred.max() > 1:
        pred = pred.sigmoid()
    pred_np = (pred.squeeze().numpy() > threshold).astype("uint8") * 255
    out_name = Path(image_path).stem + ".png"
    cv2.imwrite(str(save_dir / out_name), pred_np)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.base_model == "dummy":
        base_model = DummyBBoxBaseModel().to(args.device)
    else:
        base_model = build_medsam3(
            checkpoint_path=args.checkpoint,
            image_size=args.image_size,
            device=args.device,
            load_from_hf=False,
        )

    model = build_medsam3_extended(
        base_model=base_model,
        use_msfa=not args.disable_msfa,
        use_apg=not args.disable_apg,
        use_brh=not args.disable_brh,
        use_tga=not args.disable_tga,
        image_size=args.image_size,
    )
    return model.to(args.device)


def run_validation(args: argparse.Namespace) -> Dict[str, object]:
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

    model = build_model(args)
    model.eval()

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "boundary_f1": 0.0,
        "mean_confidence": 0.0,
        "mean_refinement_gate": 0.0,
        "mean_error_confidence": 0.0,
        "mean_shape_prior": 0.0,
    }
    stratified_totals = {
        "all": init_group_totals(),
        "low_contrast": init_group_totals(),
        "blurry_boundary": init_group_totals(),
        "small_polyp": init_group_totals(),
        "hard_case": init_group_totals(),
    }
    total_samples = 0
    save_dir = Path(args.save_dir) if args.save_dir else None

    progress = tqdm(dataloader, desc=f"Validate {args.dataset} ext", unit="batch")
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in progress:
            images = batch["image"].float()
            if images.max() > 1.0:
                images = images / 255.0

            images = images.to(args.device, non_blocking=True)
            targets = squeeze_mask_dims(batch["mask"].float().to(args.device, non_blocking=True))
            text_prompts = batch.get("text_prompt")

            if args.prompt_mode == "apg-only":
                bboxes = None
            else:
                bboxes = batch["bbox"].float().to(args.device, non_blocking=True)

            outputs = model(
                images,
                bboxes=bboxes,
                text_prompt=text_prompts,
            )
            pred_tensor = outputs.get("mask_logits", outputs["masks"])
            pred_masks = squeeze_mask_dims(pred_tensor.float())
            pred_prob = to_probability(pred_masks)
            pred_binary = (pred_prob > args.threshold).float()
            target_binary = targets.float()

            metrics = compute_all_metrics(pred_masks, targets, threshold=args.threshold)
            boundary_f1_values = boundary_f1_per_sample(pred_binary, target_binary)
            batch_size = images.shape[0]
            total_samples += batch_size

            for key in ["dice", "iou", "precision", "recall"]:
                totals[key] += metrics[key] * batch_size
            totals["boundary_f1"] += boundary_f1_values.mean().item() * batch_size

            dice_values = dice_coefficient(pred_binary.squeeze(1), target_binary.squeeze(1))
            iou_values = iou_score(pred_binary.squeeze(1), target_binary.squeeze(1))
            precision_values = precision_score(pred_binary.squeeze(1), target_binary.squeeze(1))
            recall_values = recall_score(pred_binary.squeeze(1), target_binary.squeeze(1))

            confidence = outputs["iou_predictions"].detach().float().mean().item()
            totals["mean_confidence"] += confidence * batch_size

            if "refinement_gate" in outputs:
                totals["mean_refinement_gate"] += outputs["refinement_gate"].detach().mean().item() * batch_size
            if "error_confidence" in outputs:
                totals["mean_error_confidence"] += outputs["error_confidence"].detach().mean().item() * batch_size
            if "shape_prior" in outputs:
                totals["mean_shape_prior"] += outputs["shape_prior"].detach().mean().item() * batch_size

            if save_dir is not None:
                image_paths = batch["image_path"]
                masks_to_save = outputs["masks"]
                for idx in range(batch_size):
                    save_prediction(save_dir, image_paths[idx], masks_to_save[idx], args.threshold)

            for idx in range(batch_size):
                tags = estimate_sample_tags(
                    images[idx].detach().cpu(),
                    target_binary[idx].detach().cpu(),
                    small_polyp_ratio=args.small_polyp_ratio,
                    low_contrast_threshold=args.low_contrast_threshold,
                    blurry_boundary_threshold=args.blurry_boundary_threshold,
                )
                metric_values = {
                    "dice": dice_values[idx].item(),
                    "iou": iou_values[idx].item(),
                    "precision": precision_values[idx].item(),
                    "recall": recall_values[idx].item(),
                    "boundary_f1": boundary_f1_values[idx].item(),
                }

                group_names = ["all"]
                for name in ["low_contrast", "blurry_boundary", "small_polyp", "hard_case"]:
                    if tags[name]:
                        group_names.append(name)

                for group_name in group_names:
                    group = stratified_totals[group_name]
                    group["count"] += 1.0
                    for metric_name, metric_value in metric_values.items():
                        group[metric_name] += metric_value

            averaged = {k: v / total_samples for k, v in totals.items()}
            progress.set_postfix(dice=f"{averaged['dice']:.4f}", iou=f"{averaged['iou']:.4f}")

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
            "base_model": args.base_model,
            "prompt_mode": args.prompt_mode,
            "elapsed_seconds": elapsed,
            "samples_per_second": total_samples / elapsed if elapsed > 0 else 0.0,
            "stratified": summarize_group_totals(stratified_totals),
            "stratification_thresholds": {
                "small_polyp_ratio": args.small_polyp_ratio,
                "low_contrast_threshold": args.low_contrast_threshold,
                "blurry_boundary_threshold": args.blurry_boundary_threshold,
            },
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
        summary_path = Path(__file__).resolve().parent / "outputs" / f"val_{args.dataset}_ext_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())