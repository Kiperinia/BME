"""PolypGen 零样本评估脚本。

面向选定 PolypGen-SAM3 序列执行 MedicalSAM3 零样本验证，并区分有无 prompt 的样本统计结果。
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.medsam3_base import build_medsam3
from utils.metrics import compute_all_metrics


DEFAULT_SEQUENCES = ["seq8", "seq10", "seq15", "seq16", "seq23"]


class PolypGenZeroShotDataset(Dataset):
    """读取零样本评估所需的图像、mask 与可选 bbox prompt。"""

    def __init__(
        self,
        data_dir: str,
        sequences: List[str],
        image_size: int = 1024,
        prompted_only: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        self.prompts_path = self.data_dir / "prompts.json"
        self.image_size = image_size
        self.sequences = sequences
        self.prompted_only = prompted_only

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir() or not self.prompts_path.is_file():
            raise FileNotFoundError("Dataset directory must contain images/, masks/, and prompts.json")

        self.prompts = json.loads(self.prompts_path.read_text(encoding="utf-8"))
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file():
                continue
            if not any(image_path.name.startswith(seq) for seq in self.sequences):
                continue

            mask_path = self.mask_dir / image_path.name
            if not mask_path.is_file():
                continue

            bbox = self.prompts.get(image_path.name)
            if self.prompted_only and bbox is None:
                continue

            seq_match = re.match(r"^(seq\d+)", image_path.stem)
            sequence = seq_match.group(1) if seq_match else "unknown"
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "bbox": bbox,
                    "has_prompt": bbox is not None,
                    "sequence": sequence,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        bbox = sample["bbox"] if sample["bbox"] is not None else [0, 0, 0, 0]

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "image_path": str(image_path),
            "sequence": sample["sequence"],
            "has_prompt": bool(sample["has_prompt"]),
        }


def build_parser() -> argparse.ArgumentParser:
    """构建零样本评估脚本的命令行参数。"""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Zero-shot MedicalSAM3 evaluation on selected PolypGen-SAM3 sequences."
    )
    parser.add_argument(
        "--data-dir",
        default=str(script_dir / "data" / "PolypGen-SAM3-seq8_10_15_16_23"),
        help="Directory containing the extracted PolypGen-SAM3 subset.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(script_dir / "checkpoint" / "MedSAM3.pt"),
        help="Path to the MedicalSAM3 checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Image size used by the evaluation dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size. Batch size 1 is recommended for prompt-aware zero-shot evaluation.",
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
        help="Binary threshold for mask evaluation.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        help="Sequence prefixes to evaluate.",
    )
    parser.add_argument(
        "--prompted-only",
        action="store_true",
        help="Evaluate only samples with bbox prompts in prompts.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save predicted masks.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary JSON path.",
    )
    return parser


def squeeze_mask_dims(mask: torch.Tensor) -> torch.Tensor:
    """将 mask 统一整理为 (B, 1, H, W) 形状。"""

    while mask.dim() > 4:
        mask = mask.squeeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
    return mask


def save_prediction(save_dir: Path, image_path: str, pred_mask: torch.Tensor) -> None:
    """将预测 mask 保存为 PNG，便于离线检查。"""

    save_dir.mkdir(parents=True, exist_ok=True)
    pred_np = pred_mask.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    cv2.imwrite(str(save_dir / (Path(image_path).stem + ".png")), pred_np)


def empty_prediction_like(target: torch.Tensor, device: str) -> Dict[str, torch.Tensor]:
    """为无 prompt 样本生成与目标形状一致的空预测。"""

    pred_mask = torch.zeros_like(target, device=device)
    confidence = torch.zeros((target.shape[0], 1), dtype=torch.float32, device=device)
    return {
        "masks": pred_mask,
        "iou_predictions": confidence,
    }


def to_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def update_stats(bucket: Dict[str, float], metrics: Dict[str, float], confidence: float) -> None:
    """将单样本指标累计到统计桶中。"""

    bucket["count"] += 1
    bucket["dice"] += metrics["dice"]
    bucket["iou"] += metrics["iou"]
    bucket["precision"] += metrics["precision"]
    bucket["recall"] += metrics["recall"]
    bucket["mean_confidence"] += confidence


def finalize_stats(raw_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """把累计统计转换为最终平均值。"""

    final = {}
    for key, values in raw_stats.items():
        count = int(values["count"])
        if count == 0:
            continue
        final[key] = {
            "count": count,
            "dice": values["dice"] / count,
            "iou": values["iou"] / count,
            "precision": values["precision"] / count,
            "recall": values["recall"] / count,
            "mean_confidence": values["mean_confidence"] / count,
        }
    return final


def run_evaluation(args: argparse.Namespace) -> Dict[str, object]:
    """执行 PolypGen 零样本评估并输出整体与分序列摘要。"""

    dataset = PolypGenZeroShotDataset(
        data_dir=args.data_dir,
        sequences=args.sequences,
        image_size=args.image_size,
        prompted_only=args.prompted_only,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No PolypGen samples found under {args.data_dir}")

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

    raw_stats = defaultdict(
        lambda: {
            "count": 0,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_confidence": 0.0,
        }
    )
    prompt_stats = {"with_prompt": 0, "without_prompt": 0}
    total_samples = 0
    save_dir = Path(args.save_dir) if args.save_dir else None

    progress = tqdm(dataloader, desc="PolypGen zeroshot", unit="sample")
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in progress:
            images = batch["image"].to(args.device, non_blocking=True).float()
            targets = squeeze_mask_dims(batch["mask"].to(args.device, non_blocking=True).float())
            bboxes = batch["bbox"].to(args.device, non_blocking=True).float()
            sequences = list(batch["sequence"])
            image_paths = list(batch["image_path"])
            has_prompt_list = [to_bool_flag(value) for value in batch["has_prompt"]]

            for idx in range(images.shape[0]):
                image = images[idx : idx + 1]
                target = targets[idx : idx + 1]
                sequence = sequences[idx]
                image_path = image_paths[idx]
                has_prompt = has_prompt_list[idx]

                if has_prompt:
                    output = model(image, bboxes=bboxes[idx : idx + 1])
                    prompt_stats["with_prompt"] += 1
                else:
                    output = empty_prediction_like(target, args.device)
                    prompt_stats["without_prompt"] += 1

                pred_mask = squeeze_mask_dims(output["masks"]).float()
                metrics = compute_all_metrics(pred_mask, target, threshold=args.threshold)
                confidence = float(output["iou_predictions"].detach().mean().item())

                update_stats(raw_stats["overall"], metrics, confidence)
                update_stats(raw_stats[sequence], metrics, confidence)
                total_samples += 1

                if save_dir is not None:
                    save_prediction(save_dir, image_path, pred_mask[0])

                overall = raw_stats["overall"]
                progress.set_postfix(
                    dice=f"{overall['dice'] / overall['count']:.4f}",
                    iou=f"{overall['iou'] / overall['count']:.4f}",
                )

                if args.max_samples is not None and total_samples >= args.max_samples:
                    break

            if args.max_samples is not None and total_samples >= args.max_samples:
                break

    elapsed = time.perf_counter() - start_time
    final_stats = finalize_stats(raw_stats)
    summary: Dict[str, object] = {
        "dataset": str(Path(args.data_dir).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
        "image_size": args.image_size,
        "sequences": args.sequences,
        "prompted_only": args.prompted_only,
        "num_samples": total_samples,
        "prompt_stats": prompt_stats,
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed if elapsed > 0 else 0.0,
        "overall": final_stats.pop("overall"),
        "per_sequence": final_stats,
    }
    return summary


def main() -> int:
    """零样本评估入口。"""

    args = build_parser().parse_args()
    summary = run_evaluation(args)

    print("Zero-shot evaluation summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        summary_path = (
            Path(__file__).resolve().parent
            / "outputs"
            / "val_polypgen_zeroshot_summary.json"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())