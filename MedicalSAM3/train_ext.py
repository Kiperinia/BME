"""MedicalSAM3 扩展训练脚本。

负责构建数据集、扩展模型与 BRH 监督损失，并在当前高层包装器约束下训练外围扩展模块。
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from models.medsam3_base import build_medsam3
from models.medsam3_ext import build_medsam3_extended
from utils.dataset import create_dataset
from utils.losses import BoundaryLoss, DiceLoss
from utils.metrics import compute_all_metrics

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class ResizeOnlyTransform:
    """在缺少 albumentations 时提供最小化的 resize 变换。"""

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


def build_train_transform(image_size: int):
    """构建训练阶段增强；不可用时退化为纯 resize。"""

    if not HAS_ALBUMENTATIONS:
        return ResizeOnlyTransform(image_size)

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                p=0.4,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                    A.CLAHE(clip_limit=3.0, p=1.0),
                ],
                p=0.2,
            ),
        ]
    )


def squeeze_mask_dims(mask: torch.Tensor) -> torch.Tensor:
    """将 mask 统一整理为 (B, 1, H, W) 形状。"""

    while mask.dim() > 4:
        mask = mask.squeeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
    return mask


def to_probability(mask: torch.Tensor) -> torch.Tensor:
    """将 logits 或概率图统一转换为概率。"""

    if mask.min() < 0 or mask.max() > 1:
        return mask.sigmoid()
    return mask


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    """通过简单形态学近似构造边界带。"""

    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) == 9).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0).float()
    return (dilated - eroded).clamp(0, 1)


def boundary_f1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """按样本计算边界 F1，便于训练和验证汇总。"""

    pred_b = boundary_band(pred)
    target_b = boundary_band(target)
    tp = (pred_b * target_b).flatten(1).sum(dim=1)
    pred_sum = pred_b.flatten(1).sum(dim=1)
    target_sum = target_b.flatten(1).sum(dim=1)
    precision = (tp + 1e-6) / (pred_sum + 1e-6)
    recall = (tp + 1e-6) / (target_sum + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    """从单个二值 mask 提取 xyxy 边界框；空 mask 返回占位框。"""

    if mask.dim() == 3:
        mask = mask.squeeze(0)
    positive = mask > 0.5
    ys, xs = torch.where(positive)
    if len(xs) == 0 or len(ys) == 0:
        return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=mask.device)
    x1 = xs.min().float()
    y1 = ys.min().float()
    x2 = (xs.max() + 1).float()
    y2 = (ys.max() + 1).float()
    return torch.stack([x1, y1, x2, y2])


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """批量从 mask 张量推导边界框。"""

    masks = squeeze_mask_dims(masks)
    return torch.stack([mask_to_box(mask) for mask in masks], dim=0)


class TransformSubset(Dataset):
    """为拆分后的子集补充增强、张量化和 bbox 重算。"""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.subset[idx]
        image = sample["image"]
        mask = sample["mask"]

        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze(0).cpu().numpy()
        else:
            mask_np = mask

        image_np = image_np.astype(np.uint8)
        mask_np = mask_np.astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed["image"]
            mask_np = transformed["mask"]

        if image_np.ndim == 3 and image_np.shape[-1] == 3:
            image_np = np.transpose(image_np, (2, 0, 1))

        image_tensor = torch.from_numpy(image_np).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
        bbox_tensor = mask_to_box(mask_tensor)

        return {
            **sample,
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
        }


class DummyBBoxBaseModel(nn.Module):
    """烟雾测试占位骨干，直接把 bbox 区域映射成粗分割结果。"""

    def forward(
        self,
        images: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
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


class BRHTrainingLoss(nn.Module):
    """组合分割主损失与 BRH 辅助监督的训练损失。"""

    def __init__(
        self,
        seg_bce_weight: float = 1.0,
        seg_dice_weight: float = 1.0,
        seg_boundary_weight: float = 0.25,
        err_weight: float = 0.5,
        boundary_align_weight: float = 0.5,
        delta_weight: float = 0.25,
        apg_bbox_weight: float = 0.2,
    ):
        super().__init__()
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        self.seg_boundary_weight = seg_boundary_weight
        self.err_weight = err_weight
        self.boundary_align_weight = boundary_align_weight
        self.delta_weight = delta_weight
        self.apg_bbox_weight = apg_bbox_weight

        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss(weight=1.0)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        targets = squeeze_mask_dims(targets).float()
        mask_logits = squeeze_mask_dims(outputs["mask_logits"]).float()

        total_loss = mask_logits.new_tensor(0.0)
        logs: Dict[str, float] = {}

        if self.seg_bce_weight > 0:
            seg_bce = F.binary_cross_entropy_with_logits(mask_logits, targets)
            total_loss = total_loss + self.seg_bce_weight * seg_bce
            logs["seg_bce"] = seg_bce.item()
        if self.seg_dice_weight > 0:
            seg_dice = self.dice_loss(mask_logits, targets)
            total_loss = total_loss + self.seg_dice_weight * seg_dice
            logs["seg_dice"] = seg_dice.item()
        if self.seg_boundary_weight > 0:
            seg_boundary = self.boundary_loss(mask_logits, targets)
            total_loss = total_loss + self.seg_boundary_weight * seg_boundary
            logs["seg_boundary"] = seg_boundary.item()

        training_targets = outputs.get("training_targets")
        if training_targets is not None:
            if self.err_weight > 0 and "error_confidence" in outputs:
                err_target = training_targets["error_region"].detach().clamp(0, 1)
                err_loss = F.binary_cross_entropy(
                    outputs["error_confidence"].clamp(1e-6, 1 - 1e-6),
                    err_target,
                )
                total_loss = total_loss + self.err_weight * err_loss
                logs["err_bce"] = err_loss.item()

            if self.boundary_align_weight > 0 and "boundary_mask" in outputs:
                gt_boundary = training_targets["gt_boundary"].detach().clamp(0, 1)
                boundary_align = F.binary_cross_entropy(
                    outputs["boundary_mask"].clamp(1e-6, 1 - 1e-6),
                    gt_boundary,
                )
                total_loss = total_loss + self.boundary_align_weight * boundary_align
                logs["boundary_align"] = boundary_align.item()

            if self.delta_weight > 0 and "delta" in outputs and "refinement_gate" in outputs:
                signed_error = training_targets["signed_error"].detach()
                delta_pred = outputs["delta"] * outputs["refinement_gate"]
                delta_loss = F.smooth_l1_loss(delta_pred, signed_error)
                total_loss = total_loss + self.delta_weight * delta_loss
                logs["delta_l1"] = delta_loss.item()

        apg_output = outputs.get("apg_output")
        if (
            self.apg_bbox_weight > 0
            and isinstance(apg_output, dict)
            and "bbox_loss" in apg_output
        ):
            apg_bbox = apg_output["bbox_loss"]
            total_loss = total_loss + self.apg_bbox_weight * apg_bbox
            logs["apg_bbox"] = apg_bbox.item()

        logs["total"] = total_loss.item()
        return total_loss, logs


def build_parser() -> argparse.ArgumentParser:
    """定义扩展训练脚本的命令行参数。"""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train MedSAM3Extended on KvasirCVC or PolypGen external test")
    parser.add_argument(
        "--dataset",
        default="kvasircvc",
        choices=["kvasircvc", "polypgen"],
        help="Dataset name.",
    )
    parser.add_argument("--data-root", default=str(script_dir / "data"), help="Root directory containing dataset folders.")
    parser.add_argument("--checkpoint", default=str(script_dir / "checkpoint" / "MedSAM3.pt"), help="Path to MedSAM3 checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--base-model", default="medsam3", choices=["medsam3", "dummy"], help="Choose real MedSAM3 or a dummy backbone for smoke tests.")
    parser.add_argument("--image-size", type=int, default=1024, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for trainable extension modules.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--jitter-bbox-ratio", type=float, default=0.05, help="Reserved bbox jitter flag for future low-level wrapper training.")
    parser.add_argument("--prompt-mode", default="dataset-bbox", choices=["dataset-bbox", "apg-only"], help="Use dataset bbox prompts or let APG generate prompts.")
    parser.add_argument("--train-scope", default="brh-only", choices=["brh-only", "ext-modules"], help="Current high-level wrapper only supports training outer extension modules.")
    parser.add_argument("--disable-msfa", action="store_true", help="Disable MSFA.")
    parser.add_argument("--disable-apg", action="store_true", help="Disable APG.")
    parser.add_argument("--disable-brh", action="store_true", help="Disable BRH.")
    parser.add_argument("--disable-tga", action="store_true", help="Disable TGA.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold used in validation.")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional cap per epoch for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap for validation smoke tests.")
    parser.add_argument("--save-dir", default=str(script_dir / "outputs" / "train_ext"), help="Directory for checkpoints and summaries.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--seg-bce-weight", type=float, default=1.0, help="Segmentation BCE weight.")
    parser.add_argument("--seg-dice-weight", type=float, default=1.0, help="Segmentation Dice weight.")
    parser.add_argument("--seg-boundary-weight", type=float, default=0.25, help="Boundary-weighted segmentation loss weight.")
    parser.add_argument("--err-weight", type=float, default=0.5, help="BRH error-confidence supervision weight.")
    parser.add_argument("--boundary-align-weight", type=float, default=0.5, help="Boundary-mask alignment weight.")
    parser.add_argument("--delta-weight", type=float, default=0.25, help="Signed correction regression weight.")
    parser.add_argument("--apg-bbox-weight", type=float, default=0.2, help="APG bbox regression weight.")
    return parser


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 Torch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_raw_dataset(args: argparse.Namespace):
    """按数据集名称构建未增强的原始数据集实例。"""

    return create_dataset(
        args.dataset,
        args.data_root,
        transform=None,
        image_size=args.image_size,
    )


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """划分训练/验证集并返回对应 DataLoader。"""

    dataset = build_raw_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {args.dataset} is empty under {args.data_root}")

    n_total = len(dataset)
    n_train = max(1, int(n_total * args.train_ratio))
    n_val = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_train = n_total - 1
        n_val = 1

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=generator)

    train_dataset = TransformSubset(train_subset, build_train_transform(args.image_size))
    val_dataset = TransformSubset(val_subset, ResizeOnlyTransform(args.image_size))

    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace) -> nn.Module:
    """根据参数构建基础模型和扩展模块封装。"""

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


def set_trainable_scope(model: nn.Module, scope: str) -> List[str]:
    """冻结全模型后，仅打开指定范围内模块的训练开关。"""

    for param in model.parameters():
        param.requires_grad = False

    trainable_prefixes: List[str] = []
    if scope == "brh-only":
        if not hasattr(model, "brh"):
            raise ValueError("BRH is disabled, but train-scope is brh-only.")
        modules = ["brh"]
    elif scope == "ext-modules":
        modules = ["image_feature_stem", "fallback_text_embedding", "msfa", "apg", "brh", "tga"]
    else:
        raise ValueError(f"Unsupported train scope: {scope}")

    for name in modules:
        if hasattr(model, name):
            getattr(model, name).train()
            for param in getattr(model, name).parameters():
                param.requires_grad = True
            trainable_prefixes.append(name)

    if hasattr(model, "base"):
        model.base.eval()

    return trainable_prefixes


def collect_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for the selected train scope.")
    return params


def activate_training_scope(model: nn.Module, scope: str) -> None:
    model.eval()
    active_names = set_trainable_scope(model, scope)
    for name in active_names:
        getattr(model, name).train()


def prepare_prompt_boxes(args: argparse.Namespace, targets: torch.Tensor, batch_bbox: torch.Tensor) -> Optional[torch.Tensor]:
    if args.prompt_mode == "apg-only":
        return None
    return batch_bbox


def summarize_logs(log_sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        return {key: 0.0 for key in log_sums}
    return {key: value / count for key, value in log_sums.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BRHTrainingLoss,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """执行单轮训练并返回平均损失日志。"""

    activate_training_scope(model, args.train_scope)

    log_sums: Dict[str, float] = {}
    sample_count = 0
    progress = tqdm(dataloader, desc=f"Train epoch {epoch}", unit="batch")

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].float().to(args.device, non_blocking=True)
        targets = squeeze_mask_dims(batch["mask"].float().to(args.device, non_blocking=True))
        batch_bbox = batch["bbox"].float().to(args.device, non_blocking=True)
        prompt_boxes = prepare_prompt_boxes(args, targets, batch_bbox)
        text_prompts = batch.get("text_prompt")

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            images,
            bboxes=prompt_boxes,
            text_prompt=text_prompts,
            gt_masks=targets,
        )
        loss, logs = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        sample_count += batch_size
        for key, value in logs.items():
            log_sums[key] = log_sums.get(key, 0.0) + value * batch_size

        averages = summarize_logs(log_sums, sample_count)
        progress.set_postfix(loss=f"{averages.get('total', 0.0):.4f}")

        if args.max_train_steps is not None and step >= args.max_train_steps:
            break

    return summarize_logs(log_sums, sample_count)


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: BRHTrainingLoss,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """在验证集上评估扩展模型并汇总损失与指标。"""

    model.eval()

    total_samples = 0
    loss_sums: Dict[str, float] = {}
    metric_sums = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "boundary_f1": 0.0,
        "mean_confidence": 0.0,
    }

    progress = tqdm(dataloader, desc="Validate", unit="batch")
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].float().to(args.device, non_blocking=True)
        targets = squeeze_mask_dims(batch["mask"].float().to(args.device, non_blocking=True))
        batch_bbox = batch["bbox"].float().to(args.device, non_blocking=True)
        prompt_boxes = prepare_prompt_boxes(args, targets, batch_bbox)
        text_prompts = batch.get("text_prompt")

        outputs = model(
            images,
            bboxes=prompt_boxes,
            text_prompt=text_prompts,
            gt_masks=targets,
        )
        _, loss_logs = criterion(outputs, targets)

        pred_tensor = outputs.get("mask_logits", outputs["masks"])
        pred_masks = squeeze_mask_dims(pred_tensor.float())
        pred_prob = to_probability(pred_masks)
        pred_binary = (pred_prob > args.threshold).float()

        metrics = compute_all_metrics(pred_masks, targets, threshold=args.threshold)
        boundary_f1 = boundary_f1_per_sample(pred_binary, targets).mean().item()
        batch_size = images.shape[0]
        total_samples += batch_size

        for key, value in loss_logs.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + value * batch_size
        for key in ["dice", "iou", "precision", "recall"]:
            metric_sums[key] += metrics[key] * batch_size
        metric_sums["boundary_f1"] += boundary_f1 * batch_size
        metric_sums["mean_confidence"] += outputs["iou_predictions"].detach().float().mean().item() * batch_size

        progress.set_postfix(dice=f"{metric_sums['dice'] / total_samples:.4f}")

        if args.max_val_batches is not None and step >= args.max_val_batches:
            break

    summary = summarize_logs(loss_sums, total_samples)
    summary.update({key: value / total_samples for key, value in metric_sums.items()})
    return summary


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
        },
        path,
    )


def save_brh_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    if not hasattr(model, "brh"):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.brh.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
            "module": "brh",
        },
        path,
    )


def maybe_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Optional[str],
    device: str,
) -> Tuple[int, float]:
    if not resume_path:
        return 0, float("-inf")

    state = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), float(state.get("best_metric", float("-inf")))


def main() -> int:
    """训练入口：解析参数、训练、验证并保存检查点。"""

    parser = build_parser()
    args = parser.parse_args()
    if args.disable_brh and args.train_scope == "brh-only":
        raise ValueError("BRH is disabled, but train-scope is brh-only.")
    if args.train_scope == "ext-modules" and not args.disable_msfa and args.batch_size < 2:
        raise ValueError("ext-modules scope with MSFA enabled requires batch-size >= 2 because MSFA contains BatchNorm branches.")

    set_seed(args.seed)
    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args)
    trainable_names = set_trainable_scope(model, args.train_scope)
    optimizer = torch.optim.AdamW(
        collect_trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = BRHTrainingLoss(
        seg_bce_weight=args.seg_bce_weight,
        seg_dice_weight=args.seg_dice_weight,
        seg_boundary_weight=args.seg_boundary_weight,
        err_weight=args.err_weight,
        boundary_align_weight=args.boundary_align_weight,
        delta_weight=args.delta_weight,
        apg_bbox_weight=args.apg_bbox_weight,
    ).to(args.device)

    start_epoch, best_dice = maybe_resume(model, optimizer, args.resume, args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float | int]] = []
    start_time = time.perf_counter()
    print(f"Trainable modules: {', '.join(trainable_names)}")
    print("Note: current MedSAM3Wrapper uses processor/PIL conversion, so training is limited to outer extension modules.")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_summary = train_one_epoch(model, train_loader, optimizer, criterion, args, epoch)

        epoch_record: Dict[str, float | int] = {"epoch": epoch, **{f"train_{k}": v for k, v in train_summary.items()}}
        if epoch % args.val_every == 0:
            val_summary = validate(model, val_loader, criterion, args)
            epoch_record.update({f"val_{k}": v for k, v in val_summary.items()})

            current_dice = float(val_summary.get("dice", 0.0))
            if current_dice >= best_dice:
                best_dice = current_dice
                save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_dice, args)
                save_brh_checkpoint(save_dir / "brh_best.pt", model, epoch, best_dice, args)

        history.append(epoch_record)
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, best_dice, args)
        save_brh_checkpoint(save_dir / "brh_last.pt", model, epoch, best_dice, args)

        print(json.dumps(epoch_record, ensure_ascii=False))

    elapsed = time.perf_counter() - start_time
    summary = {
        "dataset": args.dataset,
        "base_model": args.base_model,
        "train_scope": args.train_scope,
        "prompt_mode": args.prompt_mode,
        "epochs": args.epochs,
        "best_dice": best_dice if best_dice != float("-inf") else None,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    summary_path = save_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Training summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())