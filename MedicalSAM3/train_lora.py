"""基于 SAM3 的轻量 LoRA 训练脚本。

这份脚本面向当前仓库的数据组织方式，只使用 bbox prompt 训练
SAM3 的 prompt encoder 和 mask decoder 路径。
"""

import argparse
import copy
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sam3 import build_sam3_image_model

from utils.dataset import BUSIDataset, KvasirSEGDataset
from utils.losses import CombinedSegLoss
from utils.metrics import compute_all_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: Dict[str, Any] = {
	"model": {
		"sam3_checkpoint": None,
		"custom_checkpoint": str(SCRIPT_DIR / "checkpoint" / "MedSAM3.pt"),
		"load_from_hf": True,
		"use_lora": True,
		"lora_rank": 16,
		"lora_alpha": 32,
		"lora_dropout": 0.1,
		"target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
		"trainable_modules": ["sam_prompt_encoder", "sam_mask_decoder"],
	},
	"data": {
		"dataset": "kvasir",
		"data_root": str(SCRIPT_DIR / "data"),
		"image_size": 1008,
		"batch_size": 1,
		"num_workers": 0,
		"train_ratio": 0.85,
		"jitter_bbox_ratio": 0.05,
	},
	"training": {
		"epochs": 20,
		"learning_rate": 1e-4,
		"weight_decay": 0.01,
		"warmup_steps": 50,
		"gradient_accumulation_steps": 4,
		"mixed_precision": "fp16",
		"output_dir": str(SCRIPT_DIR / "outputs" / "train_lora"),
		"save_every_n_epochs": 5,
		"evaluation_interval": 1,
		"seed": 42,
		"max_grad_norm": 1.0,
	},
	"optimizer": {
		"betas": [0.9, 0.999],
	},
}


class SAM3ResizeNormalize:
	"""仅做 resize 和 SAM3 风格归一化，避免 bbox 与几何增强失配。"""

	def __init__(self, image_size: int):
		self.image_size = image_size

	def __call__(self, image=None, mask=None, **kwargs):
		result = {}
		if image is not None:
			image = cv2.resize(
				image,
				(self.image_size, self.image_size),
				interpolation=cv2.INTER_LINEAR,
			)
			image = image.astype(np.float32) / 255.0
			image = (image - 0.5) / 0.5
			result["image"] = image
		if mask is not None:
			result["mask"] = cv2.resize(
				mask,
				(self.image_size, self.image_size),
				interpolation=cv2.INTER_NEAREST,
			)
		return result


class LoRALinear(nn.Module):
	def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
		super().__init__()
		if rank <= 0:
			raise ValueError("LoRA rank 必须大于 0")

		self.in_features = linear.in_features
		self.out_features = linear.out_features
		self.rank = rank
		self.scaling = alpha / rank
		self.weight = linear.weight
		self.bias = linear.bias
		self.weight.requires_grad = False
		if self.bias is not None:
			self.bias.requires_grad = False

		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.lora_down = nn.Linear(self.in_features, rank, bias=False)
		self.lora_up = nn.Linear(rank, self.out_features, bias=False)
		nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
		nn.init.zeros_(self.lora_up.weight)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		base = F.linear(x, self.weight, self.bias)
		delta = self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
		return base + delta


class SAM3BoxSegModel(nn.Module):
	"""直接使用 SAM3 的图像编码器、prompt encoder 和 mask decoder。"""

	def __init__(self, sam3_model: nn.Module):
		super().__init__()
		self.sam3 = sam3_model

	def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
		batch_size = images.shape[0]
		backbone_out = self.sam3.forward_image(images)
		_, vision_feats, _, feat_sizes = self.sam3._prepare_backbone_features(backbone_out)
		vision_feats[-1] = vision_feats[-1] + self.sam3.no_mem_embed

		feats = [
			feat.permute(1, 2, 0).reshape(batch_size, -1, feat_size[0], feat_size[1])
			for feat, feat_size in zip(vision_feats, feat_sizes)
		]
		image_embed = feats[-1]
		high_res_feats = feats[:-1]

		box_coords = boxes.reshape(-1, 2, 2).float()
		box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=boxes.device)
		box_labels = box_labels.repeat(batch_size, 1)

		sparse_embeddings, dense_embeddings = self.sam3.sam_prompt_encoder(
			points=(box_coords, box_labels),
			boxes=None,
			masks=None,
		)
		low_res_masks, iou_predictions, _, _ = self.sam3.sam_mask_decoder(
			image_embeddings=image_embed,
			image_pe=self.sam3.sam_prompt_encoder.get_dense_pe(),
			sparse_prompt_embeddings=sparse_embeddings,
			dense_prompt_embeddings=dense_embeddings,
			multimask_output=False,
			repeat_image=False,
			high_res_features=high_res_feats,
		)

		if low_res_masks.shape[-2:] != images.shape[-2:]:
			masks = F.interpolate(
				low_res_masks.float(),
				size=images.shape[-2:],
				mode="bilinear",
				align_corners=False,
			)
		else:
			masks = low_res_masks.float()

		return {
			"masks": masks,
			"low_res_masks": low_res_masks.float(),
			"iou_predictions": iou_predictions.float(),
		}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train MedicalSAM3 with SAM3 backbone")
	parser.add_argument(
		"--config",
		default=str(SCRIPT_DIR / "configs" / "config_lora.yaml"),
		help="Path to a YAML/JSON config file.",
	)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Torch device used for training.",
	)
	parser.add_argument("--resume", default=None, help="Optional checkpoint to resume.")
	parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
	parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
	parser.add_argument("--dataset", default=None, choices=["kvasir", "busi"], help="Override dataset.")
	parser.add_argument("--data-root", default=None, help="Override data root.")
	parser.add_argument("--output-dir", default=None, help="Override output directory.")
	return parser


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
	for key, value in updates.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			deep_update(base[key], value)
		else:
			base[key] = value
	return base


def load_config_file(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {}
	if path.suffix.lower() == ".json":
		return json.loads(path.read_text(encoding="utf-8"))

	try:
		import yaml
	except ImportError as exc:
		raise RuntimeError("读取 YAML 配置需要先安装 pyyaml") from exc

	with path.open("r", encoding="utf-8") as handle:
		loaded = yaml.safe_load(handle) or {}
	if not isinstance(loaded, dict):
		raise ValueError("配置文件顶层必须是一个字典")
	return loaded


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
	config = copy.deepcopy(DEFAULT_CONFIG)
	config_path = Path(args.config)
	deep_update(config, load_config_file(config_path))

	if args.dataset is not None:
		config["data"]["dataset"] = args.dataset
	if args.data_root is not None:
		config["data"]["data_root"] = args.data_root
	if args.batch_size is not None:
		config["data"]["batch_size"] = args.batch_size
	if args.epochs is not None:
		config["training"]["epochs"] = args.epochs
	if args.output_dir is not None:
		config["training"]["output_dir"] = args.output_dir
	config["training"]["device"] = args.device
	config["training"]["resume"] = args.resume
	return config


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_custom_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
	ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	if "model" in ckpt and isinstance(ckpt["model"], dict):
		state_dict = ckpt["model"]
	else:
		state_dict = ckpt

	sample_key = next(iter(state_dict), "")
	if "detector." in sample_key:
		state_dict = {
			key.replace("detector.", ""): value
			for key, value in state_dict.items()
			if "detector" in key
		}

	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"Checkpoint missing keys: {len(missing)}")
	if unexpected:
		print(f"Checkpoint unexpected keys: {len(unexpected)}")


def freeze_model(model: nn.Module) -> None:
	for parameter in model.parameters():
		parameter.requires_grad = False


def replace_module(root_module: nn.Module, module_name: str, new_module: nn.Module) -> None:
	parts = module_name.split(".")
	parent = root_module
	for part in parts[:-1]:
		parent = getattr(parent, part)
	setattr(parent, parts[-1], new_module)


def apply_lora(
	model: nn.Module,
	target_modules: Sequence[str],
	rank: int,
	alpha: float,
	dropout: float,
) -> List[str]:
	replaced: List[str] = []
	for module_name, module in list(model.named_modules()):
		if not module_name or not isinstance(module, nn.Linear):
			continue
		leaf_name = module_name.split(".")[-1]
		if not any(target in leaf_name or target in module_name for target in target_modules):
			continue
		replace_module(model, module_name, LoRALinear(module, rank, alpha, dropout))
		replaced.append(module_name)
	return replaced


def enable_named_modules(model: nn.Module, allowed_prefixes: Sequence[str]) -> List[str]:
	trainable = []
	for name, parameter in model.named_parameters():
		if any(prefix in name for prefix in allowed_prefixes):
			parameter.requires_grad = True
			trainable.append(name)
	return trainable


def get_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
	trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
	return {
		name: tensor.detach().cpu()
		for name, tensor in model.state_dict().items()
		if name in trainable_names
	}


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
	total = sum(parameter.numel() for parameter in model.parameters())
	trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	return trainable, total


def build_model(config: Dict[str, Any], device: str) -> Tuple[SAM3BoxSegModel, List[str]]:
	model_cfg = config["model"]
	sam3_checkpoint = model_cfg.get("sam3_checkpoint")
	custom_checkpoint = model_cfg.get("custom_checkpoint")

	checkpoint_path = None
	if sam3_checkpoint:
		candidate = Path(sam3_checkpoint)
		if candidate.is_file():
			checkpoint_path = str(candidate)

	base_model = build_sam3_image_model(
		checkpoint_path=checkpoint_path,
		load_from_HF=bool(model_cfg.get("load_from_hf", True)) if checkpoint_path is None else False,
		enable_inst_interactivity=True,
		eval_mode=False,
		device=device,
	)

	if custom_checkpoint:
		custom_path = Path(custom_checkpoint)
		if custom_path.is_file():
			load_custom_checkpoint(base_model, custom_path)
		else:
			print(f"Skip custom checkpoint, file not found: {custom_path}")

	freeze_model(base_model)
	applied_modules: List[str] = []
	if model_cfg.get("use_lora", True):
		applied_modules = apply_lora(
			base_model,
			target_modules=model_cfg.get("target_modules", []),
			rank=int(model_cfg.get("lora_rank", 16)),
			alpha=float(model_cfg.get("lora_alpha", 32)),
			dropout=float(model_cfg.get("lora_dropout", 0.0)),
		)
		if not applied_modules:
			raise RuntimeError("没有匹配到任何可注入 LoRA 的线性层")
	else:
		applied_modules = enable_named_modules(
			base_model,
			model_cfg.get("trainable_modules", ["sam_prompt_encoder", "sam_mask_decoder"]),
		)
		if not applied_modules:
			raise RuntimeError("没有匹配到任何可训练模块")

	model = SAM3BoxSegModel(base_model).to(device)
	return model, applied_modules


def build_base_dataset(
	dataset_name: str,
	data_root: str,
	image_size: int,
	jitter_bbox_ratio: float,
):
	transform = SAM3ResizeNormalize(image_size)
	if dataset_name == "kvasir":
		return KvasirSEGDataset(
			data_root,
			transform=transform,
			image_size=image_size,
			jitter_bbox_ratio=jitter_bbox_ratio,
		)
	if dataset_name == "busi":
		return BUSIDataset(
			data_root,
			transform=transform,
			image_size=image_size,
			jitter_bbox_ratio=jitter_bbox_ratio,
		)
	raise ValueError(f"不支持的数据集: {dataset_name}")


def split_indices(total_size: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
	if total_size < 2:
		raise RuntimeError("数据量不足，至少需要 2 个样本才能划分训练/验证集")
	indices = list(range(total_size))
	random.Random(seed).shuffle(indices)
	train_count = max(1, int(total_size * train_ratio))
	train_count = min(train_count, total_size - 1)
	return indices[:train_count], indices[train_count:]


def build_dataloaders(config: Dict[str, Any], image_size: int, device: str) -> Tuple[DataLoader, DataLoader]:
	data_cfg = config["data"]
	dataset_name = str(data_cfg["dataset"]).lower()
	data_root = str(data_cfg["data_root"])
	train_ratio = float(data_cfg.get("train_ratio", 0.85))
	seed = int(config["training"].get("seed", 42))

	train_dataset = build_base_dataset(
		dataset_name=dataset_name,
		data_root=data_root,
		image_size=image_size,
		jitter_bbox_ratio=float(data_cfg.get("jitter_bbox_ratio", 0.0)),
	)
	val_dataset = build_base_dataset(
		dataset_name=dataset_name,
		data_root=data_root,
		image_size=image_size,
		jitter_bbox_ratio=0.0,
	)

	if len(train_dataset) == 0:
		raise RuntimeError(f"数据集为空: {dataset_name} @ {data_root}")

	train_indices, val_indices = split_indices(len(train_dataset), train_ratio, seed)
	batch_size = int(data_cfg.get("batch_size", 1))
	num_workers = int(data_cfg.get("num_workers", 0))
	pin_memory = device.startswith("cuda")

	train_loader = DataLoader(
		Subset(train_dataset, train_indices),
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=len(train_indices) >= batch_size,
	)
	val_loader = DataLoader(
		Subset(val_dataset, val_indices),
		batch_size=1,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)
	return train_loader, val_loader


def prepare_batch(batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	images = batch["image"].float().to(device, non_blocking=True)
	boxes = batch["bbox"].float().to(device, non_blocking=True)
	masks = batch["mask"].float().to(device, non_blocking=True)
	return images, boxes, masks


def create_autocast_context(device: str, mixed_precision: str):
	if not device.startswith("cuda"):
		return nullcontext()
	if mixed_precision == "fp16":
		return torch.autocast(device_type="cuda", dtype=torch.float16)
	if mixed_precision == "bf16":
		return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
	return nullcontext()


def create_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
	def lr_lambda(step: int) -> float:
		if total_steps <= 0:
			return 1.0
		if warmup_steps > 0 and step < warmup_steps:
			return float(step + 1) / float(max(1, warmup_steps))
		progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
		progress = min(max(progress, 0.0), 1.0)
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	return LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: str,
	mixed_precision: str,
) -> Dict[str, float]:
	model.eval()
	total_loss = 0.0
	total_samples = 0
	totals = {
		"dice": 0.0,
		"iou": 0.0,
		"precision": 0.0,
		"recall": 0.0,
		"mean_confidence": 0.0,
	}

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Validate", leave=False):
			images, boxes, targets = prepare_batch(batch, device)
			with create_autocast_context(device, mixed_precision):
				outputs = model(images, boxes)
				loss = criterion(outputs["masks"], targets)

			metrics = compute_all_metrics(outputs["masks"], targets)
			batch_size = images.size(0)
			total_loss += loss.item() * batch_size
			total_samples += batch_size

			for key in ["dice", "iou", "precision", "recall"]:
				totals[key] += metrics[key] * batch_size
			totals["mean_confidence"] += outputs["iou_predictions"].mean().item() * batch_size

	summary = {key: value / max(1, total_samples) for key, value in totals.items()}
	summary["loss"] = total_loss / max(1, total_samples)
	return summary


def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	optimizer: AdamW,
	scheduler: LambdaLR,
	scaler: GradScaler,
	criterion: nn.Module,
	device: str,
	mixed_precision: str,
	gradient_accumulation_steps: int,
	max_grad_norm: float,
) -> Dict[str, float]:
	model.train()
	optimizer.zero_grad(set_to_none=True)

	total_loss = 0.0
	total_samples = 0
	step_count = 0
	progress = tqdm(dataloader, desc="Train", leave=False)

	for batch_index, batch in enumerate(progress, start=1):
		images, boxes, targets = prepare_batch(batch, device)
		with create_autocast_context(device, mixed_precision):
			outputs = model(images, boxes)
			loss = criterion(outputs["masks"], targets)
			scaled_loss = loss / gradient_accumulation_steps

		scaler.scale(scaled_loss).backward()

		should_step = (
			batch_index % gradient_accumulation_steps == 0
			or batch_index == len(dataloader)
		)
		if should_step:
			scaler.unscale_(optimizer)
			if max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)
			scheduler.step()
			step_count += 1

		batch_size = images.size(0)
		total_loss += loss.item() * batch_size
		total_samples += batch_size
		progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.4f}")

	return {
		"loss": total_loss / max(1, total_samples),
		"optimizer_steps": step_count,
	}


def save_checkpoint(
	output_dir: Path,
	checkpoint_name: str,
	epoch: int,
	model: nn.Module,
	optimizer: AdamW,
	scheduler: LambdaLR,
	scaler: GradScaler,
	best_dice: float,
	config: Dict[str, Any],
) -> None:
	checkpoint = {
		"epoch": epoch,
		"best_dice": best_dice,
		"config": config,
		"trainable_state_dict": get_trainable_state_dict(model),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"scaler_state_dict": scaler.state_dict(),
	}
	torch.save(checkpoint, output_dir / checkpoint_name)


def load_resume_checkpoint(
	checkpoint_path: Path,
	model: nn.Module,
	optimizer: AdamW,
	scheduler: LambdaLR,
	scaler: GradScaler,
) -> Tuple[int, float]:
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	state_dict = checkpoint.get("trainable_state_dict") or checkpoint.get("model_state_dict")
	if state_dict is None:
		raise RuntimeError("恢复 checkpoint 缺少 trainable_state_dict/model_state_dict")
	model.load_state_dict(state_dict, strict=False)
	if "optimizer_state_dict" in checkpoint:
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	if "scheduler_state_dict" in checkpoint:
		scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
	if "scaler_state_dict" in checkpoint:
		scaler.load_state_dict(checkpoint["scaler_state_dict"])
	return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_dice", 0.0))


def main() -> int:
	args = build_parser().parse_args()
	config = resolve_config(args)

	device = str(config["training"]["device"])
	seed_everything(int(config["training"].get("seed", 42)))

	model, tuned_modules = build_model(config, device)
	image_size = int(getattr(model.sam3, "image_size", config["data"].get("image_size", 1008)))
	config["data"]["image_size"] = image_size

	train_loader, val_loader = build_dataloaders(config, image_size=image_size, device=device)

	trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not trainable_params:
		raise RuntimeError("当前没有可训练参数")

	optimizer = AdamW(
		trainable_params,
		lr=float(config["training"]["learning_rate"]),
		weight_decay=float(config["training"].get("weight_decay", 0.0)),
		betas=tuple(config["optimizer"].get("betas", [0.9, 0.999])),
	)

	gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
	total_updates_per_epoch = max(1, math.ceil(len(train_loader) / gradient_accumulation_steps))
	total_steps = total_updates_per_epoch * int(config["training"]["epochs"])
	scheduler = create_scheduler(
		optimizer,
		total_steps=total_steps,
		warmup_steps=int(config["training"].get("warmup_steps", 0)),
	)

	mixed_precision = str(config["training"].get("mixed_precision", "none")).lower()
	scaler = GradScaler(enabled=device.startswith("cuda") and mixed_precision == "fp16")
	criterion = CombinedSegLoss(dice_weight=1.0, focal_weight=1.0, bce_weight=1.0)

	start_epoch = 1
	best_dice = 0.0
	resume_path = config["training"].get("resume")
	if resume_path:
		start_epoch, best_dice = load_resume_checkpoint(
			Path(resume_path),
			model,
			optimizer,
			scheduler,
			scaler,
		)

	output_dir = Path(config["training"]["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)
	trainable_count, total_count = count_trainable_parameters(model)

	print(f"Device: {device}")
	print(f"Dataset: {config['data']['dataset']}")
	print(f"Image size: {image_size}")
	print(f"Trainable params: {trainable_count:,} / {total_count:,}")
	print(f"Tuned modules: {len(tuned_modules)}")

	history: List[Dict[str, Any]] = []
	max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))
	eval_interval = int(config["training"].get("evaluation_interval", 1))
	save_interval = int(config["training"].get("save_every_n_epochs", 0))

	for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
		train_metrics = train_one_epoch(
			model=model,
			dataloader=train_loader,
			optimizer=optimizer,
			scheduler=scheduler,
			scaler=scaler,
			criterion=criterion,
			device=device,
			mixed_precision=mixed_precision,
			gradient_accumulation_steps=gradient_accumulation_steps,
			max_grad_norm=max_grad_norm,
		)

		epoch_record: Dict[str, Any] = {
			"epoch": epoch,
			"train": train_metrics,
		}

		if epoch % eval_interval == 0:
			val_metrics = evaluate(
				model=model,
				dataloader=val_loader,
				criterion=criterion,
				device=device,
				mixed_precision=mixed_precision,
			)
			epoch_record["val"] = val_metrics
			print(
				f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
				f"val_loss={val_metrics['loss']:.4f}, val_dice={val_metrics['dice']:.4f}, "
				f"val_iou={val_metrics['iou']:.4f}"
			)

			if val_metrics["dice"] >= best_dice:
				best_dice = val_metrics["dice"]
				save_checkpoint(
					output_dir=output_dir,
					checkpoint_name="best.pt",
					epoch=epoch,
					model=model,
					optimizer=optimizer,
					scheduler=scheduler,
					scaler=scaler,
					best_dice=best_dice,
					config=config,
				)
		else:
			print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}")

		if save_interval > 0 and epoch % save_interval == 0:
			save_checkpoint(
				output_dir=output_dir,
				checkpoint_name=f"epoch_{epoch:03d}.pt",
				epoch=epoch,
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				scaler=scaler,
				best_dice=best_dice,
				config=config,
			)

		history.append(epoch_record)
		(output_dir / "train_history.json").write_text(
			json.dumps(history, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)

	save_checkpoint(
		output_dir=output_dir,
		checkpoint_name="last.pt",
		epoch=int(config["training"]["epochs"]),
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		scaler=scaler,
		best_dice=best_dice,
		config=config,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
