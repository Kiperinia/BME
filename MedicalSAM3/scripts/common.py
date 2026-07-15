"""MedEx-SAM3 数据、指标与轻量训练的共享辅助工具。"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from MedicalSAM3.adapters.boundary_adapter import BoundaryAwareAdapter
from MedicalSAM3.adapters.medical_adapter import MedicalImageAdapter
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper


def seed_everything(seed: int) -> None:
    """设置全局随机种子以保证实验可复现。

    参数：
        - seed: 随机种子值

    返回：
        - 无返回值，仅执行设置随机种子的副作用
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runtime_environment_report(requested_device: str = "auto") -> dict[str, Any]:
    """构建运行时环境报告，包含 Python、Torch 及 CUDA 信息。

    参数：
        - requested_device: 请求的设备类型，可选 "auto"/"cuda"/"cpu"

    返回：
        - 包含运行时环境信息的字典
    """
    normalized = requested_device.strip().lower()
    report: dict[str, Any] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": str(getattr(torch.version, "cuda", "") or ""),
        "requested_device": normalized,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        report["device_count"] = 0
        report["torch_arch_list"] = []
        report["cuda_capability"] = None
        report["cuda_capability_supported"] = None
        return report

    report["device_count"] = int(torch.cuda.device_count())
    arch_list = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
    report["torch_arch_list"] = arch_list
    try:
        props = torch.cuda.get_device_properties(0)
        capability = f"sm_{props.major}{props.minor}"
        report["device_name"] = props.name
        report["cuda_capability"] = capability
        report["cuda_capability_supported"] = capability in arch_list if arch_list else None
    except Exception as exc:
        report["device_name"] = None
        report["cuda_capability"] = None
        report["cuda_capability_supported"] = None
        report["device_property_error"] = repr(exc)
    return report


def resolve_runtime_device(requested_device: str = "auto") -> str:
    """解析实际运行设备，在 CUDA 不可用时回退到 CPU。

    参数：
        - requested_device: 请求的设备类型，可选 "auto"/"cuda"/"cpu"

    返回：
        - 解析后实际使用的设备字符串，"cuda" 或 "cpu"
    """
    normalized = requested_device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device: {requested_device}")
    report = build_runtime_environment_report(normalized)
    failure_reason: Optional[str] = None
    if not bool(report.get("cuda_available", False)):
        failure_reason = "torch.cuda.is_available() is False"
    elif report.get("cuda_capability_supported") is False:
        failure_reason = (
            f"current GPU capability {report.get('cuda_capability')} is not in torch arch list "
            f"{report.get('torch_arch_list', [])}"
        )

    if failure_reason is not None:
        if normalized == "cuda":
            raise RuntimeError(failure_reason)
        return "cpu"
    try:
        probe = torch.randn(16, 16, device="cuda")
        _ = float((probe @ probe.transpose(0, 1)).mean().item())
        _ = float(probe.min().item())
        _ = float(probe.max().item())
        torch.cuda.synchronize()
        return "cuda"
    except Exception as exc:
        if normalized == "cuda":
            raise RuntimeError(f"CUDA probe failed: {exc.__class__.__name__}: {exc}") from exc
        return "cpu"


def log_runtime_environment(
    script_name: str,
    *,
    requested_device: str = "auto",
    resolved_device: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """打印并返回运行时环境日志信息。

    参数：
        - script_name: 当前脚本名称
        - requested_device: 请求的设备类型
        - resolved_device: 已解析的设备，为 None 时自动解析
        - extra: 额外需要合并到报告中的字段

    返回：
        - 包含运行时环境信息的字典
    """
    report = build_runtime_environment_report(requested_device)
    report["script"] = script_name
    report["resolved_device"] = resolved_device if resolved_device is not None else resolve_runtime_device(requested_device)
    if extra:
        report.update(extra)
    print(json.dumps({"runtime": report}, ensure_ascii=True), flush=True)
    return report


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则递归创建。

    参数：
        - path: 目标目录路径

    返回：
        - 创建或已存在的目录 Path 对象
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: Optional[str | Path]) -> dict[str, Any]:
    """加载 YAML 或简易文本格式的配置文件。

    参数：
        - path: 配置文件路径，为 None 时返回空字典

    返回：
        - 解析后的配置字典
    """
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:
        config: dict[str, Any] = {}
        for line in target.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip()
            lowered = value.lower()
            if lowered in {"true", "false"}:
                parsed: Any = lowered == "true"
            else:
                try:
                    parsed = json.loads(value)
                except Exception:
                    try:
                        parsed = int(value)
                    except ValueError:
                        try:
                            parsed = float(value)
                        except ValueError:
                            parsed = value
            config[key.strip()] = parsed
        return config


def apply_config_overrides(args: Any, config: dict[str, Any], defaults: dict[str, Any]) -> Any:
    """当命令行参数等于默认值时，用配置文件的值覆盖该参数。

    参数：
        - args: 命令行参数对象（具备属性读写能力）
        - config: 配置字典
        - defaults: 参数默认值字典

    返回：
        - 覆盖后的 args 对象
    """
    for key, default in defaults.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == default and key in config:
            setattr(args, key, config[key])
    return args


def dump_config(path: str | Path, config: dict[str, Any]) -> Path:
    """将配置字典转储为 YAML 或 JSON 文件。

    参数：
        - path: 输出文件路径
        - config: 要写入的配置字典

    返回：
        - 写入后的文件 Path 对象
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except Exception:
        destination.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return destination


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """从文件读取记录列表，支持 JSON 行和制表符分隔格式。

    参数：
        - path: 记录文件路径

    返回：
        - 记录字典列表；文件不存在时返回空列表
    """
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            records.append(json.loads(stripped))
        else:
            image_path, mask_path, dataset_name, image_id = stripped.split("\t")
            records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "dataset_name": dataset_name,
                    "image_id": image_id,
                }
            )
    return records


def write_records(path: str | Path, records: list[dict[str, Any]]) -> Path:
    """将记录列表以 JSON 行格式写入文件。

    参数：
        - path: 输出文件路径
        - records: 记录字典列表

    返回：
        - 写入后的文件 Path 对象
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records),
        encoding="utf-8",
    )
    return destination


def infer_source_domain(
    dataset_name: str = "",
    image_id: str = "",
    image_path: str = "",
    mask_path: str = "",
) -> str:
    """根据数据集名称和文件路径推断来源域（如 Kvasir、CVC、PolypGen）。

    参数：
        - dataset_name: 数据集名称
        - image_id: 图像标识
        - image_path: 图像路径
        - mask_path: 掩码路径

    返回：
        - 推断出的来源域字符串
    """
    dataset_lower = str(dataset_name).strip().lower()
    compact_dataset = re.sub(r"[^a-z0-9]+", "", dataset_lower)
    image_id_lower = str(image_id).lower()
    image_id_leaf = image_id_lower.split("__", 1)[-1] if "__" in image_id_lower else image_id_lower
    stem_candidates = [
        Path(str(image_path)).stem.lower(),
        Path(str(mask_path)).stem.lower(),
        image_id_leaf,
    ]
    searchable = " ".join(
        part for part in [dataset_lower, image_id_leaf, str(image_path).lower(), str(mask_path).lower()] if part
    )

    if "polypgen" in searchable:
        return "PolypGen"

    if compact_dataset in {"kvasircvc", "dataset504kvasircvc"}:
        for stem in stem_candidates:
            if stem.startswith("kvasir_"):
                return "Kvasir"
            if stem.startswith("cvc_"):
                return "CVC"

    if compact_dataset in {"kvasir", "kvasirseg"} or any("kvasir" in stem for stem in stem_candidates):
        return "Kvasir"
    if compact_dataset in {"cvc", "clinicdb", "cvcclinicdb"} or any("cvc" in stem or "clinicdb" in stem for stem in stem_candidates):
        return "CVC"

    if dataset_name:
        return dataset_name.strip()
    return "unknown"


def synthetic_polyp_sample(image_size: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成合成的息肉图像与掩码样本。

    参数：
        - image_size: 输出图像的边长（像素）
        - index: 样本索引，用于控制颜色与位置变化

    返回：
        - 由 (图像张量, 掩码张量) 组成的元组
    """
    image = Image.new("RGB", (image_size, image_size), color=(30 + index * 3 % 80, 20, 20))
    mask = Image.new("L", (image_size, image_size), color=0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    radius = image_size // 6 + (index % 5) * max(image_size // 40, 1)
    cx = image_size // 2 + (index % 3 - 1) * image_size // 10
    cy = image_size // 2 + (index % 4 - 2) * image_size // 12
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw_image.ellipse(bbox, fill=(170, 80 + index * 7 % 60, 90))
    draw_mask.ellipse(bbox, fill=255)
    image_tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
    mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype("float32")).unsqueeze(0)
    return image_tensor, mask_tensor


def load_record_tensors(record: dict[str, Any], image_size: int, fallback_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """从记录加载图像和掩码张量，加载失败时回退到合成样本。

    参数：
        - record: 包含 image_path 和 mask_path 的记录字典
        - image_size: 目标图像尺寸
        - fallback_index: 加载失败时用于生成合成样本的索引

    返回：
        - 由 (图像张量, 掩码张量) 组成的元组
    """
    image_path = Path(record.get("image_path", ""))
    mask_path = Path(record.get("mask_path", ""))
    if image_path.is_file() and mask_path.is_file():
        if image_path.stem.endswith("_0000"):
            channel_paths = [image_path.with_name(image_path.name.replace("_0000", f"_000{i}")) for i in range(3)]
            if all(channel_path.exists() for channel_path in channel_paths):
                channels = [
                    np.asarray(Image.open(channel_path).convert("L").resize((image_size, image_size))).astype("float32")
                    for channel_path in channel_paths
                ]
                stacked = np.stack(channels, axis=-1) / 255.0
                image_tensor = torch.from_numpy(stacked).permute(2, 0, 1)
            else:
                image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
                image_tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
        else:
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
            image_tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
        mask = Image.open(mask_path).convert("L").resize((image_size, image_size), resample=Image.NEAREST)
        mask_array = np.asarray(mask)
        threshold = 0 if mask_array.max() <= 1 else 127
        mask_tensor = torch.from_numpy((mask_array > threshold).astype("float32")).unsqueeze(0)
        return image_tensor, mask_tensor
    return synthetic_polyp_sample(image_size, fallback_index)


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    """从二值掩码提取最小外接边界框 [x1, y1, x2, y2]。

    参数：
        - mask: 二值掩码张量

    返回：
        - 边界框张量 [x1, y1, x2, y2]
    """
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        _, height, width = mask.shape
        return torch.tensor([0.0, 0.0, float(width), float(height)])
    y1 = coords[:, -2].min().float()
    x1 = coords[:, -1].min().float()
    y2 = coords[:, -2].max().float() + 1.0
    x2 = coords[:, -1].max().float() + 1.0
    return torch.tensor([x1, y1, x2, y2])


def pad_box(box: torch.Tensor, image_size: int, ratio: float) -> torch.Tensor:
    """按比例向外扩展边界框，并限制在图像范围内。

    参数：
        - box: 原始边界框 [x1, y1, x2, y2]
        - image_size: 图像边长
        - ratio: 扩展比例

    返回：
        - 扩展后的边界框张量
    """
    if ratio <= 0:
        return box.float()
    x1, y1, x2, y2 = box.float().tolist()
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = width * ratio
    pad_y = height * ratio
    return torch.tensor(
        [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(image_size), x2 + pad_x),
            min(float(image_size), y2 + pad_y),
        ],
        dtype=torch.float32,
    )


def jitter_box(box: torch.Tensor, image_size: int, ratio: float) -> torch.Tensor:
    """对边界框施加随机抖动扰动，并限制在图像范围内。

    参数：
        - box: 原始边界框 [x1, y1, x2, y2]
        - image_size: 图像边长
        - ratio: 抖动幅度比例

    返回：
        - 抖动后的边界框张量
    """
    if ratio <= 0:
        return box.float()
    x1, y1, x2, y2 = box.float().tolist()
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    delta_x = width * ratio * (torch.rand(1).item() * 2.0 - 1.0)
    delta_y = height * ratio * (torch.rand(1).item() * 2.0 - 1.0)
    return torch.tensor(
        [
            max(0.0, x1 + delta_x),
            max(0.0, y1 + delta_y),
            min(float(image_size), x2 + delta_x),
            min(float(image_size), y2 + delta_y),
        ],
        dtype=torch.float32,
    )


def full_image_box(image_size: int) -> torch.Tensor:
    """生成覆盖整幅图像的边界框 [0, 0, image_size, image_size]。

    参数：
        - image_size: 图像边长

    返回：
        - 全图边界框张量
    """
    return torch.tensor([0.0, 0.0, float(image_size), float(image_size)], dtype=torch.float32)


def removed_box_sentinel() -> torch.Tensor:
    """生成表示提示被移除的哨兵边界框（全 NaN）。

    参数：
        - 无

    返回：
        - 全 NaN 的哨兵边界框张量
    """
    return torch.full((4,), float("nan"), dtype=torch.float32)


def augment_box_prompt(
    box: torch.Tensor,
    image_size: int,
    *,
    base_padding_ratio: float = 0.0,
    corruption_prob: float = 1.0,
    jitter_ratio: float = 0.0,
    jitter_prob: float = 1.0,
    loose_ratio: float = 0.0,
    loose_prob: float = 0.0,
    dropout_prob: float = 0.0,
    prompt_removal_prob: float = 0.0,
) -> tuple[torch.Tensor, dict[str, bool]]:
    """对边界框提示进行数据增强，包括填充、抖动、松弛、丢弃和移除。

    参数：
        - box: 原始边界框
        - image_size: 图像边长
        - base_padding_ratio: 基础填充比例
        - corruption_prob: 触发增强的概率
        - jitter_ratio: 抖动幅度比例
        - jitter_prob: 触发抖动的概率
        - loose_ratio: 松弛幅度比例
        - loose_prob: 触发松弛的概率
        - dropout_prob: 触发丢弃（替换为全图框）的概率
        - prompt_removal_prob: 触发移除（替换为哨兵框）的概率

    返回：
        - 由 (增强后的边界框, 增强操作标记字典) 组成的元组
    """
    updated = pad_box(box, image_size, base_padding_ratio)
    applied = {"corrupted": False, "removed": False, "dropout": False, "loose": False, "jitter": False}
    if corruption_prob <= 0 or torch.rand(1).item() > corruption_prob:
        return updated, applied
    applied["corrupted"] = True
    if prompt_removal_prob > 0 and torch.rand(1).item() < prompt_removal_prob:
        applied["removed"] = True
        return removed_box_sentinel(), applied
    if dropout_prob > 0 and torch.rand(1).item() < dropout_prob:
        applied["dropout"] = True
        return full_image_box(image_size), applied
    if loose_ratio > 0 and loose_prob > 0 and torch.rand(1).item() < loose_prob:
        updated = pad_box(updated, image_size, loose_ratio)
        applied["loose"] = True
    if jitter_ratio > 0 and jitter_prob > 0 and torch.rand(1).item() < jitter_prob:
        updated = jitter_box(updated, image_size, jitter_ratio)
        applied["jitter"] = True
    return updated, applied


def boundary_band(mask: torch.Tensor) -> torch.Tensor:
    """计算掩码的边界带（膨胀区域减去腐蚀区域）。

    参数：
        - mask: 二值掩码张量

    返回：
        - 边界带张量，取值在 [0, 1]
    """
    kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)
    eroded = (F.conv2d(mask, kernel, padding=1) >= 9.0).float()
    dilated = (F.conv2d(mask, kernel, padding=1) > 0.0).float()
    return (dilated - eroded).clamp(0, 1)


def _surface_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> tuple[float, float]:
    """计算预测与真值掩码之间的表面距离指标 HD95 和 ASSD。

    参数：
        - pred_mask: 预测二值掩码
        - gt_mask: 真值二值掩码

    返回：
        - 由 (HD95, ASSD) 组成的元组
    """
    pred_points = torch.nonzero(boundary_band(pred_mask) > 0.5, as_tuple=False).float()
    gt_points = torch.nonzero(boundary_band(gt_mask) > 0.5, as_tuple=False).float()
    if pred_points.numel() == 0 or gt_points.numel() == 0:
        return 0.0, 0.0
    distances = torch.cdist(pred_points[:, -2:], gt_points[:, -2:])
    symmetric = torch.cat([distances.min(dim=1).values, distances.min(dim=0).values])
    hd95 = float(torch.quantile(symmetric, 0.95).item())
    assd = float(symmetric.mean().item())
    return hd95, assd


def compute_segmentation_metrics(mask_logits: torch.Tensor, gt_mask: torch.Tensor) -> dict[str, float]:
    """计算分割指标，包括 Dice、IoU、精确率、召回率、边界 F1、HD95、ASSD 等。

    参数：
        - mask_logits: 预测的掩码 logits 张量
        - gt_mask: 真值掩码张量

    返回：
        - 包含各项分割指标的字典
    """
    if gt_mask.shape != mask_logits.shape:
        gt_mask = F.interpolate(gt_mask.float(), size=mask_logits.shape[-2:], mode="nearest")
    prob = torch.sigmoid(mask_logits)
    pred = (prob > 0.5).float()
    target = gt_mask.float()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = pred_sum + target_sum
    pred_or_target = (pred + target).clamp(0, 1).sum()

    if pred_sum == 0 and target_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = float(((2.0 * intersection + 1e-6) / (union + 1e-6)).item())
        iou = float(((intersection + 1e-6) / (pred_or_target + 1e-6)).item())
        precision = float(((intersection + 1e-6) / (pred_sum + 1e-6)).item())
        recall = float(((intersection + 1e-6) / (target_sum + 1e-6)).item())
    pred_boundary = boundary_band(pred)
    gt_boundary = boundary_band(target)
    boundary_intersection = (pred_boundary * gt_boundary).sum()
    boundary_pred_sum = pred_boundary.sum()
    boundary_gt_sum = gt_boundary.sum()
    boundary_union = boundary_pred_sum + boundary_gt_sum
    if boundary_pred_sum == 0 and boundary_gt_sum == 0:
        boundary_f1 = 1.0
    else:
        boundary_f1 = float(((2.0 * boundary_intersection + 1e-6) / (boundary_union + 1e-6)).item())
    hd95, assd = _surface_metrics(pred, target)
    fp = ((pred == 1) & (target == 0)).float().sum()
    fn = ((pred == 0) & (target == 1)).float().sum()
    tn = ((pred == 0) & (target == 0)).float().sum()
    tp = ((pred == 1) & (target == 1)).float().sum()
    fpr = float((fp / (fp + tn + 1e-6)).item())
    fnr = 0.0 if target_sum == 0 else float((fn / (fn + tp + 1e-6)).item())
    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Boundary F1": boundary_f1,
        "HD95": hd95,
        "ASSD": assd,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
    }


class SplitSegmentationDataset(Dataset):
    """分割数据集，支持边界框提示增强的样本加载。

    参数：
        - records: 记录字典列表
        - image_size: 图像尺寸
        - box_padding_ratio: 边界框基础填充比例
        - prompt_corruption_prob: 触发提示增强的概率
        - box_jitter_ratio: 边界框抖动幅度比例
        - box_jitter_prob: 触发抖动的概率
        - loose_box_ratio: 松弛框幅度比例
        - loose_box_prob: 触发松弛的概率
        - box_dropout_prob: 边界框丢弃概率
        - prompt_removal_prob: 提示移除概率
        - box_provider: 自定义边界框提供器

    返回：
        - 提供 __getitem__ 加载增强后的样本字典
    """
    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: int,
        box_padding_ratio: float = 0.0,
        prompt_corruption_prob: float = 1.0,
        box_jitter_ratio: float = 0.0,
        box_jitter_prob: float = 1.0,
        loose_box_ratio: float = 0.0,
        loose_box_prob: float = 0.0,
        box_dropout_prob: float = 0.0,
        prompt_removal_prob: float = 0.0,
        box_provider: Any | None = None,
    ) -> None:
        """初始化分割数据集。

        参数：
            - records: 记录字典列表
            - image_size: 图像尺寸
            - box_padding_ratio: 边界框基础填充比例
            - prompt_corruption_prob: 触发提示增强的概率
            - box_jitter_ratio: 边界框抖动幅度比例
            - box_jitter_prob: 触发抖动的概率
            - loose_box_ratio: 松弛框幅度比例
            - loose_box_prob: 触发松弛的概率
            - box_dropout_prob: 边界框丢弃概率
            - prompt_removal_prob: 提示移除概率
            - box_provider: 自定义边界框提供器

        返回：
            - 无返回值，仅初始化实例属性
        """
        self.records = records
        self.image_size = image_size
        self.box_padding_ratio = box_padding_ratio
        self.prompt_corruption_prob = prompt_corruption_prob
        self.box_jitter_ratio = box_jitter_ratio
        self.box_jitter_prob = box_jitter_prob
        self.loose_box_ratio = loose_box_ratio
        self.loose_box_prob = loose_box_prob
        self.box_dropout_prob = box_dropout_prob
        self.prompt_removal_prob = prompt_removal_prob
        self.box_provider = box_provider

    def __len__(self) -> int:
        """返回数据集中样本数量。

        参数：
            - 无

        返回：
            - 样本数量
        """
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """加载指定索引的样本，并进行边界框提示增强。

        参数：
            - index: 样本索引

        返回：
            - 包含图像、掩码、边界框等信息的样本字典
        """
        record = self.records[index]
        image, mask = load_record_tensors(record, self.image_size, fallback_index=index)
        if self.box_provider is not None:
            box = self.box_provider.get_box(
                record,
                self.image_size,
                image=image,
                mask=mask,
                fallback_index=index,
            )
        else:
            box = mask_to_box(mask)
        box, prompt_aug = augment_box_prompt(
            box,
            self.image_size,
            base_padding_ratio=self.box_padding_ratio,
            corruption_prob=self.prompt_corruption_prob,
            jitter_ratio=self.box_jitter_ratio,
            jitter_prob=self.box_jitter_prob,
            loose_ratio=self.loose_box_ratio,
            loose_prob=self.loose_box_prob,
            dropout_prob=self.box_dropout_prob,
            prompt_removal_prob=self.prompt_removal_prob,
        )
        return {
            "image": image,
            "mask": mask,
            "box": box,
            "text_prompt": ["polyp"],
            "record": record,
            "prompt_aug": prompt_aug,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """将样本列表整理为批次字典，对张量进行堆叠。

    参数：
        - batch: 样本字典列表

    返回：
        - 整理后的批次字典
    """
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "boxes": torch.stack([item["box"] for item in batch], dim=0),
        "text_prompt": [item["text_prompt"][0] for item in batch],
        "records": [item["record"] for item in batch],
        "prompt_aug": [item.get("prompt_aug", {}) for item in batch],
    }


def resolve_feature_map(feature: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    """将特征张量解析为 4D 特征图格式，缺失时用 fallback 均值填充。

    参数：
        - feature: 原始特征张量，可能为 None
        - fallback: 备用特征张量

    返回：
        - 4D 特征图张量
    """
    if feature is None:
        return fallback.mean(dim=1, keepdim=True).repeat(1, 128, 1, 1)
    if feature.dim() == 4:
        return feature
    if feature.dim() == 3:
        batch_size, tokens, channels = feature.shape
        side = int(math.sqrt(tokens))
        if side * side == tokens:
            return feature.transpose(1, 2).reshape(batch_size, channels, side, side)
        feature = feature.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        return feature.repeat(1, 1, fallback.shape[-2] // 4, fallback.shape[-1] // 4)
    raise ValueError("Unsupported feature shape")


class MedExSam3SegmentationModel(nn.Module):
    """MedEx-SAM3 分割模型，封装 SAM3 包装器并可选医学适配器与边界适配器。

    参数：
        - wrapper: SAM3 张量前向包装器
        - enable_medical_adapter: 是否启用医学图像适配器
        - enable_boundary_adapter: 是否启用边界感知适配器
        - embed_dim: 嵌入维度

    返回：
        - 提供 forward 方法输出分割结果
    """
    def __init__(
        self,
        wrapper: Sam3TensorForwardWrapper,
        enable_medical_adapter: bool = False,
        enable_boundary_adapter: bool = False,
        embed_dim: int = 128,
    ) -> None:
        """初始化 MedEx-SAM3 分割模型。

        参数：
            - wrapper: SAM3 张量前向包装器
            - enable_medical_adapter: 是否启用医学图像适配器
            - enable_boundary_adapter: 是否启用边界感知适配器
            - embed_dim: 嵌入维度

        返回：
            - 无返回值，仅初始化模型组件
        """
        super().__init__()
        self.wrapper = wrapper
        self.medical_adapter = MedicalImageAdapter(embed_dim, max(embed_dim // 4, 8)) if enable_medical_adapter else None
        self.boundary_adapter = BoundaryAwareAdapter(embed_dim) if enable_boundary_adapter else None
        self.refine_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        text_prompt: Optional[list[str]] = None,
        exemplar_prompt_tokens: Optional[torch.Tensor] = None,
        retrieval_prior: Optional[dict[str, Any]] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """执行分割前向推理，输出掩码 logits、掩码概率和查询嵌入等。

        参数：
            - images: 输入图像张量
            - boxes: 边界框提示张量
            - text_prompt: 文本提示列表
            - exemplar_prompt_tokens: 示例提示令牌张量
            - retrieval_prior: 检索先验字典
            - gt_mask: 真值掩码（训练时用于边界适配器）

        返回：
            - 包含 mask_logits、masks、query_embedding 等的输出字典
        """
        outputs = self.wrapper(
            images=images,
            text_prompt=text_prompt,
            boxes=boxes,
            exemplar_prompt_tokens=exemplar_prompt_tokens,
            retrieval_prior=retrieval_prior,
        )
        feature_map = resolve_feature_map(outputs.get("image_embeddings"), images)
        if self.medical_adapter is not None:
            feature_map = self.medical_adapter(feature_map)
        aux = {}
        if self.boundary_adapter is not None:
            feature_map, aux = self.boundary_adapter(
                feature_map,
                coarse_mask_logits=outputs["mask_logits"],
                gt_mask=gt_mask if self.training else None,
            )
        delta = self.refine_head(feature_map)
        delta = F.interpolate(delta, size=outputs["mask_logits"].shape[-2:], mode="bilinear", align_corners=False)
        outputs["mask_logits"] = outputs["mask_logits"] + 0.1 * delta
        outputs["masks"] = torch.sigmoid(outputs["mask_logits"])
        outputs["query_embedding"] = F.normalize(F.adaptive_avg_pool2d(feature_map, 1).flatten(1), dim=1)
        outputs["adapter_aux"] = aux
        outputs["image_embeddings"] = feature_map
        return outputs
