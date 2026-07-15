"""从 MedEx-SAM3 分割记录构建 YOLO 检测数据集。

该脚本将分割掩码转换为带填充的边界框标签，用于训练被 SAM3 消费的提示检测器。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.scripts.common import read_records


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _safe_stem(value: str, fallback: str) -> str:
    """将字符串安全转换为文件系统友好的 stem 名称。

    替换非字母数字字符为下划线，去除首尾特殊字符。

    参数：
        - value: 原始字符串。
        - fallback: 若转换后为空时的备用名称。

    返回：
        - 安全的文件系统名称。
    """
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    stem = stem.strip("._")
    return stem or fallback


def _record_key(record: dict[str, Any], index: int) -> str:
    """从记录字典生成唯一且安全的文件键名。

    优先使用 image_id，其次为图像 stem 名称，最后使用索引。

    参数：
        - record: 数据记录字典。
        - index: 记录序号，用于生成后备键名。

    返回：
        - 唯一且安全的文件键名字符串。
    """
    image_path = str(record.get("image_path", ""))
    image_id = str(record.get("image_id", ""))
    dataset_name = str(record.get("dataset_name", ""))
    raw = image_id or Path(image_path).stem or f"case_{index:06d}"
    if dataset_name and not raw.lower().startswith(dataset_name.lower()):
        raw = f"{dataset_name}__{raw}"
    return _safe_stem(raw, f"case_{index:06d}")


def _load_rgb_image(path: Path) -> Image.Image:
    """从路径加载 RGB 图像，支持 _0000 多通道格式。

    对于以 _0000 结尾的文件名，尝试加载 _0000、_0001、_0002 三个通道并合并。

    参数：
        - path: 图像文件路径。

    返回：
        - RGB 模式的 PIL 图像。
    """
    if path.stem.endswith("_0000"):
        channel_paths = [path.with_name(path.name.replace("_0000", f"_000{i}")) for i in range(3)]
        if all(channel_path.exists() for channel_path in channel_paths):
            channels = [np.asarray(Image.open(channel_path).convert("L")) for channel_path in channel_paths]
            return Image.fromarray(np.stack(channels, axis=-1).astype(np.uint8), mode="RGB")
    return Image.open(path).convert("RGB")


def _mask_to_xyxy(mask_path: Path) -> tuple[float, float, float, float] | None:
    """将掩码图像转换为 xyxy 边界框坐标。

    参数：
        - mask_path: 掩码图像文件路径。

    返回：
        - (x1, y1, x2, y2) 元组；若掩码为空则返回 None。
    """
    mask = Image.open(mask_path).convert("L")
    array = np.asarray(mask)
    threshold = 0 if array.max() <= 1 else 127
    coords = np.argwhere(array > threshold)
    if coords.size == 0:
        return None
    y1 = float(coords[:, 0].min())
    x1 = float(coords[:, 1].min())
    y2 = float(coords[:, 0].max() + 1)
    x2 = float(coords[:, 1].max() + 1)
    return x1, y1, x2, y2


def _pad_xyxy(
    xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding_ratio: float,
    min_size: float,
) -> tuple[float, float, float, float] | None:
    """对 xyxy 边界框应用填充并确保最小尺寸。

    参数：
        - xyxy: 原始边界框 (x1, y1, x2, y2)。
        - width: 图像宽度。
        - height: 图像高度。
        - padding_ratio: 填充比例（相对框边长）。
        - min_size: 填充后框的最小边长。

    返回：
        - 填充后的边界框；若无效则返回 None。
    """
    x1, y1, x2, y2 = xyxy
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(width), x2 + pad_x)
    y2 = min(float(height), y2 + pad_y)

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_size:
        center = (x1 + x2) * 0.5
        half = min_size * 0.5
        x1 = max(0.0, center - half)
        x2 = min(float(width), center + half)
    if box_h < min_size:
        center = (y1 + y2) * 0.5
        half = min_size * 0.5
        y1 = max(0.0, center - half)
        y2 = min(float(height), center + half)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _to_yolo_line(xyxy: tuple[float, float, float, float], width: int, height: int) -> str:
    """将 xyxy 坐标转换为 YOLO 格式标签行。

    格式：class_id x_center y_center box_width box_height（归一化到 [0, 1]）。

    参数：
        - xyxy: 边界框坐标 (x1, y1, x2, y2)。
        - width: 图像宽度。
        - height: 图像高度。

    返回：
        - YOLO 格式的标签字符串。
    """
    x1, y1, x2, y2 = xyxy
    x_center = ((x1 + x2) * 0.5) / float(width)
    y_center = ((y1 + y2) * 0.5) / float(height)
    box_w = (x2 - x1) / float(width)
    box_h = (y2 - y1) / float(height)
    return f"0 {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}"


def _materialize_image(source: Path, destination: Path, *, link_mode: str) -> None:
    """将源图像复制或链接到目标路径。

    对于 _0000 格式的多通道图像，先合并再保存为 PNG。

    参数：
        - source: 源图像路径。
        - destination: 目标路径。
        - link_mode: 链接模式，"symlink" / "hardlink" / "copy"。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if source.stem.endswith("_0000"):
        image = _load_rgb_image(source)
        image.save(destination)
        return

    if link_mode == "symlink":
        os.symlink(source.resolve(), destination)
    elif link_mode == "hardlink":
        os.link(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def _prepare_split(
    *,
    split_name: str,
    split_file: Path,
    output_dir: Path,
    padding_ratio: float,
    min_box_size: float,
    link_mode: str,
    include_empty_labels: bool,
) -> dict[str, Any]:
    """处理单个数据集划分（训练/验证/测试）。

    读取记录，提取/填充边界框，写入图像和标签文件，返回统计数据。

    参数：
        - split_name: 划分名称（如 train / val / test）。
        - split_file: 划分记录文件路径。
        - output_dir: 输出根目录。
        - padding_ratio: 边界框填充比例。
        - min_box_size: 最小框尺寸。
        - link_mode: 图像文件处理模式。
        - include_empty_labels: 是否包含空标签文件。

    返回：
        - 包含处理统计信息的字典。
    """
    records = read_records(split_file)
    stats: dict[str, Any] = {
        "split_file": str(split_file),
        "records": len(records),
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "empty_masks": 0,
        "missing_files": 0,
        "errors": [],
    }
    for index, record in enumerate(records):
        image_path = Path(str(record.get("image_path", "")))
        mask_path = Path(str(record.get("mask_path", "")))
        case_key = _record_key(record, index)
        if not image_path.is_file() or not mask_path.is_file():
            stats["missing_files"] += 1
            stats["errors"].append({"case_id": case_key, "error": "missing image or mask"})
            continue
        try:
            image = _load_rgb_image(image_path)
            width, height = image.size
            raw_box = _mask_to_xyxy(mask_path)
            if raw_box is None:
                stats["empty_masks"] += 1
                if not include_empty_labels:
                    continue
                label_line = ""
            else:
                padded_box = _pad_xyxy(
                    raw_box,
                    width=width,
                    height=height,
                    padding_ratio=padding_ratio,
                    min_size=min_box_size,
                )
                if padded_box is None:
                    stats["empty_masks"] += 1
                    if not include_empty_labels:
                        continue
                    label_line = ""
                else:
                    label_line = _to_yolo_line(padded_box, width, height)

            image_suffix = ".png" if image_path.stem.endswith("_0000") else image_path.suffix.lower()
            if image_suffix not in IMAGE_EXTENSIONS:
                image_suffix = ".png"
            output_image = output_dir / "images" / split_name / f"{case_key}{image_suffix}"
            output_label = output_dir / "labels" / split_name / f"{case_key}.txt"
            _materialize_image(image_path, output_image, link_mode=link_mode)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text((label_line + "\n") if label_line else "", encoding="utf-8")

            stats["images"] += 1
            stats["labels"] += 1
            stats["boxes"] += 1 if label_line else 0
        except Exception as exc:
            stats["errors"].append({"case_id": case_key, "image_path": str(image_path), "error": repr(exc)})
    return stats


def _write_data_yaml(output_dir: Path, splits: dict[str, Path]) -> Path:
    """写入 YOLO data.yaml 配置文件。

    参数：
        - output_dir: 数据集根目录。
        - splits: 划分名称到分割文件的映射。

    返回：
        - 生成的 YAML 文件路径。
    """
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        lines.append("test: images/test")
    lines.extend(["nc: 1", "names: ['polyp']", ""])
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def main() -> int:
    """命令行入口，将 MedEx-SAM3 分割记录转换为 YOLO 检测数据集。

    解析命令行参数，处理训练/验证/测试划分，写入图像、标签和配置文件。

    返回：
        - 退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(description="Prepare a YOLO polyp detection dataset from split records.")
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--include-empty-labels", action="store_true")
    parser.add_argument("--clear", action="store_true", help="Remove the output directory before writing.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clear and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Path] = {
        "train": Path(args.train_split),
        "val": Path(args.val_split),
    }
    if args.test_split:
        splits["test"] = Path(args.test_split)

    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "padding_ratio": args.padding_ratio,
        "min_box_size": args.min_box_size,
        "link_mode": args.link_mode,
        "splits": {},
    }
    for split_name, split_file in splits.items():
        summary["splits"][split_name] = _prepare_split(
            split_name=split_name,
            split_file=split_file,
            output_dir=output_dir,
            padding_ratio=args.padding_ratio,
            min_box_size=args.min_box_size,
            link_mode=args.link_mode,
            include_empty_labels=args.include_empty_labels,
        )

    data_yaml = _write_data_yaml(output_dir, splits)
    summary["data_yaml"] = str(data_yaml)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
