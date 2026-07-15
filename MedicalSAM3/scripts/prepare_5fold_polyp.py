"""准备 Kvasir+CVC 的 5 折交叉验证划分，并将 PolypGen 作为仅外部测试集。"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image
import numpy as np

from MedicalSAM3.scripts.common import ensure_dir, infer_source_domain, write_records


def _dataset_prefix(dataset_name: str) -> str:
    """将数据集名称转换为文件名安全的前缀。

    参数：
        - dataset_name: 原始数据集名称

    返回：
        - 规范化后的前缀字符串
    """
    prefix = re.sub(r"[^a-zA-Z0-9]+", "_", dataset_name.strip()).strip("_")
    return prefix or "dataset"


def _make_record(image_path: Path, mask_path: Path, dataset_name: str, stem: str) -> dict[str, str]:
    """构建包含路径、来源域和图像标识的记录字典。

    参数：
        - image_path: 图像文件路径
        - mask_path: 掩码文件路径
        - dataset_name: 数据集名称
        - stem: 文件名主干

    返回：
        - 记录字典
    """
    source_domain = infer_source_domain(
        dataset_name=dataset_name,
        image_id=stem,
        image_path=str(image_path),
        mask_path=str(mask_path),
    )
    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "dataset_name": source_domain,
        "source_group": dataset_name,
        "image_id": f"{_dataset_prefix(source_domain)}__{stem}",
    }


def _create_dummy_dataset(root: Path, dataset_name: str, count: int, external: bool = False) -> list[dict[str, str]]:
    """创建合成 dummy 数据集用于测试。

    参数：
        - root: 根目录路径
        - dataset_name: 数据集名称
        - count: 生成样本数量
        - external: 是否为外部测试集

    返回：
        - 记录字典列表
    """
    dataset_root = root / dataset_name
    image_dir = ensure_dir(dataset_root / ("imagesTs" if external else "images"))
    mask_dir = ensure_dir(dataset_root / ("labelsTs" if external else "masks"))
    records = []
    for index in range(count):
        image_id = f"{dataset_name.lower()}_{index:03d}"
        image_path = image_dir / f"{image_id}.png"
        mask_path = mask_dir / f"{image_id}.png"
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        canvas[..., 0] = 40 + (index * 11) % 90
        canvas[..., 1] = 25 + (index * 7) % 60
        canvas[..., 2] = 20 + (index * 5) % 80
        yy, xx = np.mgrid[:96, :96]
        cx = 48 + (index % 5 - 2) * 6
        cy = 48 + (index % 4 - 1) * 5
        r = 14 + index % 7
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).astype(np.uint8) * 255
        canvas[mask > 0] = np.array([170, 80, 90], dtype=np.uint8)
        Image.fromarray(canvas).save(image_path)
        Image.fromarray(mask).save(mask_path)
        records.append(_make_record(image_path, mask_path, dataset_name, image_id))
    return records


def _scan_standard_pairs(root: Path, dataset_name: str) -> list[dict[str, str]]:
    """brief:
        Handle scan standard pairs.

    parameter:
        - root: Input value for root.
        - dataset_name: Input value for dataset_name.

    retrival:
        - Returns the computed value for the caller or command workflow.
    """
    records: list[dict[str, str]] = []
    lower_dataset = dataset_name.lower()
    for mask_dir_name in ["labelsTr", "labelsTs", "masks", "mask"]:
        for mask_dir in root.rglob(mask_dir_name):
            if lower_dataset not in str(mask_dir).lower():
                continue
            image_dir_candidates = [
                mask_dir.parent / "imagesTr",
                mask_dir.parent / "imagesTs",
                mask_dir.parent / "images",
                mask_dir.parent / "image",
            ]
            image_dir = next((candidate for candidate in image_dir_candidates if candidate.exists()), None)
            if image_dir is None:
                continue
            for mask_path in sorted(mask_dir.glob("*.*")):
                if not mask_path.is_file():
                    continue
                stem = mask_path.stem
                image_candidates = [
                    image_dir / f"{stem}_0000.png",
                    image_dir / f"{stem}.png",
                    image_dir / f"{stem}.jpg",
                ]
                image_path = next((candidate for candidate in image_candidates if candidate.exists()), None)
                if image_path is None:
                    continue
                records.append(_make_record(image_path, mask_path, dataset_name, stem))
    return records


def _scan_nnunet_raw_dataset(
    dataset_root: Path,
    dataset_name: str,
    image_dir_name: str,
    mask_dir_name: str,
) -> list[dict[str, str]]:
    """brief:
        Handle scan nnunet raw dataset.

    parameter:
        - dataset_root: Input value for dataset_root.
        - dataset_name: Input value for dataset_name.
        - image_dir_name: Input value for image_dir_name.
        - mask_dir_name: Input value for mask_dir_name.

    retrival:
        - Returns the computed value for the caller or command workflow.
    """
    image_dir = dataset_root / image_dir_name
    mask_dir = dataset_root / mask_dir_name
    if not image_dir.exists() or not mask_dir.exists():
        return []

    records: list[dict[str, str]] = []
    for mask_path in sorted(mask_dir.glob("*.*")):
        if not mask_path.is_file():
            continue
        stem = mask_path.stem
        image_candidates = [
            image_dir / f"{stem}_0000.png",
            image_dir / f"{stem}_0000.jpg",
            image_dir / f"{stem}.png",
            image_dir / f"{stem}.jpg",
        ]
        image_path = next((candidate for candidate in image_candidates if candidate.exists()), None)
        if image_path is None:
            continue
        records.append(_make_record(image_path, mask_path, dataset_name, stem))
    return records


def _scan_image_mask_dataset(dataset_root: Path, dataset_name: str) -> list[dict[str, str]]:
    """brief:
        Handle scan image mask dataset.

    parameter:
        - dataset_root: Input value for dataset_root.
        - dataset_name: Input value for dataset_name.

    retrival:
        - Returns the computed value for the caller or command workflow.
    """
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []

    records: list[dict[str, str]] = []
    for image_path in sorted(image_dir.glob("*.*")):
        if not image_path.is_file():
            continue
        stem = image_path.stem
        mask_candidates = [
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.jpg",
            mask_dir / image_path.name,
        ]
        mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), None)
        if mask_path is None:
            continue
        records.append(_make_record(image_path, mask_path, dataset_name, stem))
    return records


def _deduplicate_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """brief:
        Handle deduplicate records.

    parameter:
        - records: Input value for records.

    retrival:
        - Returns the computed value for the caller or command workflow.
    """
    unique: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for record in records:
        image_id = record["image_id"]
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        unique.append(record)
    return unique


def _build_folds(records: list[dict[str, str]], seed: int) -> list[tuple[list[dict[str, str]], list[dict[str, str]]]]:
    """brief:
        Build folds.

    parameter:
        - records: Input value for records.
        - seed: Input value for seed.

    retrival:
        - Returns the computed value for the caller or command workflow.
    """
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    folds = []
    for fold_id in range(5):
        val = shuffled[fold_id::5]
        val_ids = {record["image_id"] for record in val}
        train = [record for record in shuffled if record["image_id"] not in val_ids]
        folds.append((train, val))
    return folds


def main() -> int:
    """brief:
        Run the command-line entry point for this script.

    parameter:
        - None.

    retrival:
        - Returns the computed value for the caller or command workflow.
    """
    parser = argparse.ArgumentParser(description="Prepare 5-fold Kvasir+CVC splits with PolypGen as external-only.")
    parser.add_argument("--data-root", default="MedicalSAM3/data")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = ensure_dir(args.output_dir)
    warnings_list: list[str] = []

    if args.dummy:
        dummy_root = ensure_dir(output_dir.parent / "dummy_dataset")
        kvasir_records = _create_dummy_dataset(dummy_root, "Kvasir-SEG", 15)
        cvc_records = _create_dummy_dataset(dummy_root, "CVC-ClinicDB", 10)
        kvasircvc_records: list[dict[str, str]] = []
        polypgen_records = _create_dummy_dataset(dummy_root, "PolypGen", 6, external=True)
    else:
        kvasir_records = _scan_image_mask_dataset(data_root / "Kvasir-SEG", "Kvasir-SEG")
        cvc_records = _scan_image_mask_dataset(data_root / "CVC-ClinicDB", "CVC-ClinicDB")
        kvasircvc_records = _scan_nnunet_raw_dataset(
            data_root / "KvasirCVC-nnunet_raw" / "Dataset504_KvasirCVC",
            "KvasirCVC",
            "imagesTr",
            "labelsTr",
        )
        if not kvasir_records:
            kvasir_records = _scan_standard_pairs(data_root, "Kvasir-SEG")
        if not cvc_records:
            cvc_records = _scan_standard_pairs(data_root, "CVC-ClinicDB")
        if not kvasircvc_records:
            kvasircvc_records = _scan_standard_pairs(data_root, "KvasirCVC")

        polypgen_records = _scan_nnunet_raw_dataset(
            data_root / "PolypGen_external_test" / "Dataset502_PolypGen",
            "PolypGen",
            "imagesTs",
            "labelsTs",
        )
        if not polypgen_records:
            polypgen_records = _scan_standard_pairs(data_root / "PolypGen_external_test", "PolypGen")

        kvasir_records = _deduplicate_records(kvasir_records)
        cvc_records = _deduplicate_records(cvc_records)
        kvasircvc_records = _deduplicate_records(kvasircvc_records)
        polypgen_records = _deduplicate_records(polypgen_records)

        if not polypgen_records:
            warnings_list.append("external_polypgen_count=0; external validation must stay blocked until data is available")

    merged = _deduplicate_records(kvasir_records + cvc_records + kvasircvc_records)
    if not merged:
        raise FileNotFoundError("No Kvasir/CVC training records found. Expected Dataset504_KvasirCVC or standalone Kvasir/CVC folders.")

    folds = _build_folds(merged, seed=args.seed)

    source_domain_counts: dict[str, int] = {}
    for record in merged:
        source_domain = str(record.get("dataset_name", "unknown"))
        source_domain_counts[source_domain] = source_domain_counts.get(source_domain, 0) + 1

    summary = {
        "seed": args.seed,
        "train_val_count": len(merged),
        "source_domain_counts": source_domain_counts,
        "source_group_counts": {
            "Kvasir-SEG": len(kvasir_records),
            "CVC-ClinicDB": len(cvc_records),
            "KvasirCVC": len(kvasircvc_records),
        },
        "external_polypgen_count": len(polypgen_records),
        "folds": [],
        "leakage_check_passed": True,
        "warnings": warnings_list,
    }
    all_train_ids = set()
    all_val_ids = set()
    external_ids = {record["image_id"] for record in polypgen_records}

    for fold_id, (train_records, val_records) in enumerate(folds):
        fold_dir = ensure_dir(output_dir / f"fold_{fold_id}")
        write_records(fold_dir / "train_ids.txt", train_records)
        write_records(fold_dir / "val_ids.txt", val_records)
        train_ids = {record["image_id"] for record in train_records}
        val_ids = {record["image_id"] for record in val_records}
        if train_ids & val_ids:
            summary["leakage_check_passed"] = False
            raise RuntimeError(f"Train/val overlap detected in fold {fold_id}")
        if train_ids & external_ids or val_ids & external_ids:
            summary["leakage_check_passed"] = False
            raise RuntimeError(f"External PolypGen leakage detected in fold {fold_id}")
        all_train_ids |= train_ids
        all_val_ids |= val_ids
        summary["folds"].append(
            {
                "fold_id": fold_id,
                "train_count": len(train_records),
                "val_count": len(val_records),
            }
        )

    write_records(output_dir / "external_polypgen_ids.txt", polypgen_records)
    if external_ids & all_train_ids or external_ids & all_val_ids:
        summary["leakage_check_passed"] = False
        raise RuntimeError("PolypGen samples leaked into train/val splits")

    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
