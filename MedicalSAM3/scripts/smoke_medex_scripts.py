"""Run a bounded smoke suite for MedEx-SAM3 command-line scripts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRACKED_RUNTIME_ARTIFACTS = [
    WORKSPACE_ROOT / "MedicalSAM3" / "sam3_lora_targets.json",
    WORKSPACE_ROOT / "MedicalSAM3" / "sam3_modules.txt",
    WORKSPACE_ROOT / "MedicalSAM3" / "sam3_module_tree.json",
    WORKSPACE_ROOT / "MedicalSAM3" / "sam3_module_tree.txt",
]


@dataclass(frozen=True)
class SmokeCommand:
    """表示冒烟测试矩阵中的一个脚本调用。"""

    label: str
    args: list[str]
    timeout_seconds: int = 180


def _workspace_relative(path: Path) -> str:
    """将工作区下的路径转换为 POSIX 风格的项目相对路径。

    参数：
        - path: 绝对路径或工作区相对路径

    返回：
        - 适用于命令行的项目相对路径字符串
    """
    return path.resolve().relative_to(WORKSPACE_ROOT).as_posix()


def _write_synthetic_pair(image_path: Path, mask_path: Path, index: int, image_size: int) -> None:
    """生成合成 RGB 图像和二值掩码，用于冒烟测试。

    参数：
        - image_path: 输出图像路径
        - mask_path: 输出掩码路径
        - index: 合成样本索引
        - image_size: 正方形图像的宽高

    返回：
        - 无
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_size, image_size), color=(35 + index * 24, 42, 54))
    mask = Image.new("L", (image_size, image_size), color=0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    margin = 7 + index
    bbox = [margin, margin + 2, image_size - margin, image_size - margin + 1]
    draw_image.ellipse(bbox, fill=(178, 82 + index * 12, 98))
    draw_mask.ellipse(bbox, fill=255)
    image.save(image_path)
    mask.save(mask_path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    """写入 JSONL 行数据，自动创建父目录。

    参数：
        - path: 输出的 JSONL 文件路径
        - rows: 可序列化字典的可迭代对象

    返回：
        - 写入后的文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")
    return path


def _make_records(output_dir: Path, image_size: int) -> dict[str, Path]:
    """为所有冒烟命令创建合成记录、指标和拆分文件。

    参数：
        - output_dir: 冒烟测试输出根目录
        - image_size: 合成图像尺寸

    返回：
        - 以语义名称为键的重要夹具路径字典
    """
    fixtures = output_dir / "fixtures"
    images_dir = fixtures / "images"
    masks_dir = fixtures / "masks"
    records: list[dict[str, object]] = []
    for index, dataset_name in enumerate(["Kvasir", "CVC", "Kvasir"]):
        image_path = images_dir / f"smoke_{index}.png"
        mask_path = masks_dir / f"smoke_{index}.png"
        _write_synthetic_pair(image_path, mask_path, index, image_size)
        records.append(
            {
                "image_path": _workspace_relative(image_path),
                "mask_path": _workspace_relative(mask_path),
                "dataset_name": dataset_name,
                "image_id": f"smoke_{index}",
            }
        )

    internal_split = _write_jsonl(fixtures / "internal_split.jsonl", records[:2])
    external_records = [
        {
            **records[2],
            "dataset_name": "PolypGen",
            "image_id": "smoke_external_0",
        }
    ]
    external_split = _write_jsonl(fixtures / "external_split.jsonl", external_records)
    lora_split_dir = fixtures / "splits" / "fold_0"
    _write_jsonl(lora_split_dir / "train_ids.txt", records[:2])
    _write_jsonl(lora_split_dir / "val_ids.txt", records[1:2])
    _write_jsonl(fixtures / "splits" / "external_polypgen_ids.txt", external_records)
    (fixtures / "splits" / "split_summary.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "train_val_count": 2,
                "external_polypgen_count": len(external_records),
                "leakage_check_passed": True,
                "folds": [{"fold_id": 0, "train_count": 2, "val_count": 1}],
                "source_domain_counts": {"Kvasir": 1, "CVC": 1},
                "source_group_counts": {"Kvasir-SEG": 1, "CVC-ClinicDB": 1, "KvasirCVC": 0},
                "warnings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_rows = []
    for index, record in enumerate(records[:2]):
        baseline_dice = 0.42 + index * 0.08
        result_dice = baseline_dice + 0.12 - index * 0.03
        metric_rows.append(
            {
                **record,
                "metrics": {
                    "Dice": result_dice,
                    "IoU": result_dice * 0.85,
                    "Boundary F1": result_dice * 0.9,
                    "HD95": 2.0 + index,
                    "ASSD": 0.5 + index * 0.1,
                    "False Positive Rate": 0.01 + index * 0.02,
                    "False Negative Rate": 0.08 + index * 0.02,
                },
                "baseline_metrics": {
                    "Dice": baseline_dice,
                    "IoU": baseline_dice * 0.85,
                    "Boundary F1": baseline_dice * 0.9,
                    "HD95": 3.0 + index,
                    "ASSD": 0.7 + index * 0.1,
                    "False Positive Rate": 0.02 + index * 0.02,
                    "False Negative Rate": 0.13 + index * 0.02,
                },
                "delta_dice": result_dice - baseline_dice,
                "retrieval_vs_baseline": {
                    "Dice Delta": result_dice - baseline_dice,
                    "Boundary F1 Delta": result_dice * 0.9 - baseline_dice * 0.9,
                    "FNR Delta": -0.04,
                    "FPR Delta": -0.01 if index == 0 else 0.03,
                    "HD95 Delta": -0.5,
                    "ASSD Delta": -0.1,
                },
                "selected_exemplars": {
                    "positive_ids": [f"smoke_{index}_positive"],
                    "negative_ids": [f"smoke_{index}_negative"],
                    "boundary_ids": [f"smoke_{index}_boundary"],
                },
                "retrieval_sensitivity": {},
                "prompt_sensitivity_score": 0.04 + index * 0.02,
                "retrieval_influence_strength": 0.35 + index * 0.1,
                "lesion_area": 64.0 + index,
                "prediction_area": 70.0 + index * 10.0,
                "feature_vector": [1.0, 0.0, float(index) / 10.0],
            }
        )
    metrics_jsonl = _write_jsonl(fixtures / "per_image_metrics.jsonl", metric_rows)
    metrics_json = fixtures / "per_image_metrics.json"
    metrics_json.write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")

    return {
        "fixtures": fixtures,
        "internal_split": internal_split,
        "external_split": external_split,
        "metrics_jsonl": metrics_jsonl,
        "metrics_json": metrics_json,
        "lora_split_root": fixtures / "splits",
        "image": images_dir / "smoke_0.png",
        "mask": masks_dir / "smoke_0.png",
    }


def _snapshot_artifacts() -> dict[Path, bytes | None]:
    """快照可能被预检脚本原地重写的生成文件。

    参数：
        - 无

    返回：
        - 以路径为键的文件内容字典，缺失文件值为 None
    """
    snapshot: dict[Path, bytes | None] = {}
    for path in TRACKED_RUNTIME_ARTIFACTS:
        snapshot[path] = path.read_bytes() if path.exists() else None
    return snapshot


def _restore_artifacts(snapshot: dict[Path, bytes | None]) -> None:
    """恢复快照文件，确保冒烟运行不污染受追踪的运行时产物。

    参数：
        - snapshot: _snapshot_artifacts 返回的文件内容字典

    返回：
        - 无
    """
    for path, content in snapshot.items():
        if content is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _build_matrix(paths: dict[str, Path], output_dir: Path, timeout_seconds: int, suite: str) -> list[SmokeCommand]:
    """构建限界的训练/测试/评估冒烟命令矩阵。

    参数：
        - paths: _make_records 创建的夹具路径
        - output_dir: 冒烟输出根目录
        - timeout_seconds: 每个命令的默认超时
        - suite: 冒烟测试套件（quick 或 full）

    返回：
        - 有序的冒烟命令列表
    """
    internal = _workspace_relative(paths["internal_split"])
    external = _workspace_relative(paths["external_split"])
    metrics_jsonl = _workspace_relative(paths["metrics_jsonl"])
    metrics_json = _workspace_relative(paths["metrics_json"])
    lora_split_root = _workspace_relative(paths["lora_split_root"])
    smoke_image = _workspace_relative(paths["image"])
    smoke_mask = _workspace_relative(paths["mask"])
    out = _workspace_relative(output_dir)

    commands = [
        SmokeCommand(
            "prepare_5fold_polyp",
            ["MedicalSAM3/scripts/prepare_5fold_polyp.py", "--dummy", "--output-dir", f"{out}/splits_prepare"],
            timeout_seconds,
        ),
        SmokeCommand(
            "build_exemplar_bank",
            [
                "MedicalSAM3/scripts/build_exemplar_bank.py",
                "--split-file",
                internal,
                "--output-dir",
                f"{out}/exemplar_bank",
                "--dummy",
                "--max-items",
                "2",
                "--image-size",
                "32",
                "--min-positive-quality",
                "0",
                "--min-negative-quality",
                "0",
                "--min-diversity",
                "0",
                "--max-uncertainty",
                "1",
                "--min-negative-false-positive-risk",
                "0",
                "--min-items-per-type",
                "0",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "update_memory_from_review",
            [
                "MedicalSAM3/scripts/update_memory_from_review.py",
                "--memory-bank",
                f"{out}/exemplar_bank/memory_v0.json",
                "--review-csv",
                f"{out}/exemplar_bank/review_queue.csv",
                "--output-dir",
                f"{out}/exemplar_bank_reviewed",
                "--dummy",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "curate_exemplar_bank_from_delta",
            [
                "MedicalSAM3/scripts/curate_exemplar_bank_from_delta.py",
                "--memory-bank",
                f"{out}/exemplar_bank/memory_v0.json",
                "--per-image-metrics",
                metrics_jsonl,
                "--output-dir",
                f"{out}/exemplar_bank_curated",
                "--min-used",
                "1",
                "--min-items-per-type",
                "0",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "build_rssda_bank",
            [
                "MedicalSAM3/scripts/build_rssda_bank.py",
                "--split-file",
                internal,
                "--output-dir",
                f"{out}/rssda_bank",
                "--dummy",
                "--max-items",
                "2",
                "--image-size",
                "32",
                "--device",
                "cpu",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "preflight_medex_sam3",
            [
                "MedicalSAM3/scripts/preflight_medex_sam3.py",
                "--allow-dummy",
                "--no-require-official-sam3",
                "--image-size",
                "32",
                "--precision",
                "fp32",
                "--report-dir",
                f"{out}/preflight",
                "--split-dir",
                lora_split_root,
                "--results-dir",
                f"{out}/preflight_results",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "train_lora_medical",
            [
                "MedicalSAM3/scripts/train_lora_medical.py",
                "--dummy",
                "--allow-dummy",
                "--no-require-official-sam3",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--image-size",
                "32",
                "--device",
                "cpu",
                "--max-train-steps",
                "1",
                "--max-val-steps",
                "1",
                "--split-dir",
                lora_split_root,
                "--output-dir",
                f"{out}/lora",
                "--report-dir",
                f"{out}/lora_reports",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "train_exemplar_prompt",
            [
                "MedicalSAM3/scripts/train_exemplar_prompt.py",
                "--memory-bank",
                f"{out}/exemplar_bank",
                "--split-file",
                internal,
                "--val-split-file",
                internal,
                "--output-dir",
                f"{out}/exemplar_prompt",
                "--dummy",
                "--preflight-only",
                "--image-size",
                "32",
                "--device",
                "cpu",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "train_rssda",
            [
                "MedicalSAM3/scripts/train_rssda.py",
                "--split-file",
                internal,
                "--memory-bank",
                f"{out}/rssda_bank",
                "--output-dir",
                f"{out}/rssda_train",
                "--dummy",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--image-size",
                "32",
                "--device",
                "cpu",
                "--max-steps",
                "1",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "validate_medex_sam3",
            [
                "MedicalSAM3/scripts/validate_medex_sam3.py",
                "--split-file",
                external,
                "--output-dir",
                f"{out}/validate_medex",
                "--dummy",
                "--image-size",
                "32",
                "--no-visualizations",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "validate_rssda",
            [
                "MedicalSAM3/scripts/validate_rssda.py",
                "--split-file",
                external,
                "--memory-bank",
                f"{out}/rssda_bank",
                "--output-dir",
                f"{out}/validate_rssda",
                "--dummy",
                "--image-size",
                "32",
                "--device",
                "cpu",
                "--no-visualizations",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "analyze_hard_case_delta",
            [
                "MedicalSAM3/scripts/analyze_hard_case_delta.py",
                "--per-image-metrics",
                metrics_jsonl,
                "--output",
                f"{out}/hard_case_delta/report.json",
                "--hard-cases-csv",
                f"{out}/hard_case_delta/hard_cases.csv",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "select_bank_candidates",
            [
                "MedicalSAM3/scripts/select_bank_candidates.py",
                "--per-image-metrics",
                metrics_jsonl,
                "--output-dir",
                f"{out}/bank_candidates",
                "--positive-limit",
                "1",
                "--negative-limit",
                "1",
                "--per-site-limit",
                "1",
                "--max-hash-distance",
                "0",
                "--feature-similarity-threshold",
                "1.1",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "prepare_continual_adaptation",
            [
                "MedicalSAM3/scripts/prepare_continual_adaptation.py",
                "--per-image-metrics",
                metrics_jsonl,
                "--split-file",
                internal,
                "--output-dir",
                f"{out}/continual_adaptation",
                "--max-cases",
                "2",
            ],
            timeout_seconds,
        ),
        SmokeCommand(
            "run_ablation",
            ["MedicalSAM3/scripts/run_ablation.py", "--dummy", "--output-dir", f"{out}/ablation"],
            timeout_seconds,
        ),
        SmokeCommand(
            "summarize_cv_results",
            [
                "MedicalSAM3/scripts/summarize_cv_results.py",
                "--results-dir",
                out,
                "--ablation-dir",
                f"{out}/ablation",
            ],
            timeout_seconds,
        ),
    ]

    if suite == "full":
        commands.extend(
            [
                SmokeCommand(
                    "run_retrieval_inference",
                    [
                        "MedicalSAM3/scripts/run_retrieval_inference.py",
                        "--input-path",
                        smoke_image,
                        "--bbox",
                        "4,4,28,28",
                        "--memory-bank",
                        f"{out}/rssda_bank",
                        "--output-dir",
                        f"{out}/retrieval_inference",
                        "--dummy",
                        "--image-size",
                        "32",
                        "--device",
                        "cpu",
                    ],
                    timeout_seconds,
                ),
                SmokeCommand(
                    "prompt_sensitivity_case",
                    [
                        "MedicalSAM3/scripts/prompt_sensitivity_case.py",
                        "--input-path",
                        smoke_image,
                        "--bbox",
                        "4,4,28,28",
                        "--mask-path",
                        smoke_mask,
                        "--memory-bank",
                        f"{out}/rssda_bank",
                        "--output-dir",
                        f"{out}/prompt_sensitivity",
                        "--dummy",
                        "--image-size",
                        "32",
                        "--device",
                        "cpu",
                    ],
                    timeout_seconds,
                ),
                SmokeCommand(
                    "report_rssda_behavior",
                    [
                        "MedicalSAM3/scripts/report_rssda_behavior.py",
                        "--internal-split-file",
                        internal,
                        "--external-split-file",
                        external,
                        "--memory-bank",
                        f"{out}/rssda_bank",
                        "--output-dir",
                        f"{out}/rssda_behavior_report",
                        "--dummy",
                        "--image-size",
                        "32",
                        "--device",
                        "cpu",
                        "--max-samples-per-split",
                        "1",
                    ],
                    timeout_seconds,
                ),
            ]
        )
    return commands


def _run_command(command: SmokeCommand, python_executable: str, env: dict[str, str]) -> dict[str, object]:
    """运行单个冒烟命令，输出继承的终端信息。

    参数：
        - command: 命令描述符
        - python_executable: 使用的 Python 解释器
        - env: 子进程的环境变量

    返回：
        - 用于最终冒烟摘要的结果字典
    """
    argv = [python_executable, *command.args]
    print(json.dumps({"smoke_start": command.label, "argv": argv}, ensure_ascii=True), flush=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=WORKSPACE_ROOT,
            env=env,
            check=False,
            timeout=command.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"smoke_timeout": command.label, "timeout_seconds": command.timeout_seconds}, ensure_ascii=True), flush=True)
        return {"label": command.label, "status": "timeout", "code": None}
    status = "ok" if completed.returncode == 0 else "failed"
    print(json.dumps({"smoke_done": command.label, "status": status, "code": completed.returncode}, ensure_ascii=True), flush=True)
    return {"label": command.label, "status": status, "code": completed.returncode}


def main() -> int:
    """命令行入口：运行 MedEx-SAM3 脚本冒烟测试套件。

    参数：
        - 无（通过 argparse 解析命令行参数）

    返回：
        - 进程退出码（全部通过为 0，否则为 1）
    """
    parser = argparse.ArgumentParser(description="Run bounded MedEx-SAM3 script smoke checks with synthetic fixtures.")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/script_smoke")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    output_dir = (WORKSPACE_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    try:
        output_dir.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"output directory must stay inside the project root: {output_dir}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _make_records(output_dir, args.image_size)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    snapshot = _snapshot_artifacts()
    results: list[dict[str, object]] = []
    try:
        for command in _build_matrix(paths, output_dir, args.timeout_seconds, args.suite):
            result = _run_command(command, args.python, env)
            results.append(result)
            if result["status"] != "ok":
                break
    finally:
        _restore_artifacts(snapshot)

    summary = {
        "suite": args.suite,
        "output_dir": _workspace_relative(output_dir),
        "passed": sum(1 for result in results if result["status"] == "ok"),
        "total_run": len(results),
        "results": results,
    }
    (output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"smoke_summary": summary}, indent=2), flush=True)
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
