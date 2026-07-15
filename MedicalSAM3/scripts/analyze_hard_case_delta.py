"""分析低 Dice 困难案例上的示例/检索增益。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    """从 JSON 或 JSONL 文件加载逐图像指标行。

    参数：
        - path: 指标文件路径

    返回：
        - 指标字典列表
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"per-image metrics not found: {target}")
    if target.suffix == ".jsonl":
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("per-image metrics must be a JSON array or JSONL rows")
    return payload


def _dice(row: dict[str, Any], field: str) -> float:
    """从行数据中提取指定字段的 Dice 系数。

    参数：
        - row: 指标数据行字典
        - field: 字段名，支持 "delta_dice" 或指标字典字段名

    返回：
        - Dice 系数浮点值
    """
    if field == "delta_dice":
        return float(row.get("delta_dice", 0.0))
    payload = row.get(field, {})
    if not isinstance(payload, dict):
        raise TypeError(f"{field} must be a metrics dictionary")
    return float(payload.get("Dice", 0.0))


def _summarize_subset(rows: list[dict[str, Any]], *, min_gain: float, rescue_threshold: float) -> dict[str, Any]:
    """汇总子集（如低 Dice 子集）的各项统计指标。

    参数：
        - rows: 指标行列表
        - min_gain: 被视为有意义增益的最小增量阈值
        - rescue_threshold: 抢救阈值（基线低于此、结果高于此视为抢救成功）

    返回：
        - 包含样本数、均值、中位数、增益率、抢救率等指标的字典
    """
    if not rows:
        return {
            "count": 0,
            "baseline_dice_mean": 0.0,
            "result_dice_mean": 0.0,
            "mean_delta_dice": 0.0,
            "median_delta_dice": 0.0,
            "positive_delta_rate": 0.0,
            "negative_delta_rate": 0.0,
            "rescue_rate": 0.0,
            "meaningful_gain_rate": 0.0,
            "low_dice_error_reduction": 0.0,
            "severe_harm_rate": 0.0,
        }

    baseline = [_dice(row, "baseline_metrics") for row in rows]
    result = [_dice(row, "metrics") for row in rows]
    delta = [res - base for base, res in zip(baseline, result)]
    sorted_delta = sorted(delta)
    mid = len(sorted_delta) // 2
    median = sorted_delta[mid] if len(sorted_delta) % 2 else 0.5 * (sorted_delta[mid - 1] + sorted_delta[mid])
    error_reductions = [
        (res - base) / max(1.0 - base, 1e-6)
        for base, res in zip(baseline, result)
    ]
    return {
        "count": len(rows),
        "baseline_dice_mean": sum(baseline) / len(rows),
        "result_dice_mean": sum(result) / len(rows),
        "mean_delta_dice": sum(delta) / len(rows),
        "median_delta_dice": median,
        "positive_delta_rate": sum(1 for value in delta if value > 0.0) / len(rows),
        "negative_delta_rate": sum(1 for value in delta if value < 0.0) / len(rows),
        "rescue_rate": sum(1 for base, res in zip(baseline, result) if base < rescue_threshold <= res) / len(rows),
        "meaningful_gain_rate": sum(1 for value in delta if value >= min_gain) / len(rows),
        "low_dice_error_reduction": sum(error_reductions) / len(rows),
        "severe_harm_rate": sum(1 for value in delta if value <= -min_gain) / len(rows),
    }


def _bottom_quantile(rows: list[dict[str, Any]], quantile: float) -> list[dict[str, Any]]:
    """按基线 Dice 排序，取出最低分位数子集。

    参数：
        - rows: 指标行列表
        - quantile: 分位数（如 0.1 表示底部 10%）

    返回：
        - 排序后的底部子集列表
    """
    if not rows:
        return []
    ranked = sorted(rows, key=lambda row: _dice(row, "baseline_metrics"))
    count = max(1, int(round(len(ranked) * quantile)))
    return ranked[:count]


def _weighted_hard_case_gain(rows: list[dict[str, Any]], gamma: float) -> float:
    """计算加权困难案例增益，权重为 (1 - baseline Dice) 的 gamma 次幂。

    参数：
        - rows: 指标行列表
        - gamma: 权重指数

    返回：
        - 加权后的平均增益
    """
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        baseline = _dice(row, "baseline_metrics")
        delta = _dice(row, "metrics") - baseline
        weight = max(1.0 - baseline, 0.0) ** gamma
        numerator += weight * delta
        denominator += weight
    return numerator / max(denominator, 1e-6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """将困难案例按基线 Dice 排序后写入 CSV 文件。

    参数：
        - path: CSV 输出文件路径
        - rows: 指标行列表

    返回：
        - 无返回值，仅执行写入文件的副作用
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "baseline_dice",
                "result_dice",
                "delta_dice",
                "selected_positive",
                "selected_negative",
                "selected_boundary",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: _dice(item, "baseline_metrics")):
            selected = row.get("selected_exemplars") or {}
            writer.writerow(
                {
                    "image_id": row.get("image_id", ""),
                    "baseline_dice": _dice(row, "baseline_metrics"),
                    "result_dice": _dice(row, "metrics"),
                    "delta_dice": _dice(row, "metrics") - _dice(row, "baseline_metrics"),
                    "selected_positive": ";".join(selected.get("positive_ids", []) or []),
                    "selected_negative": ";".join(selected.get("negative_ids", []) or []),
                    "selected_boundary": ";".join(selected.get("boundary_ids", []) or []),
                }
            )


def main() -> int:
    """脚本命令行入口，分析低 Dice 困难案例的增量指标。

    参数：
        - 无

    返回：
        - 进程退出码，0 表示成功
    """
    parser = argparse.ArgumentParser(description="Analyze low-Dice hard-case delta metrics.")
    parser.add_argument("--per-image-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--quantiles", default="0.1,0.2")
    parser.add_argument("--min-gain", type=float, default=0.03)
    parser.add_argument("--rescue-threshold", type=float, default=0.5)
    parser.add_argument("--hard-weight-gamma", type=float, default=2.0)
    parser.add_argument("--hard-cases-csv", default=None)
    args = parser.parse_args()

    rows = _load_rows(args.per_image_metrics)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    quantiles = [float(value) for value in args.quantiles.split(",") if value.strip()]

    threshold_report = {}
    hard_rows_for_csv: list[dict[str, Any]] = []
    for threshold in thresholds:
        subset = [row for row in rows if _dice(row, "baseline_metrics") < threshold]
        threshold_report[f"baseline_dice<{threshold:g}"] = _summarize_subset(
            subset,
            min_gain=args.min_gain,
            rescue_threshold=args.rescue_threshold,
        )
        if threshold == max(thresholds):
            hard_rows_for_csv = subset

    quantile_report = {}
    for quantile in quantiles:
        subset = _bottom_quantile(rows, quantile)
        quantile_report[f"bottom_{int(quantile * 100)}pct_by_baseline_dice"] = _summarize_subset(
            subset,
            min_gain=args.min_gain,
            rescue_threshold=args.rescue_threshold,
        )

    report = {
        "count": len(rows),
        "definitions": {
            "hard_case_dice_gain@tau": "mean(result Dice - baseline Dice) on cases with baseline Dice < tau",
            "low_dice_error_reduction": "mean((result Dice - baseline Dice) / (1 - baseline Dice)) on the subset",
            "rescue_rate": f"fraction of subset crossing Dice >= {args.rescue_threshold}",
            "meaningful_gain_rate": f"fraction of subset with delta Dice >= {args.min_gain}",
            "severe_harm_rate": f"fraction of subset with delta Dice <= -{args.min_gain}",
            "weighted_hard_case_gain": f"delta Dice weighted by (1 - baseline Dice)^{args.hard_weight_gamma}",
        },
        "overall": _summarize_subset(rows, min_gain=args.min_gain, rescue_threshold=args.rescue_threshold),
        "weighted_hard_case_gain": _weighted_hard_case_gain(rows, gamma=args.hard_weight_gamma),
        "threshold_subsets": threshold_report,
        "quantile_subsets": quantile_report,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.hard_cases_csv:
        _write_csv(Path(args.hard_cases_csv), hard_rows_for_csv)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
