"""聚合检索校准诊断和假阴性分析。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from MedicalSAM3.evaluation.retrieval_diagnostics import summarize_retrieval_diagnostics


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件并返回字典列表。

    参数：
        - path: JSONL 文件路径。

    返回：
        - 解析后的字典列表。
    """
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _mean(values: list[float]) -> float:
    """计算浮点数列表的平均值。

    参数：
        - values: 浮点数列表。

    返回：
        - 平均值，空列表返回 0.0。
    """
    if not values:
        return 0.0
    return float(sum(values) / max(len(values), 1))


def _quantile(values: list[float], q: float) -> float:
    """计算浮点数列表的指定分位数。

    参数：
        - values: 浮点数列表。
        - q: 分位数值（0.0 ~ 1.0）。

    返回：
        - 分位数结果。
    """
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def summarize_false_negative_analysis(
    diagnostics_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    small_lesion_quantile: float = 0.25,
) -> dict[str, Any]:
    """总结假阴性分析结果，按病变大小和高负样本影响分组统计 FNR。

    参数：
        - diagnostics_rows: 诊断行列表。
        - metric_rows: 指标行列表。
        - small_lesion_quantile: 小病变分位阈值，默认为 0.25。

    返回：
        - 包含假阴性分析摘要的字典。
    """
    diagnostics_by_image = {str(row.get("image_id", "")): row for row in diagnostics_rows}
    lesion_areas = [float(row.get("lesion_area", 0.0)) for row in metric_rows if float(row.get("lesion_area", 0.0)) > 0.0]
    small_lesion_threshold = _quantile(lesion_areas, small_lesion_quantile) if lesion_areas else 0.0

    small_lesion_fnr: list[float] = []
    large_lesion_fnr: list[float] = []
    high_negative_influence_fnr: list[float] = []
    high_negative_cases: list[dict[str, Any]] = []

    for row in metric_rows:
        image_id = str(row.get("image_id", ""))
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        fnr = float(metrics.get("FNR", metrics.get("False Negative Rate", 0.0)))
        lesion_area = float(row.get("lesion_area", 0.0))
        diagnostics = diagnostics_by_image.get(image_id, {})
        gate_diag = diagnostics.get("gate_diagnostics", {}) if isinstance(diagnostics.get("gate_diagnostics"), dict) else {}
        negative_weight = float(gate_diag.get("negative_calibrated_weight", 0.0))
        positive_weight = float(gate_diag.get("positive_calibrated_weight", 0.0))
        influence = float(diagnostics.get("retrieval_influence_strength", 0.0))

        if lesion_area > 0.0 and lesion_area <= small_lesion_threshold:
            small_lesion_fnr.append(fnr)
        elif lesion_area > small_lesion_threshold:
            large_lesion_fnr.append(fnr)

        if negative_weight >= positive_weight:
            high_negative_influence_fnr.append(fnr)
            high_negative_cases.append(
                {
                    "image_id": image_id,
                    "lesion_area": lesion_area,
                    "fnr": fnr,
                    "negative_calibrated_weight": negative_weight,
                    "positive_calibrated_weight": positive_weight,
                    "retrieval_influence_strength": influence,
                }
            )

    high_negative_cases.sort(key=lambda item: (item["fnr"], item["negative_calibrated_weight"]), reverse=True)
    return {
        "small_lesion_area_threshold": small_lesion_threshold,
        "small_lesion_count": len(small_lesion_fnr),
        "large_lesion_count": len(large_lesion_fnr),
        "small_lesion_mean_fnr": _mean(small_lesion_fnr),
        "large_lesion_mean_fnr": _mean(large_lesion_fnr),
        "high_negative_influence_count": len(high_negative_influence_fnr),
        "high_negative_influence_mean_fnr": _mean(high_negative_influence_fnr),
        "top_false_negative_risk_cases": high_negative_cases[:10],
    }


def summarize_retrieval_calibration(
    diagnostics_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    bins: int = 10,
    small_lesion_quantile: float = 0.25,
) -> dict[str, Any]:
    """汇总检索校准诊断结果，包含检索诊断摘要和假阴性分析。

    参数：
        - diagnostics_rows: 诊断行列表。
        - metric_rows: 指标行列表。
        - bins: 直方图柱数，默认为 10。
        - small_lesion_quantile: 小病变分位阈值，默认为 0.25。

    返回：
        - 包含检索诊断和假阴性分析的字典。
    """
    return {
        "retrieval_diagnostics": summarize_retrieval_diagnostics(diagnostics_rows, bins=bins),
        "false_negative_analysis": summarize_false_negative_analysis(
            diagnostics_rows,
            metric_rows,
            small_lesion_quantile=small_lesion_quantile,
        ),
    }


def write_retrieval_calibration_report(
    output_dir: str | Path,
    diagnostics_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    bins: int = 10,
    small_lesion_quantile: float = 0.25,
) -> Path:
    """生成并保存检索校准报告 JSON 文件。

    参数：
        - output_dir: 输出目录。
        - diagnostics_rows: 诊断行列表。
        - metric_rows: 指标行列表。
        - bins: 直方图柱数。
        - small_lesion_quantile: 小病变分位阈值。

    返回：
        - 生成的报告文件路径。
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / "retrieval_calibration_summary.json"
    destination.write_text(
        json.dumps(
            summarize_retrieval_calibration(
                diagnostics_rows,
                metric_rows,
                bins=bins,
                small_lesion_quantile=small_lesion_quantile,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    """命令行入口：汇总检索校准诊断并保存报告。

    参数：
        - 无。

    返回：
        - 退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(description="Summarize retrieval calibration diagnostics.")
    parser.add_argument("--diagnostics-path", default="MedicalSAM3/outputs/medex_sam3/rssda_eval/retrieval_diagnostics.jsonl")
    parser.add_argument("--metrics-path", default="MedicalSAM3/outputs/medex_sam3/rssda_eval/per_image_metrics.jsonl")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/rssda_eval")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--small-lesion-quantile", type=float, default=0.25)
    args = parser.parse_args()

    diagnostics_rows = _read_jsonl(args.diagnostics_path)
    metric_rows = _read_jsonl(args.metrics_path)
    destination = write_retrieval_calibration_report(
        args.output_dir,
        diagnostics_rows,
        metric_rows,
        bins=args.bins,
        small_lesion_quantile=args.small_lesion_quantile,
    )
    print(json.dumps({"rows": len(diagnostics_rows), "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
