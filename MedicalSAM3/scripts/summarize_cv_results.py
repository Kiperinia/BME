"""汇总 MedEx-SAM3 的交叉验证和消融实验结果。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MedicalSAM3.scripts.common import ensure_dir


def _collect_val_metrics(results_dir: Path) -> list[dict[str, float]]:
    """收集各折的验证集指标。

    参数：
        - results_dir: 结果根目录

    返回：
        - 各折指标字典列表
    """
    metrics = []
    for path in sorted(results_dir.glob("fold_*/val_metrics.json")):
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    return metrics


def _collect_external_metrics(eval_dir: Path) -> dict[str, float]:
    """收集外部测试集指标。

    参数：
        - eval_dir: 评估目录

    返回：
        - 指标字典
    """
    summary_path = eval_dir / "summary_metrics.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    legacy_path = eval_dir / "external_polypgen_metrics.json"
    if legacy_path.exists():
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    return {}


def _mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """计算多个字典中各数值字段的均值和标准差。

    参数：
        - rows: 字典列表

    返回：
        - 字段名到 {mean, std} 的字典
    """
    if not rows:
        return {}
    keys = [key for key in rows[0].keys() if isinstance(rows[0][key], (int, float))]
    return {
        key: {
            "mean": statistics.mean(row[key] for row in rows),
            "std": statistics.pstdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        }
        for key in keys
    }


def main() -> int:
    """脚本命令行入口，汇总交叉验证和消融结果并输出表格。

    参数：
        - 无

    返回：
        - 进程退出码，0 表示成功
    """
    parser = argparse.ArgumentParser(description="Summarize MedEx-SAM3 CV and ablation results.")
    parser.add_argument("--results-dir", default="MedicalSAM3/outputs/medex_sam3")
    parser.add_argument("--ablation-dir", default="MedicalSAM3/outputs/medex_sam3/ablation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(results_dir / "summary")
    fold_metrics = _collect_val_metrics(results_dir)
    cv_summary = _mean_std(fold_metrics)
    (output_dir / "cv_mean_std.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")

    external_metrics = _collect_external_metrics(results_dir / "eval")
    (output_dir / "external_polypgen_metrics.json").write_text(json.dumps(external_metrics, indent=2), encoding="utf-8")

    ablation_dir = Path(args.ablation_dir)
    rows = []
    for metrics_path in sorted(ablation_dir.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"method": payload.get("method", metrics_path.parent.name), "fold": payload.get("fold", "aggregate"), **payload["metrics"]})
    if rows:
        csv_path = output_dir / "ablation_table.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        md_lines = ["| method | fold | Dice | IoU | Precision | Recall | Boundary F1 | HD95 | ASSD | FPR | FNR |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            md_lines.append(
                f"| {row['method']} | {row['fold']} | {row['Dice']} | {row['IoU']} | {row['Precision']} | {row['Recall']} | {row['Boundary F1']} | {row['HD95']} | {row['ASSD']} | {row['FPR']} | {row['FNR']} |"
            )
        (output_dir / "ablation_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"cv_mean_std": str(output_dir / 'cv_mean_std.json'), "external": str(output_dir / 'external_polypgen_metrics.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
