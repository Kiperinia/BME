"""Summarize cross-validation and ablation results for MedEx-SAM3."""

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
    metrics = []
    for path in sorted(results_dir.glob("fold_*/val_metrics.json")):
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    return metrics


def _mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
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
    parser = argparse.ArgumentParser(description="Summarize MedEx-SAM3 CV and ablation results.")
    parser.add_argument("--results-dir", default="MedicalSAM3/outputs/medex_sam3")
    parser.add_argument("--ablation-dir", default="MedicalSAM3/outputs/medex_sam3/ablation_runs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(results_dir)
    fold_metrics = _collect_val_metrics(results_dir)
    cv_summary = _mean_std(fold_metrics)
    (output_dir / "cv_mean_std.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")

    external_path = results_dir / "validation" / "external_polypgen_metrics.json"
    external_metrics = json.loads(external_path.read_text(encoding="utf-8")) if external_path.exists() else {}
    (output_dir / "external_polypgen_metrics.json").write_text(json.dumps(external_metrics, indent=2), encoding="utf-8")

    ablation_dir = Path(args.ablation_dir)
    rows = []
    for metrics_path in sorted(ablation_dir.glob("run_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"Config": payload["config"], **payload["metrics"]})
    if rows:
        csv_path = output_dir / "ablation_table.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        md_lines = ["| Config | Dice | IoU | Precision | Recall | Boundary F1 | HD95 | ASSD | FPR | FNR | Prompt Sensitivity |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            md_lines.append(
                f"| {row['Config']} | {row['Dice']} | {row['IoU']} | {row['Precision']} | {row['Recall']} | {row['Boundary F1']} | {row['HD95']} | {row['ASSD']} | {row['FPR']} | {row['FNR']} | {row['Prompt Sensitivity']} |"
            )
        (output_dir / "ablation_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"cv_mean_std": str(output_dir / 'cv_mean_std.json'), "external": str(output_dir / 'external_polypgen_metrics.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
