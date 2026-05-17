import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from MedicalSAM3.scripts.select_bank_candidates import select_bank_candidates


def _write_image(path: Path, color: tuple[int, int, int], pattern: int = 0) -> None:
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[:] = color
    offset = max(min(pattern, 8), 0)
    array[offset : offset + 4, offset : offset + 4] = 255
    array[12 - offset : 16 - offset, :4] = (pattern * 20) % 255
    Image.fromarray(array).save(path)


class TestSelectBankCandidates(unittest.TestCase):
    def test_balances_candidates_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            vis_dir = root / "visualizations"
            image_dir.mkdir(parents=True)
            vis_dir.mkdir(parents=True)

            ids = {
                "p1": "PolypGen__C1_PATA_0001",
                "p2": "PolypGen__C1_PATB_0001",
                "p3": "PolypGen__C2_PATC_0001",
                "n1": "PolypGen__C1_PATD_0001",
                "n2": "PolypGen__C2_PATE_0001",
                "n3": "PolypGen__C2_PATF_0001",
            }
            images = {}
            masks = {}
            for key, image_id in ids.items():
                image_path = image_dir / f"{image_id}.png"
                mask_path = image_dir / f"{image_id}_mask.png"
                color = {
                    "p1": (240, 50, 50),
                    "p2": (220, 60, 60),
                    "p3": (50, 240, 50),
                    "n1": (50, 50, 240),
                    "n2": (230, 230, 80),
                    "n3": (230, 230, 80),
                }[key]
                pattern = {
                    "p1": 1,
                    "p2": 3,
                    "p3": 5,
                    "n1": 7,
                    "n2": 2,
                    "n3": 2,
                }[key]
                _write_image(image_path, color, pattern=pattern)
                _write_image(mask_path, (255, 255, 255))
                _write_image(vis_dir / f"{image_id}_query.png", color, pattern=pattern)
                _write_image(vis_dir / f"{image_id}_gt.png", (255, 255, 255))
                _write_image(vis_dir / f"{image_id}_pred.png", color, pattern=pattern)
                images[key] = image_path
                masks[key] = mask_path

            metrics_rows = [
                {
                    "image_id": ids["p1"],
                    "image_path": str(images["p1"]),
                    "mask_path": str(masks["p1"]),
                    "dataset_name": "PolypGen",
                    "metrics": {"Dice": 0.92, "Boundary F1": 0.81, "False Positive Rate": 0.01, "False Negative Rate": 0.09, "HD95": 1.0, "ASSD": 0.1},
                    "baseline_metrics": {"Dice": 0.82, "Boundary F1": 0.70, "False Positive Rate": 0.02, "False Negative Rate": 0.15, "HD95": 2.0, "ASSD": 0.3},
                    "retrieval_vs_baseline": {"Dice Delta": 0.10, "Boundary F1 Delta": 0.11, "FNR Delta": -0.06, "FPR Delta": -0.01, "HD95 Delta": -1.0, "ASSD Delta": -0.2},
                    "retrieval_sensitivity": {"Dice Delta": 0.01},
                    "lesion_area": 120.0,
                    "prediction_area": 118.0,
                    "selected_positive": ["pos_a", "pos_b"],
                    "selected_negative": ["neg_a"],
                    "feature_vector": [1.0, 0.0, 0.0],
                },
                {
                    "image_id": ids["p2"],
                    "image_path": str(images["p2"]),
                    "mask_path": str(masks["p2"]),
                    "dataset_name": "PolypGen",
                    "metrics": {"Dice": 0.90, "Boundary F1": 0.78, "False Positive Rate": 0.01, "False Negative Rate": 0.11, "HD95": 1.2, "ASSD": 0.12},
                    "baseline_metrics": {"Dice": 0.79, "Boundary F1": 0.66, "False Positive Rate": 0.02, "False Negative Rate": 0.18, "HD95": 2.4, "ASSD": 0.32},
                    "retrieval_vs_baseline": {"Dice Delta": 0.11, "Boundary F1 Delta": 0.12, "FNR Delta": -0.07, "FPR Delta": -0.01, "HD95 Delta": -1.2, "ASSD Delta": -0.2},
                    "retrieval_sensitivity": {"Dice Delta": 0.02},
                    "lesion_area": 122.0,
                    "prediction_area": 120.0,
                    "selected_positive": ["pos_a", "pos_b"],
                    "selected_negative": ["neg_a"],
                    "feature_vector": [1.0, 0.0, 0.0],
                },
                {
                    "image_id": ids["p3"],
                    "image_path": str(images["p3"]),
                    "mask_path": str(masks["p3"]),
                    "dataset_name": "PolypGen",
                    "metrics": {"Dice": 0.89, "Boundary F1": 0.79, "False Positive Rate": 0.01, "False Negative Rate": 0.10, "HD95": 1.1, "ASSD": 0.11},
                    "baseline_metrics": {"Dice": 0.80, "Boundary F1": 0.68, "False Positive Rate": 0.02, "False Negative Rate": 0.16, "HD95": 2.2, "ASSD": 0.28},
                    "retrieval_vs_baseline": {"Dice Delta": 0.09, "Boundary F1 Delta": 0.11, "FNR Delta": -0.06, "FPR Delta": -0.01, "HD95 Delta": -1.1, "ASSD Delta": -0.17},
                    "retrieval_sensitivity": {"Dice Delta": 0.01},
                    "lesion_area": 110.0,
                    "prediction_area": 109.0,
                    "selected_positive": ["pos_c", "pos_d"],
                    "selected_negative": ["neg_b"],
                    "feature_vector": [0.0, 1.0, 0.0],
                },
                {
                    "image_id": ids["n1"],
                    "image_path": str(images["n1"]),
                    "mask_path": str(masks["n1"]),
                    "dataset_name": "PolypGen",
                    "metrics": {"Dice": 0.60, "Boundary F1": 0.40, "False Positive Rate": 0.20, "False Negative Rate": 0.30, "HD95": 7.0, "ASSD": 1.8},
                    "baseline_metrics": {"Dice": 0.72, "Boundary F1": 0.52, "False Positive Rate": 0.08, "False Negative Rate": 0.24, "HD95": 5.0, "ASSD": 1.1},
                    "retrieval_vs_baseline": {"Dice Delta": -0.12, "Boundary F1 Delta": -0.12, "FNR Delta": 0.06, "FPR Delta": 0.12, "HD95 Delta": 2.0, "ASSD Delta": 0.7},
                    "retrieval_sensitivity": {"Dice Delta": 0.05},
                    "lesion_area": 50.0,
                    "prediction_area": 110.0,
                    "selected_positive": ["pos_x"],
                    "selected_negative": ["neg_x"],
                    "feature_vector": [0.0, 0.0, 1.0],
                },
                {
                    "image_id": ids["n2"],
                    "image_path": str(images["n2"]),
                    "mask_path": str(masks["n2"]),
                    "dataset_name": "PolypGen",
                    "metrics": {"Dice": 0.58, "Boundary F1": 0.38, "False Positive Rate": 0.24, "False Negative Rate": 0.28, "HD95": 7.4, "ASSD": 1.9},
                    "baseline_metrics": {"Dice": 0.70, "Boundary F1": 0.50, "False Positive Rate": 0.10, "False Negative Rate": 0.22, "HD95": 5.2, "ASSD": 1.2},
                    "retrieval_vs_baseline": {"Dice Delta": -0.12, "Boundary F1 Delta": -0.12, "FNR Delta": 0.06, "FPR Delta": 0.14, "HD95 Delta": 2.2, "ASSD Delta": 0.7},
                    "retrieval_sensitivity": {"Dice Delta": 0.04},
                    "lesion_area": 45.0,
                    "prediction_area": 120.0,
                    "selected_positive": ["pos_y"],
                    "selected_negative": ["neg_y"],
                },
                {
                    "image_id": ids["n3"],
                    "image_path": str(images["n3"]),
                    "mask_path": str(masks["n3"]),
                    "dataset_name": "PolypGen",
                    "metrics": {"Dice": 0.57, "Boundary F1": 0.37, "False Positive Rate": 0.25, "False Negative Rate": 0.28, "HD95": 7.5, "ASSD": 2.0},
                    "baseline_metrics": {"Dice": 0.69, "Boundary F1": 0.49, "False Positive Rate": 0.09, "False Negative Rate": 0.22, "HD95": 5.1, "ASSD": 1.2},
                    "retrieval_vs_baseline": {"Dice Delta": -0.12, "Boundary F1 Delta": -0.12, "FNR Delta": 0.06, "FPR Delta": 0.16, "HD95 Delta": 2.4, "ASSD Delta": 0.8},
                    "retrieval_sensitivity": {"Dice Delta": 0.04},
                    "lesion_area": 45.0,
                    "prediction_area": 122.0,
                    "selected_positive": ["pos_y"],
                    "selected_negative": ["neg_y"],
                },
            ]
            diagnostics_rows = [
                {"image_id": ids["p1"], "retrieval_influence_strength": 0.7},
                {"image_id": ids["p2"], "retrieval_influence_strength": 0.68},
                {"image_id": ids["p3"], "retrieval_influence_strength": 0.65},
                {"image_id": ids["n1"], "retrieval_influence_strength": 0.75},
                {"image_id": ids["n2"], "retrieval_influence_strength": 0.72},
                {"image_id": ids["n3"], "retrieval_influence_strength": 0.73},
            ]
            region_rows = [
                {"image_id": ids["p1"], "mean_confidence_in_region": 0.88, "selected_site_id": "C1"},
                {"image_id": ids["p2"], "mean_confidence_in_region": 0.86, "selected_site_id": "C1"},
                {"image_id": ids["p3"], "mean_confidence_in_region": 0.83, "selected_site_id": "C2"},
                {"image_id": ids["n1"], "mean_entropy_in_region": 0.52, "selected_site_id": "C1"},
                {"image_id": ids["n2"], "mean_entropy_in_region": 0.61, "selected_site_id": "C2"},
                {"image_id": ids["n3"], "mean_entropy_in_region": 0.60, "selected_site_id": "C2"},
            ]
            prompt_rows = [
                {"image_id": ids["p1"], "prompt_sensitivity": {"prompt_sensitivity_score": 0.01}},
                {"image_id": ids["p2"], "prompt_sensitivity": {"prompt_sensitivity_score": 0.02}},
                {"image_id": ids["p3"], "prompt_sensitivity": {"prompt_sensitivity_score": 0.02}},
                {"image_id": ids["n1"], "prompt_sensitivity": {"prompt_sensitivity_score": 0.18}},
                {"image_id": ids["n2"], "prompt_sensitivity": {"prompt_sensitivity_score": 0.21}},
                {"image_id": ids["n3"], "prompt_sensitivity": {"prompt_sensitivity_score": 0.19}},
            ]

            metrics_path = root / "per_image_metrics.jsonl"
            metrics_path.write_text("\n".join(json.dumps(row) for row in metrics_rows), encoding="utf-8")
            diagnostics_path = root / "retrieval_diagnostics.jsonl"
            diagnostics_path.write_text("\n".join(json.dumps(row) for row in diagnostics_rows), encoding="utf-8")
            region_path = root / "region_retrieval_diagnostics.jsonl"
            region_path.write_text("\n".join(json.dumps(row) for row in region_rows), encoding="utf-8")
            prompt_path = root / "prompt_sensitivity.jsonl"
            prompt_path.write_text("\n".join(json.dumps(row) for row in prompt_rows), encoding="utf-8")

            summary = select_bank_candidates(
                per_image_metrics_path=metrics_path,
                retrieval_diagnostics_path=diagnostics_path,
                region_retrieval_diagnostics_path=region_path,
                prompt_sensitivity_path=prompt_path,
                output_dir=root / "review",
                bank_root=root / "continual_bank",
                train_bank_root=root / "train_bank",
                positive_limit=2,
                negative_limit=2,
                per_site_limit=1,
                max_hash_distance=0,
                feature_similarity_threshold=0.99,
                visualization_root=vis_dir,
            )

            self.assertEqual(summary["selected_counts"]["positive"], 2)
            self.assertEqual(summary["selected_counts"]["negative"], 2)

            positive_rows = [json.loads(line) for line in (root / "review" / "positive_candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            negative_rows = [json.loads(line) for line in (root / "review" / "negative_candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

            positive_ids = {row["image_id"] for row in positive_rows}
            self.assertIn(ids["p3"], positive_ids)
            self.assertEqual(len(positive_ids & {ids["p1"], ids["p2"]}), 1)
            negative_ids = {row["image_id"] for row in negative_rows}
            self.assertIn(ids["n1"], negative_ids)
            self.assertEqual(len(negative_ids & {ids["n2"], ids["n3"]}), 1)
            self.assertTrue((root / "review" / "approved_copy_commands.sh").exists())
            self.assertTrue(any(Path(row["review_dir"]).exists() for row in positive_rows))
            review_files = {path.name for path in Path(positive_rows[0]["review_dir"]).iterdir()}
            self.assertTrue(any(name.endswith("_pred.png") for name in review_files))
            self.assertTrue(all("target_bank_path" in row for row in positive_rows + negative_rows))
            self.assertTrue(all(row["recommended_priority"] in {"high", "medium", "low"} for row in positive_rows + negative_rows))
            self.assertTrue(all(Path(row["target_bank_path"]).name == "images" for row in positive_rows + negative_rows))
            copy_commands = (root / "review" / "approved_copy_commands.sh").read_text(encoding="utf-8")
            self.assertIn("# cp ", copy_commands)


if __name__ == "__main__":
    unittest.main()