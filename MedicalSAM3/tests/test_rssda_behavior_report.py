import unittest

import torch

from MedicalSAM3.scripts.report_rssda_behavior import _report_gap, summarize_heatmap


class TestRSSDABehaviorReport(unittest.TestCase):
    def test_summarize_heatmap_reports_overlap_and_entropy(self) -> None:
        heatmap = torch.tensor(
            [
                [0.1, 0.2, 0.9, 0.8],
                [0.1, 0.2, 0.7, 0.6],
                [0.0, 0.1, 0.2, 0.3],
                [0.0, 0.0, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        gt_mask = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        stats = summarize_heatmap(heatmap, gt_mask, top_percent=0.25)

        self.assertAlmostEqual(stats["max"], 0.9, places=5)
        self.assertGreater(stats["entropy"], 0.0)
        self.assertGreater(stats["top_percent_activation"], stats["mean"])
        self.assertGreater(stats["hotspot_overlap_ratio"], 0.9)

    def test_report_gap_tracks_internal_external_delta(self) -> None:
        internal = {
            "variant_metrics": {
                "correct_positive": {
                    "Dice": 0.91,
                    "Precision": 0.93,
                    "False Positive Rate": 0.03,
                }
            }
        }
        external = {
            "variant_metrics": {
                "correct_positive": {
                    "Dice": 0.84,
                    "Precision": 0.88,
                    "False Positive Rate": 0.08,
                }
            }
        }

        gap = _report_gap(internal, external)

        self.assertAlmostEqual(gap["correct_positive"]["dice_gap"], 0.07, places=5)
        self.assertAlmostEqual(gap["correct_positive"]["precision_gap"], 0.05, places=5)
        self.assertAlmostEqual(gap["correct_positive"]["fpr_gap"], -0.05, places=5)


if __name__ == "__main__":
    unittest.main()