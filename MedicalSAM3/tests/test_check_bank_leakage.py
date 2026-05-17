import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from MedicalSAM3.utils.check_bank_leakage import run_bank_leakage_check


class TestCheckBankLeakage(unittest.TestCase):
    def test_detects_patient_and_perceptual_hash_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_dir = root / "eval"
            bank_dir = root / "bank" / "positive"
            bank_mask_dir = root / "bank" / "positive" / "masks"
            eval_dir.mkdir(parents=True)
            bank_dir.mkdir(parents=True)
            bank_mask_dir.mkdir(parents=True)

            array = np.zeros((16, 16, 3), dtype=np.uint8)
            array[4:12, 4:12] = 255
            eval_image = eval_dir / "C1_123PAT_0001.png"
            eval_mask = eval_dir / "C1_123PAT_0001_mask.png"
            bank_image = bank_dir / "C1_123PAT_9999.png"
            bank_mask = bank_mask_dir / "C1_123PAT_9999.png"
            Image.fromarray(array).save(eval_image)
            Image.fromarray(array).save(bank_image)
            Image.fromarray(array).save(eval_mask)
            Image.fromarray(array).save(bank_mask)

            records_path = root / "eval_records.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "image_id": "PolypGen__C1_123PAT_0001",
                        "image_path": str(eval_image),
                        "mask_path": str(eval_mask),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_bank_leakage_check(
                eval_records_path=records_path,
                bank_root=root / "bank",
                max_hash_distance=0,
            )

            self.assertTrue(summary["leakage_detected"])
            self.assertEqual(summary["total_eval_samples"], 1)
            self.assertEqual(summary["total_bank_samples"], 1)
            self.assertEqual(len(summary["patient_overlap"]), 1)
            self.assertEqual(len(summary["hash_overlap"]), 1)
            self.assertEqual(len(summary["mask_overlap"]), 1)
            self.assertEqual(len(summary["suspicious_pairs"]), 1)
            self.assertIn("patient_overlap", summary["suspicious_pairs"][0]["reasons"])
            self.assertIn("hash_overlap", summary["suspicious_pairs"][0]["reasons"])
            self.assertIn("mask_overlap", summary["suspicious_pairs"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()