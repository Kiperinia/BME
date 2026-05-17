import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from MedicalSAM3.retrieval.mask_prior import attach_retrieved_mask_priors


class _Entry:
    def __init__(self, mask_path: str | None) -> None:
        self.mask_path = mask_path


class TestMaskPrior(unittest.TestCase):
    def test_attach_retrieved_mask_priors_builds_soft_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "mask.png"
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            Image.fromarray(mask).save(mask_path)
            retrieval = {
                "positive_features": torch.randn(1, 1, 4),
                "negative_features": torch.randn(1, 0, 4),
                "positive_weights": torch.tensor([[1.0]], dtype=torch.float32),
                "negative_weights": torch.zeros(1, 0, dtype=torch.float32),
                "positive_entries": [[_Entry(str(mask_path))]],
                "negative_entries": [[]],
            }
            updated = attach_retrieved_mask_priors(retrieval, spatial_size=(8, 8))

            self.assertTrue(updated["mask_prior_available"])
            self.assertEqual(tuple(updated["positive_mask_prior"].shape), (1, 1, 8, 8))
            self.assertGreater(float(updated["positive_mask_prior"][0, 0, 3, 3].item()), 0.5)


if __name__ == "__main__":
    unittest.main()