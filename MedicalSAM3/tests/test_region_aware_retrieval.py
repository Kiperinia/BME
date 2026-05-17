import unittest

import torch

from MedicalSAM3.models.prompt_adapter import GatedRetrievalFusion
from MedicalSAM3.retrieval.region_gate import build_retrieval_region_mask
from MedicalSAM3.retrieval.region_uncertainty import build_region_uncertainty_maps


class TestRegionAwareRetrieval(unittest.TestCase):
    def test_region_gate_blocks_high_confidence_regions(self) -> None:
        logits = torch.full((1, 1, 8, 8), -8.0)
        logits[:, :, 3:5, 3:5] = 0.0
        logits[:, :, 0:2, 0:2] = 8.0
        maps = build_region_uncertainty_maps(logits)
        gate = build_retrieval_region_mask(
            probability_map=maps["probability_map"],
            confidence_map=maps["confidence_map"],
            entropy_map=maps["entropy_map"],
            boundary_uncertainty_map=maps["boundary_uncertainty_map"],
            low_confidence_lesion_map=maps["low_confidence_lesion_map"],
        )

        self.assertAlmostEqual(float(gate["retrieval_region_mask"][0, 0, 0, 0].item()), 0.0, places=5)
        self.assertGreater(float(gate["retrieval_region_mask"][0, 0, 3, 3].item()), 0.0)
        self.assertGreater(float(gate["high_confidence_preserve_mask"][0, 0, 0, 0].item()), 0.0)

    def test_region_aware_policy_localizes_retrieval_delta(self) -> None:
        fusion = GatedRetrievalFusion(
            dim=8,
            positive_weight=1.0,
            negative_weight=0.25,
            similarity_threshold=0.4,
            confidence_scale=8.0,
            similarity_weighting="soft",
            similarity_temperature=0.1,
            retrieval_policy="region-aware",
            uncertainty_threshold=0.35,
            uncertainty_scale=12.0,
        )
        baseline_mask_logits = torch.full((1, 1, 8, 8), -8.0)
        baseline_mask_logits[:, :, 3:5, 3:5] = 0.0
        outputs = fusion(
            feature_map=torch.randn(1, 8, 8, 8),
            positive_tokens=torch.randn(1, 1, 8),
            negative_tokens=torch.randn(1, 1, 8),
            positive_similarity=torch.ones(1, 1, 8, 8) * 0.8,
            negative_similarity=torch.ones(1, 1, 8, 8) * 0.2,
            positive_weights=torch.ones(1, 1),
            negative_weights=torch.ones(1, 1),
            positive_scores=torch.tensor([[0.8]]),
            negative_scores=torch.tensor([[0.2]]),
            baseline_mask_logits=baseline_mask_logits,
            spatial_prior=torch.ones(1, 1, 8, 8),
        )[2]

        self.assertGreater(float(outputs["retrieval_activation_ratio"].mean().item()), 0.0)
        self.assertLess(float(outputs["retrieval_activation_ratio"].mean().item()), 0.5)
        self.assertLess(float(outputs["high_confidence_region_modification_ratio"].mean().item()), 1e-4)
        self.assertGreater(float(outputs["retrieval_region_mask"][0, 0, 3, 3].item()), 0.0)


if __name__ == "__main__":
    unittest.main()