import unittest
from pathlib import Path

import torch

from MedicalSAM3.retrieval.multi_bank_fusion import fuse_multi_bank_retrieval
from MedicalSAM3.retrieval.site_bank_resolver import SiteBankResolution


class _Entry:
    def __init__(self, prototype_id: str, polarity: str) -> None:
        self.prototype_id = prototype_id
        self.crop_path = None
        self.source_dataset = "PolypGen"
        self.polarity = polarity


def _retrieval_fixture(score: float, prefix: str) -> dict[str, object]:
    positive_features = torch.tensor([[[score, 0.0], [score * 0.9, 0.1]]], dtype=torch.float32)
    negative_features = torch.tensor([[[0.0, score * 0.2]]], dtype=torch.float32)
    positive_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
    negative_weights = torch.tensor([[1.0]], dtype=torch.float32)
    return {
        "projected_query": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "positive_features": positive_features,
        "negative_features": negative_features,
        "positive_weights": positive_weights,
        "negative_weights": negative_weights,
        "positive_score_tensor": torch.tensor([[score, score * 0.95]], dtype=torch.float32),
        "negative_score_tensor": torch.tensor([[score * 0.2]], dtype=torch.float32),
        "positive_prototype": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "negative_prototype": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "similarity_score": {
            "positive_topk_mean": torch.tensor([score], dtype=torch.float32),
            "negative_topk_mean": torch.tensor([score * 0.2], dtype=torch.float32),
            "margin": torch.tensor([score * 0.8], dtype=torch.float32),
        },
        "retrieval_stability": {
            "positive_similarity_mean": torch.tensor([score], dtype=torch.float32),
            "negative_similarity_mean": torch.tensor([score * 0.2], dtype=torch.float32),
            "margin": torch.tensor([score * 0.8], dtype=torch.float32),
            "positive_similarity_std": torch.tensor([0.05], dtype=torch.float32),
            "negative_similarity_std": torch.tensor([0.02], dtype=torch.float32),
            "positive_weight_entropy": torch.tensor([0.4], dtype=torch.float32),
            "negative_weight_entropy": torch.tensor([0.1], dtype=torch.float32),
        },
        "top_k_positive": 2,
        "top_k_negative": 1,
        "positive_entries": [[_Entry(f"{prefix}_p0", "positive"), _Entry(f"{prefix}_p1", "positive")]],
        "negative_entries": [[_Entry(f"{prefix}_n0", "negative")]],
        "positive_scores": [torch.tensor([score, score * 0.95], dtype=torch.float32)],
        "negative_scores": [torch.tensor([score * 0.2], dtype=torch.float32)],
    }


class TestMultiBankFusion(unittest.TestCase):
    def test_fusion_keeps_train_site_separate_diagnostics(self) -> None:
        train_retrieval = _retrieval_fixture(0.55, "train")
        site_retrieval = _retrieval_fixture(0.9, "site")
        resolution = SiteBankResolution(
            mode="train_plus_site",
            site_id="C3",
            train_bank_path=Path("MedicalSAM3/banks/train_bank"),
            continual_bank_root=Path("MedicalSAM3/banks/continual_bank"),
            site_bank_path=Path("MedicalSAM3/banks/continual_bank/C3"),
            expected_site_bank=Path("MedicalSAM3/banks/continual_bank/C3"),
            selected_bank_paths=[Path("MedicalSAM3/banks/train_bank"), Path("MedicalSAM3/banks/continual_bank/C3")],
            fallback_reason=None,
        )
        fused = fuse_multi_bank_retrieval(
            train_retrieval=train_retrieval,
            site_retrieval=site_retrieval,
            resolution=resolution,
            train_bank_path="MedicalSAM3/banks/train_bank",
            site_bank_path="MedicalSAM3/banks/continual_bank/C3",
        )

        self.assertEqual(fused["multi_bank_fusion"]["site_id"], "C3")
        self.assertGreater(float(fused["multi_bank_fusion"]["site_contribution"][0].item()), 0.5)
        self.assertLess(float(fused["multi_bank_fusion"]["train_contribution"][0].item()), 0.5)
        self.assertEqual(fused["positive_features"].shape[1], 4)
        self.assertEqual(len(fused["positive_entries"][0]), 4)
        self.assertAlmostEqual(float(fused["positive_weights"].sum().item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()