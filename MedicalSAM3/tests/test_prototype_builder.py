import tempfile
import unittest
from pathlib import Path

import torch

from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank
from MedicalSAM3.exemplar.prototype_builder import PrototypeBuilder


class TestPrototypeBuilder(unittest.TestCase):
    def test_prototype_modes_and_topk(self) -> None:
        builder = PrototypeBuilder(variance_threshold=0.01)
        embeddings = torch.randn(5, 8)
        mean_proto = builder.build_mean_prototype(embeddings)
        self.assertEqual(tuple(mean_proto.shape), (8,))

        weighted_proto, weights = builder.build_weighted_prototype(embeddings, torch.randn(5))
        self.assertEqual(tuple(weighted_proto.shape), (8,))
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=5)

        query = torch.randn(8)
        _, attn_weights = builder.build_attention_fused_prototype(query, embeddings)
        self.assertAlmostEqual(float(attn_weights.sum().item()), 1.0, places=5)

        clustered = builder.build_clustered_subprototypes(embeddings, n_clusters=2)
        self.assertEqual(clustered.shape[1], 8)
        variance = builder.compute_prototype_variance(embeddings, mean_proto)
        self.assertTrue(builder.reject_if_high_variance(variance, threshold=0.0))

    def test_build_positive_negative_boundary_prototypes(self) -> None:
        builder = PrototypeBuilder()
        bank = ExemplarMemoryBank()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index, exemplar_type in enumerate(["positive", "negative", "boundary", "positive", "positive"]):
                emb_path = root / f"emb_{index}.pt"
                torch.save(torch.randn(8), emb_path)
                bank.add_item(
                    ExemplarItem(
                        item_id=f"item-{index}",
                        image_id=f"img-{index}",
                        crop_path="crop.png",
                        mask_path=None,
                        bbox=[0.0, 0.0, 10.0, 10.0],
                        embedding_path=str(emb_path),
                        type=exemplar_type,
                        source_dataset="Kvasir-SEG",
                        fold_id=0,
                        human_verified=True,
                        quality_score=0.9,
                        boundary_score=0.8,
                        diversity_score=0.7,
                        difficulty_score=0.6,
                        uncertainty_score=0.1,
                        false_positive_risk=0.1,
                        created_at="2026-05-07T00:00:00Z",
                        version="v0",
                        notes="",
                    )
                )
            for top_k in [1, 3, 5, 10]:
                result = builder.build_positive_negative_boundary_prototypes(torch.randn(8), bank, top_k=top_k)
                self.assertIn("positive", result)
                self.assertIn("selected_item_ids", result["positive"])


if __name__ == "__main__":
    unittest.main()
