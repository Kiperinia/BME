import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.evaluation.retrieval_analysis import _prompt_sensitivity
from MedicalSAM3.evaluation.retrieval_influence import _mask_difference_ratio
from MedicalSAM3.exemplar_bank import PrototypeBankEntry, PrototypeExtractor, RSSDABank, masked_average_pool
from MedicalSAM3.models.prompt_adapter import GatedRetrievalFusion
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import DirectoryBankLoader, load_retrieval_bank
from MedicalSAM3.scripts.common import SplitSegmentationDataset, infer_source_domain


class TestRSSDAModules(unittest.TestCase):
    def test_bank_roundtrip_and_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bank = RSSDABank()
            for index, polarity in enumerate(["positive", "positive", "negative"]):
                feature_path = root / f"feature_{index}.pt"
                torch.save({"prototype": torch.randn(8)}, feature_path)
                bank.add_entry(
                    PrototypeBankEntry(
                        prototype_id=f"proto_{index}",
                        feature_path=str(feature_path),
                        polarity=polarity,
                        source_dataset="Kvasir" if index < 2 else "CVC",
                        polyp_type="flat" if polarity == "positive" else "mucosa",
                        boundary_quality=0.8,
                        confidence=0.9,
                    )
                )
            bank.save(root)
            loaded = RSSDABank.load(root)
            retriever = PrototypeRetriever(loaded, feature_dim=8, top_k_positive=2, top_k_negative=1)
            outputs = retriever(torch.randn(1, 8, 8, 8))
            self.assertEqual(outputs["positive_features"].shape[1], 2)
            self.assertEqual(outputs["negative_features"].shape[1], 1)
            self.assertEqual(outputs["projected_query"].shape[-1], 8)

    def test_infer_source_domain_from_merged_kvasircvc_record(self) -> None:
        source = infer_source_domain(
            dataset_name="KvasirCVC",
            image_id="KvasirCVC__cvc_57",
            image_path="MedicalSAM3/data/KvasirCVC-nnunet_raw/Dataset504_KvasirCVC/imagesTr/cvc_57_0000.png",
            mask_path="MedicalSAM3/data/KvasirCVC-nnunet_raw/Dataset504_KvasirCVC/labelsTr/cvc_57.png",
        )
        self.assertEqual(source, "CVC")

    def test_retriever_prefers_cross_domain_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bank = RSSDABank()
            fixtures = [
                ("kvasir_positive", "positive", "Kvasir", torch.tensor([1.0, 0.0, 0.0, 0.0])),
                ("cvc_positive", "positive", "CVC", torch.tensor([0.9, 0.1, 0.0, 0.0])),
                ("negative", "negative", "CVC", torch.tensor([0.0, 1.0, 0.0, 0.0])),
            ]
            for prototype_id, polarity, source_dataset, feature in fixtures:
                feature_path = root / f"{prototype_id}.pt"
                torch.save({"prototype": feature}, feature_path)
                bank.add_entry(
                    PrototypeBankEntry(
                        prototype_id=prototype_id,
                        feature_path=str(feature_path),
                        polarity=polarity,
                        source_dataset=source_dataset,
                        polyp_type="polyp",
                        boundary_quality=0.8,
                        confidence=0.9,
                    )
                )

            retriever = PrototypeRetriever(bank, feature_dim=4, top_k_positive=1, top_k_negative=1)
            query_feature = torch.zeros(1, 4, 2, 2)
            query_feature[:, 0] = 1.0
            outputs = retriever(
                query_feature,
                query_source_datasets=["Kvasir"],
                prefer_cross_domain_positive=True,
            )
            self.assertEqual(outputs["positive_entries"][0][0].source_dataset, "CVC")

    def test_similarity_and_adapter_outputs(self) -> None:
        feature_map = torch.randn(2, 8, 8, 8)
        mask = torch.ones(2, 1, 8, 8)
        pooled = masked_average_pool(feature_map, mask)
        self.assertEqual(tuple(pooled.shape), (2, 8))

        similarity_builder = SimilarityHeatmapBuilder(lambda_negative=0.5)
        positive_prototypes = torch.randn(2, 2, 8)
        negative_prototypes = torch.randn(2, 1, 8)
        weights = torch.softmax(torch.randn(2, 2), dim=-1)
        similarity = similarity_builder(feature_map, positive_prototypes, negative_prototypes, weights, torch.ones(2, 1))
        adapter = RetrievalSpatialSemanticAdapter(dim=8)
        adapted, prior, aux = adapter(
            feature_map=feature_map,
            similarity_map=similarity["fused_similarity"],
            positive_prototype=torch.randn(2, 8),
            negative_prototype=torch.randn(2, 8),
            positive_heatmap=similarity["positive_heatmap"],
            negative_heatmap=similarity["negative_heatmap"],
            mode="joint",
        )
        self.assertEqual(adapted.shape, feature_map.shape)
        self.assertIn("spatial_bias_map", prior)
        self.assertIn("semantic_prototype_map", prior)
        self.assertEqual(prior["semantic_prototype_map"].shape, feature_map.shape)
        self.assertEqual(aux["semantic_prototype"].shape[-1], 8)
        self.assertIn("semantic_prototype", prior)
        self.assertTrue(any(parameter.requires_grad for parameter in similarity_builder.parameters()))

    def test_gated_retrieval_fusion_calibrates_low_confidence_negative_retrieval(self) -> None:
        fusion = GatedRetrievalFusion(
            dim=8,
            positive_weight=1.0,
            negative_weight=0.25,
            similarity_threshold=0.5,
            confidence_scale=10.0,
            similarity_weighting="hard",
            retrieval_policy="uncertainty-aware",
            uncertainty_threshold=0.35,
            uncertainty_scale=12.0,
        )
        feature_map = torch.randn(1, 8, 4, 4)
        positive_tokens = torch.randn(1, 1, 8)
        negative_tokens = torch.randn(1, 1, 8)
        spatial_prior = torch.full((1, 1, 4, 4), 0.5)
        low_segmentation_confidence = fusion(
            feature_map=feature_map,
            positive_tokens=positive_tokens,
            negative_tokens=negative_tokens,
            positive_similarity=torch.ones(1, 1, 4, 4) * 0.8,
            negative_similarity=torch.ones(1, 1, 4, 4) * 0.8,
            positive_weights=torch.ones(1, 1),
            negative_weights=torch.ones(1, 1),
            positive_scores=torch.tensor([[0.8]]),
            negative_scores=torch.tensor([[0.8]]),
            baseline_mask_logits=torch.zeros(1, 1, 4, 4),
            spatial_prior=spatial_prior,
        )[2]
        high_segmentation_confidence = fusion(
            feature_map=feature_map,
            positive_tokens=positive_tokens,
            negative_tokens=negative_tokens,
            positive_similarity=torch.ones(1, 1, 4, 4) * 0.8,
            negative_similarity=torch.ones(1, 1, 4, 4) * 0.8,
            positive_weights=torch.ones(1, 1),
            negative_weights=torch.ones(1, 1),
            positive_scores=torch.tensor([[0.8]]),
            negative_scores=torch.tensor([[0.8]]),
            baseline_mask_logits=torch.full((1, 1, 4, 4), 8.0),
            spatial_prior=spatial_prior,
        )[2]
        self.assertGreater(
            float(low_segmentation_confidence["retrieval_activation_ratio"].mean().item()),
            float(high_segmentation_confidence["retrieval_activation_ratio"].mean().item()),
        )
        self.assertLess(
            float(low_segmentation_confidence["negative_calibrated_weight"].mean().item()),
            float(low_segmentation_confidence["positive_calibrated_weight"].mean().item()),
        )

    def test_gated_retrieval_fusion_similarity_threshold_can_disable_retrieval(self) -> None:
        fusion = GatedRetrievalFusion(
            dim=8,
            positive_weight=1.0,
            negative_weight=0.25,
            similarity_threshold=0.5,
            confidence_scale=10.0,
            similarity_weighting="hard",
            retrieval_policy="similarity-threshold",
        )
        feature_map = torch.randn(1, 8, 4, 4)
        outputs = fusion(
            feature_map=feature_map,
            positive_tokens=torch.randn(1, 1, 8),
            negative_tokens=torch.randn(1, 1, 8),
            positive_similarity=torch.ones(1, 1, 4, 4) * 0.2,
            negative_similarity=torch.ones(1, 1, 4, 4) * 0.2,
            positive_weights=torch.ones(1, 1),
            negative_weights=torch.ones(1, 1),
            positive_scores=torch.tensor([[0.2]]),
            negative_scores=torch.tensor([[0.2]]),
            spatial_prior=torch.ones(1, 1, 4, 4),
        )[2]
        self.assertEqual(float(outputs["retrieval_activation_ratio"].mean().item()), 0.0)
        self.assertEqual(float(outputs["positive_calibrated_weight"].mean().item()), 0.0)
        self.assertEqual(float(outputs["negative_calibrated_weight"].mean().item()), 0.0)

    def test_gated_retrieval_fusion_soft_similarity_weighting_keeps_partial_activation(self) -> None:
        fusion = GatedRetrievalFusion(
            dim=8,
            positive_weight=1.0,
            negative_weight=0.25,
            similarity_threshold=0.5,
            confidence_scale=8.0,
            similarity_weighting="soft",
            similarity_temperature=0.1,
            retrieval_policy="uncertainty-aware",
            uncertainty_threshold=0.35,
            uncertainty_scale=12.0,
        )
        outputs = fusion(
            feature_map=torch.randn(1, 8, 4, 4),
            positive_tokens=torch.randn(1, 1, 8),
            negative_tokens=torch.randn(1, 1, 8),
            positive_similarity=torch.ones(1, 1, 4, 4) * 0.45,
            negative_similarity=torch.ones(1, 1, 4, 4) * 0.4,
            positive_weights=torch.ones(1, 1),
            negative_weights=torch.ones(1, 1),
            positive_scores=torch.tensor([[0.45]]),
            negative_scores=torch.tensor([[0.4]]),
            baseline_mask_logits=torch.zeros(1, 1, 4, 4),
            spatial_prior=torch.ones(1, 1, 4, 4),
        )[2]
        self.assertGreater(float(outputs["retrieval_similarity_weight"].mean().item()), 0.0)
        self.assertLess(float(outputs["retrieval_similarity_weight"].mean().item()), 1.0)
        self.assertGreater(float(outputs["positive_calibrated_weight"].mean().item()), 0.0)
        self.assertGreater(float(outputs["retrieval_activation_ratio"].mean().item()), 0.0)

    def test_prompt_dropout_uses_full_image_box(self) -> None:
        dataset = SplitSegmentationDataset(
            [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": "dropout_case"}],
            image_size=64,
            box_dropout_prob=1.0,
        )
        sample = dataset[0]
        self.assertTrue(sample["prompt_aug"]["dropout"])
        self.assertTrue(torch.equal(sample["box"], torch.tensor([0.0, 0.0, 64.0, 64.0])))

    def test_prompt_removal_uses_nan_sentinel(self) -> None:
        dataset = SplitSegmentationDataset(
            [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": "removed_case"}],
            image_size=64,
            prompt_corruption_prob=1.0,
            prompt_removal_prob=1.0,
        )
        sample = dataset[0]
        self.assertTrue(sample["prompt_aug"]["removed"])
        self.assertTrue(torch.isnan(sample["box"]).all())

    def test_prompt_corruption_gate_can_disable_corruption(self) -> None:
        dataset = SplitSegmentationDataset(
            [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": "clean_case"}],
            image_size=64,
            prompt_corruption_prob=0.0,
            prompt_removal_prob=1.0,
            box_dropout_prob=1.0,
            loose_box_ratio=0.3,
            loose_box_prob=1.0,
            box_jitter_ratio=0.3,
            box_jitter_prob=1.0,
        )
        sample = dataset[0]
        self.assertFalse(sample["prompt_aug"]["corrupted"])
        self.assertFalse(sample["prompt_aug"]["removed"])
        self.assertFalse(sample["prompt_aug"]["dropout"])

    def test_loose_bbox_expands_prompt_box(self) -> None:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": "loose_case"}]
        baseline = SplitSegmentationDataset(records, image_size=64)
        loose = SplitSegmentationDataset(records, image_size=64, loose_box_ratio=0.3, loose_box_prob=1.0)
        baseline_box = baseline[0]["box"]
        loose_box = loose[0]["box"]
        baseline_area = float((baseline_box[2] - baseline_box[0]) * (baseline_box[3] - baseline_box[1]))
        loose_area = float((loose_box[2] - loose_box[0]) * (loose_box[3] - loose_box[1]))
        self.assertGreater(loose_area, baseline_area)
        self.assertTrue(loose[0]["prompt_aug"]["loose"])

    def test_prompt_sensitivity_reports_logit_shift(self) -> None:
        zeros = torch.zeros(1, 1, 4, 4)
        variants = {
            "positive_exemplar": {"mask_logits": zeros + 0.3, "metrics": {"Dice": 0.7, "IoU": 0.55}},
            "negative_exemplar": {"mask_logits": zeros - 0.3, "metrics": {"Dice": 0.5, "IoU": 0.35}},
            "random_exemplar": {"mask_logits": zeros + 0.1, "metrics": {"Dice": 0.62, "IoU": 0.44}},
            "empty_exemplar": {"mask_logits": zeros, "metrics": {"Dice": 0.6, "IoU": 0.4}},
        }
        summary = _prompt_sensitivity(variants)
        self.assertGreater(summary["mean_logit_difference"], 0.0)
        self.assertGreater(summary["prompt_sensitivity_score"], 0.0)

    def test_mask_difference_ratio_uses_xor_over_union(self) -> None:
        mask_a = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
        mask_b = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
        ratio = _mask_difference_ratio(mask_a, mask_b)
        self.assertAlmostEqual(ratio, 2.0 / 3.0, places=5)

    def test_extractor_saves_pt_and_json(self) -> None:
        extractor = PrototypeExtractor()
        feature_map = torch.randn(1, 8, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        prototype = extractor.extract_from_feature_map(feature_map, mask)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = PrototypeBankEntry(
                prototype_id="proto_a",
                feature_path="",
                polarity="positive",
                source_dataset="Kvasir",
                polyp_type="bloody",
                boundary_quality=0.7,
                confidence=0.9,
            )
            stored = extractor.save_prototype(tmpdir, prototype, entry)
            self.assertTrue(Path(stored.feature_path).exists())
            self.assertTrue((Path(tmpdir) / "positive_bank" / "proto_a.json").exists())

    def test_directory_bank_loader_builds_cache_and_supports_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_root = Path(tmpdir) / "banks" / "train_bank"
            for polarity, color in (("positive", (220, 40, 40)), ("negative", (40, 220, 40))):
                target_dir = bank_root / polarity
                target_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), color=color).save(target_dir / f"{polarity}_0.png")

            loader = DirectoryBankLoader(
                bank_root,
                image_size=32,
                precision="fp32",
                allow_dummy_fallback=True,
            )
            context = loader.build_context()
            self.assertEqual(context.stats["positive_count"], 1)
            self.assertEqual(context.stats["negative_count"], 1)
            self.assertEqual(context.stats["cache_misses"], 2)
            self.assertTrue(all(Path(entry.feature_path).exists() for entry in context.bank.entries))

            feature_dim = int(RSSDABank.load_feature(context.bank.entries[0]).shape[-1])
            retrieval = loader.retrieve(torch.randn(1, feature_dim, 4, 4), top_k=1)
            self.assertEqual(tuple(retrieval["positive_prototype_feature"].shape), (1, feature_dim))
            self.assertEqual(tuple(retrieval["negative_prototype_feature"].shape), (1, feature_dim))
            self.assertIn("margin", retrieval["similarity_score"])

            reloaded = load_retrieval_bank(
                bank_root,
                purpose="external-eval",
                image_size=32,
                precision="fp32",
                allow_dummy_fallback=True,
            )
            self.assertEqual(reloaded.stats["cache_misses"], 0)


if __name__ == "__main__":
    unittest.main()