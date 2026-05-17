import tempfile
import unittest
import warnings
from pathlib import Path

from MedicalSAM3.retrieval.site_bank_resolver import resolve_site_bank_paths
from MedicalSAM3.utils.polypgen_site import resolve_polypgen_site


class TestPolypGenSiteRouting(unittest.TestCase):
    def test_resolve_polypgen_site_from_multiple_aliases(self) -> None:
        self.assertEqual(
            resolve_polypgen_site(image_path="MedicalSAM3/data/PolypGen_external_test/Dataset502_PolypGen/imagesTs/C1_100S0001_0000.png"),
            "C1",
        )
        self.assertEqual(resolve_polypgen_site(sample_id="center_2_case_11"), "C2")
        self.assertEqual(resolve_polypgen_site(dataset_name="PolypGen-C3"), "C3")
        self.assertEqual(resolve_polypgen_site(metadata={"site_id": "polypgen_c4"}), "C4")
        self.assertEqual(resolve_polypgen_site(metadata={"center": "center5"}), "C5")
        self.assertEqual(resolve_polypgen_site(sample_id="case_for_c6_patient"), "C6")

    def test_resolve_polypgen_site_warns_on_failure(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_polypgen_site(metadata={"dataset_name": "PolypGen"})
        self.assertIsNone(resolved)
        self.assertTrue(any("Unable to resolve PolypGen site id" in str(item.message) for item in caught))

    def test_train_plus_site_resolution_prefers_matching_site_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_bank = root / "train_bank"
            continual_root = root / "continual_bank"
            (train_bank / "positive").mkdir(parents=True)
            (train_bank / "negative").mkdir(parents=True)
            (continual_root / "C1" / "positive").mkdir(parents=True)
            (continual_root / "C1" / "negative").mkdir(parents=True)
            (train_bank / "positive" / "train_case.png").write_bytes(b"x")
            (continual_root / "C1" / "positive" / "site_case.png").write_bytes(b"x")

            resolution = resolve_site_bank_paths(
                sample_metadata={
                    "dataset_name": "PolypGen",
                    "image_path": "imagesTs/C1_100H0050_0000.png",
                    "image_id": "C1_100H0050",
                },
                train_bank=train_bank,
                continual_bank_root=continual_root,
                mode="train_plus_site",
            )

            self.assertEqual(resolution.site_id, "C1")
            self.assertEqual(resolution.expected_site_bank, continual_root / "C1")
            self.assertEqual(resolution.selected_bank_paths, [train_bank, continual_root / "C1"])
            self.assertFalse(resolution.fallback_to_train_bank)
            self.assertIsNone(resolution.fallback_reason)

    def test_empty_site_bank_falls_back_to_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_bank = root / "train_bank"
            continual_root = root / "continual_bank"
            (train_bank / "positive").mkdir(parents=True)
            (train_bank / "negative").mkdir(parents=True)
            (continual_root / "C1" / "positive").mkdir(parents=True)
            (continual_root / "C1" / "negative").mkdir(parents=True)
            (train_bank / "positive" / "train_case.png").write_bytes(b"x")

            resolution = resolve_site_bank_paths(
                sample_metadata={
                    "dataset_name": "PolypGen",
                    "image_path": "imagesTs/C1_100H0050_0000.png",
                    "image_id": "C1_100H0050",
                },
                train_bank=train_bank,
                continual_bank_root=continual_root,
                mode="train_plus_site",
            )

            self.assertEqual(resolution.selected_bank_paths, [train_bank])
            self.assertTrue(resolution.fallback_to_train_bank)
            self.assertEqual(resolution.fallback_reason, "site_bank_empty")

    def test_missing_site_bank_falls_back_to_train_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_bank = root / "train_bank"
            continual_root = root / "continual_bank"
            (train_bank / "positive").mkdir(parents=True)
            (train_bank / "negative").mkdir(parents=True)
            (train_bank / "positive" / "train_case.png").write_bytes(b"x")

            resolution = resolve_site_bank_paths(
                sample_metadata={
                    "dataset_name": "PolypGen",
                    "image_id": "C4_17",
                },
                train_bank=train_bank,
                continual_bank_root=continual_root,
                mode="site_only",
            )

            self.assertEqual(resolution.site_id, "C4")
            self.assertEqual(resolution.selected_bank_paths, [train_bank])
            self.assertTrue(resolution.fallback_to_train_bank)
            self.assertEqual(resolution.fallback_reason, "site_bank_missing")
            self.assertTrue(any("falling back to train_bank" in warning for warning in resolution.warnings))


if __name__ == "__main__":
    unittest.main()