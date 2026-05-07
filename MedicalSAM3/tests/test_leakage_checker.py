import unittest

from MedicalSAM3.agents.leakage_checker import LeakageChecker
from MedicalSAM3.exemplar.memory_bank import ExemplarItem


def _item(item_id: str, image_id: str, dataset: str, fold_id: int) -> ExemplarItem:
    return ExemplarItem(
        item_id=item_id,
        image_id=image_id,
        crop_path="crop.png",
        mask_path=None,
        bbox=[0.0, 0.0, 10.0, 10.0],
        embedding_path=None,
        type="positive",
        source_dataset=dataset,
        fold_id=fold_id,
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


class TestLeakageChecker(unittest.TestCase):
    def test_rejects_polypgen_and_duplicates_and_fold_leakage(self) -> None:
        checker = LeakageChecker(external_test_ids=["ext-1"])
        ok, reason = checker.check_item(_item("i1", "img-1", "Kvasir-SEG", 0))
        self.assertTrue(ok)
        self.assertIsNone(reason)

        ok, reason = checker.check_item(_item("i2", "img-2", "PolypGen", 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "external_dataset_leakage")

        ok, reason = checker.check_item(_item("i1", "img-3", "Kvasir-SEG", 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "duplicate_item")

        ok, reason = checker.check_item(_item("i3", "img-1", "CVC-ClinicDB", 1))
        self.assertFalse(ok)
        self.assertEqual(reason, "fold_leakage")

        ok, reason = checker.check_item(_item("i4", "ext-1", "Kvasir-SEG", 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "external_test_id_leakage")


if __name__ == "__main__":
    unittest.main()