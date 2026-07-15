import tempfile
import unittest
from pathlib import Path

from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank


def _make_item(item_id: str, dataset: str, human_verified: bool) -> ExemplarItem:
    """创建用于测试的 ExemplarItem 辅助函数。

    参数：
        - item_id: 样本 ID
        - dataset: 数据集名称
        - human_verified: 人工验证标志

    返回：
        - 配置好的 ExemplarItem 实例
    """
    return ExemplarItem(
        item_id=item_id,
        image_id=f"img-{item_id}",
        crop_path=f"crops/{item_id}.png",
        mask_path=None,
        bbox=[0.0, 0.0, 10.0, 10.0],
        embedding_path=None,
        type="positive",
        source_dataset=dataset,
        fold_id=0,
        human_verified=human_verified,
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


class TestExemplarMemoryBank(unittest.TestCase):
    """测试示例记忆库的添加、拒绝、保存和加载功能。"""
    def test_add_reject_save_and_load(self) -> None:
        """验证记忆库的添加项、拒绝外部数据集、保存和加载往返功能。"""
        bank = ExemplarMemoryBank()
        bank.add_item(_make_item("p1", "Kvasir-SEG", True))
        self.assertEqual(len(bank.trainable_items), 1)

        with self.assertRaises(ValueError):
            bank.add_item(_make_item("pg", "PolypGen", True))

        bank.add_item(_make_item("c1", "CVC-ClinicDB", False))
        self.assertEqual(len(bank.trainable_items), 1)
        self.assertEqual(len(bank.get_items(human_verified=False)), 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = bank.save(tmpdir)
            self.assertTrue(output.exists())
            loaded = ExemplarMemoryBank.load(tmpdir)
            self.assertEqual(len(loaded.items), len(bank.items))
            self.assertEqual(len(loaded.trainable_items), 1)


if __name__ == "__main__":
    unittest.main()
