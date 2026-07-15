import unittest
import warnings

from MedicalSAM3.utils.polypgen_site import normalize_polypgen_site_id, resolve_polypgen_site


class TestPolypGenSite(unittest.TestCase):
    """测试 PolypGen 站点 ID 标准化和解析工具。"""
    def test_normalize_site_aliases(self) -> None:
        """验证站点别名标准化（如 center_2 → C2）的正确性。"""
        self.assertEqual(normalize_polypgen_site_id("C1"), "C1")
        self.assertEqual(normalize_polypgen_site_id("center_2"), "C2")
        self.assertEqual(normalize_polypgen_site_id("PolypGen-C3"), "C3")

    def test_resolve_common_path_and_metadata_patterns(self) -> None:
        """验证通过常见的路径和元数据模式解析站点 ID。"""
        cases = [
            ({"image_path": "images/C1_100S0001.png"}, "C1"),
            ({"sample_id": "center1_case_05"}, "C1"),
            ({"sample_id": "center_2_case_11"}, "C2"),
            ({"sample_id": "center-3_case_08"}, "C3"),
            ({"dataset_name": "PolypGen-C4"}, "C4"),
            ({"metadata": {"dataset_name": "polypgen_c5"}}, "C5"),
            ({"sample_id": "seq9_C1_frame_0042"}, "C1"),
            ({"image_path": "EndoCV2021_C5_50000208.jpg"}, "C5"),
            ({"image_path": "seq16_seq1_C6_340.jpg"}, "C6"),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(resolve_polypgen_site(warn=False, **kwargs), expected)

    def test_returns_none_and_warns_when_unresolved(self) -> None:
        """验证无法解析时返回 None 并发出警告。"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_polypgen_site(dataset_name="PolypGen", sample_id="unknown_case")
        self.assertIsNone(resolved)
        self.assertTrue(any("Unable to resolve PolypGen site id" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
