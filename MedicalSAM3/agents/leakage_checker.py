"""Leakage checks for exemplar and split management."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def _get_field(item: Any, key: str, default: Any = None) -> Any:
    """安全地从数据类、字典或普通对象中读取字段值。

    参数：
        - item: 数据源对象
        - key: 字段名称
        - default: 字段不存在时的默认值

    返回：
        - 字段值或默认值
    """
    if is_dataclass(item):
        return getattr(item, key, default)
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class LeakageChecker:
    """数据泄漏检测器，用于检查范例在数据集拆分中的泄漏风险。

    支持检测外部数据集泄漏、重复条目、图像折叠冲突等。
    """
    def __init__(
        self,
        external_dataset_names: Optional[list[str]] = None,
        external_test_ids: Optional[list[str]] = None,
    ) -> None:
        """初始化泄漏检测器。

        参数：
            - external_dataset_names: 外部数据集名称列表（默认 ["PolypGen"]）
            - external_test_ids: 外部测试图像 ID 集合

        返回：
            - None
        """
        self.external_dataset_names = [name.lower() for name in (external_dataset_names or ["PolypGen"])]
        self.external_test_ids = set(external_test_ids or [])
        self.item_ids: set[str] = set()
        self.image_fold_map: dict[str, Optional[int]] = {}

    def _reason_for_item(self, item: Any) -> Optional[str]:
        """返回条目被判定为泄漏的原因，无泄漏则返回 None。

        参数：
            - item: 待检查的条目对象

        返回：
            - 泄漏原因字符串，或 None
        """
        source_dataset = str(_get_field(item, "source_dataset", "")).lower()
        item_id = str(_get_field(item, "item_id", ""))
        image_id = str(_get_field(item, "image_id", ""))
        fold_id = _get_field(item, "fold_id", None)

        if any(dataset in source_dataset for dataset in self.external_dataset_names):
            return "external_dataset_leakage"
        if image_id in self.external_test_ids:
            return "external_test_id_leakage"
        if item_id in self.item_ids:
            return "duplicate_item"
        if image_id in self.image_fold_map and self.image_fold_map[image_id] != fold_id:
            return "fold_leakage"
        return None

    def check_item(self, item: Any) -> tuple[bool, Optional[str]]:
        """检查单个条目是否通过泄漏检测。

        通过则记录其 item_id 和 image_id，失败则返回原因。

        参数：
            - item: 待检查的条目对象

        返回：
            - (是否通过, 泄漏原因或 None) 元组
        """
        reason = self._reason_for_item(item)
        if reason is not None:
            return False, reason
        item_id = str(_get_field(item, "item_id", ""))
        image_id = str(_get_field(item, "image_id", ""))
        fold_id = _get_field(item, "fold_id", None)
        self.item_ids.add(item_id)
        self.image_fold_map[image_id] = fold_id
        return True, None

    def reject_or_raise(self, item: Any) -> None:
        """检查条目并通过抛出异常来拒绝违规条目。

        参数：
            - item: 待检查的条目对象

        返回：
            - None；若检测到泄漏则抛出 ValueError
        """
        ok, reason = self.check_item(item)
        if not ok:
            raise ValueError(reason)
