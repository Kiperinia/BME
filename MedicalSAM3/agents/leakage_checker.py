"""Leakage checks for exemplar and split management."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def _get_field(item: Any, key: str, default: Any = None) -> Any:
    if is_dataclass(item):
        return getattr(item, key, default)
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class LeakageChecker:
    def __init__(
        self,
        external_dataset_names: Optional[list[str]] = None,
        external_test_ids: Optional[list[str]] = None,
    ) -> None:
        self.external_dataset_names = [name.lower() for name in (external_dataset_names or ["PolypGen"])]
        self.external_test_ids = set(external_test_ids or [])
        self.item_ids: set[str] = set()
        self.image_fold_map: dict[str, Optional[int]] = {}

    def _reason_for_item(self, item: Any) -> Optional[str]:
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
        ok, reason = self.check_item(item)
        if not ok:
            raise ValueError(reason)
