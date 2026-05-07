"""Versioned human-verified exemplar memory bank."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ExemplarItem:
    item_id: str
    image_id: str
    crop_path: str
    mask_path: Optional[str]
    bbox: list[float]
    embedding_path: Optional[str]
    type: str
    source_dataset: str
    fold_id: Optional[int]
    human_verified: bool
    quality_score: float
    boundary_score: float
    diversity_score: float
    difficulty_score: float
    uncertainty_score: float
    false_positive_risk: float
    created_at: str
    version: str
    notes: str


class ExemplarMemoryBank:
    def __init__(self, items: Optional[list[ExemplarItem]] = None) -> None:
        self.items: list[ExemplarItem] = items or []
        self.rejected_items: list[dict[str, object]] = []
        self.changelog: list[dict[str, object]] = []
        self.version = "v0"

    @property
    def trainable_items(self) -> list[ExemplarItem]:
        return [item for item in self.items if item.human_verified]

    @classmethod
    def load(cls, path: str | Path) -> "ExemplarMemoryBank":
        target = Path(path)
        if target.is_dir():
            candidates = sorted(target.glob("memory_v*.json"))
            if not candidates:
                return cls()
            target = candidates[-1]
        if not target.exists():
            return cls()

        payload = json.loads(target.read_text(encoding="utf-8"))
        items = [ExemplarItem(**item) for item in payload.get("items", [])]
        bank = cls(items=items)
        bank.version = payload.get("version", target.stem.replace("memory_", ""))

        rejected_path = target.parent / "rejected_items.json"
        changelog_path = target.parent / "changelog.json"
        if rejected_path.exists():
            bank.rejected_items = json.loads(rejected_path.read_text(encoding="utf-8"))
        if changelog_path.exists():
            bank.changelog = json.loads(changelog_path.read_text(encoding="utf-8"))
        return bank

    def _record_change(self, action: str, item_id: str, details: Optional[dict[str, object]] = None) -> None:
        self.changelog.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.version,
                "action": action,
                "item_id": item_id,
                "details": details or {},
            }
        )

    def _validate_item(self, item: ExemplarItem, external_dataset_names: Optional[list[str]] = None) -> None:
        external_dataset_names = external_dataset_names or ["PolypGen"]
        if any(name.lower() in item.source_dataset.lower() for name in external_dataset_names):
            raise ValueError(f"External dataset leakage detected for item {item.item_id}: {item.source_dataset}")

    def add_item(self, item: ExemplarItem) -> None:
        self._validate_item(item)
        self.items = [existing for existing in self.items if existing.item_id != item.item_id]
        self.items.append(item)
        self._record_change("add", item.item_id, {"human_verified": item.human_verified, "type": item.type})

    def remove_item(self, item_id: str) -> None:
        self.items = [item for item in self.items if item.item_id != item_id]
        self._record_change("remove", item_id)

    def get_items(
        self,
        type: Optional[str] = None,
        source_dataset: Optional[str] = None,
        human_verified: Optional[bool] = None,
    ) -> list[ExemplarItem]:
        items = self.items
        if type is not None:
            items = [item for item in items if item.type == type]
        if source_dataset is not None:
            items = [item for item in items if item.source_dataset == source_dataset]
        if human_verified is not None:
            items = [item for item in items if item.human_verified == human_verified]
        return items

    def reject_item(self, item_id: str, reason: str) -> None:
        matched = None
        remaining = []
        for item in self.items:
            if item.item_id == item_id:
                matched = item
            else:
                remaining.append(item)
        self.items = remaining
        record = {
            "item_id": item_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "item": asdict(matched) if matched is not None else None,
        }
        self.rejected_items.append(record)
        self._record_change("reject", item_id, {"reason": reason})

    def _next_version_path(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        version_index = 0
        while (directory / f"memory_v{version_index}.json").exists():
            version_index += 1
        self.version = f"v{version_index}"
        for item in self.items:
            item.version = self.version
        return directory / f"memory_{self.version}.json"

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.stem.startswith("memory_v"):
                self.version = destination.stem.replace("memory_", "")
        else:
            destination = self._next_version_path(destination)
        payload = {
            "version": self.version,
            "items": [asdict(item) for item in self.items],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        (destination.parent / "rejected_items.json").write_text(
            json.dumps(self.rejected_items, indent=2), encoding="utf-8"
        )
        (destination.parent / "changelog.json").write_text(
            json.dumps(self.changelog, indent=2), encoding="utf-8"
        )
        return destination

    def export_changelog(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.changelog, indent=2), encoding="utf-8")
        return destination

    def check_no_external_leakage(self, external_dataset_names: list[str] = ["PolypGen"]) -> bool:
        return not any(
            any(dataset.lower() in item.source_dataset.lower() for dataset in external_dataset_names)
            for item in self.items
        )
