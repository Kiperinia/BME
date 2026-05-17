"""Prototype bank storage for retrieval-conditioned spatial-semantic adaptation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class PrototypeBankEntry:
    prototype_id: str
    feature_path: str
    polarity: str
    source_dataset: str
    polyp_type: str
    boundary_quality: float
    confidence: float
    image_id: str = ""
    crop_path: Optional[str] = None
    mask_path: Optional[str] = None
    device_metadata: dict[str, Any] = field(default_factory=dict)
    human_verified: bool = True
    notes: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class RSSDABank:
    def __init__(self, entries: Optional[list[PrototypeBankEntry]] = None, version: str = "rssda_v0") -> None:
        self.entries = entries or []
        self.version = version

    @classmethod
    def load(cls, path: str | Path) -> "RSSDABank":
        target = Path(path)
        if not target.exists():
            return cls()
        metadata_path = target if target.is_file() else target / "metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return cls(
                entries=[PrototypeBankEntry(**item) for item in payload.get("entries", [])],
                version=str(payload.get("version", "rssda_v0")),
            )

        entries: list[PrototypeBankEntry] = []
        for pattern in ["positive_bank/*.json", "negative_bank/*.json"]:
            for item_path in sorted(target.glob(pattern)):
                entries.append(PrototypeBankEntry(**json.loads(item_path.read_text(encoding="utf-8"))))
        return cls(entries=entries)

    def save(self, root: str | Path) -> Path:
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "positive_bank").mkdir(parents=True, exist_ok=True)
        (destination / "negative_bank").mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "entries": [asdict(entry) for entry in self.entries],
        }
        metadata_path = destination / "metadata.json"
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for entry in self.entries:
            bank_dir = destination / ("positive_bank" if entry.polarity == "positive" else "negative_bank")
            (bank_dir / f"{entry.prototype_id}.json").write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")
        return metadata_path

    def add_entry(self, entry: PrototypeBankEntry) -> None:
        if entry.polarity not in {"positive", "negative"}:
            raise ValueError(f"Unsupported polarity: {entry.polarity}")
        self.entries = [item for item in self.entries if item.prototype_id != entry.prototype_id]
        self.entries.append(entry)

    def get_entries(
        self,
        polarity: Optional[str] = None,
        source_dataset: Optional[str] = None,
        human_verified: Optional[bool] = None,
    ) -> list[PrototypeBankEntry]:
        entries = self.entries
        if polarity is not None:
            entries = [entry for entry in entries if entry.polarity == polarity]
        if source_dataset is not None:
            entries = [entry for entry in entries if entry.source_dataset == source_dataset]
        if human_verified is not None:
            entries = [entry for entry in entries if entry.human_verified == human_verified]
        return entries

    @staticmethod
    def load_feature(entry: PrototypeBankEntry, device: str | torch.device = "cpu") -> torch.Tensor:
        payload = torch.load(Path(entry.feature_path), map_location=device, weights_only=False)
        if isinstance(payload, dict):
            for key in ["prototype", "feature", "embedding"]:
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    return value.float()
        if isinstance(payload, torch.Tensor):
            return payload.float()
        raise TypeError(f"Unsupported feature payload for {entry.prototype_id}")

    def stack_features(
        self,
        entries: list[PrototypeBankEntry],
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if not entries:
            return torch.empty(0, 0, device=device)
        features = [self.load_feature(entry, device=device) for entry in entries]
        target_dim = max((int(feature.shape[-1]) for feature in features), default=0)
        aligned = []
        for feature in features:
            if feature.shape[-1] == target_dim:
                aligned.append(feature)
                continue
            if feature.shape[-1] > target_dim:
                aligned.append(feature[..., :target_dim])
                continue
            aligned.append(F.pad(feature, (0, target_dim - feature.shape[-1])))
        return torch.stack(aligned, dim=0)