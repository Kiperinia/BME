"""Version snapshots and rollback for exemplar memory banks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank


class MemoryVersionManager:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def list_versions(self) -> list[str]:
        return sorted(path.stem.replace("memory_", "") for path in self.root_dir.glob("memory_v*.json"))

    def save_new_version(self, memory_bank: ExemplarMemoryBank) -> Path:
        return memory_bank.save(self.root_dir)

    def rollback(self, version: str) -> ExemplarMemoryBank:
        target = self.root_dir / f"memory_{version}.json"
        if not target.exists():
            raise FileNotFoundError(f"Memory version not found: {version}")
        bank = ExemplarMemoryBank.load(target)
        bank.changelog.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "rollback",
                "from_version": bank.version,
                "requested_version": version,
            }
        )
        bank.save(self.root_dir)
        return bank
