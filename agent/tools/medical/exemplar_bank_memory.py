from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .exemplar_bank_schemas import (
    ExemplarEmbeddingRecord,
    ExemplarLifecycleState,
    FeatureCentroid,
    ExemplarPolarity,
    MemoryBankSnapshot,
    MedicalExemplarRecord,
    PrototypeClusterRecord,
    RetrievalStatistics,
    compute_dedup_signature,
)

logger = logging.getLogger(__name__)


class ExemplarMemoryManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.snapshot_path = self.root / "snapshot.json"
        self.audit_path = self.root / "audit_log.json"

    def load(self, bank_id: str = "default-bank") -> MemoryBankSnapshot:
        if not self.snapshot_path.exists():
            return MemoryBankSnapshot(bank_id=bank_id, version="v0")

        payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        return MemoryBankSnapshot(
            bank_id=payload.get("bank_id", bank_id),
            version=payload.get("version", "v0"),
            generated_at=payload.get("generated_at", ""),
            positive_items=[self._deserialize_record(item) for item in payload.get("positive_items", [])],
            negative_items=[self._deserialize_record(item) for item in payload.get("negative_items", [])],
            boundary_items=[self._deserialize_record(item) for item in payload.get("boundary_items", [])],
            clusters=[PrototypeClusterRecord(**item) for item in payload.get("clusters", [])],
            stats=payload.get("stats", {}),
        )

    def save(self, snapshot: MemoryBankSnapshot) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = asdict(snapshot)
        payload["positive_items"] = [item.to_json_dict() for item in snapshot.positive_items]
        payload["negative_items"] = [item.to_json_dict() for item in snapshot.negative_items]
        payload["boundary_items"] = [item.to_json_dict() for item in snapshot.boundary_items]
        self.snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.snapshot_path

    def append_audit_log(self, event: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        logs = []
        if self.audit_path.exists():
            logs = json.loads(self.audit_path.read_text(encoding="utf-8"))
        logs.append(event)
        self.audit_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def deduplicate(records: Iterable[MedicalExemplarRecord]) -> list[MedicalExemplarRecord]:
        seen: set[str] = set()
        unique: list[MedicalExemplarRecord] = []
        for record in records:
            signature = record.dedup_hash or compute_dedup_signature(record)
            if signature in seen:
                continue
            seen.add(signature)
            record.dedup_hash = signature
            unique.append(record)
        return unique

    @staticmethod
    def partition(records: Iterable[MedicalExemplarRecord]) -> tuple[list[MedicalExemplarRecord], list[MedicalExemplarRecord], list[MedicalExemplarRecord]]:
        positive: list[MedicalExemplarRecord] = []
        negative: list[MedicalExemplarRecord] = []
        boundary: list[MedicalExemplarRecord] = []
        for record in records:
            if record.state == ExemplarLifecycleState.REJECTED:
                continue
            if record.polarity == ExemplarPolarity.POSITIVE:
                positive.append(record)
            elif record.polarity == ExemplarPolarity.NEGATIVE:
                negative.append(record)
            else:
                boundary.append(record)
        return positive, negative, boundary

    @staticmethod
    def _deserialize_record(payload: dict[str, object]) -> MedicalExemplarRecord:
        normalized = dict(payload)
        normalized["polarity"] = ExemplarPolarity(normalized.get("polarity", ExemplarPolarity.POSITIVE.value))
        normalized["state"] = ExemplarLifecycleState(normalized.get("state", ExemplarLifecycleState.CANDIDATE.value))
        normalized["retrieval_statistics"] = RetrievalStatistics(**dict(normalized.get("retrieval_statistics", {})))
        normalized["centroid"] = FeatureCentroid(**dict(normalized.get("centroid", {})))
        normalized["embeddings"] = ExemplarEmbeddingRecord(**dict(normalized.get("embeddings", {})))
        return MedicalExemplarRecord(**normalized)
