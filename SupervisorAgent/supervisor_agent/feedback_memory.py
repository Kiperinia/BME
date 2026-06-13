from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from .models import Decision, Feedback, Issue, RiskLevel


@dataclass
class FeedbackRecord:
    report_id: str
    decision_status: str
    risk_level: str
    issues: List[Issue]
    feedback: Optional[Feedback] = None
    created_at: str = ""
    hard_case: bool = False
    routing: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def from_decision(
        decision: Decision,
        *,
        feedback: Optional[Feedback] = None,
        metadata: Optional[dict] = None,
    ) -> "FeedbackRecord":
        created_at = datetime.now(timezone.utc).isoformat()
        return FeedbackRecord(
            report_id=decision.report_id,
            decision_status=decision.status.value,
            risk_level=decision.risk_level.value if isinstance(decision.risk_level, RiskLevel) else str(decision.risk_level),
            issues=decision.issues,
            feedback=feedback,
            created_at=created_at,
            hard_case=decision.hard_case,
            routing=decision.routing,
            metadata=metadata or decision.metadata,
        )


class FeedbackMemory(Protocol):
    def record(self, record: FeedbackRecord) -> None:
        ...

    def list_recent(self, limit: int = 200) -> List[FeedbackRecord]:
        ...


class InMemoryFeedbackMemory:
    def __init__(self, max_records: int = 1000) -> None:
        self._max_records = max_records
        self._records: List[FeedbackRecord] = []

    def record(self, record: FeedbackRecord) -> None:
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    def list_recent(self, limit: int = 200) -> List[FeedbackRecord]:
        return list(self._records[-limit:])


class JsonlFeedbackMemory:
    def __init__(self, path: str | Path, max_records: int = 5000) -> None:
        self._path = Path(path)
        self._max_records = max_records
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: FeedbackRecord) -> None:
        payload = _to_json(record)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        self._truncate()

    def list_recent(self, limit: int = 200) -> List[FeedbackRecord]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        records: List[FeedbackRecord] = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(_from_json(data))
        return records

    def _truncate(self) -> None:
        if not self._path.exists():
            return
        lines = self._path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self._max_records:
            return
        self._path.write_text("\n".join(lines[-self._max_records :]) + "\n", encoding="utf-8")


def _to_json(record: FeedbackRecord) -> str:
    payload = asdict(record)
    payload["issues"] = [asdict(issue) for issue in record.issues]
    if record.feedback is not None:
        payload["feedback"] = asdict(record.feedback)
    return json.dumps(payload, ensure_ascii=True)


def _from_json(payload: dict) -> FeedbackRecord:
    issues = [Issue(**item) for item in payload.get("issues", [])]
    feedback_payload = payload.get("feedback")
    feedback = Feedback(**feedback_payload) if feedback_payload else None
    return FeedbackRecord(
        report_id=payload.get("report_id", ""),
        decision_status=payload.get("decision_status", ""),
        risk_level=payload.get("risk_level", ""),
        issues=issues,
        feedback=feedback,
        created_at=payload.get("created_at", ""),
        hard_case=bool(payload.get("hard_case", False)),
        routing=list(payload.get("routing", []) or []),
        metadata=payload.get("metadata", {}) or {},
    )
