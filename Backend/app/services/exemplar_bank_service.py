from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import WORKSPACE_DIR
from app.core.exceptions import AppException
from app.schemas.workspace import ExemplarBankDecisionSchema, ExemplarBankRequestSchema


@dataclass(slots=True)
class BankRecord:
    sample_id: str
    created_at: str
    image_hash: str
    filename: str
    patient_id: str
    paris_classification: str
    lesion_type: str
    pathology_classification: str
    mask_area_ratio: float
    score: float
    image_path: str
    metadata_path: str


class ExemplarBankService:
    threshold = 0.6

    def __init__(self) -> None:
        self.bank_root = (WORKSPACE_DIR / "agent" / "memory" / "exemplar_bank").resolve()
        self.assets_root = self.bank_root / "assets"
        self.metadata_root = self.bank_root / "metadata"
        self.index_path = self.bank_root / "index.json"

    def evaluate_and_store(self, payload: ExemplarBankRequestSchema) -> ExemplarBankDecisionSchema:
        image_bytes = self._decode_image_source(payload.image.dataUrl)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        records = self._load_records()
        duplicate = next((record for record in records if record.image_hash == image_hash), None)
        score, reasons = self._score_candidate(payload=payload, records=records, duplicate=duplicate is not None)

        if duplicate is not None:
            return ExemplarBankDecisionSchema(
                sampleId=duplicate.sample_id,
                accepted=False,
                score=round(score, 4),
                threshold=self.threshold,
                reasons=[*reasons, "Rejected because the same image content already exists in the exemplar bank."],
                duplicateOf=duplicate.sample_id,
                bankSize=len(records),
                storedAt=datetime.fromisoformat(duplicate.created_at),
            )

        if score < self.threshold:
            return ExemplarBankDecisionSchema(
                accepted=False,
                score=round(score, 4),
                threshold=self.threshold,
                reasons=[*reasons, "Rejected because the current value score did not cross the exemplar-bank threshold."],
                bankSize=len(records),
            )

        sample_id = f"sample-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc)
        suffix = Path(payload.image.filename).suffix.lower() or ".png"
        image_path = self.assets_root / f"{sample_id}{suffix}"
        metadata_path = self.metadata_root / f"{sample_id}.json"

        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        metadata_path.write_text(
            json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        record = BankRecord(
            sample_id=sample_id,
            created_at=created_at.isoformat(),
            image_hash=image_hash,
            filename=payload.image.filename,
            patient_id=payload.patient.patientId,
            paris_classification=payload.expertConfig.parisClassification,
            lesion_type=payload.expertConfig.lesionType,
            pathology_classification=payload.expertConfig.pathologyClassification,
            mask_area_ratio=payload.segmentation.maskAreaRatio,
            score=round(score, 4),
            image_path=str(image_path),
            metadata_path=str(metadata_path),
        )
        records.append(record)
        self._save_records(records)

        return ExemplarBankDecisionSchema(
            sampleId=sample_id,
            accepted=True,
            score=round(score, 4),
            threshold=self.threshold,
            reasons=[*reasons, "Accepted and stored as a high-value exemplar candidate for future curation."],
            bankSize=len(records),
            storedAt=created_at,
        )

    def _score_candidate(
        self,
        *,
        payload: ExemplarBankRequestSchema,
        records: list[BankRecord],
        duplicate: bool,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        label_coverage = 0.0
        if payload.expertConfig.parisClassification.strip():
            label_coverage += 0.12
        if payload.expertConfig.lesionType.strip():
            label_coverage += 0.12
        if payload.expertConfig.pathologyClassification.strip():
            label_coverage += 0.16
        if payload.expertConfig.expertNotes.strip():
            label_coverage += 0.08
        score += label_coverage
        reasons.append(f"Label completeness contributed {label_coverage:.2f}.")

        segmentation_score = 0.0
        if payload.segmentation.pointCount >= 6:
            segmentation_score += 0.08
        if 0.01 <= payload.segmentation.maskAreaRatio <= 0.45:
            segmentation_score += 0.12
        if payload.segmentation.boundingBox[2] > payload.segmentation.boundingBox[0]:
            segmentation_score += 0.05
        score += segmentation_score
        reasons.append(f"Segmentation quality contributed {segmentation_score:.2f}.")

        report_score = 0.0
        if payload.findings.strip():
            report_score += 0.08
        if payload.conclusion.strip():
            report_score += 0.08
        if payload.reportMarkdown.strip():
            report_score += 0.04
        score += report_score
        reasons.append(f"Report completeness contributed {report_score:.2f}.")

        novelty_score = 0.0
        signature = self._classification_signature(payload)
        existing_signatures = {
            f"{record.paris_classification}|{record.lesion_type}|{record.pathology_classification}"
            for record in records
        }
        if signature and signature not in existing_signatures:
            novelty_score += 0.18
        elif signature:
            novelty_score += 0.06
        score += novelty_score
        reasons.append(f"Novelty scoring contributed {novelty_score:.2f}.")

        note_bonus = min(len(payload.expertConfig.expertNotes.strip()) / 400.0, 0.08)
        score += note_bonus
        reasons.append(f"Expert note richness contributed {note_bonus:.2f}.")

        if duplicate:
            score = min(score, 0.2)

        return min(score, 1.0), reasons

    @staticmethod
    def _classification_signature(payload: ExemplarBankRequestSchema) -> str:
        parts = [
            payload.expertConfig.parisClassification.strip(),
            payload.expertConfig.lesionType.strip(),
            payload.expertConfig.pathologyClassification.strip(),
        ]
        return "|".join(parts)

    def _load_records(self) -> list[BankRecord]:
        if not self.index_path.exists():
            return []

        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        samples = raw.get("samples", []) if isinstance(raw, dict) else []
        records: list[BankRecord] = []
        for sample in samples:
            try:
                records.append(BankRecord(**sample))
            except TypeError as exc:
                raise AppException(500, 50061, f"invalid exemplar bank record format: {exc}") from exc
        return records

    def _save_records(self, records: list[BankRecord]) -> None:
        self.bank_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "samples": [asdict(record) for record in records],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _decode_image_source(image_source: str) -> bytes:
        if not image_source.startswith("data:"):
            raise AppException(400, 40062, "workspace image must be sent as a data URL")

        try:
            _, encoded = image_source.split(",", 1)
        except ValueError as exc:
            raise AppException(400, 40063, "invalid image data URL payload") from exc

        try:
            return base64.b64decode(encoded)
        except Exception as exc:
            raise AppException(400, 40064, "failed to decode uploaded image data") from exc
