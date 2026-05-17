from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.nn.functional as F

from app.core.config import WORKSPACE_DIR
from app.core.exceptions import AppException
from app.schemas.workspace import (
    ExemplarBankDecisionSchema,
    ExemplarBankRequestSchema,
    ExemplarFeedbackRequestSchema,
    ExemplarFeedbackResponseSchema,
    ExemplarRetrievalCandidateSchema,
    ExemplarRetrievalRequestSchema,
    ExemplarRetrievalResponseSchema,
)


class ExemplarBankService:
    threshold = 0.6

    def __init__(self, *, hidden_dim: int = 256) -> None:
        self.hidden_dim = hidden_dim
        self.agent_root = (WORKSPACE_DIR / "agent").resolve()
        self.bank_root = (self.agent_root / "memory" / "exemplar_bank").resolve()
        self._ensure_agent_path()

        from core.agent import build_exemplar_bank_agent

        self.agent = build_exemplar_bank_agent(
            memory_root=str(self.bank_root),
            hidden_dim=hidden_dim,
        )

        self.assets_root = self.bank_root / "assets"
        self.metadata_root = self.bank_root / "metadata"

    def evaluate_and_store(self, payload: ExemplarBankRequestSchema) -> ExemplarBankDecisionSchema:
        from tools.medical.exemplar_bank_schemas import ExemplarPolarity

        image_bytes = self._decode_image_source(payload.image.dataUrl)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        snapshot = self.agent.memory.load(bank_id="default-bank")
        duplicate = self._find_duplicate(snapshot, image_hash=image_hash)
        score, reasons = self._score_candidate(payload=payload, duplicate=duplicate is not None)

        if duplicate is not None:
            return ExemplarBankDecisionSchema(
                sampleId=duplicate.exemplar_id,
                accepted=False,
                score=round(score, 4),
                threshold=self.threshold,
                reasons=[*reasons, "Rejected because the same image content already exists in the exemplar bank."],
                duplicateOf=duplicate.exemplar_id,
                bankSize=self._snapshot_size(snapshot),
                storedAt=datetime.fromisoformat(duplicate.created_at),
                bankId=snapshot.bank_id,
                memoryState=duplicate.state.value,
            )

        if score < self.threshold:
            return ExemplarBankDecisionSchema(
                accepted=False,
                score=round(score, 4),
                threshold=self.threshold,
                reasons=[*reasons, "Rejected because the current value score did not cross the exemplar-bank threshold."],
                bankSize=self._snapshot_size(snapshot),
                bankId=snapshot.bank_id,
            )

        sample_id = f"sample-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc)
        suffix = Path(payload.image.filename).suffix.lower() or ".png"
        image_path = self.assets_root / f"{sample_id}{suffix}"
        metadata_path = self.metadata_root / f"{sample_id}.json"
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

        record = self._build_record(
            payload=payload,
            sample_id=sample_id,
            created_at=created_at,
            image_hash=image_hash,
            image_path=image_path,
            metadata_path=metadata_path,
            polarity=ExemplarPolarity(payload.polarityHint),
            score=score,
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "workspacePayload": payload.model_dump(mode="json"),
                    "record": record.to_json_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result = self.agent.ingest(record, bank_id=snapshot.bank_id)
        quality = result.evolution.quality if result.evolution is not None else None
        updated_snapshot = self.agent.memory.load(bank_id=snapshot.bank_id)
        quality_breakdown = self._quality_to_dict(quality)
        reasons_out = list(reasons)
        if result.evolution is not None:
            reasons_out.extend(result.evolution.reasons)
        reasons_out.append("Accepted and stored as a high-value exemplar candidate for future retrieval and continual learning.")

        return ExemplarBankDecisionSchema(
            sampleId=sample_id,
            accepted=True,
            score=round(score, 4),
            threshold=self.threshold,
            reasons=reasons_out,
            bankSize=self._snapshot_size(updated_snapshot),
            storedAt=created_at,
            bankId=snapshot.bank_id,
            memoryState=result.stored_record.state.value if result.stored_record is not None else None,
            qualityBreakdown=quality_breakdown,
        )

    def retrieve_prior(self, payload: ExemplarRetrievalRequestSchema) -> ExemplarRetrievalResponseSchema:
        response, _ = self.build_retrieval_artifacts(payload)
        return response

    def build_retrieval_artifacts(self, payload: ExemplarRetrievalRequestSchema) -> tuple[ExemplarRetrievalResponseSchema, Any | None]:
        from tools.medical.exemplar_bank_retrieval import RetrievedFeatureSet
        from tools.medical.exemplar_bank_schemas import QueryFeatureBatch, RetrievalCandidate

        snapshot = self.agent.memory.load(bank_id=payload.bankId)
        records = [*snapshot.positive_items, *snapshot.negative_items, *snapshot.boundary_items]
        if not records:
            return (
                ExemplarRetrievalResponseSchema(
                    bankId=payload.bankId,
                    confidence=0.0,
                    uncertainty=1.0,
                    promptTokenShape=(0,),
                    priorKeys=[],
                    candidateCount=0,
                    candidates=[],
                    diagnostics={"warning": "exemplar bank is empty"},
                ),
                None,
            )

        query_semantic = self._semantic_embedding_from_payload(payload)
        query_boundary = self._boundary_embedding_from_payload(payload)
        query_spatial = self._spatial_embedding_from_payload(payload, query_semantic, query_boundary)
        query_hidden = query_semantic.unsqueeze(1).repeat(1, 4, 1)

        query = QueryFeatureBatch(
            query_id=f"{payload.patient.patientId}:{payload.image.filename}",
            semantic_embedding=query_semantic,
            spatial_embedding=query_spatial,
            boundary_embedding=query_boundary,
            hidden_states=query_hidden,
            metadata={"top_k": payload.topK},
        )

        selected = self._select_candidates(records=records, query_semantic=query_semantic, query_boundary=query_boundary, top_k=payload.topK)
        positive_tokens = self._stack_token_group(selected["positive"], fallback=query_semantic)
        negative_tokens = self._stack_token_group(selected["negative"], fallback=query_semantic)
        boundary_tokens = self._stack_token_group(selected["boundary"], fallback=query_boundary)
        positive_map = self._stack_map_group(selected["positive"], fallback=query_semantic)
        negative_map = self._stack_map_group(selected["negative"], fallback=query_semantic)
        boundary_map = self._stack_map_group(selected["boundary"], fallback=query_boundary)

        retrieved = RetrievedFeatureSet(
            positive_tokens=positive_tokens,
            negative_tokens=negative_tokens,
            boundary_tokens=boundary_tokens,
            positive_map=positive_map,
            negative_map=negative_map,
            boundary_map=boundary_map,
            candidate_metadata=selected["candidates"],
        )
        result = self.agent.retrieve_prior(query=query, retrieved=retrieved, bank_id=payload.bankId)
        if result.retrieval is None:
            raise AppException(500, 50081, "exemplar retrieval pipeline returned no retrieval package")

        retrieval = result.retrieval
        candidates = [
            ExemplarRetrievalCandidateSchema(
                exemplarId=item.exemplar_id,
                polarity=item.polarity.value,
                similarity=round(float(item.similarity), 4),
                rankScore=round(float(item.rank_score), 4),
                uncertaintyPenalty=round(float(item.uncertainty_penalty), 4),
                tags=list(item.metadata.get("tags", [])),
            )
            for item in retrieval.selected_candidates
        ]
        return (
            ExemplarRetrievalResponseSchema(
                bankId=payload.bankId,
                confidence=round(float(retrieval.confidence.mean().item()), 4),
                uncertainty=round(float(retrieval.uncertainty.mean().item()), 4),
                promptTokenShape=tuple(int(value) for value in retrieval.prompt_tokens.shape),
                priorKeys=list(retrieval.retrieval_prior.keys()),
                candidateCount=len(candidates),
                candidates=candidates,
                diagnostics=self._sanitize_diagnostics(retrieval.diagnostics),
            ),
            retrieval,
        )

    def apply_feedback(self, payload: ExemplarFeedbackRequestSchema) -> ExemplarFeedbackResponseSchema:
        feedback: dict[str, Any] = {"failure_mode": payload.failureMode, **payload.metadata}
        if payload.qualityScore is not None:
            feedback["quality_score"] = payload.qualityScore
        if payload.uncertainty is not None:
            feedback["uncertainty"] = payload.uncertainty

        result = self.agent.update_with_feedback(
            exemplar_id=payload.exemplarId,
            feedback=feedback,
            bank_id=payload.bankId,
        )
        if result.stored_record is None or result.evolution is None:
            raise AppException(404, 40481, f"exemplar {payload.exemplarId} not found in bank {payload.bankId}")

        return ExemplarFeedbackResponseSchema(
            exemplarId=payload.exemplarId,
            bankId=payload.bankId,
            updatedState=result.stored_record.state.value,
            reasons=list(result.evolution.reasons),
            qualityBreakdown=self._quality_to_dict(result.evolution.quality),
        )

    def _build_record(
        self,
        *,
        payload: ExemplarBankRequestSchema,
        sample_id: str,
        created_at: datetime,
        image_hash: str,
        image_path: Path,
        metadata_path: Path,
        polarity: Any,
        score: float,
    ) -> Any:
        from tools.medical.exemplar_bank_schemas import (
            ExemplarEmbeddingRecord,
            FeatureCentroid,
            MedicalExemplarRecord,
            RetrievalStatistics,
        )

        semantic = self._semantic_embedding_from_payload(payload)
        boundary = self._boundary_embedding_from_payload(payload)
        spatial = self._spatial_embedding_from_payload(payload, semantic, boundary)
        semantic_list = semantic.squeeze(0).tolist()
        boundary_list = boundary.squeeze(0).tolist()

        return MedicalExemplarRecord(
            exemplar_id=sample_id,
            image_path=str(image_path),
            mask_path=str(metadata_path),
            boundary_mask_path=str(metadata_path),
            domain_source="workspace",
            polarity=polarity,
            morphology_tags=self._morphology_tags(payload),
            pathology_tags=self._pathology_tags(payload),
            semantic_tags=self._semantic_tags(payload),
            difficulty_score=round(self._difficulty_score(payload), 4),
            boundary_complexity=round(self._boundary_complexity(payload), 4),
            quality_score=round(score, 4),
            uncertainty_score=round(self._uncertainty_score(payload), 4),
            usage_frequency=0,
            retrieval_statistics=RetrievalStatistics(),
            centroid=FeatureCentroid(
                semantic_centroid=semantic_list,
                spatial_centroid_path=str(metadata_path),
                boundary_centroid=boundary_list,
                momentum=0.95,
            ),
            embeddings=ExemplarEmbeddingRecord(
                sam_embedding_path=str(metadata_path),
                medical_semantic_embedding_path=str(metadata_path),
                spatial_embedding_path=str(metadata_path),
                boundary_embedding_path=str(metadata_path),
                embedding_dim=self.hidden_dim,
                feature_shape=tuple(int(value) for value in spatial.shape),
            ),
            cluster_id=f"workspace::{payload.expertConfig.parisDetail.subtypeCode or 'unknown'}",
            dedup_hash=image_hash,
            created_at=created_at.isoformat(),
            updated_at=created_at.isoformat(),
            metadata={
                "patient_id": payload.patient.patientId,
                "filename": payload.image.filename,
                "paris_classification": payload.expertConfig.parisClassification,
                "lesion_type": payload.expertConfig.lesionType,
                "pathology_classification": payload.expertConfig.pathologyClassification,
                "surface_pattern": payload.expertConfig.surfacePattern,
                "mask_area_ratio": payload.segmentation.maskAreaRatio,
                "point_count": payload.segmentation.pointCount,
                "semantic_map_mean": float(spatial.mean().item()),
            },
        )

    def _score_candidate(self, *, payload: ExemplarBankRequestSchema, duplicate: bool) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        label_coverage = 0.0
        if payload.expertConfig.parisClassification.strip():
            label_coverage += 0.10
        if payload.expertConfig.parisDetail.subtypeCode.strip():
            label_coverage += 0.08
        if payload.expertConfig.lesionType.strip():
            label_coverage += 0.10
        if payload.expertConfig.pathologyClassification.strip():
            label_coverage += 0.14
        if payload.expertConfig.expertNotes.strip():
            label_coverage += 0.08
        score += label_coverage
        reasons.append(f"Label completeness contributed {label_coverage:.2f}.")

        segmentation_score = 0.0
        if payload.segmentation.pointCount >= 6:
            segmentation_score += 0.08
        if 0.01 <= payload.segmentation.maskAreaRatio <= 0.45:
            segmentation_score += 0.10
        if payload.segmentation.boundingBox[2] > payload.segmentation.boundingBox[0]:
            segmentation_score += 0.04
        segmentation_score += min(self._boundary_complexity(payload) * 0.16, 0.16)
        score += segmentation_score
        reasons.append(f"Segmentation quality contributed {segmentation_score:.2f}.")

        report_score = 0.0
        if payload.findings.strip():
            report_score += 0.08
        if payload.conclusion.strip():
            report_score += 0.08
        if payload.reportMarkdown.strip():
            report_score += 0.05
        score += report_score
        reasons.append(f"Report completeness contributed {report_score:.2f}.")

        novelty_score = min((len(self._semantic_tags(payload)) + len(self._morphology_tags(payload))) / 16.0, 0.17)
        score += novelty_score
        reasons.append(f"Novelty scoring contributed {novelty_score:.2f}.")

        note_bonus = min(len(payload.expertConfig.expertNotes.strip()) / 400.0, 0.08)
        score += note_bonus
        reasons.append(f"Expert note richness contributed {note_bonus:.2f}.")

        if duplicate:
            score = min(score, 0.2)

        return min(score, 1.0), reasons

    def _select_candidates(
        self,
        *,
        records: list[Any],
        query_semantic: torch.Tensor,
        query_boundary: torch.Tensor,
        top_k: int,
    ) -> dict[str, list[Any]]:
        from tools.medical.exemplar_bank_schemas import ExemplarPolarity, RetrievalCandidate

        grouped: dict[str, list[tuple[float, float, Any]]] = {"positive": [], "negative": [], "boundary": []}
        query_sem = F.normalize(query_semantic.squeeze(0), dim=0)
        query_bnd = F.normalize(query_boundary.squeeze(0), dim=0)

        for record in records:
            semantic = self._tensor_from_list(record.centroid.semantic_centroid)
            boundary = self._tensor_from_list(record.centroid.boundary_centroid)
            semantic_sim = float(F.cosine_similarity(query_sem.unsqueeze(0), semantic.unsqueeze(0)).item())
            boundary_sim = float(F.cosine_similarity(query_bnd.unsqueeze(0), boundary.unsqueeze(0)).item())
            uncertainty_penalty = float(min(record.uncertainty_score, 1.0) * 0.25)
            rank_score = 0.72 * semantic_sim + 0.28 * boundary_sim - uncertainty_penalty + 0.05 * float(record.quality_score)
            candidate = RetrievalCandidate(
                exemplar_id=record.exemplar_id,
                polarity=record.polarity,
                similarity=semantic_sim,
                rank_score=rank_score,
                uncertainty_penalty=uncertainty_penalty,
                feature_path=record.embeddings.medical_semantic_embedding_path,
                metadata={
                    "tags": [*record.morphology_tags[:3], *record.pathology_tags[:2], *record.semantic_tags[:3]],
                    "boundary_similarity": round(boundary_sim, 4),
                    "state": record.state.value,
                },
            )
            grouped[record.polarity.value].append((rank_score, boundary_sim, candidate))

        selected: dict[str, list[Any]] = {"positive": [], "negative": [], "boundary": [], "candidates": []}
        for polarity_name in ("positive", "negative", "boundary"):
            ranked = sorted(grouped[polarity_name], key=lambda item: (item[0], item[1]), reverse=True)[:top_k]
            candidates = [item[2] for item in ranked]
            selected[polarity_name] = candidates
            selected["candidates"].extend(candidates)
        return selected

    def _stack_token_group(self, candidates: list[Any], *, fallback: torch.Tensor) -> torch.Tensor:
        vectors = [self._candidate_vector(candidate) for candidate in candidates]
        if not vectors:
            vectors = [fallback.squeeze(0)]
        return torch.stack(vectors, dim=0).unsqueeze(0)

    def _stack_map_group(self, candidates: list[Any], *, fallback: torch.Tensor) -> torch.Tensor:
        base = fallback.squeeze(0).view(self.hidden_dim, 1, 1).expand(self.hidden_dim, 16, 16)
        maps = [base.clone() * (1.0 + max(candidate.rank_score, 0.0)) for candidate in candidates]
        if not maps:
            maps = [base]
        return torch.stack(maps, dim=0).mean(dim=0, keepdim=True)

    def _candidate_vector(self, candidate: Any) -> torch.Tensor:
        signature = f"{candidate.exemplar_id}|{candidate.rank_score:.4f}|{candidate.similarity:.4f}"
        return self._hash_to_unit_vector(signature)

    def _semantic_embedding_from_payload(self, payload: Any) -> torch.Tensor:
        tokens = [
            payload.image.filename,
            payload.patient.patientId,
            payload.expertConfig.parisClassification,
            payload.expertConfig.parisDetail.subtypeCode,
            payload.expertConfig.parisDetail.featureSummary,
            payload.expertConfig.lesionType,
            payload.expertConfig.pathologyClassification,
            payload.expertConfig.surfacePattern,
            payload.expertConfig.expertNotes[:512],
        ]
        return self._hash_to_unit_vector("|".join(part.strip() for part in tokens if part.strip())).unsqueeze(0)

    def _boundary_embedding_from_payload(self, payload: Any) -> torch.Tensor:
        bbox = payload.segmentation.boundingBox
        signature = "|".join(
            [
                ",".join(str(int(value)) for value in bbox),
                f"{payload.segmentation.maskAreaRatio:.4f}",
                str(int(payload.segmentation.pointCount)),
                payload.expertConfig.parisDetail.morphologyGroup,
            ]
        )
        return self._hash_to_unit_vector(signature).unsqueeze(0)

    def _spatial_embedding_from_payload(self, payload: Any, semantic: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        base = 0.65 * semantic.squeeze(0) + 0.35 * boundary.squeeze(0)
        feature = base.view(self.hidden_dim, 1, 1).expand(self.hidden_dim, 16, 16).clone()
        bbox = payload.segmentation.boundingBox
        width = max(payload.image.width, 1)
        height = max(payload.image.height, 1)
        x1, y1, x2, y2 = bbox
        norm_x = (x1 + x2) / 2.0 / width
        norm_y = (y1 + y2) / 2.0 / height
        feature[:16] *= float(1.0 + norm_x)
        feature[16:32] *= float(1.0 + norm_y)
        feature[32:48] *= float(1.0 + payload.segmentation.maskAreaRatio)
        return feature.unsqueeze(0)

    def _morphology_tags(self, payload: ExemplarBankRequestSchema) -> list[str]:
        detail = payload.expertConfig.parisDetail
        tags = [
            payload.expertConfig.parisClassification,
            detail.subtypeCode,
            detail.displayLabel,
            detail.featureSummary,
            detail.morphologyGroup,
        ]
        return [item.strip() for item in tags if item.strip()]

    def _pathology_tags(self, payload: ExemplarBankRequestSchema) -> list[str]:
        tags = [payload.expertConfig.lesionType, payload.expertConfig.pathologyClassification, payload.expertConfig.surfacePattern]
        return [item.strip() for item in tags if item.strip()]

    def _semantic_tags(self, payload: ExemplarBankRequestSchema) -> list[str]:
        tags = [payload.findings, payload.conclusion, payload.expertConfig.expertNotes]
        normalized: list[str] = []
        for tag in tags:
            if not tag.strip():
                continue
            normalized.extend(part.strip() for part in tag.replace("。", " ").replace("，", " ").split() if part.strip())
        return normalized[:12]

    def _difficulty_score(self, payload: ExemplarBankRequestSchema) -> float:
        complexity = self._boundary_complexity(payload)
        area = payload.segmentation.maskAreaRatio
        subtype_bonus = 0.08 if payload.expertConfig.parisDetail.subtypeCode in {"0-IIc", "0-Is", "0-IIa+IIc"} else 0.0
        return min(0.35 * complexity + 0.25 * (1.0 - min(area / 0.35, 1.0)) + subtype_bonus + 0.18, 1.0)

    def _boundary_complexity(self, payload: ExemplarBankRequestSchema) -> float:
        point_count = max(payload.segmentation.pointCount, len(payload.segmentation.maskCoordinates), 1)
        area = max(payload.segmentation.maskAreaRatio, 1e-4)
        raw = min(point_count / 24.0, 1.0) * 0.6 + min(abs(math.log(area + 1e-4)) / 8.0, 1.0) * 0.4
        return min(max(raw, 0.0), 1.0)

    def _uncertainty_score(self, payload: ExemplarBankRequestSchema) -> float:
        missing = 0.0
        if not payload.expertConfig.pathologyClassification.strip():
            missing += 0.2
        if not payload.expertConfig.parisClassification.strip():
            missing += 0.15
        if payload.segmentation.pointCount < 4:
            missing += 0.25
        if payload.segmentation.maskAreaRatio <= 0.0:
            missing += 0.3
        return min(missing + 0.1, 1.0)

    def _find_duplicate(self, snapshot: Any, *, image_hash: str) -> Any | None:
        for record in [*snapshot.positive_items, *snapshot.negative_items, *snapshot.boundary_items]:
            if record.dedup_hash == image_hash:
                return record
        return None

    @staticmethod
    def _snapshot_size(snapshot: Any) -> int:
        return len(snapshot.positive_items) + len(snapshot.negative_items) + len(snapshot.boundary_items)

    def _tensor_from_list(self, values: list[float]) -> torch.Tensor:
        if not values:
            return self._hash_to_unit_vector("empty")
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() < self.hidden_dim:
            tensor = F.pad(tensor, (0, self.hidden_dim - tensor.numel()))
        elif tensor.numel() > self.hidden_dim:
            tensor = tensor[: self.hidden_dim]
        return F.normalize(tensor, dim=0)

    def _hash_to_unit_vector(self, text: str) -> torch.Tensor:
        if not text:
            text = "empty"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        seed = digest
        while len(values) < self.hidden_dim:
            values.extend(seed)
            seed = hashlib.sha256(seed).digest()
        vector = torch.tensor(values[: self.hidden_dim], dtype=torch.float32)
        vector = vector / 255.0
        vector = vector * 2.0 - 1.0
        return F.normalize(vector, dim=0)

    @staticmethod
    def _sanitize_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                sanitized[key] = value.detach().cpu().tolist()
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def _quality_to_dict(quality: Any | None) -> dict[str, float]:
        if quality is None:
            return {}
        return {key: round(float(value), 4) for key, value in asdict(quality).items()}

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

    def _ensure_agent_path(self) -> None:
        agent_root_str = str(self.agent_root)
        if agent_root_str not in sys.path:
            sys.path.insert(0, agent_root_str)
