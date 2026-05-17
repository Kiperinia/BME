from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch


class ExemplarPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"


class ExemplarLifecycleState(str, Enum):
    CANDIDATE = "candidate"
    INDEXED = "indexed"
    ACTIVE = "active"
    HARD_POSITIVE = "hard_positive"
    HARD_NEGATIVE = "hard_negative"
    UNCERTAIN = "uncertain"
    AGED = "aged"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class VectorBackend(str, Enum):
    FAISS = "faiss"
    MILVUS = "milvus"
    CHROMA = "chroma"


@dataclass(slots=True)
class RetrievalStatistics:
    retrieval_count: int = 0
    positive_hits: int = 0
    negative_hits: int = 0
    boundary_hits: int = 0
    cumulative_similarity: float = 0.0
    cumulative_gain: float = 0.0
    last_retrieved_at: str = ""
    failure_count: int = 0
    false_positive_count: int = 0
    false_negative_count: int = 0


@dataclass(slots=True)
class FeatureCentroid:
    semantic_centroid: list[float] = field(default_factory=list)
    spatial_centroid_path: str = ""
    boundary_centroid: list[float] = field(default_factory=list)
    momentum: float = 0.95


@dataclass(slots=True)
class ExemplarEmbeddingRecord:
    sam_embedding_path: str = ""
    medical_semantic_embedding_path: str = ""
    spatial_embedding_path: str = ""
    boundary_embedding_path: str = ""
    embedding_dim: int = 0
    feature_shape: tuple[int, ...] = ()


@dataclass(slots=True)
class MedicalExemplarRecord:
    exemplar_id: str
    image_path: str
    mask_path: str
    boundary_mask_path: str
    domain_source: str
    polarity: ExemplarPolarity
    morphology_tags: list[str] = field(default_factory=list)
    pathology_tags: list[str] = field(default_factory=list)
    semantic_tags: list[str] = field(default_factory=list)
    difficulty_score: float = 0.0
    boundary_complexity: float = 0.0
    quality_score: float = 0.0
    uncertainty_score: float = 0.0
    usage_frequency: int = 0
    retrieval_statistics: RetrievalStatistics = field(default_factory=RetrievalStatistics)
    centroid: FeatureCentroid = field(default_factory=FeatureCentroid)
    embeddings: ExemplarEmbeddingRecord = field(default_factory=ExemplarEmbeddingRecord)
    state: ExemplarLifecycleState = ExemplarLifecycleState.CANDIDATE
    cluster_id: str = ""
    dedup_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    lineage_parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["polarity"] = self.polarity.value
        payload["state"] = self.state.value
        return payload


@dataclass(slots=True)
class PrototypeClusterRecord:
    cluster_id: str
    centroid_exemplar_id: str
    member_ids: list[str] = field(default_factory=list)
    prototype_quality: float = 0.0
    intra_cluster_variance: float = 0.0
    domain_mix: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class MemoryBankSnapshot:
    bank_id: str
    version: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    positive_items: list[MedicalExemplarRecord] = field(default_factory=list)
    negative_items: list[MedicalExemplarRecord] = field(default_factory=list)
    boundary_items: list[MedicalExemplarRecord] = field(default_factory=list)
    clusters: list[PrototypeClusterRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorDocument:
    id: str
    embedding: list[float]
    metadata: dict[str, Any]
    document: str = ""


@dataclass(slots=True)
class CacheRecord:
    cache_key: str
    exemplar_ids: list[str]
    retrieval_prior_path: str = ""
    confidence: float = 0.0
    uncertainty: float = 0.0
    expires_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryFeatureBatch:
    query_id: str
    semantic_embedding: torch.Tensor
    spatial_embedding: torch.Tensor
    boundary_embedding: torch.Tensor
    morphology_embedding: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalCandidate:
    exemplar_id: str
    polarity: ExemplarPolarity
    similarity: float
    rank_score: float
    uncertainty_penalty: float
    feature_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalPackage:
    prompt_tokens: torch.Tensor
    retrieval_prior: dict[str, torch.Tensor]
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    selected_candidates: list[RetrievalCandidate] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_vector_document(record: MedicalExemplarRecord, embedding: list[float], backend: VectorBackend) -> VectorDocument:
    base_metadata = {
        "domain_source": record.domain_source,
        "polarity": record.polarity.value,
        "state": record.state.value,
        "difficulty_score": record.difficulty_score,
        "boundary_complexity": record.boundary_complexity,
        "quality_score": record.quality_score,
        "morphology_tags": record.morphology_tags,
        "semantic_tags": record.semantic_tags,
        "pathology_tags": record.pathology_tags,
        "cluster_id": record.cluster_id,
    }
    if backend == VectorBackend.MILVUS:
        base_metadata["pk"] = record.exemplar_id
    if backend == VectorBackend.CHROMA:
        base_metadata["source"] = record.image_path
    return VectorDocument(
        id=record.exemplar_id,
        embedding=embedding,
        metadata=base_metadata,
        document=" | ".join([*record.morphology_tags, *record.semantic_tags, *record.pathology_tags]),
    )


def compute_dedup_signature(record: MedicalExemplarRecord) -> str:
    normalized = [
        Path(record.image_path).name.lower(),
        Path(record.mask_path).name.lower(),
        record.domain_source.lower(),
        record.polarity.value,
        ",".join(sorted(tag.lower() for tag in record.morphology_tags)),
    ]
    return "|".join(normalized)
