from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from agents.exemplar_bank_agent import ExemplarBankAgent
from tools.medical.exemplar_bank_retrieval import RetrievedFeatureSet
from tools.medical.exemplar_bank_schemas import (
    ExemplarEmbeddingRecord,
    ExemplarPolarity,
    FeatureCentroid,
    MedicalExemplarRecord,
    QueryFeatureBatch,
)


def _mock_record(exemplar_id: str, polarity: ExemplarPolarity) -> MedicalExemplarRecord:
    return MedicalExemplarRecord(
        exemplar_id=exemplar_id,
        image_path=f"images/{exemplar_id}.png",
        mask_path=f"masks/{exemplar_id}.png",
        boundary_mask_path=f"boundaries/{exemplar_id}.png",
        domain_source="kvasir",
        polarity=polarity,
        morphology_tags=["0-IIb", "flat"],
        pathology_tags=["adenoma"],
        semantic_tags=["polyp", "endoscopy"],
        difficulty_score=0.35,
        boundary_complexity=0.62,
        quality_score=0.78,
        uncertainty_score=0.14,
        centroid=FeatureCentroid(semantic_centroid=[0.1] * 8, boundary_centroid=[0.2] * 8),
        embeddings=ExemplarEmbeddingRecord(embedding_dim=256),
    )


def main() -> None:
    agent = ExemplarBankAgent(memory_root=Path("agent") / "memory" / "exemplar_agent_bank_test", hidden_dim=256)
    ingest_result = agent.ingest(_mock_record("positive-1", ExemplarPolarity.POSITIVE))
    assert ingest_result.stored_record is not None

    query = QueryFeatureBatch(
        query_id="case-001",
        semantic_embedding=torch.randn(1, 256),
        spatial_embedding=torch.randn(1, 256, 32, 32),
        boundary_embedding=torch.randn(1, 256),
        hidden_states=torch.randn(1, 8, 256),
    )
    retrieved = RetrievedFeatureSet(
        positive_tokens=torch.randn(1, 4, 256),
        negative_tokens=torch.randn(1, 4, 256),
        boundary_tokens=torch.randn(1, 4, 256),
        positive_map=torch.randn(1, 256, 32, 32),
        negative_map=torch.randn(1, 256, 32, 32),
        boundary_map=torch.randn(1, 256, 32, 32),
        candidate_metadata=[],
    )
    retrieval_result = agent.retrieve_prior(query, retrieved)
    assert retrieval_result.retrieval is not None
    assert retrieval_result.retrieval.prompt_tokens.shape == (1, 1, 256)

    feedback_result = agent.update_with_feedback(
        "positive-1",
        {"failure_mode": "false_negative", "quality_score": 0.83, "uncertainty": 0.21},
    )
    assert feedback_result.evolution is not None
    print("exemplar-bank-agent-smoke: ok")


if __name__ == "__main__":
    main()
