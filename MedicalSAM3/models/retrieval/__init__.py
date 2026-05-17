"""Retrieval modules for RSS-DA."""

from .prototype_retriever import PrototypeRetriever
from .similarity import SimilarityHeatmapBuilder, cosine_similarity_map

__all__ = [
    "PrototypeRetriever",
    "SimilarityHeatmapBuilder",
    "cosine_similarity_map",
]