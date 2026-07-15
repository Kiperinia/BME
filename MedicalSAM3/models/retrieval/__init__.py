"""RSS-DA 检索子模块统一导出。"""

from .prototype_retriever import PrototypeRetriever
from .similarity import SimilarityHeatmapBuilder, cosine_similarity_map

__all__ = [
    "PrototypeRetriever",
    "SimilarityHeatmapBuilder",
    "cosine_similarity_map",
]