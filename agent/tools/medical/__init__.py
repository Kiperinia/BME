# agent/tools/medical — 医学专用工具集
#
# 提供从 SAM3 分割结果到结构化诊断报告的完整流水线：
#   feature_extractor  →  morphology_classifier  →  paris_typing  →  risk_assessor  →  report_generator
#
# 设计原则：
#   1. 纯算法优先：形态分类、Paris 分型、风险评估均以 OpenCV 特征 + 规则引擎为主
#   2. LLM 增强：当规则置信度不足时，调用 LLM 做二次判断
#   3. 可独立运行：每个模块均可单独调用，也可通过 MedicalPipeline 串联

from .feature_extractor import FeatureExtractor
from .morphology_classifier import MorphologyClassifier
from .paris_typing import ParisTypingEngine
from .risk_assessor import RiskAssessor
from .report_generator import ReportGenerator
from .exemplar_bank_memory import ExemplarMemoryManager
from .exemplar_bank_quality import (
    PrototypeEvolutionConfig,
    PrototypeEvolutionDecision,
    PrototypeEvolutionManager,
    PrototypeQualityController,
    PrototypeQualityScore,
)
from .exemplar_bank_retrieval import (
    BoundaryEmbeddingHead,
    CrossAttentionReranker,
    ExemplarRetrievalPipeline,
    GatedRetrievalFusion,
    MultiScaleFeatureAligner,
    RetrievalConfidenceEstimator,
    RetrievedFeatureSet,
    SemanticProjector,
)
from .exemplar_bank_schemas import (
    CacheRecord,
    ExemplarEmbeddingRecord,
    ExemplarLifecycleState,
    ExemplarPolarity,
    FeatureCentroid,
    MemoryBankSnapshot,
    MedicalExemplarRecord,
    PrototypeClusterRecord,
    QueryFeatureBatch,
    RetrievalCandidate,
    RetrievalPackage,
    RetrievalStatistics,
    VectorBackend,
    VectorDocument,
    build_vector_document,
    compute_dedup_signature,
)
from .report_tools import (
    ConclusionComposerTool,
    FindingsComposerTool,
    LayoutSuggestionTool,
    ReportKeywordSuggestionTool,
    ReportToolRegistry,
    ToolParameterSchema,
    create_default_report_tool_registry,
)

__all__ = [
    "FeatureExtractor",
    "MorphologyClassifier",
    "ParisTypingEngine",
    "RiskAssessor",
    "ReportGenerator",
    "ExemplarMemoryManager",
    "PrototypeEvolutionConfig",
    "PrototypeEvolutionDecision",
    "PrototypeEvolutionManager",
    "PrototypeQualityController",
    "PrototypeQualityScore",
    "BoundaryEmbeddingHead",
    "CrossAttentionReranker",
    "ExemplarRetrievalPipeline",
    "GatedRetrievalFusion",
    "MultiScaleFeatureAligner",
    "RetrievalConfidenceEstimator",
    "RetrievedFeatureSet",
    "SemanticProjector",
    "CacheRecord",
    "ExemplarEmbeddingRecord",
    "ExemplarLifecycleState",
    "ExemplarPolarity",
    "FeatureCentroid",
    "MemoryBankSnapshot",
    "MedicalExemplarRecord",
    "PrototypeClusterRecord",
    "QueryFeatureBatch",
    "RetrievalCandidate",
    "RetrievalPackage",
    "RetrievalStatistics",
    "VectorBackend",
    "VectorDocument",
    "build_vector_document",
    "compute_dedup_signature",
    "FindingsComposerTool",
    "ConclusionComposerTool",
    "LayoutSuggestionTool",
    "ReportKeywordSuggestionTool",
    "ToolParameterSchema",
    "ReportToolRegistry",
    "create_default_report_tool_registry",
]
