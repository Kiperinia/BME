# agent/tools/medical — 医学专用工具集
#
# 提供从 SAM3 分割结果到结构化诊断报告的完整流水线：
#   feature_extractor  →  morphology_classifier  →  paris_typing  →  risk_assessor  →  report_generator
#
# 设计原则：
#   1. 纯算法优先：形态分类、Paris 分型、风险评估均以 OpenCV 特征 + 规则引擎为主
#   2. LLM 增强：当规则置信度不足时，调用 LLM 做二次判断
#   3. 可独立运行：每个模块均可单独调用，也可通过 MedicalPipeline 串联

from agent.tools.medical.feature_extractor import FeatureExtractor
from agent.tools.medical.morphology_classifier import MorphologyClassifier
from agent.tools.medical.paris_typing import ParisTypingEngine
from agent.tools.medical.risk_assessor import RiskAssessor
from agent.tools.medical.report_generator import ReportGenerator

__all__ = [
    "FeatureExtractor",
    "MorphologyClassifier",
    "ParisTypingEngine",
    "RiskAssessor",
    "ReportGenerator",
]
