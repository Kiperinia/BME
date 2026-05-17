#!/usr/bin/env python3
"""
test_report_reflection_agent.py — 测试 ReAct 反思 Agent

演示完整的 ReAct 循环：
  1. Agent 思考报告问题
  2. Agent 决策该执行哪个工具
  3. 工具执行
  4. Agent 观察结果
  5. 决策是否继续
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add agent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import Config
from core.llm import MyLLM, RuleOnlyLLM
from tools.medical.report_generator import ReportData
from tools.medical.morphology_classifier import MorphologyResult, PedicleType, SizeGrade
from tools.medical.paris_typing import ParisTypingResult, ParisType, InvasionRisk
from tools.medical.risk_assessor import RiskAssessmentResult, RiskLevel
from agents.report_reflection_agent import ReportReflectionAgent, ReflectionStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_mock_report() -> ReportData:
    """创建模拟的初步报告（有问题的）"""
    return ReportData(
        patient_id="PAT001",
        study_id="STU001",
        exam_date="2024-01-15",
        findings="内镜下见结肠病变一枚。",  # 太简略
        conclusion="建议临床诊断。",  # 太模糊
    )


def create_mock_morphology() -> MorphologyResult:
    """创建模拟的形态分类结果"""
    return MorphologyResult(
        pedicle_type=PedicleType.SESSILE,
        size_grade=SizeGrade.MEDIUM,
        estimated_size_mm=12.5,
        surface_pattern="pit_pattern_IV",
        dominant_color="pale_yellow",
        shape_description="flat",
        confidence=0.87,
        used_llm=False,
        llm_reasoning="",
    )


def create_mock_paris() -> ParisTypingResult:
    """创建模拟的 Paris 分型结果"""
    return ParisTypingResult(
        paris_type=ParisType.IIB,
        invasion_risk=InvasionRisk.HIGH,
        confidence=0.82,
    )


def create_mock_risk() -> RiskAssessmentResult:
    """创建模拟的风险评估结果"""
    from tools.medical.risk_assessor import Disposition
    return RiskAssessmentResult(
        risk_level=RiskLevel.HIGH,
        total_score=7.5,
        disposition=Disposition.ENDOSCOPIC_RESECTION,
        disposition_reason="病变大小超过10mm，需要内镜治疗",
        confidence=0.82,
    )


async def main():
    """Main test function"""
    logger.info("=" * 70)
    logger.info("ReAct Report Reflection Agent Test")
    logger.info("=" * 70)
    
    # Try to initialize with real LLM
    try:
        config = Config.from_env()
        logger.info(f"Initializing with LLM provider: {config.default_provider}")
        llm = MyLLM(config=config)
    except Exception as exc:
        logger.warning(f"LLM initialization failed: {exc}")
        logger.info("Falling back to rule-only mode (no reflection)")
        llm = None
    
    # Create reflection agent
    agent = ReportReflectionAgent(
        llm=llm,
        max_iterations=3,
        quality_threshold=8.0,
    )
    
    logger.info(f"Reflection enabled: {agent.reflection_enabled}")
    logger.info(f"Max iterations: {agent.max_iterations}")
    logger.info("")
    
    # Create mock data
    initial_report = create_mock_report()
    morphology = create_mock_morphology()
    paris = create_mock_paris()
    risk = create_mock_risk()
    
    logger.info("Initial Report:")
    logger.info(f"  Findings: {initial_report.findings}")
    logger.info(f"  Conclusion: {initial_report.conclusion}")
    logger.info("")
    
    logger.info("Diagnostic Context:")
    logger.info(f"  Paris Type: {paris.paris_type.value}")
    logger.info(f"  Invasion Risk: {paris.invasion_risk.value}")
    logger.info(f"  Risk Level: {risk.risk_level.value}")
    logger.info("")
    
    # Run reflection
    logger.info("=" * 70)
    logger.info("Starting Report Reflection (ReAct Loop)")
    logger.info("=" * 70)
    logger.info("")
    
    result = agent.reflect(
        report=initial_report,
        morphology=morphology,
        paris=paris,
        risk=risk,
    )
    
    # Report results
    logger.info("=" * 70)
    logger.info("Reflection Results")
    logger.info("=" * 70)
    logger.info("")
    
    logger.info(f"Total Iterations: {result.total_iterations}")
    logger.info(f"Completion Reason: {result.completion_reason}")
    logger.info(f"Final Quality Score: {result.final_quality_score}")
    logger.info("")
    
    if result.reflection_steps:
        logger.info("Reflection Steps:")
        for step in result.reflection_steps:
            logger.info(f"\n--- Iteration {step.iteration} ---")
            logger.info(f"Action: {step.action}")
            logger.info(f"Thinking: {step.thinking[:200]}...")
            logger.info(f"Decision: {step.decision}")
            if step.quality_score is not None:
                logger.info(f"Quality Score: {step.quality_score}")
            logger.info(f"Should Continue: {step.should_continue}")
    
    logger.info("")
    logger.info("Final Report:")
    logger.info(f"  Findings: {result.final_report.findings[:150]}...")
    logger.info(f"  Conclusion: {result.final_report.conclusion[:150]}...")
    
    if result.final_report.report_score:
        logger.info(f"  Quality Score: {result.final_report.report_score.get('overall_score')}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test Complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
