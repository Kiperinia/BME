"""
report_reflection_agent.py — 报告反思 Agent

实现真实的 ReAct 范式（Reasoning + Acting）：
  1. Thinking：Agent 思考当前报告的问题
  2. Acting：Agent 选择并执行改进工具
  3. Observing：Agent 观察结果，决策是否继续
  4. 循环：基于观察结果反复优化

Agent 的思考过程完全由 LLM 驱动，不是硬编码的工具调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hello_agents import HelloAgentsLLM
from hello_agents.core.agent import Agent as HelloAgent

from core.llm import MyLLM, RuleOnlyLLM
from tools.medical.morphology_classifier import MorphologyResult
from tools.medical.paris_typing import ParisTypingResult
from tools.medical.report_generator import ReportData
from tools.medical.report_tools import ReportToolRegistry, create_default_report_tool_registry
from tools.medical.risk_assessor import RiskAssessmentResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReflectionStep:
    """单次反思循环的记录"""
    iteration: int
    thinking: str  # Agent 的思考
    decision: str  # Agent 的决策（该做什么）
    action: str  # 实际执行的工具
    observation: str  # 工具执行结果的观察
    report_before: str  # 改动前的报告片段
    report_after: str  # 改动后的报告片段
    quality_score: float | None = None
    should_continue: bool = False


@dataclass(slots=True)
class ReflectionResult:
    """完整反思流程的结果"""
    initial_report: ReportData
    final_report: ReportData
    reflection_steps: list[ReflectionStep] = field(default_factory=list)
    total_iterations: int = 0
    final_quality_score: float | None = None
    completion_reason: str = ""


class ReportReflectionAgent(HelloAgent):
    """
    报告反思 Agent。
    
    实现完整的 ReAct 循环，让 Agent 自己思考如何改进报告。
    Agent 的思考由 LLM 驱动，不是硬编码逻辑。
    
    Usage::
    
        agent = ReportReflectionAgent(llm=my_llm)
        result = agent.reflect(
            report=initial_report,
            morphology=morph,
            paris=paris,
            risk=risk,
        )
    """

    def __init__(
        self,
        llm: HelloAgentsLLM | None = None,
        *,
        max_iterations: int = 3,
        quality_threshold: float = 8.0,
        report_tool_registry: ReportToolRegistry | None = None,
    ):
        """
        Args:
            llm: LLM 客户端。如为 None，使用规则模式（无反思）。
            max_iterations: 最大反思迭代次数。
            quality_threshold: 质量评分满足度（0-10）。达到则停止。
            report_tool_registry: 报告工具注册中心。
        """
        resolved_llm = llm or RuleOnlyLLM(
            model="rule-only",
            provider="rule-only",
        )
        
        super().__init__(
            name="report-reflection-agent",
            llm=resolved_llm,
            system_prompt="你是一名医学诊断报告质量改进专家。你的任务是通过反思和改进使诊断报告更加准确、完整和清晰。",
        )
        
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.report_tool_registry = report_tool_registry or create_default_report_tool_registry(llm_client=resolved_llm)
        self.llm_client = llm  # 保存原始 LLM，用于工具调用
        self.reflection_enabled = not isinstance(llm, RuleOnlyLLM)

    def reflect(
        self,
        report: ReportData,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
    ) -> ReflectionResult:
        """
        对报告进行反思和改进。

        Args:
            report: 初步生成的报告。
            morphology: 形态分类结果。
            paris: Paris 分型结果。
            risk: 风险评估结果。

        Returns:
            ReflectionResult：包含所有反思步骤和最终改进的报告。
        """
        if not self.reflection_enabled:
            logger.warning("Reflection disabled (LLM not available); returning original report")
            return ReflectionResult(
                initial_report=report,
                final_report=report,
                total_iterations=0,
                completion_reason="LLM not available",
            )

        current_report = report
        reflection_steps: list[ReflectionStep] = []
        
        logger.info(f"Starting report reflection (max {self.max_iterations} iterations)")
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"=== Reflection iteration {iteration} ===")
            
            # ---- Step 1: Thinking ----
            thinking = self._generate_thinking(
                current_report,
                morphology,
                paris,
                risk,
                iteration,
            )
            logger.info(f"Agent thinking: {thinking[:150]}...")
            
            # ---- Step 2: Acting ----
            action = self._decide_action(thinking)
            logger.info(f"Agent decided action: {action}")
            
            if action == "stop":
                reflection_steps.append(
                    ReflectionStep(
                        iteration=iteration,
                        thinking=thinking,
                        decision=action,
                        action="no_action",
                        observation="报告质量已满足要求",
                        report_before=current_report.findings[:100],
                        report_after=current_report.findings[:100],
                        should_continue=False,
                    )
                )
                logger.info(f"Agent decided to stop (quality satisfied)")
                break
            
            # ---- Step 3: Execute tool ----
            report_before_findings = current_report.findings
            report_before_conclusion = current_report.conclusion
            
            try:
                if action == "analyze":
                    observation = self._execute_analyze(current_report, paris, risk)
                    current_report.react_analysis = observation
                    
                elif action == "refine_findings":
                    observation = self._execute_refine_findings(
                        current_report,
                        current_report.react_analysis or {},
                    )
                    if observation.get("refined_text"):
                        current_report.findings = observation["refined_text"]
                    
                elif action == "refine_conclusion":
                    observation = self._execute_refine_conclusion(
                        current_report,
                        current_report.react_analysis or {},
                    )
                    if observation.get("refined_text"):
                        current_report.conclusion = observation["refined_text"]
                    
                elif action == "score":
                    observation = self._execute_score(
                        current_report,
                        morphology,
                        paris,
                        risk,
                        current_report.react_analysis or {},
                    )
                    current_report.report_score = observation
                    
                else:
                    observation = {"error": f"Unknown action: {action}"}
                
                logger.info(f"Tool {action} executed: {str(observation)[:150]}")
                
            except Exception as exc:
                logger.warning(f"Tool execution failed: {exc}")
                observation = {"error": str(exc)}
            
            # ---- Step 4: Observing & Record ----
            quality_score = current_report.report_score.get("overall_score") if current_report.report_score else None
            should_continue = iteration < self.max_iterations and (quality_score is None or quality_score < self.quality_threshold)
            
            reflection_steps.append(
                ReflectionStep(
                    iteration=iteration,
                    thinking=thinking,
                    decision=action,
                    action=action,
                    observation=str(observation)[:500],
                    report_before=report_before_findings[:100],
                    report_after=current_report.findings[:100],
                    quality_score=quality_score,
                    should_continue=should_continue,
                )
            )
            
            if not should_continue:
                logger.info(f"Stopping reflection: quality={quality_score}, threshold={self.quality_threshold}")
                break
        
        # ---- Final: Update tool_calls ----
        current_report.tool_calls = self.report_tool_registry.get_call_logs()
        
        return ReflectionResult(
            initial_report=report,
            final_report=current_report,
            reflection_steps=reflection_steps,
            total_iterations=len(reflection_steps),
            final_quality_score=current_report.report_score.get("overall_score") if current_report.report_score else None,
            completion_reason=self._summarize_completion(reflection_steps),
        )

    def _generate_thinking(
        self,
        report: ReportData,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        iteration: int,
    ) -> str:
        """
        Agent Thinking：LLM 思考当前报告的问题。
        """
        thinking_prompt = f"""你现在在进行第 {iteration} 轮报告质量改进。

当前报告信息：
- 检查所见 ({len(report.findings)} 字符): {report.findings[:200]}...
- 诊断结论 ({len(report.conclusion)} 字符): {report.conclusion[:200]}...
- 当前质量评分: {report.report_score.get('overall_score', '未评')/10 if report.report_score else '未评'}

基础诊断信息：
- Paris分型: {paris.paris_type.value if paris.paris_type else '未明确'}
- 侵润风险: {paris.invasion_risk.value if paris.invasion_risk else '未明确'}
- 风险等级: {risk.risk_level.value if risk.risk_level else '未明确'}
- 风险评分: {risk.total_score:.1f}/10

请思考：
1. 当前报告是否存在明显问题？
2. 需要进行什么样的改进？
3. 改进的优先级是什么？
4. 是否已经达到满意质量（≥8.0/10）？

请给出你的分析和建议。"""

        try:
            response = self.llm.invoke(
                messages=[{"role": "user", "content": thinking_prompt}],
                temperature=0.6,
                max_tokens=512,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as exc:
            logger.warning(f"LLM thinking failed: {exc}")
            return f"Thinking failed: {str(exc)}"

    def _decide_action(self, thinking: str) -> str:
        """
        Agent Decision：根据 thinking 决定下一步行动。
        """
        thinking_lower = thinking.lower()
        
        # 检查 Agent 是否决定停止
        if any(phrase in thinking_lower for phrase in ["已达", "满足", "停止", "完成", "足够"]):
            return "stop"
        
        # 检查 Agent 建议的改进方向
        if "分析" in thinking or "问题" in thinking or "identify" in thinking_lower:
            return "analyze"
        
        if "findings" in thinking_lower or "所见" in thinking:
            return "refine_findings"
        
        if "conclusion" in thinking_lower or "结论" in thinking:
            return "refine_conclusion"
        
        if "评" in thinking or "score" in thinking_lower or "质量" in thinking:
            return "score"
        
        # 默认：先分析再评分
        return "analyze"

    def _execute_analyze(
        self,
        report: ReportData,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
    ) -> dict[str, Any]:
        """执行分析工具。"""
        return self.report_tool_registry.call(
            "analyze_report",
            findings=report.findings,
            conclusion=report.conclusion,
            paris=paris,
            risk=risk,
        )

    def _execute_refine_findings(
        self,
        report: ReportData,
        analysis_result: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 findings 精修工具。"""
        return self.report_tool_registry.call(
            "refine_report",
            original_text=report.findings,
            analysis_result=analysis_result,
            text_type="findings",
        )

    def _execute_refine_conclusion(
        self,
        report: ReportData,
        analysis_result: dict[str, Any],
    ) -> dict[str, Any]:
        """执行 conclusion 精修工具。"""
        return self.report_tool_registry.call(
            "refine_report",
            original_text=report.conclusion,
            analysis_result=analysis_result,
            text_type="conclusion",
        )

    def _execute_score(
        self,
        report: ReportData,
        morphology: MorphologyResult,
        paris: ParisTypingResult,
        risk: RiskAssessmentResult,
        analysis_result: dict[str, Any],
    ) -> dict[str, Any]:
        """执行评分工具。"""
        return self.report_tool_registry.call(
            "score_report",
            findings=report.findings,
            conclusion=report.conclusion,
            paris=paris,
            risk=risk,
            analysis_result=analysis_result,
        )

    def _summarize_completion(self, steps: list[ReflectionStep]) -> str:
        """总结反思完成原因。"""
        if not steps:
            return "未执行反思"
        
        last_step = steps[-1]
        if last_step.decision == "stop":
            return "质量满足要求"
        
        if len(steps) >= self.max_iterations:
            return f"达到最大迭代次数 ({self.max_iterations})"
        
        return "未知原因"

    def run(self, input_text: str, **kwargs) -> str:
        """
        实现 HelloAgent.run() 接口。
        
        支持的输入格式：
        - JSON 字符串，包含 report, morphology, paris, risk 字段
        - 或通过 kwargs 传递这些字段
        """
        import json
        
        # Parse input
        try:
            if input_text and input_text.strip().startswith("{"):
                data = json.loads(input_text)
            else:
                data = kwargs
        except Exception:
            data = kwargs
        
        # This is a minimal implementation for HelloAgent interface
        # The main entry point is still reflect() method
        return "ReportReflectionAgent: use reflect() method instead of run()"
