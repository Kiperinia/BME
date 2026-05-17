# 报告生成 Agent 工作流简明版

## 核心工作流（3步骤）

```
诊断 (Diagnosis) 
   ↓ 
报告生成 (Report Generation) 
   ↓ 
反思优化 (Reflection) [可选]
```

## 工具清单

### 必需工具（4个）- 基础报告组成

| 工具 | 类型 | 输入 | 输出 | 时间 |
|-----|------|------|------|------|
| `compose_findings` | 规则 | morphology, paris, features | findings (str) | 0.05s |
| `compose_conclusion` | 规则 | paris, risk | conclusion (str) | 0.05s |
| `suggest_layout` | 规则 | morphology, paris, risk | layout (str) | 0.05s |
| `suggest_report_keywords` | 规则 | findings, conclusion | keywords (list) | 0.1s |

### 可选工具（3个）- ReAct 精修

| 工具 | 阶段 | 输入 | 输出 | 时间 | 说明 |
|-----|------|------|------|------|------|
| `analyze_report` | Thinking | findings, conclusion, paris, risk | issues, suggestions | 8-12s | LLM分析问题 |
| `refine_report` | Acting | original_text, analysis | refined_text, changes | 3-6s | LLM精修文本 |
| `score_report` | Scoring | findings, conclusion + context | overall_score, dimensions | 6-8s | LLM评分 |

## 执行流程

### 模板模式（推荐快速）

```
输入 → compose_findings → compose_conclusion → suggest_layout 
→ suggest_keywords → [可选: ReAct精修] → 输出
```

**耗时：** ~0.3秒 (无ReAct) 或 ~25秒 (含ReAct)

### LLM模式（更自然的报告）

```
输入 → LLM Chat → suggest_layout → suggest_keywords 
→ [可选: ReAct精修] → 输出
```

**耗时：** ~5秒 (无ReAct) 或 ~30秒 (含ReAct)

## ReAct 精修流程

```
1️⃣ Thinking 阶段 (~8-12秒)
   analyze_report() 
   → 识别问题：信息缺失、逻辑矛盾、表达不清

2️⃣ Acting 阶段 (~6-12秒)
   如有问题：
   - refine_report(findings) → 精修病变描述
   - refine_report(conclusion) → 精修诊断结论

3️⃣ Scoring 阶段 (~6-8秒)
   score_report() → 评分 (0-10)
```

## 独立 ReAct Agent（可选启用）

启用条件：`enable_report_reflection=True`

**流程：** 自动迭代最多3次

```
每次迭代 (Iteration 1-3):
  ├─ 思考 (Thinking): LLM分析报告质量
  ├─ 决策 (Decision): 选择工具 (analyze/refine/score/stop)
  ├─ 执行 (Acting): 调用选定的工具
  ├─ 观察 (Observing): 评估质量评分
  └─ 控制 (Control): 质量≥8.0? → 停止；否则继续或达最大迭代 → 停止
```

**总耗时：** ~30秒 (3次迭代×8-10秒)

## 配置参数

```python
# DiagnosisAgent
enable_report_reflection = True          # 启用独立反思
reflection_max_iterations = 3           # 最大迭代数

# ReportGenerator
use_llm = False                          # 使用LLM生成（False=模板）

# ReportReflectionAgent
quality_threshold = 8.0                  # 质量评分阈值
```

## 关键数据结构

```python
ReportData:
  - findings: str                        # 检查所见
  - conclusion: str                      # 诊断结论
  - layout_suggestion: str              # 排版建议
  - react_analysis: dict                # 分析结果
  - react_refinement: dict              # 精修结果
  - report_score: dict                  # 评分结果
  - tool_calls: list[dict]              # 工具调用追踪

ReflectionResult (from ReportReflectionAgent):
  - initial_report: ReportData           # 初始报告
  - final_report: ReportData             # 最终改进报告
  - reflection_steps: list[ReflectionStep]  # 反思过程
  - total_iterations: int                # 实际迭代数
  - final_quality_score: float           # 最终评分
```

## 调用示例

### 基础用法（模板模式）

```python
from agents.diagnosis_agent import DiagnosisAgent

agent = DiagnosisAgent(llm=my_llm, use_llm_report=False)
result = agent.diagnose_single_sync(
    image=image_array,
    mask=mask_array,
    context={"patient_id": "P001"}
)

report = result.report  # ReportData
```

### 启用反思

```python
agent = DiagnosisAgent(
    llm=my_llm,
    use_llm_report=False,
    enable_report_reflection=True,
    reflection_max_iterations=3
)

result = agent.diagnose_single_sync(...)
# 报告已自动通过反思优化
```

### 直接使用反思Agent

```python
from agents.report_reflection_agent import ReportReflectionAgent

reflection_agent = ReportReflectionAgent(
    llm=my_llm,
    max_iterations=3,
    quality_threshold=8.0
)

result = reflection_agent.reflect(
    report=initial_report,
    morphology=morph,
    paris=paris,
    risk=risk
)

improved_report = result.final_report
```

## 性能对比

| 场景 | 工具调用数 | 总耗时 | 质量 |
|-----|----------|-------|------|
| 纯模板 | 4 | ~0.3s | 基础 |
| 模板 + ReAct | 7 | ~25s | 改善 |
| LLM + ReAct | 7 | ~30s | 最优 |
| 独立反思Agent | 可变 | ~30s | 动态优化 |

## 工作流图文件

生成的 PlantUML 文档：
- `REPORT_WORKFLOW.puml` - 完整工作流时序图
- `REPORT_TOOLS_DETAIL.puml` - 工具详细信息
- `REPORT_TIMELINE.puml` - 时间线图
- `TOOL_CALL_MATRIX.puml` - 工具调用矩阵

使用 PlantUML 在线编辑器（http://www.plantuml.com/plantuml/uml/）查看。

## 常见问题

**Q: 如何加快报告生成速度？**
A: 禁用 ReAct 精修 (`use_llm_report=False`, `enable_report_reflection=False`)

**Q: 如何提高报告质量？**
A: 启用 LLM 模式和独立反思 Agent (`use_llm_report=True`, `enable_report_reflection=True`)

**Q: ReAct 工具什么时候调用？**
A: 仅在 LLM 可用且启用时自动调用

**Q: 能否自定义迭代次数？**
A: 可以，通过 `reflection_max_iterations` 参数

**Q: 各工具的输入参数可选吗？**
A: 否，所有参数都必需。参数验证失败时会异常并记录警告
