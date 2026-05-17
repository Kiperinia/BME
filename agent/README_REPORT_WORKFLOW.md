# 报告生成 Agent 工作流文档总览

## 📋 文档导航

### 🎯 快速开始（推荐先看）
1. **[REPORT_WORKFLOW_QUICK_REF.md](REPORT_WORKFLOW_QUICK_REF.md)** - 5分钟快速参考
   - 工具调用表
   - 工具输入输出速查
   - 常见参数配置
   - 性能对比

### 📊 详细工作流
2. **[REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md)** - 完整工作流说明
   - 整体架构
   - 四个工作阶段详解
   - 所有工具清单
   - 执行时间估计
   - 数据流转过程

3. **[WORKFLOW_QUICK_REFERENCE.md](WORKFLOW_QUICK_REFERENCE.md)** - 快速查询表
   - 工具关键表
   - 工具输入输出速查
   - 独立循环结构
   - 参数配置
   - 常见错误解决

### 🎨 流程图（PlantUML）
4. **[REPORT_WORKFLOW.puml](REPORT_WORKFLOW.puml)** - 完整时序图
   - 诊断 → 报告生成 → 反思 全流程
   - 每个步骤的参与者和交互
   - ReAct 工具调用顺序

5. **[REPORT_TOOLS_DETAIL.puml](REPORT_TOOLS_DETAIL.puml)** - 工具详细信息卡
   - 基础报告工具（4个）
   - ReAct 工具（3个）
   - 每个工具的输入输出参数

6. **[REPORT_TIMELINE.puml](REPORT_TIMELINE.puml)** - 时间线图
   - 各阶段执行时间
   - 工具执行顺序
   - 总耗时估计

7. **[TOOL_CALL_MATRIX.puml](TOOL_CALL_MATRIX.puml)** - 工具调用矩阵
   - 必需工具（4个）
   - 可选工具（3个）
   - 两种模式的工具调用流程

8. **[REPORT_DATA_FLOW.puml](REPORT_DATA_FLOW.puml)** - 完整数据流
   - 输入数据（图像、掩码、患者信息）
   - 每个处理阶段的数据转换
   - 最终输出（DiagnosisResult）

---

## 🔍 工作流速查

### 核心流程（3步）
```
诊断 (Diagnosis)
  ├─ FeatureExtractor
  ├─ MorphologyClassifier
  ├─ ParisTypingEngine
  └─ RiskAssessor
    ↓
报告生成 (Report Generation)
  ├─ compose_findings
  ├─ compose_conclusion
  ├─ suggest_layout
  ├─ suggest_keywords
  └─ [可选] ReAct 精修
    ↓
反思优化 (Reflection) [可选]
  └─ 独立 Agent 循环 ×3
```

### 工具总数：7个

| 类型 | 数量 | 名称 |
|------|------|------|
| 必需工具 | 4 | compose_findings, compose_conclusion, suggest_layout, suggest_report_keywords |
| ReAct 工具 | 3 | analyze_report, refine_report, score_report |

### 执行时间范围
- **最快**：0.3秒（仅模板，无优化）
- **标准**：25秒（模板 + ReAct）
- **最优**：30秒（LLM + ReAct + 独立反思）

---

## 📚 按用途查找文档

### "我想了解工作流"
→ 先读 [REPORT_WORKFLOW_QUICK_REF.md](REPORT_WORKFLOW_QUICK_REF.md)
→ 再看 [REPORT_WORKFLOW.puml](REPORT_WORKFLOW.puml) 时序图

### "我想知道工具的详细信息"
→ 读 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md) 第三章
→ 查 [WORKFLOW_QUICK_REFERENCE.md](WORKFLOW_QUICK_REFERENCE.md) 第2节工具输入输出

### "我想看数据流"
→ 查看 [REPORT_DATA_FLOW.puml](REPORT_DATA_FLOW.puml) 数据流图
→ 或读 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md) 第五章

### "我想配置参数"
→ 查 [WORKFLOW_QUICK_REFERENCE.md](WORKFLOW_QUICK_REFERENCE.md) 第4节参数配置
→ 或读 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md) 第六章

### "我需要调用代码"
→ 读 [REPORT_WORKFLOW_QUICK_REF.md](REPORT_WORKFLOW_QUICK_REF.md) 第5节示例代码
→ 或查 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md) 完整示例

### "工具执行报错了"
→ 查 [WORKFLOW_QUICK_REFERENCE.md](WORKFLOW_QUICK_REFERENCE.md) 第8节常见错误

### "我想优化性能"
→ 查 [WORKFLOW_QUICK_REFERENCE.md](WORKFLOW_QUICK_REFERENCE.md) 第9节优化建议
→ 或读 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md) 第八章

---

## 🎓 概念说明

### ReAct 工作流

ReAct 是一个推理-执行范式，分三个阶段：

1. **Thinking 阶段** (~10秒)
   - 工具：`analyze_report`
   - 功能：LLM 分析报告中存在的问题
   - 输出：问题列表、改进建议

2. **Acting 阶段** (~6-12秒)
   - 工具：`refine_report`（可调用多次）
   - 功能：根据分析结果，LLM 精修报告文本
   - 输出：精修后的 findings 和 conclusion

3. **Scoring 阶段** (~7秒)
   - 工具：`score_report`
   - 功能：LLM 对改进后的报告进行多维度评分
   - 输出：0-10 评分、质量等级

### 两种报告生成模式

**模板模式** (默认)
- 使用规则引擎和模板
- 快速、可预测
- 约 0.3 秒

**LLM 模式**
- 调用 LLM 生成自然语言
- 表达更自然、多样化
- 约 5 秒

### 独立反思 Agent

在完成基本报告后，可选地启用独立的 ReportReflectionAgent：
- 自主思考报告质量
- 动态选择工具（analyze/refine/score）
- 最多循环 3 次
- 直到质量评分 ≥ 8.0/10 或达最大迭代数

---

## 🛠️ 实现细节

### 主要类和文件

```
agent/
├─ agents/
│  ├─ diagnosis_agent.py          # 主诊断 Agent
│  └─ report_reflection_agent.py  # 独立反思 Agent
│
├─ tools/medical/
│  ├─ report_generator.py         # 报告生成器
│  └─ report_tools.py             # 工具注册中心
│
└─ [本文档文件夹]
   ├─ REPORT_WORKFLOW_QUICK_REF.md
   ├─ REPORT_WORKFLOW_SUMMARY.md
   ├─ WORKFLOW_QUICK_REFERENCE.md
   ├─ *.puml (PlantUML 图表)
   └─ README.md (本文件)
```

### 关键数据结构

**ReportData** - 报告数据
- findings: 检查所见
- conclusion: 诊断结论
- layout_suggestion: 排版建议
- react_analysis: ReAct 分析结果
- react_refinement: ReAct 精修结果
- report_score: 评分结果

**ReflectionResult** - 反思结果
- initial_report: 初始报告
- final_report: 最终改进报告
- reflection_steps: 反思过程记录
- total_iterations: 实际迭代数

---

## 📖 学习路径

### 初级（10分钟）
1. 阅读本 README.md
2. 浏览 [REPORT_WORKFLOW_QUICK_REF.md](REPORT_WORKFLOW_QUICK_REF.md) 第1-2节
3. 看 [REPORT_WORKFLOW.puml](REPORT_WORKFLOW.puml) 时序图

### 中级（30分钟）
1. 完整阅读 [REPORT_WORKFLOW_QUICK_REF.md](REPORT_WORKFLOW_QUICK_REF.md)
2. 阅读 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md) 前5章
3. 查看所有 PlantUML 图表

### 高级（60分钟）
1. 完整阅读 [REPORT_WORKFLOW_SUMMARY.md](REPORT_WORKFLOW_SUMMARY.md)
2. 研读 [WORKFLOW_QUICK_REFERENCE.md](WORKFLOW_QUICK_REFERENCE.md)
3. 查看源代码实现
4. 运行测试脚本了解实际执行

---

## 🚀 快速开始示例

### 最简单的用法

```python
from agents.diagnosis_agent import DiagnosisAgent

agent = DiagnosisAgent()
result = agent.diagnose_single_sync(image=img, mask=msk)
print(result.report.findings)
print(result.report.conclusion)
```

### 启用所有优化

```python
from agents.diagnosis_agent import DiagnosisAgent
from core.llm import MyLLM
from core.config import Config

config = Config.from_env()
llm = MyLLM(config=config)

agent = DiagnosisAgent(
    llm=llm,
    use_llm_report=True,              # LLM 生成报告
    enable_report_reflection=True,    # 启用独立反思
    reflection_max_iterations=3       # 最多反思 3 次
)

result = agent.diagnose_single_sync(image=img, mask=msk)
print(f"报告质量: {result.report.report_score['overall_score']}/10")
```

---

## 📞 文档维护

- **最后更新**：2026-05-17
- **版本**：1.0
- **相关文件**：
  - agent/REACT_IMPLEMENTATION.md - ReAct Agent 实现细节
  - agent/test_report_reflection_agent.py - 测试脚本

---

## 📝 文档检查清单

- ✅ REPORT_WORKFLOW_QUICK_REF.md - 5分钟速查
- ✅ REPORT_WORKFLOW_SUMMARY.md - 完整详解
- ✅ WORKFLOW_QUICK_REFERENCE.md - 快速查询表
- ✅ REPORT_WORKFLOW.puml - 完整时序图
- ✅ REPORT_TOOLS_DETAIL.puml - 工具详情卡
- ✅ REPORT_TIMELINE.puml - 时间线图
- ✅ TOOL_CALL_MATRIX.puml - 工具矩阵
- ✅ REPORT_DATA_FLOW.puml - 数据流图
- ✅ README.md (本文件) - 导航和概述
