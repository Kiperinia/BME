# 报告生成 Agent 工作流 - 方案对比和选择指南

## 方案对比表

### 六种执行方案

| 方案 | 模板 | LLM报告 | ReAct精修 | 独立反思 | 总耗时 | 质量 | 推荐场景 |
|------|------|--------|---------|--------|-------|------|--------|
| **方案1：快速** | ✅ | ❌ | ❌ | ❌ | 0.3s | ⭐⭐ | 高并发、实时响应需求 |
| **方案2：质量** | ❌ | ✅ | ❌ | ❌ | 5s | ⭐⭐⭐ | 需要自然语言表达 |
| **方案3：平衡** | ✅ | ❌ | ✅ | ❌ | 25s | ⭐⭐⭐⭐ | 标准生产场景（推荐） |
| **方案4：最优** | ❌ | ✅ | ✅ | ❌ | 30s | ⭐⭐⭐⭐⭐ | 质量优先，可接受延迟 |
| **方案5：自适** | ✅ | ❌ | ✅ | ✅ | 30s | ⭐⭐⭐⭐⭐ | 自主优化，智能迭代 |
| **方案6：完整** | ❌ | ✅ | ✅ | ✅ | 35s | ⭐⭐⭐⭐⭐ | 终极质量，成本不敏感 |

## 配置对应表

### 方案1：快速 (0.3秒)

```python
DiagnosisAgent(
    use_llm_report=False,               # ✅ 模板模式
    enable_report_reflection=False      # ✅ 无反思
)
# 仅调用基础工具
# → compose_findings, compose_conclusion, suggest_layout, suggest_keywords
```

**工具调用顺序：**
```
compose_findings (0.05s) 
  → compose_conclusion (0.05s) 
  → suggest_layout (0.05s) 
  → suggest_keywords (0.1s) 
  = 0.25s
```

**适用：** 高并发、演示、原型开发

---

### 方案2：质量 (5秒)

```python
DiagnosisAgent(
    use_llm_report=True,                # ✅ LLM 生成
    enable_report_reflection=False      # ✅ 无反思
)
# 调用 LLM，但不进行精修
```

**工具调用顺序：**
```
LLM Chat (5s) 
  → suggest_layout (0.05s) 
  → suggest_keywords (0.1s) 
  = ~5.15s
```

**适用：** 需要自然语言、接受短延迟的场景

---

### 方案3：平衡 (25秒) ⭐ 推荐

```python
DiagnosisAgent(
    use_llm_report=False,               # ✅ 模板模式
    enable_report_reflection=False      # ❌ 无独立反思
)
# 在 ReportGenerator 中启用 ReAct 精修
# llm_client 必需
```

**工具调用顺序：**
```
compose_findings (0.05s)
  → compose_conclusion (0.05s)
  → suggest_layout (0.05s)
  → suggest_keywords (0.1s)
  → [ReAct 精修]
    ├─ analyze_report (10s)
    ├─ refine_report/findings (5s)
    ├─ refine_report/conclusion (5s)
    └─ score_report (7s)
  = ~27s (含ReAct)
```

**适用：** 大多数生产场景（标准选择）

---

### 方案4：最优 (30秒)

```python
DiagnosisAgent(
    use_llm_report=True,                # ✅ LLM 生成
    enable_report_reflection=False      # ❌ 无独立反思
)
# LLM 生成 + ReAct 精修
# llm_client 必需
```

**工具调用顺序：**
```
LLM Chat (5s)
  → suggest_layout (0.05s)
  → suggest_keywords (0.1s)
  → [ReAct 精修]
    ├─ analyze_report (10s)
    ├─ refine_report/findings (5s)
    ├─ refine_report/conclusion (5s)
    └─ score_report (7s)
  = ~32s (含ReAct)
```

**适用：** 质量优先、允许较长延迟的场景

---

### 方案5：自适 (30秒)

```python
DiagnosisAgent(
    use_llm_report=False,               # ✅ 模板模式
    enable_report_reflection=True,      # ✅ 启用独立反思
    reflection_max_iterations=3         # ⚙️ 最多3次迭代
)
# ReportGenerator 进行初步 ReAct
# ReportReflectionAgent 进行智能迭代
# llm_client 必需
```

**工具调用顺序：**
```
[基础报告] (0.3s)
  → [初步ReAct] (27s)
    └─ analyze + refine + score
      → [独立Agent循环]
         ├─ 迭代1: thinking (8s) + tool (8s)
         ├─ 迭代2: thinking (8s) + tool (8s) [条件]
         └─ 迭代3: thinking (8s) + tool (8s) [条件]
  = ~30-40s (取决于质量评分)
```

**特点：** Agent 自主决策工具调用

**适用：** 需要智能优化、可接受动态时间的场景

---

### 方案6：完整 (35秒)

```python
DiagnosisAgent(
    use_llm_report=True,                # ✅ LLM 生成
    enable_report_reflection=True,      # ✅ 启用独立反思
    reflection_max_iterations=3         # ⚙️ 最多3次迭代
)
# 最完整的流程：LLM + ReAct + 独立反思
# llm_client 必需
```

**工具调用顺序：**
```
[LLM报告] (5s)
  → [初步ReAct] (32s)
    └─ LLM + analyze + refine + score
      → [独立Agent循环] (可能的额外迭代)
  = ~35-45s
```

**特点：** 最高质量，每个步骤都优化

**适用：** 关键临床诊断、成本不敏感的场景

---

## 选择决策树

```
我需要快速处理大量报告吗?
  └─ 是 → 方案1 (0.3秒)
  └─ 否 → 下一步
  
我需要 LLM 生成自然语言吗?
  └─ 是 → 下一步
  └─ 否 → 下一步
  
我愿意等待多长时间?
  ├─ < 1秒 → 方案1 (快速)
  ├─ < 10秒 → 方案2 (LLM)
  ├─ < 30秒 → 方案3 (平衡) 或 方案4 (最优)
  └─ > 30秒 → 方案5 (自适) 或 方案6 (完整)

质量和时间的权衡:
  ├─ 时间优先 → 方案1 (0.3s) 或 方案2 (5s)
  ├─ 平衡 → 方案3 (25s) ⭐ 推荐
  ├─ 质量优先 → 方案4 (30s) 或 方案5 (30s)
  └─ 质量最高 → 方案6 (35s)
```

---

## 功能矩阵

### 工具调用范围

| 工具 | 方案1 | 方案2 | 方案3 | 方案4 | 方案5 | 方案6 |
|------|------|------|------|------|------|------|
| **基础工具** |
| compose_findings | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| compose_conclusion | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| suggest_layout | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| suggest_keywords | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ReAct工具** |
| analyze_report | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| refine_report | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| score_report | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Agent工具** |
| LLM Chat | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| ReflectionAgent | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |

---

## 参数配置速览

### 快速 (方案1)
```python
DiagnosisAgent(
    use_llm_report=False,
    enable_report_reflection=False
)
```

### 质量 (方案2)
```python
DiagnosisAgent(
    use_llm_report=True,
    enable_report_reflection=False,
    llm=my_llm
)
```

### 平衡 (方案3) ⭐
```python
DiagnosisAgent(
    use_llm_report=False,
    enable_report_reflection=False,
    llm=my_llm  # ReAct会自动启用
)
```

### 最优 (方案4)
```python
DiagnosisAgent(
    use_llm_report=True,
    enable_report_reflection=False,
    llm=my_llm
)
```

### 自适 (方案5)
```python
DiagnosisAgent(
    use_llm_report=False,
    enable_report_reflection=True,
    reflection_max_iterations=3,
    llm=my_llm
)
```

### 完整 (方案6)
```python
DiagnosisAgent(
    use_llm_report=True,
    enable_report_reflection=True,
    reflection_max_iterations=3,
    llm=my_llm
)
```

---

## 成本分析

### 假设：
- LLM 调用成本 $0.001/请求
- 模板工具成本 $0

| 方案 | LLM调用数 | 总成本/报告 | 1000份成本 |
|------|----------|----------|----------|
| 方案1 | 0 | $0 | $0 |
| 方案2 | 1 | $0.001 | $1 |
| 方案3 | 3 | $0.003 | $3 |
| 方案4 | 4 | $0.004 | $4 |
| 方案5 | 3-9* | $0.003-0.009 | $3-9 |
| 方案6 | 4-10* | $0.004-0.010 | $4-10 |

*取决于迭代和质量评分

---

## 监控建议

### 关键指标

```python
# 每个报告需要监控：
metrics = {
    "generation_time": result_report_data.generated_at,  # 生成时间
    "quality_score": result_report_data.report_score['overall_score'],  # 质量评分
    "tool_calls_count": len(result_report_data.tool_calls),  # 工具调用数
    "llm_calls": sum(1 for t in result_report_data.tool_calls if t['uses_llm']),  # LLM调用数
}

# 如果启用反思Agent:
reflection_metrics = {
    "reflection_iterations": reflection_result.total_iterations,  # 实际迭代数
    "final_quality": reflection_result.final_quality_score,  # 最终评分
    "completion_reason": reflection_result.completion_reason,  # 完成原因
}
```

---

## 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|--------|--------|
| LLM 超时 | 网络延迟/API不可用 | 降级到方案1/2，或检查API |
| 报告质量低 | 模板不够完善 | 切换到方案4/6 (LLM) |
| 成本过高 | 频繁调用LLM | 切换到方案1/3 (模板) |
| 报告不自然 | 使用模板模式 | 切换到方案2/4/6 (LLM) |
| 处理太慢 | 使用了重方案 | 切换到方案1/2 (轻量) |
| 反思无效果 | 质量评分逻辑 | 调整 quality_threshold |

---

## 推荐方案

### 对于新项目：
**方案3 (平衡)** ⭐ 推荐
- 好的质量 (⭐⭐⭐⭐)
- 可接受的延迟 (25秒)
- 合理的成本 ($0.003)

### 对于生产系统：
**方案1 (快速) 或 方案3 (平衡)**
- 取决于吞吐量需求 vs 质量需求

### 对于关键诊断：
**方案6 (完整)**
- 最高质量
- 成本可控

### 对于研究场景：
**方案5 或 方案6 (自适/完整)**
- 探索最优报告质量
- 记录 Agent 决策过程
