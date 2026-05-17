# 报告生成 Agent 工作流 - 快速参考表

## 1. 工具调用关键表

### 组合方案

| 方案 | 模板 | LLM生成 | ReAct精修 | 反思Agent | 总耗时 | 适用场景 |
|-----|------|--------|---------|----------|-------|--------|
| **快速** | ✅ | ❌ | ❌ | ❌ | 0.3s | 高并发场景 |
| **质量** | ❌ | ✅ | ❌ | ❌ | 5s | 需要自然语言 |
| **平衡** | ✅ | ❌ | ✅ | ❌ | 25s | 标准场景 |
| **最优** | ❌ | ✅ | ✅ | ❌ | 30s | 质量优先 |
| **自适** | ✅ | ❌ | ✅ | ✅ | 30s | 自主优化 |

### 工具触发条件

```
compose_findings          : 总是 (模板模式)
compose_conclusion        : 总是 (模板模式)
suggest_layout            : 总是
suggest_report_keywords   : 总是
    ↓
analyze_report           : IF LLM可用 且 (use_llm OR ReAct启用)
    ↓ (条件分支)
refine_report (findings) : IF analyze结果.has_issues = True
refine_report (conc)     : IF analyze结果.has_issues = True
    ↓
score_report             : IF analyze已执行 OR ReAct启用
```

## 2. 工具输入输出速查

### compose_findings (0.05s)

```
输入:  morphology, paris, features
输出:  findings: str (病变描述)

例: "左结肠中段见扁平隆起型病变，大小10×8mm..."
```

### compose_conclusion (0.05s)

```
输入:  paris, risk
输出:  conclusion: str (诊断结论)

例: "Paris分型0-IIb，浸润风险高，建议内镜下活检或切除"
```

### suggest_layout (0.05s)

```
输入:  morphology, paris, risk
输出:  layout_suggestion: str (排版建议)

例: "标题|病变位置|形态描述|处置意见"
```

### suggest_report_keywords (0.1s)

```
输入:  findings, conclusion, max_keywords=6
输出:  keywords: list[str]

例: ["扁平隆起型", "Paris0-IIb", "浸润风险高", "内镜活检"]
```

### analyze_report (~10s, LLM)

```
输入:  findings, conclusion, paris, risk
输出:  {
         has_issues: bool,
         issues: list[str],
         suggestions: list[str],
         confidence: float,
         thinking: str
       }

例: {
  has_issues: True,
  issues: [
    "病变位置描述不够具体",
    "未提及大小估计"
  ],
  suggestions: [
    "补充"左结肠中段"等具体位置",
    "添加"大小约10×8mm"等尺寸"
  ]
}
```

### refine_report (~5s, LLM)

```
输入:  original_text, analysis_result, text_type ("findings"/"conclusion")
输出:  {
         refined_text: str,
         changes: list[str]
       }

例: {
  refined_text: "左结肠中段见扁平隆起型病变，大小约10×8mm，无蒂...",
  changes: [
    "添加位置信息：左结肠中段",
    "补充大小估计：10×8mm",
    "详述形态特征：无蒂、表面光滑"
  ]
}
```

### score_report (~7s, LLM)

```
输入:  findings, conclusion, paris, risk, analysis_result
输出:  {
         overall_score: float,     # 0-10
         quality_level: str,       # poor/fair/good/excellent
         dimensions: {
           completeness: float,
           accuracy: float,
           clarity: float,
           professionalism: float
         }
       }

例: {
  overall_score: 7.8,
  quality_level: "good",
  dimensions: {
    completeness: 8.0,
    accuracy: 8.5,
    clarity: 7.5,
    professionalism: 7.5
  }
}
```

## 3. ReportReflectionAgent 独立循环

### 循环结构

```
迭代 1-3:
  Thinking (~8s):   Agent 通过 LLM 思考当前报告质量
  ↓ 
  Decision:         解析思考，选择 [analyze|refine|score|stop]
  ↓
  Acting:           执行选定的工具
  ↓
  Observing:        评估质量评分，决策继续或停止
  ↓
  Loop Control:     quality ≥ 8.0? YES → STOP
                    iteration ≥ 3? YES → STOP
                    否则继续下一迭代
```

### 中止条件

| 条件 | 优先级 | 动作 |
|------|-------|------|
| 质量评分 ≥ 8.0 | 高 | 立即停止 |
| 迭代次数 ≥ max_iterations | 高 | 立即停止 |
| LLM 不可用 | 中 | 返回初步报告 |
| 工具执行失败 | 低 | 继续下一迭代 |

## 4. 参数配置速查

### DiagnosisAgent.__init__()

```python
enable_report_reflection: bool = True       # 启用独立反思
reflection_max_iterations: int = 3        # 最大反思迭代数
use_llm_report: bool = False               # LLM生成报告
```

### ReportGenerator.__init__()

```python
use_llm: bool = False                      # 使用LLM模式
llm_client: LLMClient | None = None       # LLM客户端
report_tool_registry: ReportToolRegistry = None  # 工具注册
```

### ReportReflectionAgent.__init__()

```python
max_iterations: int = 3                    # 最大迭代次数
quality_threshold: float = 8.0            # 质量满足度
report_tool_registry: ReportToolRegistry = None
```

## 5. 环境变量

```bash
# LLM 配置
LLM_PROVIDER=deepseek           # 提供商: deepseek/tencent/modelscope
LLM_API_KEY=your_api_key       # API密钥
LLM_BASE_URL=...               # 服务地址 (可选)
LLM_MODEL=deepseek-chat        # 模型名称 (可选)
```

## 6. 执行流程图（文本版）

### 模板 + ReAct 方案

```
START
  ↓
[诊断阶段]
  ├─ FeatureExtractor
  ├─ MorphologyClassifier  
  ├─ ParisTypingEngine
  └─ RiskAssessor
  ↓
[报告生成 - 模板模式]
  ├─ compose_findings (0.05s)
  ├─ compose_conclusion (0.05s)
  ├─ suggest_layout (0.05s)
  └─ suggest_report_keywords (0.1s)
  ↓
[ReAct 精修 - 可选]
  ├─ analyze_report (10s)
  │   └─ IF has_issues=True:
  │       ├─ refine_report/findings (5s)
  │       └─ refine_report/conclusion (5s)
  └─ score_report (7s)
  ↓
[独立反思 Agent - 可选]
  ├─ 迭代1: thinking(8s) + tool(8s)
  ├─ 迭代2: thinking(8s) + tool(8s)  
  └─ 迭代3: thinking(8s) + tool(8s)
  ↓
END (输出 ReportData)

总耗时: 0.3s (仅模板) ~ 30s (全启用)
```

## 7. 工具调用示例代码

### 基础用法

```python
from tools.medical.report_tools import create_default_report_tool_registry

registry = create_default_report_tool_registry(llm_client=my_llm)

# 调用工具
findings = registry.call("compose_findings", 
    morphology=morph, paris=paris, features=features)

analysis = registry.call("analyze_report",
    findings=findings, conclusion=conclusion, paris=paris, risk=risk)

refined = registry.call("refine_report",
    original_text=findings, analysis_result=analysis, text_type="findings")

score = registry.call("score_report",
    findings=refined['refined_text'], conclusion=conclusion,
    paris=paris, risk=risk, analysis_result=analysis)
```

### 查看工具信息

```python
# 列出所有工具
tools = registry.list_tool_specs()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")

# 查看调用日志
call_logs = registry.get_call_logs()
for log in call_logs:
    print(f"{log['tool_name']} - {log['execution_time']:.2f}s")
```

## 8. 常见错误及解决

| 错误 | 原因 | 解决 |
|------|------|------|
| `LLM unavailable` | LLM未配置 | 检查环境变量或传入 `llm_client` |
| `Parameter validation failed` | 工具参数不完整 | 确保传入所有必需参数 |
| `Tool execution timeout` | 工具执行超时 | 增加超时时间或启用快速模式 |
| `analysis.has_issues=null` | LLM调用失败 | 检查API密钥和网络连接 |

## 9. 性能优化建议

| 优化 | 方法 | 效果 |
|-----|------|------|
| 加快生成 | 禁用 ReAct, 使用模板模式 | 0.3s vs 25s |
| 提高质量 | 启用 LLM + 独立反思 | 质量+20% |
| 并发处理 | 使用快速模式 + 批量处理 | 吞吐量×N |
| 缓存优化 | 缓存频繁特征计算 | 减少重复工作 |

## 10. 监控指标

```python
# 从 ReportData 中获取指标
report_score = report.report_score.get('overall_score')
quality_level = report.report_score.get('quality_level')
tool_calls_count = len(report.tool_calls)
execution_time = ...  # 从日志中获取

# 从 ReflectionResult 中获取指标
total_iterations = reflection_result.total_iterations
completion_reason = reflection_result.completion_reason
final_score = reflection_result.final_quality_score
```
