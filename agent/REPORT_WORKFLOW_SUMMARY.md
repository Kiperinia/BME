# 报告生成 Agent 工作流总结

## 一、整体架构

```
DiagnosisAgent (诊断主Agent)
    ├─ FeatureExtractor → LesionFeatures
    ├─ MorphologyClassifier → MorphologyResult (可选LLM)
    ├─ ParisTypingEngine → ParisTypingResult (可选LLM)
    ├─ RiskAssessor → RiskAssessmentResult (可选LLM)
    ├─ ReportGenerator → ReportData
    │   ├─ 模板模式 或 LLM模式 生成初步报告
    │   └─ ReAct工具链（analyze → refine → score）
    └─ ReportReflectionAgent (可选) → 改进的ReportData
        └─ 自主思考循环（最多3次迭代）
```

## 二、工作流阶段

### 阶段1：特征提取与诊断 (DiagnosisAgent)

| 组件 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| FeatureExtractor | Image + Mask | LesionFeatures | 量化特征提取 |
| MorphologyClassifier | Features | MorphologyResult | 形态分类 (可选LLM) |
| ParisTypingEngine | Morphology + Features | ParisTypingResult | Paris分型推理 (可选LLM) |
| RiskAssessor | Morphology + Paris + Features | RiskAssessmentResult | 风险评估 (可选LLM) |

### 阶段2：报告初期生成 (ReportGenerator)

**模式选择：**
- `use_llm=False` (默认): 使用规则模板
- `use_llm=True`: 调用LLM生成

#### 2.1 模板模式工具链：

| 工具名 | 输入 | 输出 | 优先级 |
|------|------|------|-------|
| `compose_findings` | morphology, paris, features | findings (str) | 必需 |
| `compose_conclusion` | paris, risk | conclusion (str) | 必需 |
| `suggest_layout` | morphology, paris, risk | layout_suggestion (str) | 必需 |
| `suggest_report_keywords` | findings, conclusion | keywords (list) | 必需 |

**示例调用：**
```python
findings = registry.call(
    "compose_findings",
    morphology=morph,
    paris=paris,
    features=features
)
# 输出: "内镜下见结肠病变一枚，大小约12mm，扁平隆起型，表面光滑，血管消失..."
```

#### 2.2 LLM模式工具链：

| 工具名 | 输入 | 输出 | 说明 |
|------|------|------|------|
| LLM Chat | Prompt含诊断信息 | findings + conclusion + layout | 生成自然语言 |
| `suggest_layout` | morphology, paris, risk | layout_suggestion | 补充排版建议 |
| `suggest_report_keywords` | findings, conclusion | keywords | 提取关键词 |

**示例Prompt：**
```
基于以下诊断信息，生成专业的消化内镜诊断报告：
- Paris分型: 0-IIb (浅表平坦型)
- 浸润风险: 高
- 风险评分: 7.5/10
- 建议处置: 内镜下切除

请生成：
1. 检查所见（findings）
2. 诊断结论（conclusion）
3. 排版建议（layoutSuggestion）
```

### 阶段3：ReAct 反思与精修 (在ReportGenerator中)

**工作流：** Thinking → Acting → Scoring

#### 3.1 Thinking Phase - 分析工具

| 工具名 | 类型 | 输入 | 输出 | 执行时间 |
|------|------|------|------|---------|
| `analyze_report` | ReAct-Thinking | findings, conclusion, paris, risk | has_issues, issues[], suggestions[], thinking | ~8-12秒 |

**内部LLM Prompt：**
```
分析以下报告是否存在问题：

检查所见: {findings}
诊断结论: {conclusion}

Paris分型: {paris}
浸润风险: {risk}

请识别：
1. 信息缺失问题
2. 逻辑矛盾
3. 表达不清
4. 改进建议
```

**输出示例：**
```json
{
  "has_issues": true,
  "issues": [
    "病变位置描述不够具体",
    "诊断结论缺乏临床建议"
  ],
  "suggestions": [
    "补充病变位置（升结肠、横结肠等）",
    "添加处置意见（活检、切除等）"
  ],
  "confidence": 0.85,
  "thinking": "通过分析发现..."
}
```

#### 3.2 Acting Phase - 精修工具

| 工具名 | 类型 | 输入 | 输出 | 执行时间 |
|------|------|------|------|---------|
| `refine_report` (findings) | ReAct-Acting | original_text, analysis_result, type="findings" | refined_text, changes[] | ~3-6秒 |
| `refine_report` (conclusion) | ReAct-Acting | original_text, analysis_result, type="conclusion" | refined_text, changes[] | ~3-6秒 |

**内部LLM Prompt（findings）：**
```
基于以下分析结果，改进病变描述：

原文: {original_text}
识别的问题: {analysis.issues}
改进建议: {analysis.suggestions}

请生成更完整、准确的病变描述，包括：
- 病变位置
- 大小估计
- 形态特征
- 边界与表面
- 血管形态
```

**输出示例：**
```json
{
  "refined_text": "结肠降部见扁平隆起型病变，大小约12×10mm，无蒂，表面光滑，色泽浅黄，血管清晰...",
  "changes": [
    "添加具体位置：降部",
    "补充大小估计：12×10mm",
    "详述形态：扁平隆起、光滑、浅黄色、血管清晰"
  ]
}
```

#### 3.3 Scoring Phase - 评分工具

| 工具名 | 类型 | 输入 | 输出 | 执行时间 |
|------|------|------|------|---------|
| `score_report` | ReAct-Scoring | findings, conclusion, paris, risk, analysis | overall_score, quality_level, dimensions | ~6-8秒 |

**内部LLM Prompt：**
```
对以下诊断报告进行多维度评分（0-10）：

检查所见: {findings}
诊断结论: {conclusion}

评分维度：
1. 完整性（10分）：是否包含所有关键信息
2. 准确性（10分）：是否与诊断数据一致
3. 清晰性（10分）：表达是否易于理解
4. 专业性（10分）：术语使用是否规范

请给出各维度评分及总体评分。
```

**输出示例：**
```json
{
  "overall_score": 7.8,
  "quality_level": "good",
  "dimensions": {
    "completeness": 8.0,
    "accuracy": 8.5,
    "clarity": 7.5,
    "professionalism": 7.5
  },
  "feedback": "报告基本完整，建议补充..."
}
```

### 阶段4：独立 ReAct Agent 循环（可选）

**启用条件：** `enable_report_reflection=True` 且 LLM 可用

**工作流：** 重复最多3次（max_iterations）

```
Iteration 1-3:
  ├─ Thinking: LLM 思考当前报告质量
  │   └─ 输入：findings, conclusion, paris, risk
  │   └─ 思考点：问题? 优先级? 是否完成?
  │   └─ 输出：thinking text (~8秒)
  │
  ├─ Decision: Agent 解析思考文本，选择行动
  │   ├─ "analyze" → 调用 analyze_report
  │   ├─ "refine_findings" → 调用 refine_report (findings)
  │   ├─ "refine_conclusion" → 调用 refine_report (conclusion)
  │   ├─ "score" → 调用 score_report
  │   └─ "stop" → 质量满足，退出循环
  │
  ├─ Acting: 执行选定的工具
  │   └─ 输出：工具执行结果
  │
  ├─ Observing: 观察结果，评估是否继续
  │   ├─ 检查质量评分是否达到阈值（≥8.0）
  │   ├─ 检查是否达到最大迭代次数
  │   └─ 决策：continue or stop
  │
  └─ 记录：ReflectionStep {iteration, thinking, decision, action, observation, quality_score}
```

**关键参数：**
- `max_iterations`: 3 (默认)
- `quality_threshold`: 8.0/10 (默认)

**完成条件：**
1. 质量评分 ≥ 8.0/10 → 停止
2. 迭代次数 ≥ 3 → 停止

## 三、工具清单

### 3.1 基础报告组成工具（模板模式）

```
ReportToolRegistry
├── compose_findings (FindingsComposerTool)
│   ├─ 输入: morphology, paris, features
│   ├─ 规则: 按Paris分型、形态、大小拼接
│   └─ 输出: findings string
│
├── compose_conclusion (ConclusionComposerTool)
│   ├─ 输入: paris, risk
│   ├─ 规则: 按风险等级生成结论
│   └─ 输出: conclusion string
│
├── suggest_layout (LayoutSuggestionTool)
│   ├─ 输入: morphology, paris, risk
│   ├─ 规则: 按信息量建议排版
│   └─ 输出: layout_suggestion string
│
└── suggest_report_keywords (ReportKeywordSuggestionTool)
    ├─ 输入: findings, conclusion, max_keywords=6
    ├─ 规则: 抽取关键医学术语
    └─ 输出: keywords list
```

### 3.2 ReAct 工具（LLM驱动）

```
ReportToolRegistry (ReAct工具)
├── analyze_report (ReportAnalysisTool)
│   ├─ 输入: findings, conclusion, paris, risk
│   ├─ 工作: LLM分析问题 (Thinking)
│   ├─ 输出: has_issues, issues[], suggestions[], thinking
│   └─ 执行时间: ~8-12秒
│
├── refine_report (ReportRefinementTool)
│   ├─ 输入: original_text, analysis_result, text_type (findings/conclusion)
│   ├─ 工作: LLM精修文本 (Acting)
│   ├─ 输出: refined_text, changes[]
│   └─ 执行时间: ~3-6秒
│
└── score_report (ReportScoringTool)
    ├─ 输入: findings, conclusion, paris, risk, analysis_result
    ├─ 工作: LLM多维度评分 (Scoring)
    ├─ 输出: overall_score, quality_level, dimensions
    └─ 执行时间: ~6-8秒
```

## 四、完整执行时间估计

### 场景1：模板模式 + ReAct精修

```
FeatureExtractor: ~0.1秒
MorphologyClassifier: ~0.1秒 (若无LLM)
ParisTypingEngine: ~0.1秒 (若无LLM)
RiskAssessor: ~0.1秒 (若无LLM)

ReportGenerator (模板模式):
  ├─ compose_findings: ~0.05秒
  ├─ compose_conclusion: ~0.05秒
  ├─ suggest_layout: ~0.05秒
  ├─ suggest_report_keywords: ~0.1秒
  └─ ReAct精修:
      ├─ analyze_report: ~10秒
      ├─ refine_report (findings): ~5秒
      ├─ refine_report (conclusion): ~5秒
      └─ score_report: ~7秒

总计: ~27秒 (含ReAct精修)
```

### 场景2：LLM模式 + ReAct精修

```
ReportGenerator (LLM模式):
  ├─ LLM 生成报告: ~5秒
  ├─ suggest_layout: ~0.05秒
  ├─ suggest_report_keywords: ~0.1秒
  └─ ReAct精修: ~27秒 (同上)

总计: ~32秒
```

### 场景3：启用 ReportReflectionAgent

```
ReportGenerator (快速模式): ~0.5秒
ReportReflectionAgent (3次迭代):
  ├─ 迭代1: ~10秒 (thinking + tool)
  ├─ 迭代2: ~10秒 (thinking + tool)
  ├─ 迭代3: ~10秒 (thinking + tool)
  └─ 停止条件满足

总计: ~30秒 (不含诊断阶段)
```

## 五、数据流转

```
输入数据：
  Image (np.ndarray)
  Mask (np.ndarray)
  Patient Context (patient_id, study_id, exam_date)

↓ (Feature Extraction & Diagnosis)

中间数据：
  LesionFeatures
  MorphologyResult
  ParisTypingResult
  RiskAssessmentResult

↓ (Report Generation)

初步报告（ReportData）：
  - findings: str
  - conclusion: str
  - layout_suggestion: str
  - lesion_summary: dict
  - risk_summary: dict
  - tool_calls: list[dict]
  - react_analysis: dict
  - react_refinement: dict
  - report_score: dict

↓ (Optional: Report Reflection)

改进报告（ReportData）：
  - findings: str (可能已精修)
  - conclusion: str (可能已精修)
  - 其他字段同上 (包含反思过程)

↓ (Output)

最终输出（DiagnosisResult）：
  - lesion_id: str
  - label: str (分类标签)
  - confidence: float
  - bbox: tuple
  - features: LesionFeatures
  - morphology: MorphologyResult
  - paris_typing: ParisTypingResult
  - risk_assessment: RiskAssessmentResult
  - report: ReportData
```

## 六、配置项汇总

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `use_llm` (ReportGenerator) | bool | False | 是否使用LLM生成报告 |
| `enable_report_reflection` (DiagnosisAgent) | bool | True | 是否启用独立反思Agent |
| `reflection_max_iterations` | int | 3 | 反思最大迭代次数 |
| `quality_threshold` (ReportReflectionAgent) | float | 8.0 | 质量评分满足阈值 |
| `LLM_API_KEY` | env | - | LLM API密钥 |
| `LLM_BASE_URL` | env | - | LLM服务地址 |
| `LLM_PROVIDER` | env | deepseek | LLM提供商 |

## 七、关键特性对比

### 工具级 ReAct（在 ReportGenerator 中）

- ✅ 自动分析 → 精修 → 评分（固定流程）
- ✅ 工具调用可追踪
- ✅ 无需单独调用
- ❌ 不够灵活（流程固定）

### Agent 级 ReAct（ReportReflectionAgent）

- ✅ 自主思考 → 决策工具 → 执行 → 观察
- ✅ 动态工具选择
- ✅ 迭代控制（质量驱动）
- ✅ 完整思考过程可观察
- ⚠️ 需要额外启用

## 八、错误处理

```
各工具级别：
├─ 参数验证失败 → 异常 + 回退到默认值
├─ LLM调用失败 → 记录警告 + 使用规则模式
└─ 超时 → 跳过ReAct, 返回初步报告

Agent级别：
├─ LLM不可用 → 自动禁用reflection
├─ 迭代失败 → 继续下一迭代
└─ 全部失败 → 返回初步报告
```
