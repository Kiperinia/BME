# 样本库评估 Agent 工作流总结

## 一、整体架构

```text
ExemplarBankAgent（样本库评估 Agent）
    ├─ ExemplarMemoryManager
    │   ├─ snapshot.json 读写
    │   ├─ audit_log.json 追踪
    │   └─ 去重 / 分桶（positive / negative / boundary）
    ├─ PrototypeQualityController
    │   ├─ quality score
    │   ├─ keep / decay / archive 决策
    │   └─ curriculum / EMA / prototype momentum
    ├─ PrototypeEvolutionManager
    │   ├─ false positive memory
    │   ├─ false negative memory
    │   ├─ uncertainty memory
    │   └─ feedback-driven state transition
    ├─ ExemplarRetrievalPipeline
    │   ├─ SemanticProjector
    │   ├─ BoundaryEmbeddingHead
    │   ├─ MultiScaleFeatureAligner
    │   ├─ CrossAttentionReranker
    │   ├─ GatedRetrievalFusion
    │   └─ RetrievalConfidenceEstimator
    └─ Backend ExemplarBankService（桥接层）
        ├─ workspace payload -> exemplar record
        ├─ retrieval prior generation
        └─ API schema / runtime integration
```

## 二、工作流阶段

### 阶段1：病例候选构建（Backend ExemplarBankService）

| 组件 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| `ExemplarBankService._build_record` | workspace image + segmentation + expertConfig + report | `MedicalExemplarRecord` | 将前端病例转为标准 exemplar 记录 |
| `_semantic_embedding_from_payload` | 文本与结构化标签 | semantic embedding | 基于病例元信息构造语义向量 |
| `_boundary_embedding_from_payload` | bbox + mask ratio + contour count | boundary embedding | 生成边界复杂度相关向量 |
| `_spatial_embedding_from_payload` | semantic + boundary + spatial stats | spatial feature map | 供 retrieval prior 融合使用 |

### 阶段2：质量评估与入库（ExemplarBankAgent.ingest）

| 组件 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| `PrototypeQualityController.score` | `MedicalExemplarRecord` | `PrototypeQualityScore` | 计算 clinical / novelty / hard value / freshness 等维度 |
| `PrototypeQualityController.decide` | quality score | `PrototypeEvolutionDecision` | 决定 ACTIVE / INDEXED / AGED / ARCHIVED |
| `ExemplarMemoryManager.deduplicate` | exemplar list | unique exemplar list | 基于 image hash / tags 进行去重 |
| `ExemplarMemoryManager.partition` | exemplar list | positive / negative / boundary | 按 polarity 分桶保存 |
| `ExemplarMemoryManager.save` | `MemoryBankSnapshot` | snapshot.json | 持久化样本库快照 |

### 阶段3：检索先验生成（ExemplarBankAgent.retrieve_prior）

| 组件 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| `_select_candidates` | query semantic + query boundary + exemplar bank | ranked candidates | 为正样本、负样本、边界样本分别排序 |
| `CrossAttentionReranker` | query tokens + memory tokens | reranked prototype | 做 cross-attention 重排 |
| `GatedRetrievalFusion` | positive / negative / boundary proto | prompt tokens + retrieval prior | 生成 prompt 注入与 decoder bias |
| `RetrievalConfidenceEstimator` | fused prototypes | confidence + uncertainty | 估计检索可信度与不确定性 |
| `ExemplarRetrievalPipeline.forward` | `QueryFeatureBatch` + `RetrievedFeatureSet` | `RetrievalPackage` | 最终输出可注入 SAM3 的先验 |

### 阶段4：反馈驱动演化（ExemplarBankAgent.update_with_feedback）

| 组件 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| `PrototypeEvolutionManager.evolve_record` | exemplarId + feedback | updated record state | 基于 false positive / false negative / uncertain 做状态切换 |
| `PrototypeQualityController.decide` | updated record | new decision | 重新评估保留、降权或淘汰 |
| `ExemplarMemoryManager.append_audit_log` | feedback event | audit log | 记录持续学习轨迹 |

## 三、工具清单

### 3.1 样本入库工具链

```text
ExemplarBankService
├─ _decode_image_source
├─ _build_record
├─ _score_candidate
├─ _find_duplicate
└─ evaluate_and_store
```

**作用说明：**
- 将前端 workspace 请求转为标准 exemplar 记录
- 先做候选打分，再交给 ExemplarBankAgent 决策
- 保持 API 层与 agent 侧低耦合

### 3.2 样本检索工具链

```text
ExemplarRetrievalPipeline
├─ SemanticProjector
├─ BoundaryEmbeddingHead
├─ MultiScaleFeatureAligner
├─ CrossAttentionReranker（positive / negative / boundary）
├─ GatedRetrievalFusion
└─ RetrievalConfidenceEstimator
```

**作用说明：**
- 对 query feature 做语义、边界和空间对齐
- 分 polarity 检索并重排
- 生成 `prompt_tokens` 与 `retrieval_prior`

### 3.3 样本演化工具链

```text
PrototypeQualityController
├─ score
├─ decide
├─ update_ema_prototype
├─ update_cluster_centroid
└─ curriculum_weight

PrototypeEvolutionManager
└─ evolve_record
```

**作用说明：**
- 根据 exemplar 质量分数做保留/降权/归档
- 根据医生反馈驱动 hard sample memory 演化

## 四、完整执行流程

### 场景1：评估并入库

```text
START
  ↓
Workspace payload（image + segmentation + expertConfig + report）
  ↓
ExemplarBankService._build_record
  ↓
PrototypeQualityController.score
  ↓
PrototypeQualityController.decide
  ↓
ExemplarMemoryManager.deduplicate
  ↓
ExemplarMemoryManager.partition
  ↓
ExemplarMemoryManager.save
  ↓
END（ExemplarBankDecisionSchema）
```

### 场景2：生成 retrieval prior

```text
START
  ↓
ExemplarRetrievalRequestSchema
  ↓
build query semantic / boundary / spatial embedding
  ↓
_select_candidates（positive / negative / boundary）
  ↓
CrossAttentionReranker
  ↓
GatedRetrievalFusion
  ↓
RetrievalConfidenceEstimator
  ↓
RetrievalPackage
  ↓
SAM3 prompt_tokens + retrieval_prior 注入
  ↓
END
```

### 场景3：反馈更新

```text
START
  ↓
医生点击反馈（success / false_negative / false_positive / uncertain）
  ↓
ExemplarBankAgent.update_with_feedback
  ↓
PrototypeEvolutionManager.evolve_record
  ↓
PrototypeQualityController.decide
  ↓
ExemplarMemoryManager.save + append_audit_log
  ↓
END（ExemplarFeedbackResponseSchema）
```

## 五、数据流转

```text
前端 /workspace
  ├─ 评估入库请求
  ├─ retrieval prior 请求
  └─ feedback 请求

Backend ExemplarBankService
  ├─ 构建 exemplar record
  ├─ 调用 ExemplarBankAgent
  └─ 返回 API schema

ExemplarBankAgent
  ├─ ingest -> Memory Snapshot
  ├─ retrieve_prior -> RetrievalPackage
  └─ update_with_feedback -> EvolutionDecision

SAM3Runtime
  ├─ predict_bytes(...)
  ├─ build_retrieval_artifacts(...)
  └─ Sam3TensorForwardWrapper(... exemplar_prompt_tokens, retrieval_prior)
```

## 六、关键输入输出

### 6.1 入库输入

```python
ExemplarBankRequestSchema(
    patient,
    image,
    segmentation,
    expertConfig,
    polarityHint,
    reportMarkdown,
    findings,
    conclusion,
)
```

### 6.2 入库输出

```python
ExemplarBankDecisionSchema(
    sampleId,
    accepted,
    score,
    threshold,
    reasons,
    bankId,
    memoryState,
    qualityBreakdown,
)
```

### 6.3 检索输出

```python
ExemplarRetrievalResponseSchema(
    bankId,
    confidence,
    uncertainty,
    promptTokenShape,
    priorKeys,
    candidateCount,
    candidates,
    diagnostics,
)
```

### 6.4 SAM3 注入输出

```python
RetrievalPackage(
    prompt_tokens,            # [B, N, C]
    retrieval_prior={         # semantic_prototype / spatial_bias_map / decoder_feature_bias_map ...
        ...
    },
    confidence,
    uncertainty,
    selected_candidates,
)
```

## 七、质量决策规则

| 分数区间 | 状态 | 动作 |
|------|------|------|
| `overall >= keep_threshold` | `ACTIVE` | 作为高价值 exemplar 保留 |
| `decay_threshold <= overall < keep_threshold` | `INDEXED` | 保留索引，但权重中性 |
| `prune_threshold <= overall < decay_threshold` | `AGED` | 降权并等待复核 |
| `overall < prune_threshold` | `ARCHIVED` | 归档，不再参与主检索 |

## 八、反馈驱动状态转移

| failure_mode | 目标状态 | 解释 |
|------|------|------|
| `success` | 维持当前状态 | 当前 exemplar 对检索有帮助 |
| `false_negative` | `HARD_POSITIVE` | 模型漏检，需要强化该类正样本 |
| `false_positive` | `HARD_NEGATIVE` | 模型误检，需要强化负样本抑制 |
| `uncertain` | `UNCERTAIN` | 当前 exemplar 存在高不确定性，需要后续复核 |

## 九、实际落地到系统的接口

### Backend API

```text
POST /api/agent/workspace/exemplar-bank
POST /api/agent/workspace/exemplar-bank/retrieve-prior
POST /api/agent/workspace/exemplar-bank/feedback
```

### 运行时注入点

```text
SAM3Engine.predict_bytes(...)
  └─ _build_retrieval_artifacts(...)
      └─ ExemplarBankService.build_retrieval_artifacts(...)
          └─ ExemplarBankAgent.retrieve_prior(...)
              └─ Sam3TensorForwardWrapper(... exemplar_prompt_tokens, retrieval_prior)
```

## 十、当前实现中实际使用的核心模块

```text
agent/agents/exemplar_bank_agent.py
agent/tools/medical/exemplar_bank_schemas.py
agent/tools/medical/exemplar_bank_memory.py
agent/tools/medical/exemplar_bank_quality.py
agent/tools/medical/exemplar_bank_retrieval.py
Backend/app/services/exemplar_bank_service.py
Backend/app/services/sam3_runtime.py
Backend/app/api/endpoints/workspace.py
```

## 十一、推荐理解顺序

1. 先看 `ExemplarBankService` 如何把 workspace payload 转成 exemplar record。
2. 再看 `ExemplarBankAgent.ingest / retrieve_prior / update_with_feedback` 三条主入口。
3. 然后看 `ExemplarRetrievalPipeline` 如何生成 `prompt_tokens + retrieval_prior`。
4. 最后看 `SAM3Runtime.predict_bytes()` 如何把 retrieval prior 注入 `Sam3TensorForwardWrapper`。
