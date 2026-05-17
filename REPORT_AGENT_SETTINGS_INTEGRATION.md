# 报告生成 Agent 配置设置页面集成文档

## 概述

已将报告生成 Agent 的关键配置参数集成到前端设置页面（System Settings），用户可以在 Web UI 中直接修改这些参数，无需手动编辑配置文件。

## 修改内容汇总

### 1️⃣ 后端修改 (Backend)

#### 1.1 `Backend/app/core/config.py` - Settings 类添加新字段

新增 4 个配置字段到 `Settings` 类：

```python
# Report generation workflow settings
report_use_llm: bool = Field(default=False, alias="REPORT_USE_LLM")
report_enable_reflection: bool = Field(default=False, alias="REPORT_ENABLE_REFLECTION")
report_reflection_max_iterations: int = Field(default=3, alias="REPORT_REFLECTION_MAX_ITERATIONS", ge=1, le=10)
report_reflection_quality_threshold: float = Field(default=8.0, alias="REPORT_REFLECTION_QUALITY_THRESHOLD", ge=0.0, le=10.0)
```

**环境变量对应：**
- `REPORT_USE_LLM` - 是否使用 LLM 生成报告
- `REPORT_ENABLE_REFLECTION` - 是否启用 ReAct 反思优化
- `REPORT_REFLECTION_MAX_ITERATIONS` - 最大迭代次数（1-10，默认3）
- `REPORT_REFLECTION_QUALITY_THRESHOLD` - 质量阈值（0-10，默认8.0）

#### 1.2 `Backend/app/schemas/system_settings.py` - AgentSettingsSchema 扩展

添加 4 个新字段到 `AgentSettingsSchema`：

```python
class AgentSettingsSchema(BaseModel):
    enableLlm: bool = True
    enableLlmReport: bool = True
    pixelSizeMm: float = Field(default=0.15, gt=0.0, le=10.0)
    # Report generation workflow settings
    useLlmReport: bool = False
    enableReportReflection: bool = False
    reflectionMaxIterations: int = Field(default=3, ge=1, le=10)
    reflectionQualityThreshold: float = Field(default=8.0, ge=0.0, le=10.0)
```

#### 1.3 `Backend/app/services/system_settings_service.py` - 两处修改

**修改 1：_build_payload 方法**

在构建 AgentSettingsSchema 时包含新字段：

```python
agent=AgentSettingsSchema(
    enableLlm=settings.agent_use_llm,
    enableLlmReport=settings.agent_use_llm_report,
    pixelSizeMm=settings.agent_pixel_size_mm,
    useLlmReport=settings.report_use_llm,                              # 新增
    enableReportReflection=settings.report_enable_reflection,          # 新增
    reflectionMaxIterations=settings.report_reflection_max_iterations, # 新增
    reflectionQualityThreshold=settings.report_reflection_quality_threshold, # 新增
),
```

**修改 2：_serialize_runtime_overrides 方法**

在保存运行时配置时包含新字段：

```python
"agent_use_llm": payload.agent.enableLlm,
"agent_use_llm_report": payload.agent.enableLlmReport,
"agent_pixel_size_mm": payload.agent.pixelSizeMm,
"report_use_llm": payload.agent.useLlmReport,                              # 新增
"report_enable_reflection": payload.agent.enableReportReflection,          # 新增
"report_reflection_max_iterations": payload.agent.reflectionMaxIterations, # 新增
"report_reflection_quality_threshold": payload.agent.reflectionQualityThreshold, # 新增
```

### 2️⃣ 前端修改 (Frontend)

#### 2.1 `Frontend/src/types/systemSettings.ts` - 类型定义扩展

添加 4 个新属性到 `AgentSettings` 接口：

```typescript
export interface AgentSettings {
  enableLlm: boolean
  enableLlmReport: boolean
  pixelSizeMm: number
  // Report generation workflow settings
  useLlmReport: boolean
  enableReportReflection: boolean
  reflectionMaxIterations: number
  reflectionQualityThreshold: number
}
```

#### 2.2 `Frontend/src/pages/SystemSettings.vue` - UI 组件新增

在系统设置页面添加新的 section：**"报告生成工作流配置"**

**功能：**
- ✅ 复选框：使用 LLM 生成自然语言报告
- ✅ 复选框：启用 ReAct 反思优化
- ✅ 条件显示：当启用反思优化时，显示参数配置面板
  - 数字输入：最大迭代次数（1-10）
  - 范围滑块：质量满足度阈值（0-10）
- ✅ 实时预估：显示所选配置的预计耗时

**UI 特点：**
- 响应式设计：桌面版两列布局，移动版单列
- 交互反馈：启用/禁用状态下显示不同的信息提示
- 实时计算：根据配置选择动态计算工作流耗时
- 配色方案：
  - 报告生成模式区域：灰色边框
  - 参数配置区域：天蓝色背景（条件显示）
  - 耗时预估区域：琥珀色提示

## 工作流配置方案

### 耗时预估对应表

| 配置 | 耗时 | 质量 | 场景 |
|------|------|------|------|
| 模板 + 无反思 | ~0.3s | ⭐⭐ | 高并发 |
| LLM + 无反思 | ~5s | ⭐⭐⭐ | 需要自然语言 |
| 模板 + ReAct | ~25s | ⭐⭐⭐⭐ | **推荐生产环境** |
| LLM + ReAct | ~35s | ⭐⭐⭐⭐⭐ | 质量优先 |

## API 端点

### 获取系统设置
```
GET /api/system/settings
Response: SystemSettingsResponse
```

### 更新系统设置
```
PUT /api/system/settings
Body: SystemSettingsPayload
Response: SystemSettingsResponse
```

## 使用指南

### 前端用户操作

1. **访问设置页面**
   - 打开应用设置（System Settings）
   - 滚动到"报告生成工作流配置"区域

2. **配置报告生成模式**
   - ☐ 不勾选"使用 LLM 生成"：快速模式（0.3s）
   - ☑ 勾选"使用 LLM 生成"：自然语言模式（5s）

3. **启用反思优化**（可选）
   - ☐ 不勾选：关闭 ReAct 优化
   - ☑ 勾选：启用 ReAct 优化
     - 设置最大迭代次数（建议 3）
     - 设置质量阈值（建议 8.0）

4. **保存配置**
   - 点击页面顶部的"保存设置"按钮
   - 配置立即同步到后端

### 后端集成

后端服务自动读取运行时配置：

```python
from app.core.config import get_settings

settings = get_settings()
use_llm_report = settings.report_use_llm
enable_reflection = settings.report_enable_reflection
max_iterations = settings.report_reflection_max_iterations
quality_threshold = settings.report_reflection_quality_threshold
```

在 Agent 初始化时使用这些配置：

```python
from agent.agents.diagnosis_agent import DiagnosisAgent
from agent.agents.report_reflection_agent import ReportReflectionAgent

agent = DiagnosisAgent(
    use_llm_report=settings.report_use_llm,
    # ... other params
)

if settings.report_enable_reflection:
    reflection_agent = ReportReflectionAgent(
        max_iterations=settings.report_reflection_max_iterations,
        quality_threshold=settings.report_reflection_quality_threshold,
    )
```

## 配置文件持久化

### 运行时配置保存位置

配置被保存到 `Backend/runtime/system_settings.json`：

```json
{
  "agent_use_llm": true,
  "agent_use_llm_report": true,
  "agent_pixel_size_mm": 0.15,
  "report_use_llm": false,
  "report_enable_reflection": false,
  "report_reflection_max_iterations": 3,
  "report_reflection_quality_threshold": 8.0,
  "...": "其他配置项"
}
```

### 环境变量覆盖

如果设置了对应的环境变量，将覆盖配置文件的值：

```bash
# .env 文件或系统环境变量
REPORT_USE_LLM=true
REPORT_ENABLE_REFLECTION=true
REPORT_REFLECTION_MAX_ITERATIONS=5
REPORT_REFLECTION_QUALITY_THRESHOLD=7.5
```

## 验证检查清单

- [x] 后端 Settings 类新增 4 个字段
- [x] 后端 AgentSettingsSchema 新增 4 个字段
- [x] 后端 _build_payload 方法更新
- [x] 后端 _serialize_runtime_overrides 方法更新
- [x] 前端 systemSettings.ts 类型定义更新
- [x] 前端 SystemSettings.vue 新增 UI section
- [x] HTML 结构正确（开标签与闭标签匹配）
- [x] 前后端数据字段对应一致

## 数据流向

```
前端 UI 修改
    ↓
SystemSettings.vue 收集表单数据
    ↓
updateSystemSettings() API 调用
    ↓
后端 PUT /api/system/settings
    ↓
SystemSettingsService._serialize_runtime_overrides()
    ↓
Backend/runtime/system_settings.json 保存
    ↓
Settings 类读取配置
    ↓
Agent 初始化时使用新配置
```

## 常见问题

### Q: 配置修改后何时生效？
A: 配置立即保存，但需要重新启动 Agent 或诊断流程才能应用新配置。

### Q: 能否同时启用两个工作流？
A: 后端会按优先级选择：
1. 如果 `enableReportReflection=true`，使用完整 ReAct 工作流
2. 否则，根据 `useLlmReport` 选择 LLM 或模板模式

### Q: 配置丢失了怎么办？
A: 
1. 检查 `Backend/runtime/system_settings.json` 是否存在
2. 检查环境变量是否被重置
3. 使用"恢复已保存"按钮恢复到上次保存的配置

### Q: 如何重置为默认配置？
A: 删除 `Backend/runtime/system_settings.json` 文件，系统将使用代码中的默认值。

## 关联文档

- [报告生成 Agent 工作流详解](./agent/REPORT_WORKFLOW_SUMMARY.md)
- [工作流方案对比](./agent/WORKFLOW_SOLUTION_COMPARISON.md)
- [快速参考](./agent/REPORT_WORKFLOW_QUICK_REF.md)
