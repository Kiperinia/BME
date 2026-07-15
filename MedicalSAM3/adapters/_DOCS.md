# adapters 目录文档

## 目录概述

`adapters/` 目录是 MedEx-SAM3 模型的**适配器模块集合**，为 SAM3 基础模型提供多种轻量化、可插拔的特征增强与参数高效微调能力。该目录下的模块围绕医学图像分割场景设计，涵盖以下几类能力：

- **参数高效微调**：通过 LoRA 低秩注入，在冻结基座权重的前提下训练少量适配参数。
- **医学特征适配**：基于瓶颈残差结构的特征变换，并可选叠加深度可分离卷积以增强纹理表达。
- **边界感知精炼**：从掩码或对比度先验提取边界信息，门控融合到图像特征中以强化组织边界。
- **示例提示生成**：将正/负/边界示例原型投影为提示 token，驱动分割解码器。
- **空间-语义检索融合**：将检索得到的原型与相似度先验，通过空间偏置与门控机制注入分割特征，实现基于检索的域自适应增强。

各适配器通过 `__init__.py` 统一对外导出，上层模型可按需组合使用。

---

## 逐文件说明

### 1. `__init__.py`

**文件功能**：适配器包的入口文件，负责统一导出本目录下所有适配器类与 LoRA 工具函数，供外部模块按需导入。

**导出内容**：
- 适配器类：`BottleneckAdapter`、`MedicalImageAdapter`、`BoundaryAwareAdapter`、`ExemplarPromptAdapter`、`RetrievalSpatialSemanticAdapter`
- LoRA 相关：`LoRAConfig`、`LoRALinear` 及 `apply_lora_to_model`、`get_lora_state_dict`、`is_target_module`、`load_lora_weights`、`mark_only_lora_as_trainable`、`merge_lora_weights`、`replace_linear_with_lora`、`save_lora_weights` 等函数

---

### 2. `lora.py`

**文件功能**：提供 LoRA（Low-Rank Adaptation）低秩适配的完整工具链，包括配置定义、线性层替换、目标模块筛选、权重保存/加载/合并以及注入报告生成，用于在 SAM3 模型的指定作用域与训练阶段中高效注入可训练增量。

**主要类与函数**：

| 名称 | 类型 | 作用 |
|------|------|------|
| `LoRAConfig` | dataclass | LoRA 注入配置，包含秩 `rank`、缩放 `alpha`、`dropout`、目标模块/作用域列表、排除关键字、训练阶段 `stage` 等参数 |
| `LoRALinear` | nn.Module | 在已有 `nn.Linear` 上注入低秩适配的核心模块；冻结基座权重，新增 `lora_A`、`lora_B` 两个低秩矩阵，前向输出为基座输出加缩放低秩增量；支持 `merge()` 将增量合并回基座权重 |
| `is_target_module` | 函数 | 判断给定模块是否满足 LoRA 注入条件（线性层、不在排除列表、作用域允许、名称命中目标关键字） |
| `apply_lora_to_model` | 函数 | 主注入入口：加载/生成目标清单、按阶段规则筛选候选、逐个替换为 `LoRALinear`、统计可训练参数并写入注入报告 |
| `replace_linear_with_lora` | 函数 | 将指定名称的 `nn.Linear` 替换为 `LoRALinear` |
| `mark_only_lora_as_trainable` | 函数 | 将模型中仅 LoRA 参数（`lora_A`/`lora_B`）标记为可训练，其余冻结 |
| `get_lora_state_dict` | 函数 | 提取模型中所有 LoRA 参数的状态字典 |
| `save_lora_weights` / `load_lora_weights` | 函数 | 保存/加载 LoRA 权重；加载时支持前缀归一化与严格匹配控制 |
| `merge_lora_weights` | 函数 | 将模型中所有 `LoRALinear` 的低秩增量合并回基座权重 |

**内部辅助函数**：`_infer_scope_from_name`（推断作用域）、`_parse_block_index`（解析 block 索引）、`_collect_scope_block_depths`（统计作用域深度）、`_matches_stage_rule`（阶段匹配规则）、`_scope_allowed`（作用域许可判断）、`_load_target_catalog`（目标清单加载/生成）、`_write_lora_report`（写入报告）、`_get_parent_module`（获取父模块）、`_extract_state_dict`（提取状态字典）、`_strip_known_prefixes`（去除已知前缀）、`_normalize_lora_state_dict`（键名归一化）。

---

### 3. `medical_adapter.py`

**文件功能**：定义医学图像特征适配器，通过瓶颈残差结构对特征进行降维-升维变换，并可选叠加深度可分离卷积以增强局部纹理表达。

**主要类**：

| 名称 | 类型 | 作用 |
|------|------|------|
| `BottleneckAdapter` | nn.Module | 瓶颈结构特征适配器：LayerNorm → 降维 → GELU → Dropout → 升维，并以可学习缩放因子 `scale` 做残差融合；前向自动适配 `[B, N, C]` 序列与 `[B, C, H, W]` 图像两种形态 |
| `MedicalImageAdapter` | nn.Module | 医学图像适配器：内部组合 `BottleneckAdapter`，并在四维图像输入时叠加深度可分离卷积（depthwise + pointwise）纹理分支，输出为两者之和 |

---

### 4. `boundary_adapter.py`

**文件功能**：实现边界感知特征精炼模块，从真值掩码、粗预测掩码或特征对比度先验中提取边界图，再通过门控机制将边界信息注入图像特征，增强分割模型对组织边界的刻画。

**主要类与函数**：

| 名称 | 类型 | 作用 |
|------|------|------|
| `_boundary_from_mask` | 函数 | 通过 3×3 卷积实现腐蚀与膨胀，取差集得到边界带，并可按 `dilation` 参数膨胀边界宽度 |
| `_contrast_prior` | 函数 | 从特征图通道均值的局部标准差计算归一化对比度先验，作为无掩码时的边界近似 |
| `BoundaryAwareAdapter` | nn.Module | 边界感知适配器：包含边界编码器（双层卷积）、边界头（1×1 卷积生成边界 logits）、门控头（融合图像与边界特征生成 sigmoid 门控）和瓶颈适配器；前向输出增强特征与辅助信息（边界图、门控图，及有真值时的边界损失） |

---

### 5. `exemplar_prompt_adapter.py`

**文件功能**：实现原型到提示 token 的投影模块，将正例、负例、边界示例原型分别投影为多 token 序列，并通过门控融合生成供分割解码器使用的提示 token。

**主要类与函数**：

| 名称 | 类型 | 作用 |
|------|------|------|
| `_reduce_proto` | 函数 | 将 `[B, K, C]` 原型沿 K 维取均值压缩为 `[B, C]` 向量，`[B, C]` 直接返回 |
| `_project_tokens` | 函数 | 调用 token 投影器将原型投影为多 token 序列，支持二维与三维原型输入 |
| `_TokenProjector` | nn.Module | token 投影器：两层 MLP（Linear → GELU → Linear）将 `[B, C]` 向量投影并重塑为 `[B, num_tokens, C]` |
| `ExemplarPromptAdapter` | nn.Module | 示例提示适配器：内含正/负/边界三个 `_TokenProjector` 与一个融合门（由查询、正、负、边界摘要生成 4 维 sigmoid 门控）；前向输出融合后的提示 token 序列与辅助信息（各类 token、融合权重、抑制门） |

---

### 6. `retrieval_spatial_semantic_adapter.py`

**文件功能**：实现空间-语义检索适配器（RSS-DA），将检索得到的正/负原型与相似度先验，通过空间偏置卷积分支与门控检索融合模块注入分割特征，支持多种融合模式（联合、仅空间、仅语义等），实现基于检索的域自适应增强。

**主要类**：

| 名称 | 类型 | 作用 |
|------|------|------|
| `RetrievalSpatialSemanticAdapter` | nn.Module | 空间-语义检索适配器：包含空间偏置卷积分支（由正/负热力图生成空间偏置）、可学习空间缩放因子 `spatial_scale`，以及核心的 `GatedRetrievalFusion` 门控融合模块；前向根据 `mode` 决定启用空间/语义路径，输出增强特征、检索先验字典与包含大量诊断指标的辅助信息字典 |

**关键参数**：
- `positive_weight` / `negative_weight`：正/负原型融合权重
- `similarity_threshold` / `similarity_weighting` / `similarity_temperature`：相似度激活与加权控制
- `retrieval_policy` / `uncertainty_threshold` / `uncertainty_scale`：不确定性感知检索策略参数
- `policy_activation_threshold` / `residual_strength`：策略激活比例与残差强度
- `mode`：融合模式，可选 `joint`/`spatial`/`semantic`/`positive-only`/`negative-only`/`positive-negative`
