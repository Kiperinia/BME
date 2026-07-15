# exemplar 模块文档

## 目录概述

`exemplar` 目录是 MedEx-SAM3 项目中的**示例记忆与原型**子包，负责示例（exemplar）的编码、存储、检索、采样、原型构建、评分以及示例感知损失的计算。其核心目标是通过对人工校验示例的管理与利用，增强模型在医学图像分割任务中的判别能力与跨域泛化能力，同时支持对外部数据集泄漏的防护。

该目录包含以下模块：损失函数（`losses.py`）、记忆库（`memory_bank.py`）、原型构建（`prototype_builder.py`）、采样（`sampler.py`）、示例编码器（`exemplar_encoder.py`）、评分（`curator.py`），并通过 `__init__.py` 统一对外导出。

## 逐文件说明

### `__init__.py`

**文件功能：** 包初始化文件，汇总并对外导出 exemplar 子包中的主要类与函数，便于上层以 `from exemplar import ...` 的方式统一引用。

**导出内容：**
- 损失类：`BoundaryBandDiceLoss`、`CrossDomainConsistencyLoss`、`ExemplarConsistencyLoss`、`ExemplarInfoNCELoss`、`NegativeSuppressionLoss`、`PrototypeVarianceLoss`、`SoftHausdorffLoss`
- 编码器：`ExemplarEncoder`
- 记忆库：`ExemplarItem`、`ExemplarMemoryBank`
- 采样器：`ExemplarSampler`
- 原型构建器：`PrototypeBuilder`
- 评分：`ExemplarScoreBreakdown`、`compute_exemplar_score`

---

### `losses.py`

**文件功能：** 提供 MedEx-SAM3 训练所用的示例感知损失函数集合，包括对比损失、边界损失、一致性损失等，以及一个统一的损失组合器。

**主要类与函数：**

- **`_soft_dice(mask_a, mask_b, eps)`**（模块级函数）：计算两个掩码之间的 Soft Dice 系数，支持软化概率输入。
- **`_boundary_band(mask)`**（模块级函数）：通过最大池化膨胀与腐蚀之差提取掩码的边界带区域。
- **`ExemplarInfoNCELoss`**：基于 InfoNCE 的示例对比损失模块，将锚点嵌入拉近正例、推远负例。
  - `__init__(temperature)`：初始化温度系数。
  - `forward(anchor_embedding, positive_embedding, negative_embeddings)`：计算对比损失。
- **`NegativeSuppressionLoss`**：负例提示抑制损失，鼓励负例提示对应的掩码预测趋近于 0。
  - `forward(negative_prompt_mask_logits)`：返回 sigmoid 后的均值损失。
- **`CrossDomainConsistencyLoss`**：跨域一致性损失，以余弦相似度约束锚点与原型嵌入方向一致。
  - `forward(anchor_embedding, prototype_embedding)`：返回 1 - 余弦相似度均值。
- **`ExemplarConsistencyLoss`**：示例一致性损失，以 Soft Dice 衡量两份掩码预测的一致性。
  - `forward(mask_logits_a, mask_logits_b)`：返回 1 - Soft Dice 均值。
- **`PrototypeVarianceLoss`**：原型方差损失，约束样本到原型的距离均值不超过裕度。
  - `__init__(margin)`：初始化距离裕度。
  - `forward(embeddings, prototype)`：返回经 ReLU 截断的方差损失。
- **`BoundaryBandDiceLoss`**：边界带 Dice 损失，仅在边界带区域上计算 Dice，强化边界刻画。
  - `forward(pred_logits, target_mask)`：返回 1 - 边界 Soft Dice 均值。
- **`SoftHausdorffLoss`**：软 Hausdorff 损失，对掩码做均值平滑后取绝对误差均值作为可微近似。
  - `forward(pred_logits, target_mask)`：返回平滑后掩码的绝对误差均值。
- **`MedExLossComposer`**：多项损失组合器，将 BCE、Dice、边界、对比、负例抑制、一致性按权重加权合成总损失。
  - `__init__(w_bce, w_dice, w_boundary, w_contrast, w_neg, w_consistency)`：初始化各项权重与子损失模块。
  - `forward(...)`：计算总损失并返回各项明细字典。

---

### `memory_bank.py`

**文件功能：** 实现带版本管理与变更日志的人工校验示例记忆库，支持条目的增删改查、磁盘持久化、外部数据集泄漏校验等功能。

**主要类与函数：**

- **`ExemplarItem`**（数据类）：描述单条示例的完整元信息，包括来源、路径、嵌入路径、类型、质量评分、版本等字段。
- **`ExemplarMemoryBank`**：示例记忆库主类。
  - `__init__(items)`：初始化条目列表、拒绝列表、变更日志与版本号。
  - `trainable_items`（属性）：返回所有人工校验通过的条目。
  - `load(path)`（类方法）：从文件或目录加载记忆库，目录时自动选取最新版本文件。
  - `_record_change(action, item_id, details)`：在变更日志中追加操作记录。
  - `_validate_item(item, external_dataset_names)`：校验条目是否包含外部数据集泄漏。
  - `add_item(item)`：添加条目（同 ID 先删后加）。
  - `remove_item(item_id)`：按 ID 移除条目。
  - `get_items(type, source_dataset, human_verified)`：按条件筛选条目。
  - `reject_item(item_id, reason)`：将条目移入拒绝列表并记录原因。
  - `_next_version_path(directory)`：生成下一个版本的文件路径并更新版本号。
  - `save(path)`：将记忆库、拒绝列表、变更日志持久化到磁盘。
  - `export_changelog(path)`：单独导出变更日志。
  - `check_no_external_leakage(external_dataset_names)`：检查是否存在外部数据集泄漏。

---

### `prototype_builder.py`

**文件功能：** 提供多种原型构建策略（均值、加权、注意力融合、聚类子原型），并支持基于方差判断的正/负/边界三类原型的端到端构建。

**主要类与函数：**

- **`PrototypeBuilder`**：原型构建器主类。
  - `__init__(variance_threshold)`：初始化方差拒绝阈值。
  - `build_mean_prototype(embeddings)`（静态方法）：构建 L2 归一化的均值原型。
  - `build_weighted_prototype(embeddings, scores)`（静态方法）：构建 softmax 分数加权原型，返回原型与权重。
  - `build_attention_fused_prototype(query_embedding, exemplar_embeddings)`（静态方法）：以相似度为注意力权重构建融合原型。
  - `build_clustered_subprototypes(embeddings, n_clusters)`（静态方法）：通过 k-means 风格聚类生成多个子原型。
  - `compute_prototype_variance(embeddings, prototype)`（静态方法）：计算嵌入到原型的距离平方均值。
  - `reject_if_high_variance(variance, threshold)`：判断方差是否过高并记录警告。
  - `_load_embedding(item)`（静态方法）：从磁盘加载示例嵌入，支持张量与字典格式。
  - `_item_score(query, embedding, item)`（静态方法）：综合相似度与分项评分计算单条示例得分。
  - `_build_single_type(query, items, top_k)`：为单一类型构建原型（评分选取 top_k、加权、方差校验、聚类降级）。
  - `build_positive_negative_boundary_prototypes(query, memory_bank, top_k)`：为正例、负例、边界三类分别构建原型。

---

### `sampler.py`

**文件功能：** 提供示例检索中的采样辅助工具，支持按类型采样与均衡采样，并使用可设置随机种子保证可复现性。

**主要类与函数：**

- **`ExemplarSampler`**：示例采样器主类。
  - `__init__(seed)`：初始化带固定种子的随机数发生器。
  - `sample_by_type(memory_bank, exemplar_type, count, human_verified)`：按类型随机采样指定数量条目，请求数量不少于候选时返回全部。
  - `sample_balanced(memory_bank, positive_count, negative_count, boundary_count)`：均衡采样正例、负例、边界三类条目，返回分类字典。

---

### `exemplar_encoder.py`

**文件功能：** 实现示例编码器，将裁剪图像（及可选掩码）编码为全局、前景、边界、背景四类嵌入，供后续原型构建与对比学习使用。

**主要类与函数：**

- **`_l2_normalize(x)`**（模块级函数）：对张量最后一维做 L2 归一化。
- **`_masked_average_pool(feature_map, mask)`**（模块级函数）：在掩码区域内对特征图做平均池化。
- **`_boundary_band(mask)`**（模块级函数）：通过卷积膨胀与腐蚀之差提取边界带。
- **`ExemplarEncoder`**（`nn.Module`）：示例编码器主类。
  - `__init__(embed_dim, backbone)`：初始化嵌入维度与可选外部骨干网络、内置 stem 与投影层。
  - `_encode_feature_map(crop_image)`：提取特征图，优先使用外部骨干，失败则回退到内置 stem。
  - `forward(crop_image, crop_mask)`：返回包含 `global_embedding`、`foreground_embedding`、`boundary_embedding`、`context_embedding` 的字典；无掩码时各类嵌入退化为全局嵌入。

---

### `curator.py`

**文件功能：** 提供示例评分的共享辅助工具，基于多维度分项加权得到示例综合评分，用于示例筛选与排序。

**主要类与函数：**

- **`ExemplarScoreBreakdown`**（数据类）：汇总示例在掩码质量、边界质量、困难样本价值、多样性增益、域迁移价值、假阳性风险等维度的分项评分。
  - `score`（属性）：基于各分项加权计算综合评分。
- **`compute_exemplar_score(mask_quality, boundary_quality, hard_case_value, diversity_gain, domain_shift_value, false_positive_risk)`**（模块级函数）：根据各维度分项计算加权综合评分，其中假阳性风险作为惩罚项。
