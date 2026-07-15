# MedicalSAM3 脚本文档

## 概述

本目录包含 MedicalSAM3 项目（MedEx-SAM3 系列）的 23 个 Python 脚本，涵盖模型训练、验证、检索推理、数据分析、记忆库策展及持续适应等流程。

## 功能分组

### 1. 训练脚本（4 个）

| 文件 | 功能说明 |
|------|----------|
| `train_lora_medical.py` | 训练 LoRA 适配器，用于 SAM3 模型的高效微调 |
| `train_exemplar_prompt.py` | 训练以示例为提示的检索增强分割模型 |
| `train_rssda.py` | 训练 RSS-DA（检索空间-语义域适应）适配器 |
| `prepare_5fold_polyp.py` | 准备 5 折交叉验证的息肉数据集拆分 |

### 2. 验证与评估脚本（4 个）

| 文件 | 功能说明 |
|------|----------|
| `validate_medex_sam3.py` | 验证 MedEx-SAM3 模型的内外部指标 |
| `validate_rssda.py` | 验证 RSS-DA 适配器的分割性能 |
| `preflight_medex_sam3.py` | 预检 MedEx-SAM3 运行环境与配置 |
| `smoke_medex_scripts.py` | 冒烟测试脚本，验证各模块基本可用性 |

### 3. 检索运行时与推理脚本（2 个）

| 文件 | 功能说明 |
|------|----------|
| `retrieval_runtime.py` | 检索运行时辅助工具：构建运行时、加载库、执行检索与推理 |
| `run_retrieval_inference.py` | 对单张图像或文件夹运行检索条件推理 |

### 4. 记忆库构建与管理脚本（4 个）

| 文件 | 功能说明 |
|------|----------|
| `build_exemplar_bank.py` | 从训练集构建示例记忆库 |
| `build_rssda_bank.py` | 构建 RSS-DA 专用的记忆库 |
| `curate_exemplar_bank_from_delta.py` | 使用验证 delta-Dice 反馈策展示例库 |
| `update_memory_from_review.py` | 根据人工审核结果更新记忆库 |

### 5. 数据分析与报告脚本（6 个）

| 文件 | 功能说明 |
|------|----------|
| `analyze_hard_case_delta.py` | 分析低 Dice 困难案例上的检索增益 |
| `prompt_sensitivity_case.py` | 运行同图不同示例的提示敏感性实验 |
| `report_rssda_behavior.py` | 生成 RSS-DA 行为及域差距的轻量数值报告 |
| `run_ablation.py` | 运行 MedEx-SAM3 消融实验 |
| `summarize_cv_results.py` | 汇总交叉验证和消融实验结果 |
| `select_bank_candidates.py` | 从评估产物中选择平衡的持续库候选 |

### 6. 持续适应脚本（2 个）

| 文件 | 功能说明 |
|------|----------|
| `prepare_continual_adaptation.py` | 挖掘困难案例并构建持续适应输入 |
| `update_memory_from_review.py` | 根据审核结果更新记忆库（见第 4 组） |

### 7. 通用工具脚本（1 个）

| 文件 | 功能说明 |
|------|----------|
| `common.py` | 共享工具函数：配置加载、指标计算、数据集处理、日志等 |

### 8. 包初始化（1 个）

| 文件 | 功能说明 |
|------|----------|
| `__init__.py` | 包初始化文件 |

## 主要类/函数说明（第二批 10 个文件）

### `analyze_hard_case_delta.py`

| 函数 | 说明 |
|------|------|
| `_load_rows(path)` | 从 JSON/JSONL 加载逐图像指标行 |
| `_dice(row, field)` | 提取指定字段的 Dice 系数 |
| `_summarize_subset(rows)` | 汇总子集统计指标（均值、中位数、增益率、抢救率） |
| `_bottom_quantile(rows, quantile)` | 按基线 Dice 取底部分位数子集 |
| `_weighted_hard_case_gain(rows, gamma)` | 计算加权困难案例增益 |
| `_write_csv(path, rows)` | 将困难案例排序后写入 CSV |
| `main()` | 命令行入口，分析困难案例增量指标 |

### `curate_exemplar_bank_from_delta.py`

| 类/函数 | 说明 |
|---------|------|
| `UsageStats` | 记录每个示例项在验证中的使用统计 |
| `_load_bank(path)` | 从文件或目录加载记忆库 |
| `_load_per_image_metrics(path)` | 加载逐图像指标 |
| `_collect_usage_stats(rows)` | 收集每个示例项的使用统计 |
| `_score_item(item, stats)` | 计算示例项的综合得分 |
| `_is_bad_item(stats)` | 判断示例项是否表现不佳 |
| `_protect_top_items(items, stats)` | 保护每类型中得分最高的项 |
| `_write_curated_bank(...)` | 将策展结果写入输出目录 |
| `main()` | 命令行入口，使用验证反馈策展示例库 |

### `prompt_sensitivity_case.py`

| 函数 | 说明 |
|------|------|
| `_load_path_mapping(path)` | 加载图像路径到掩码路径的映射 |
| `_resolve_bbox_for_image(...)` | 解析图像对应的边界框 |
| `_load_mask(mask_path, image_size)` | 加载真值掩码 |
| `_overlay_image(image, mask_logits)` | 将掩码半透明叠加到图像上 |
| `_save_influence_heatmap(path, ...)` | 保存检索影响热图 |
| `_empty_positive_like(retrieval)` | 生成空正例占位符 |
| `_override_retrieval(...)` | 用指定正/负例覆盖检索结果 |
| `_build_variants(...)` | 构建多种检索变体用于敏感性比较 |
| `_run_variant(...)` | 运行单一检索变体的前向推理 |
| `main()` | 命令行入口，运行提示敏感性实验 |

### `report_rssda_behavior.py`

| 函数 | 说明 |
|------|------|
| `_resolve_hidden_dim(model)` | 解析模型隐藏层维度 |
| `_resolve_runtime_device(...)` | 解析运行时设备 |
| `_mask_area_ratio(mask_logits)` | 计算掩码前景区域占比 |
| `_mean_confidence(outputs)` | 计算平均置信度 |
| `_load_checkpoint_payload(path, ...)` | 加载检查点文件 |
| `_maybe_load_rssda_bundle(...)` | 加载 RSS-DA 组件状态字典 |
| `_apply_retrieval_mode(...)` | 应用检索模式过滤 |
| `_dummy_records(...)` | 生成虚拟数据记录 |
| `_ensure_records(...)` | 读取或生成记录 |
| `_create_dummy_bank(...)` | 创建虚拟 RSS-DA 库 |
| `_load_or_create_bank(...)` | 加载或创建 RSS-DA 库 |
| `_entry_source_counts(bank)` | 统计各数据源条目数 |
| `_metadata_readiness(bank)` | 检查元数据完备性 |
| `_selection_from_entries(...)` | 从库中选择条目计算原型 |
| `_rank_entry_indices(...)` | 按策略排序条目 |
| `_override_retrieval(...)` | 覆盖检索结果 |
| `_empty_negative_like(...)` | 生成空负例占位符 |
| `_build_variant_retrievals(...)` | 构建多种检索变体 |
| `_safe_entropy(values)` | 安全计算归一化熵 |
| `summarize_heatmap(tensor, ...)` | 汇总热图统计信息 |
| `_variant_delta(current, baseline)` | 计算指标变化量 |
| `_sensitivity_spread(variants)` | 计算变体极差 |
| `_to_score_list(scores)` | 转换分数张量到列表 |
| `_run_variant(...)` | 运行检索变体前向推理 |
| `_accumulate_variant(...)` | 累加变体指标 |
| `_accumulate_heatmaps(...)` | 累加热图统计 |
| `_average_nested(values, count)` | 递归取均值 |
| `_report_gap(internal, external)` | 计算域差距报告 |
| `_evaluate_split(...)` | 在拆分上评估 RSS-DA 行为 |
| `main()` | 命令行入口，生成 RSS-DA 行为报告 |

### `retrieval_runtime.py`

| 类/函数 | 说明 |
|---------|------|
| `RetrievalBankBackend` | 检索库后端封装类 |
| `RetrievalRuntime` | 检索运行时数据类 |
| `resolve_hidden_dim(model)` | 解析模型隐藏层维度 |
| `apply_retrieval_mode(...)` | 应用检索模式过滤 |
| `parse_bbox(value)` | 解析边界框字符串 |
| `load_bbox_mapping(path)` | 加载边界框映射 |
| `collect_input_images(path)` | 收集输入图像文件 |
| `load_image_tensor(...)` | 加载图像张量 |
| `scale_bbox(...)` | 缩放边界框坐标 |
| `load_rssda_bundle_components(...)` | 加载 RSS-DA 捆绑组件 |
| `_build_bank_backend(...)` | 构建检索库后端 |
| `_backend_cache_key(path)` | 生成后端缓存键 |
| `_resolve_site_bank_root(...)` | 解析站点库根目录 |
| `_default_site_resolution(...)` | 生成默认站点解析 |
| `_resolve_bank_selection(...)` | 解析库选择 |
| `_get_backend(runtime, path)` | 获取后端实例 |
| `_run_backend_retrieval(...)` | 执行后端检索 |
| `resolve_effective_bank(...)` | 解析有效库 |
| `build_retrieval_runtime(...)` | 构建检索运行时 |
| `resolve_retrieval(...)` | 执行完整检索解析 |
| `run_retrieval_forward(...)` | 执行检索前向推理 |
| `infer_query_feature(...)` | 推理查询特征 |

### `run_ablation.py`

| 函数 | 说明 |
|------|------|
| `_dummy_metrics(index, ...)` | 生成虚拟消融指标 |
| `main()` | 命令行入口，运行消融实验 |

### `run_retrieval_inference.py`

| 函数 | 说明 |
|------|------|
| `_overlay_image(image, mask)` | 将掩码叠加到图像上 |
| `_save_preview(...)` | 保存检索示例预览图 |
| `_resolve_bbox_for_image(...)` | 解析图像边界框 |
| `main()` | 命令行入口，运行检索推理 |

### `select_bank_candidates.py`

| 类/函数 | 说明 |
|---------|------|
| `CandidateRecord` | 候选示例记录数据类 |
| `SelectionState` | 选择状态追踪数据类 |
| `_read_rows(path)` | 读取 JSON/JSONL 行数据 |
| `_image_id_from_row(row)` | 提取图像 ID |
| `_merge_missing_fields(...)` | 合并缺失字段 |
| `_merge_artifacts(...)` | 合并多来源行数据 |
| `_safe_float(...)` | 安全转换浮点数 |
| `_safe_dict(...)` | 安全获取字典 |
| `_safe_list(...)` | 安全获取列表 |
| `_normalize_vector(...)` | 向量 L2 归一化 |
| `_vector_from_value(...)` | 从值提取特征向量 |
| `_image_feature_vector(...)` | 计算图像感知特征向量 |
| `_extract_feature_vector(...)` | 从行提取特征向量 |
| `_average_hash(...)` | 计算平均哈希值 |
| `_hamming_distance(...)` | 计算汉明距离 |
| `_patient_id_from_value(...)` | 解析患者 ID |
| `_infer_site_id(row)` | 推断站点 ID |
| `_extract_selected_ids(...)` | 提取选中原型 ID |
| `_retrieval_delta(row)` | 计算检索增量 |
| `_prompt_sensitivity_score(...)` | 提取提示敏感性评分 |
| `_retrieval_influence_strength(...)` | 提取检索影响强度 |
| `_positive_score(row)` | 计算正例评分 |
| `_negative_score(row)` | 计算负例评分 |
| `_is_positive_candidate(...)` | 判断正例候选 |
| `_is_negative_candidate(...)` | 判断负例候选 |
| `_reason_flags(...)` | 生成候选原因标记 |
| `_build_candidate(...)` | 构建候选记录 |
| `_cosine_similarity(...)` | 计算余弦相似度 |
| `_dedup_reason(...)` | 检查去重原因 |
| `_register_candidate(...)` | 注册选中的候选 |
| `_group_candidates(...)` | 按站点分组候选 |
| `_pick_candidate(...)` | 轮询选取候选 |
| `_copy_file_if_exists(...)` | 复制存在的文件 |
| `_copy_dir_if_exists(...)` | 复制存在的目录 |
| `_target_bank_path(...)` | 确定目标库路径 |
| `_selection_reason(...)` | 生成选择原因 |
| `_recommended_priority(...)` | 推荐优先级 |
| `_selection_confidence(...)` | 计算选择置信度 |
| `_load_binary_mask(...)` | 加载二值掩码 |
| `_false_positive_bbox(...)` | 计算假阳性边界框 |
| `_save_crop(...)` | 裁剪保存图像 |
| `_stage_candidate_assets(...)` | 复制候选资产到审核目录 |
| `_candidate_payload(...)` | 生成候选输出载荷 |
| `_write_candidate_outputs(...)` | 写入候选输出 |
| `_write_copy_commands(...)` | 生成复制命令脚本 |
| `select_bank_candidates(...)` | 主选择函数，从评估产物中选择候选 |
| `main()` | 命令行入口 |

### `summarize_cv_results.py`

| 函数 | 说明 |
|------|------|
| `_collect_val_metrics(results_dir)` | 收集各折验证指标 |
| `_collect_external_metrics(eval_dir)` | 收集外部测试指标 |
| `_mean_std(rows)` | 计算均值和标准差 |
| `main()` | 命令行入口，汇总 CV 和消融结果 |

### `prepare_continual_adaptation.py`

| 函数 | 说明 |
|------|------|
| `_normalize_image_key(value)` | 标准化图像键名 |
| `_load_jsonl(path)` | 加载 JSONL 文件 |
| `_records_by_image_id(split_file)` | 按 image_id 索引记录 |
| `_suggest_polarity(record)` | 建议极性 |
| `_mine_hard_cases(...)` | 挖掘困难案例并生成审核清单 |
| `_apply_reviewed_manifest(...)` | 应用审核清单到持续库 |
| `_collect_bank_training_records(...)` | 收集持续库训练记录 |
| `main()` | 命令行入口 |

## 主要类/函数说明（首批 13 个文件，简要列出）

### `__init__.py`
包初始化，导出 `apply_config_overrides`、`compute_segmentation_metrics`、`ensure_dir`、`infer_source_domain`、`load_config`、`log_runtime_environment`、`read_records`、`resolve_feature_map`、`resolve_runtime_device`、`SplitSegmentationDataset` 等。

### `common.py`
共享工具：`SplitSegmentationDataset`（分割数据集）、`read_records`/`write_records`（记录 I/O）、`compute_segmentation_metrics`（指标计算）、`load_config`/`apply_config_overrides`/`dump_config`（配置管理）、`resolve_runtime_device`/`log_runtime_environment`（运行环境）、`resolve_feature_map`（特征图解析）、`ensure_dir`（目录确保）、`collate_batch`（批处理）、`infer_source_domain`（源域推断）。

### `train_lora_medical.py`
`run_fold`：运行单折 LoRA 训练；`main`：命令行入口。

### `train_exemplar_prompt.py`
`_sequential_add(...)`/`_batch_add(...)`：添加示例；`_score_item`：评分；`_protect_domain_diversity`：保护域多样性；`run_fold`/`main`：训练与入口。

### `train_rssda.py`
`_build_fixed_queries`/`_sample_bank_candidates`/`_rssda_step`/`run_fold`/`main`：RSS-DA 训练流程。

### `validate_medex_sam3.py`
`_evaluate_split`/`_report_gap`/`main`：内外部验证与域差距报告。

### `validate_rssda.py`
`_evaluate_split`/`_report_gap`/`main`：RSS-DA 验证与域差距报告。

### `preflight_medex_sam3.py`
`_sam3_import_ready`/`_check_retrieval_cycle`/`_check_model_build`/`main`：环境预检。

### `prepare_5fold_polyp.py`
`_build_site_aware_splits`/`_save_splits`/`main`：准备分层 5 折拆分。

### `build_exemplar_bank.py`
`_coarse_filter_crop`/`_augment_and_extract`/`_build_entry`/`main`：构建示例记忆库。

### `build_rssda_bank.py`
`_coarse_filter_crop`/`_augment_and_extract`/`_build_entry`/`_build_rssda_entries`/`main`：构建 RSS-DA 记忆库。

### `update_memory_from_review.py`
`_copy_reviewed_crops`/`_append_to_bank`/`main`：根据审核结果更新记忆库。

### `smoke_medex_scripts.py`
`_dummy_runtime`/`test_*` 系列/`main`：冒烟测试验证各模块功能。
