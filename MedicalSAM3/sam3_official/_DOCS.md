# sam3_official 目录文档

## 目录概述

`sam3_official` 是 MedicalSAM3 仓库下的一个子包，负责将**官方 SAM3 图像模型**集成进 MedEx-SAM3 工作流。
它提供统一的模型构建、冻结/解冻、参数统计接口，并以张量化的方式（tensor-native forward）封装前向推理，
同时支持在官方 SAM3 不可用时回退到自带的占位（Dummy）模型，便于在无权重或无 GPU 环境下做功能验证。

该子包主要解决以下问题：

- 调用官方 `sam3.model_builder.build_sam3_image_model` 构建图像模型，并处理权重路径解析、数据类型适配、
  CUDA 运行时库预加载、构建诊断报告等工程细节。
- 在官方模型缺失或构建失败时，使用轻量级占位模型 `DummyOfficialSam3ImageModel` 保证接口可用。
- 通过 `Sam3TensorForwardWrapper` 统一官方模型与占位模型的前向调用，输出标准化的掩码、得分及中间特征字典。
- 支持将检索先验（retrieval prior）注入到编码器记忆与掩码 logits，用于增强分割质量。
- 提供前向特征钩子与模块检查工具，辅助 LoRA 训练目标的选取与中间特征分析。

---

## 逐文件说明

### 1. `__init__.py`

**文件功能**：子包初始化文件，声明包级 docstring 并对外暴露公共 API。

**导出的主要对象**：

- `build_official_sam3_image_model`：构建官方 SAM3 图像模型（来自 `build_model`）。
- `count_trainable_parameters`：统计可训练参数数量与占比。
- `freeze_model`：冻结模型全部参数。
- `print_trainable_parameters`：打印并记录可训练参数统计。
- `unfreeze_by_keywords`：按关键词解冻参数。
- `Sam3TensorForwardWrapper`：张量化前向包装器（来自 `tensor_forward`）。

---

### 2. `build_model.py`

**文件功能**：构建并检查官方 SAM3 图像模型，提供权重解析、数据类型适配、CUDA 运行时预热、
构建诊断报告生成，以及官方构建失败时的占位回退模型实现。

**模块级常量**：

- `_DEFAULT_LOCAL_CHECKPOINTS`：默认本地权重文件名（`sam3.pt`、`MedSAM3.pt`）。
- `_DEFAULT_BUILD_REPORT`：默认构建报告输出路径。
- `_CUDA_RUNTIME_PRIMED`：CUDA 运行时库是否已预加载的全局标记。

**主要函数**：

- `_prime_cuda_runtime_libraries(device)`：在 Linux 上预加载 CUDA 运行时库（libnvrtc），避免运行期加载失败。
- `_resolve_dtype(dtype)`：将数据类型字符串（如 `fp16`、`bf16`）解析为 `torch.dtype`。
- `_find_default_local_checkpoint()`：在默认 checkpoint 目录中查找本地权重文件。
- `_resolve_checkpoint_path(checkpoint_path)`：解析权重路径，支持用户路径、仓库相对路径与默认查找。
- `_dtype_name(dtype)`：将 `torch.dtype` 反向映射为可读字符串。
- `_resolve_runtime_dtype(device, dtype)`：根据运行设备解析实际数据类型（CPU 强制回退 fp32）。
- `_default_hidden_dim(model)`：从模型推断默认隐藏维度。
- `_annotate_model(model, *, used_official_sam3, used_dummy_fallback)`：为模型附加 MedEx 元数据标注属性。
- `_build_model_report(...)`：构建模型构建过程的诊断报告字典。
- `_write_model_report(report, report_path)`：将构建报告以 JSON 写入文件。
- `_move_model(model, device, dtype)`：按运行时数据类型将模型移动到指定设备。
- `_reset_official_runtime_caches(model)`：清除官方模型模块中的坐标/尺寸缓存。
- `_move_official_model(model, device)`：将官方模型移动到设备并重置运行时缓存。
- `_build_from_official_builder(checkpoint_path, device, dtype)`：通过反射官方构建器签名构建模型。
- `build_official_sam3_image_model(...)`：**核心入口**，构建官方 SAM3 图像模型，失败时可选回退到占位模型，并生成报告。
- `freeze_model(model)`：冻结模型全部参数。
- `unfreeze_by_keywords(model, keywords)`：按关键词解冻匹配参数。
- `count_trainable_parameters(model)`：统计可训练参数数量及占比。
- `print_trainable_parameters(model)`：打印并记录可训练参数统计。

**主要类（占位回退模型组件）**：

- `DummySelfAttention`：占位自注意力模块（q/k/v/out 投影），可用作交叉注意力。
- `DummyMLP`：占位两层 MLP（GELU 激活）。
- `DummyTransformerBlock`：占位 Transformer 块（自注意力、可选交叉注意力、MLP）。
- `DummyEncoder`：占位编码器，由若干 Transformer 块堆叠而成。
- `DummyPromptEncoder`：占位提示编码器，将框/点/文本/示例提示编码为嵌入。
- `DummyMaskDecoder`：占位掩码解码器，由 token 与特征图生成掩码及得分。
- `DummyOfficialSam3ImageModel`：官方 SAM3 图像模型的完整占位回退实现，含 stem、图像编码器、
  检测编码器/解码器、提示编码器、掩码解码器与文本编码器，提供 `tensor_forward` 接口。

---

### 3. `tensor_forward.py`

**文件功能**：提供官方/占位 SAM3 图像模型的张量化前向包装器，统一前向接口、图像预处理、
提示构造、检索先验注入与中间特征采集，输出标准化结果字典。

**模块级常量**：

- `HAS_OFFICIAL_SAM3_RUNTIME`：是否成功导入官方 SAM3 运行时依赖的标志。
- `DEFAULT_TENSOR_FORWARD_REPORT`：默认冒烟测试报告输出路径。

**主要函数**：

- `_to_mask_logits(masks)`：将概率掩码转换为 logits。
- `_mean_tensor_from_feature_map(features, key_hint)`：从嵌套特征字典中按名称提示查找首个张量。
- `_is_official_sam3_model(model)`：判断模型是否为官方 SAM3 图像模型。
- `_ensure_text_prompt(text_prompt, batch_size)`：确保文本提示列表长度与批大小一致。
- `_model_device(model)`：获取模型所在设备。
- `_infer_official_resolution(model)`：从官方模型 backbone 推断期望输入分辨率。
- `_normalize_xyxy_boxes(boxes, height, width)`：将 xyxy 框归一化到 [0,1]。
- `_normalize_xy_points(points, height, width)`：将点坐标归一化到 [0,1]。
- `_resize_spatial_bias_to_tokens(spatial_bias, token_count)`：将空间偏置图缩放为 token 序列。
- `_resize_feature_bias_to_tokens(feature_bias, token_count)`：将特征偏置图缩放为 token 序列。
- `_add_token_bias(memory, token_bias)`：在形状兼容时将 token 偏置叠加到 memory。
- `_apply_retrieval_prior_to_memory(memory, retrieval_prior)`：将检索先验各类偏置应用到编码器记忆。
- `_apply_retrieval_prior_to_mask_logits(mask_logits, retrieval_prior)`：将掩码 logit 偏置应用到掩码 logits。
- `run_tensor_forward_smoke_test(...)`：运行张量化前向冒烟测试并生成报告。

**主要类**：

- `Sam3TensorForwardWrapper(nn.Module)`：核心包装器。统一官方与占位模型的前向调用，
  负责图像预处理、提示构造、检索先验注入与中间特征采集。主要方法：
  - `__init__`：初始化包装器，构建模型并按需配置预处理与特征钩子。
  - `_preprocess_official_images`：使用官方 `Sam3Processor` 预处理图像。
  - `_build_find_stage`：构建官方 SAM3 所需的 `FindStage` 提示容器。
  - `_call_official_model`：调用官方 SAM3 完成完整前向推理。
  - `_call_model`：根据模型类型分发到官方或占位模型前向。
  - `forward`：执行张量化前向并对输出做标准化后处理。

---

### 4. `feature_hooks.py`

**文件功能**：提供前向钩子辅助工具，用于提取并管理 SAM3 模块的中间特征。

**主要函数**：

- `_detach_tensor(value)`：将张量从计算图分离。
- `_sanitize_output(output)`：递归对输出对象中的张量进行分离处理。
- `register_feature_hooks(model, keywords, max_hooks)`：按关键词为模型子模块批量注册前向特征钩子。

**主要类**：

- `FeatureHookManager`：管理前向钩子的注册与特征收集。主要方法：
  - `__init__`：初始化句柄列表与特征字典。
  - `add(model, module_name)`：在指定子模块上注册前向钩子。
  - `clear()`：清空已收集的特征字典（保留钩子）。
  - `remove()`：移除所有钩子并清空特征。

---

### 5. `module_inspector.py`

**文件功能**：检查官方或占位 SAM3 模块结构，并据此建议 LoRA 插入位置，输出模块清单与 LoRA 目标文件。

**模块级常量**：

- `DEFAULT_SCOPE_ALIASES`：功能范围到别名列表的默认映射（vision_encoder、detector_encoder 等）。
- `DEFAULT_KEYWORDS`：模块查找时使用的默认关键词列表。
- 各 `DEFAULT_*` 路径常量：默认输出文件位置。

**主要函数**：

- `_module_parameter_count(module)`：统计单个模块自身（不含子模块）的参数总量。
- `_parse_block_index(name)`：从模块限定名解析 Transformer 块索引。
- `_collect_scope_depths(model, scope_aliases)`：收集各功能范围内块的最大深度。
- `classify_scope(name, scope_aliases)`：根据模块名称归类到功能范围。
- `_candidate_kind(name)`：判断 LoRA 候选层类型（attention_projection/mlp/prompt_projection/linear）。
- `_select_default_stages(name, scope, block_index, scope_depths)`：为候选层选择默认训练阶段标签。
- `list_named_modules(model, save_path, scope_aliases)`：列出所有命名子模块的元信息，可保存为 json/txt。
- `find_modules_by_keywords(model, keywords)`：按关键词查找匹配子模块名称。
- `suggest_lora_targets(model, scope_aliases)`：为 Linear 层建议 LoRA 插入目标。
- `write_inspection_outputs(modules, lora_targets)`：将检查结果写入默认输出文件。
- `main()`：命令行入口，构建模型并导出模块检查与 LoRA 目标结果。
