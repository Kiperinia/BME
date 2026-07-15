# models/prompt_adapter 目录文档

## 目录概述

本目录存放检索条件化 SAM3 的提示适配器模块，负责将检索到的正负原型通过门控机制融合到图像特征中，从而对分割解码器进行条件化引导。`__init__.py` 统一导出适配组件。

## 逐文件说明

### `__init__.py`

- **功能**：提示适配器子模块的统一导出入口。
- **导出内容**：`GatedRetrievalFusion`。

### `gated_retrieval_fusion.py`

- **功能**：轻量门控检索融合模块，用正负检索原型经门控策略调制特征图，支持多种检索策略（always-on、similarity-threshold、uncertainty-aware、region-aware、residual）。
- **主要类与函数**：
  - `_ensure_token_weights`：确保返回与 token 批次对齐的权重，缺失时使用均匀权重。
  - `_align_token_tensor`：将输入张量对齐到指定 batch 与 token 数量的形状。
  - `GatedRetrievalFusion`：核心模块，包含查询投影、原型投影、门控、delta 投影及多种策略缓冲。
    - `__init__`：初始化投影、门控、delta 投影子模块及各类策略相关缓冲与可学习参数。
    - `_context_map`：将原型 token 与相似度图聚合为空间上下文特征图。
    - `_summarize_similarity`：将相似度分数或 token 响应汇总为每批次标量相似度。
    - `build_calibration`：根据正负相似度计算置信度、激活状态与缩放等校准量。
    - `build_policy_state`：基于校准量与基线掩码不确定性构建策略门控状态。
    - `forward`：用正负检索原型经门控策略对特征图做条件化融合，返回融合后特征、先验字典与辅助字典。
