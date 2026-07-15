# models/extensions 目录文档

## 目录概述

本目录存放 MedicalSAM3 的扩展子模块，提供提示生成、边界细化、多尺度特征适配与文本引导注意力等能力，用于增强基础分割模型在医学图像上的表现。`__init__.py` 统一导出各扩展组件供上层调用。

## 逐文件说明

### `__init__.py`

- **功能**：扩展子模块的统一导出入口。
- **导出内容**：`AdaptivePromptGenerator`、`BoundaryRefinementHead`、`MultiScaleFeatureAdapter`、`TextGuidedAttention`。

### `apg.py`

- **功能**：自适应提示生成器（Adaptive Prompt Generator），从图像特征预测候选框与点提示，降低模型对外部 prompt 的依赖。
- **主要类与函数**：
  - `AdaptivePromptGenerator`：核心模块。通过 RPN 卷积、分类头、边界框头和点预测头，从特征图生成归一化候选框、点提示及其得分；提供真值框时可计算边界框 L1 损失。
    - `__init__`：构建各预测分支与可学习温度参数。
    - `forward`：输入图像特征，输出包含 `pred_bbox`、`pred_points`、`point_scores`、`cls_map`（及可选 `bbox_loss`）的字典。

### `brh.py`

- **功能**：边界细化头（Boundary Refinement Head），结合粗分割结果、图像上下文与形状先验对边界进行门控细化，并在训练时生成辅助监督目标。
- **主要类与函数**：
  - `_to_logits`：将值域 [0,1] 的掩码转换为 logits，已处于 logit 范围则原样返回。
  - `_boundary_from_binary_mask`：用 Sobel 算子从二值掩码提取边界，并可选最大池化膨胀。
  - `_local_contrast_map`：计算图像局部对比度图，低对比度区域得到更高响应。
  - `_smoothness_prior`：根据概率掩码与局部均值差异计算平滑先验。
  - `_compactness_proxy`：用大核均值池化近似估计掩码紧致度先验。
  - `build_polyp_shape_prior`：融合局部对比度、平滑度与紧致度先验构建息肉形状先验。
  - `build_error_targets`：根据粗分割与真值掩码构建误差区域等辅助训练目标。
  - `BoundaryRefinementHead`：核心模块，包含级联膨胀卷积主干、delta/error 输出头与 Sobel 边界缓冲。
    - `__init__`：构建级联膨胀卷积主干与输出头。
    - `_get_boundary_mask`：从粗分割 logits 经 Sobel 与膨胀得到边界掩码。
    - `build_training_targets`：对齐粗掩码与真值尺寸后构建误差训练目标。
    - `forward`：对粗掩码做边界门控细化，可选返回辅助中间结果。

### `msfa.py`

- **功能**：多尺度特征适配器（Multi-Scale Feature Adapter），为轻量特征支路提供多尺度上下文聚合和通道重标定能力。
- **主要类与函数**：
  - `MultiScaleFeatureAdapter`：核心模块，包含多膨胀率分支、全局池化分支、融合层与通道注意力。
    - `__init__`：构建多尺度膨胀分支、全局池化分支、融合层与通道注意力。
    - `forward`：对输入特征做多尺度聚合、通道注意力加权并叠加残差。

### `tga.py`

- **功能**：文本引导注意力（Text-Guided Attention），用文本嵌入调制图像特征，承担轻量跨模态对齐职责。
- **主要类与函数**：
  - `TextGuidedAttention`：核心模块，包含文本查询投影、交叉注意力、门控与通道注意力。
    - `__init__`：初始化各子模块。
    - `forward`：用文本嵌入通过交叉注意力与门控调制图像特征，输出与输入同形状的调制特征。
