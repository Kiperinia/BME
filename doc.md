# MedEx-SAM3 模型改进说明文档

## 1. 文档目的

本文档用于说明本项目在原始 SAM3 模型基础上进行的医学图像分割适配改进，重点描述模型结构、训练策略、Exemplar 增强机制、边界精修模块、检索增强模块以及实验工程化设计。

本项目的模型改进目标不是重新设计一个完整分割网络，而是在官方 SAM3 的基础上，通过参数高效微调、医学特征适配、边界感知增强和 Exemplar 检索提示机制，使其更适合内窥镜息肉分割任务。

## 2. 原始 SAM3 基础

原始 SAM3 作为通用分割基础模型，具备图像编码、文本/几何提示编码、检测解码和掩膜生成能力。其核心优势是通用视觉语义理解能力较强，但直接用于医学内窥镜图像时仍存在以下问题：

1. 医学图像域与自然图像域存在明显差异；
2. 息肉区域边界模糊，易受反光、黏液、褶皱、阴影影响；
3. 小目标和低对比病灶对通用分割模型不友好；
4. 单纯 box/text prompt 难以表达医学样本间的形态相似性；
5. 外部泛化测试时，模型容易出现假阳性或边界漂移。

因此，本项目在保留 SAM3 主干能力的前提下，构建了一个面向医学息肉分割的 MedEx-SAM3 改进框架。

## 3. 总体改进架构

MedEx-SAM3 的整体设计可以概括为：

> 官方 SAM3 基座 + LoRA 参数高效微调 + 医学特征 Adapter + 边界感知精修 + Exemplar Prompt + 检索空间语义增强 + 医学分割损失函数

项目中新增了 `sam3_official/build_model.py` 和 `sam3_official/tensor_forward.py`，用于将官方 SAM3 image model 包装成可直接进行 tensor 输入输出的训练/推理接口。该包装层支持官方 SAM3 构建、dummy fallback、模型构建报告、dtype 管理、forward smoke test 等功能。

`Sam3TensorForwardWrapper` 对官方 SAM3 的图像输入、文本 prompt、box prompt、point prompt、visual/exemplar prompt、encoder、decoder 和 segmentation head 进行了统一封装，使后续医学模块可以稳定接入 SAM3 推理链路。

## 4. LoRA 参数高效微调

### 4.1 改进动机

原始 SAM3 参数规模较大，如果直接全量微调，会带来以下问题：

1. 显存消耗高；
2. 训练成本高；
3. 小规模医学数据集容易过拟合；
4. 不利于多折交叉验证和快速消融实验。

因此，本项目采用 LoRA 进行参数高效微调，只训练少量低秩矩阵，在尽量保留 SAM3 原始通用能力的同时完成医学域适配。

### 4.2 实现方式

项目中实现了 `LoRAConfig`、`LoRALinear`、`apply_lora_to_model`、`mark_only_lora_as_trainable`、`save_lora_weights`、`load_lora_weights` 和 `merge_lora_weights` 等完整 LoRA 工具链。

LoRA 默认作用于以下线性层类型：

* `q_proj`
* `k_proj`
* `v_proj`
* `out_proj`
* `qkv`
* `proj`
* `fc1`
* `fc2`
* `linear1`
* `linear2`

同时排除 `text_encoder` 和 `language_backbone`，避免破坏原始文本语义能力。

### 4.3 分阶段 LoRA 策略

本项目不是对所有模块盲目插入 LoRA，而是设计了分阶段策略：

#### Stage A：视觉编码器与掩膜解码器适配

主要作用于：

* `vision_encoder`
* `mask_decoder`

在视觉编码器中，重点选择后 1/3 层的 attention projection。这样做的原因是前层通常保留通用低级视觉特征，后层更适合进行医学域语义适配。

#### Stage B：检测解码器与提示编码相关模块适配

主要作用于：

* `detector_decoder`
* `prompt_encoder`
* `exemplar_projection`

该阶段用于增强 prompt 与医学目标之间的对齐能力，尤其适合后续 Exemplar Prompt 机制。

#### Stage C：Detector MLP 适配

主要作用于：

* `detector_encoder`
* `detector_decoder` 中的 MLP 层

该阶段用于进一步增强检测分支的表达能力。

训练脚本中会先冻结 SAM3 原始参数，再注入 LoRA，并只开放 LoRA 参数训练。脚本还会检查可训练参数比例，防止误开启过多参数导致“伪 LoRA 微调”。

## 5. 医学图像特征 Adapter

### 5.1 改进动机

内窥镜息肉图像具有典型医学图像特征：

1. 局部纹理细节重要；
2. 边界区域对诊断和分割质量影响较大；
3. 病灶尺度差异明显；
4. 背景褶皱、血管、反光区域容易造成误分割。

原始 SAM3 的视觉特征来自通用图像预训练，不一定能充分表达这些医学细节。因此项目新增了医学特征适配器。

### 5.2 MedicalImageAdapter

`MedicalImageAdapter` 基于 bottleneck adapter 实现，并额外加入 depthwise convolution 和 pointwise convolution，用于增强局部纹理表达。

其作用可以理解为：

> 在不大幅改变 SAM3 主干结构的前提下，为图像特征补充医学纹理敏感性。

### 5.3 MultiScaleMedicalAdapter

`MultiScaleMedicalAdapter` 使用多个 dilation convolution 分支和 global pooling 分支，进行多尺度上下文融合。

该模块适合处理不同大小的息肉目标，尤其是小息肉、扁平息肉和形态不规则息肉。

## 6. 边界感知精修模块

### 6.1 改进动机

息肉分割的难点不仅是找到目标区域，更重要的是边界是否准确。内窥镜图像中常见以下边界问题：

1. 息肉与正常黏膜颜色接近；
2. 边缘区域模糊；
3. 反光导致边缘断裂；
4. 褶皱和阴影干扰边界判断。

因此，本项目增加了 `BoundaryAwareAdapter`，专门用于边界区域建模。

### 6.2 BoundaryAwareAdapter

该模块会从 ground truth mask 或 coarse mask logits 中提取 boundary band。如果没有 mask 信息，则使用 feature contrast prior 估计潜在边界区域。随后通过 boundary encoder、boundary head 和 gate head 对图像特征进行边界增强。

该模块输出：

* enhanced image features；
* boundary map；
* boundary gate；
* optional boundary loss。

训练时，`train_lora_medical.py` 会将 adapter 输出的 `boundary_loss` 作为辅助损失加入总损失。

### 6.3 Refine Head

在 `MedExSam3SegmentationModel` 中，经过医学 adapter 和边界 adapter 后，模型会使用 `refine_head` 生成 mask logits 的残差修正项，并将其加回原始 SAM3 输出。

其形式可以理解为：

```text
final_mask_logits = sam3_mask_logits + 0.1 * refine_delta
```

这样可以在不破坏原始 SAM3 输出稳定性的前提下，对边界和局部区域进行轻量修正。

## 7. Exemplar Prompt Adapter

### 7.1 改进动机

传统 prompt 通常包括 box、point、text，但这些 prompt 对医学图像来说表达能力有限。医学任务中，“相似病例”或“相似病灶形态”往往具有很强参考价值。

因此，本项目引入 Exemplar Prompt，让模型利用正样本、负样本和边界样本原型来辅助当前图像分割。

### 7.2 正负样本与边界原型

项目中实现了 `ExemplarPromptAdapter`，它接收：

* positive prototype；
* negative prototype；
* boundary prototype；
* query feature。

然后分别通过：

* `positive_proj`
* `negative_proj`
* `boundary_proj`
* `fusion_gate`

将原型转化为 prompt tokens。

其中 positive tokens 和 boundary tokens 会组成最终的 prompt tokens；negative tokens 不直接作为普通正向 prompt 输入，而是用于负样本抑制和后续检索融合。

### 7.3 技术意义

该模块的意义在于：

1. 让 SAM3 不只依赖当前图像的 box/text prompt；
2. 引入跨样本的医学形态先验；
3. 利用负样本减少假阳性；
4. 利用 boundary prototype 强化边界区域判断。

## 8. Exemplar Memory Bank

### 8.1 改进动机

Exemplar 机制的关键不是简单“存一些样本”，而是要保证样本库质量、可追踪、可审核，并避免外部测试集泄漏。

因此项目实现了版本化的 `ExemplarMemoryBank`。

### 8.2 样本字段设计

每个 `ExemplarItem` 包含：

* `item_id`
* `image_id`
* `crop_path`
* `mask_path`
* `bbox`
* `embedding_path`
* `type`
* `source_dataset`
* `fold_id`
* `human_verified`
* `quality_score`
* `boundary_score`
* `diversity_score`
* `difficulty_score`
* `uncertainty_score`
* `false_positive_risk`
* `created_at`
* `version`
* `notes`

这些字段说明样本库不仅记录图像，还记录了质量、边界、多样性、难度、不确定性和假阳性风险。

### 8.3 防止外部测试集泄漏

代码中显式检查 PolypGen 数据集是否进入 Exemplar Memory Bank。如果发现外部数据集泄漏，会直接抛出异常。

这对实验严谨性非常重要，因为 PolypGen 应作为外部泛化测试集，不应参与训练、验证、调参或样本库构建。

## 9. Prototype Builder

### 9.1 改进动机

单个 exemplar 可能存在偏差，直接使用所有 exemplar 又可能引入噪声。因此需要根据当前 query 动态选择高质量样本，并构建稳定 prototype。

### 9.2 原型构建方式

`PrototypeBuilder` 支持以下原型构建方式：

* mean prototype；
* weighted prototype；
* attention-fused prototype；
* clustered subprototypes。

它会综合以下因素对 exemplar 进行打分：

* query 与 exemplar embedding 的相似度；
* quality score；
* boundary score；
* diversity score；
* difficulty score；
* uncertainty score；
* false positive risk。

其中假阳性风险会被负向惩罚。

### 9.3 方差控制

如果 prototype variance 超过设定阈值，模块会触发 high-variance reject；在样本数量允许的情况下，会退化为 clustered subprototypes。

该设计用于避免 exemplar 分布过散导致 prompt 不稳定。

## 10. Retrieval Spatial-Semantic Adapter

### 10.1 改进动机

Exemplar Prompt 解决的是“提示层面的样本增强”，但对于分割任务，仅有 prompt token 仍可能不足。更强的方式是将检索结果直接转化为 feature-level 和 mask-level prior。

因此项目新增 `RetrievalSpatialSemanticAdapter`，用于实现空间-语义检索增强。

### 10.2 输入与输出

该模块输入：

* feature map；
* similarity map；
* positive prototype；
* negative prototype；
* positive tokens；
* negative tokens；
* baseline mask logits；
* positive / negative heatmap。

输出：

* adapted feature；
* retrieval prior；
* auxiliary visualization signals。

`retrieval_prior` 会包含：

* `semantic_prototype`
* `semantic_prototype_map`
* `spatial_bias_map`
* `decoder_feature_bias_map`
* `mask_logit_bias_map`
* `encoder_memory_bias`

这些 prior 可以进一步注入 SAM3 的 encoder memory、decoder feature 和 mask logits。

### 10.3 Gated Retrieval Fusion

`GatedRetrievalFusion` 是检索增强的核心门控模块。它支持多种策略：

* `always-on`
* `similarity-threshold`
* `uncertainty-aware`
* `region-aware`
* `residual`

模块会根据相似度分数、不确定性图、低置信病灶区域、高置信保护区域等信息，决定检索增强作用在哪些区域。

最终它会生成：

* semantic prototype；
* semantic prototype map；
* spatial bias map；
* decoder feature bias；
* mask logit bias；
* encoder memory bias；
* positive / negative context map；
* fusion gate map；
* uncertainty gate map；
* retrieval region mask。

这些输出使得检索增强不再是简单拼接，而是具有区域选择性和置信度约束的结构化增强。

## 11. 医学分割损失函数

### 11.1 基础损失

项目中实现了 `MedExLossComposer`，基础损失包括：

```text
L = w_bce * L_bce + w_dice * L_dice + w_boundary * L_boundary
```

其中：

* `L_bce` 用于像素级二分类监督；
* `L_dice` 用于缓解前景/背景类别不平衡；
* `L_boundary` 使用 BoundaryBandDiceLoss 强化边界区域重合度。

### 11.2 Exemplar 相关损失

损失函数中还预留了：

* `ExemplarInfoNCELoss`
* `NegativeSuppressionLoss`
* `ExemplarConsistencyLoss`
* `BoundaryBandDiceLoss`

这些损失用于支持 exemplar 对比学习、负样本抑制和一致性约束。

当前 LoRA 医学训练主流程默认关闭 contrast、negative 和 consistency 权重，主要训练 BCE + Dice + Boundary。

## 12. 训练流程改进

### 12.1 Preflight Gate

训练脚本 `train_lora_medical.py` 中加入了严格的 preflight 检查，包括：

1. split 文件是否存在；
2. train/val 是否为空；
3. PolypGen 是否泄漏到 train/val；
4. 官方 SAM3 是否构建成功；
5. LoRA 是否成功替换模块；
6. 可训练参数比例是否合理；
7. forward 是否成功；
8. backward 是否产生梯度。

如果 preflight 不通过，训练会被阻断，避免产生无效实验结果。

### 12.2 训练输出

训练过程中会保存：

* `config_used.yaml`
* `preflight_report.json`
* `train_log.jsonl`
* `val_metrics.json`
* `best_lora.pt`
* `best_adapter.pt`
* `last.pt`

这使实验具有可复现性和可追踪性。

### 12.3 指标体系

项目实现了以下分割指标：

* Dice
* IoU
* Precision
* Recall
* Boundary F1
* HD95
* ASSD
* False Positive Rate
* False Negative Rate

这些指标不仅关注区域重合度，也关注边界质量和误检/漏检情况。

## 13. 与原始 SAM3 的核心区别

| 维度        | 原始 SAM3                            | MedEx-SAM3                                          |
| --------- | ---------------------------------- | --------------------------------------------------- |
| 模型定位      | 通用分割基础模型                           | 面向内窥镜息肉分割的医学适配框架                                    |
| 参数更新      | 原始模型推理或全量微调                        | LoRA 参数高效微调                                         |
| 医学特征      | 无显式医学适配                            | Medical Adapter + MultiScale Adapter                |
| 边界处理      | 通用 mask 输出                         | BoundaryAwareAdapter + Boundary loss + Refine Head  |
| Prompt 类型 | text / box / point / visual prompt | 增加 Exemplar Prompt                                  |
| 样本先验      | 无样本库机制                             | Exemplar Memory Bank                                |
| 检索增强      | 无                                  | Retrieval Spatial-Semantic Adapter                  |
| 假阳性抑制     | 依赖模型自身判断                           | negative prototype + NegativeSuppressionLoss        |
| 实验安全      | 无专门泄漏检测                            | PolypGen 泄漏检查 + preflight gate                      |
| 可解释输出     | 常规 mask                            | fusion gate、uncertainty map、retrieval region mask 等 |

## 14. 当前改进的技术亮点

### 14.1 参数高效医学适配

通过 LoRA 避免全量微调 SAM3，在小规模医学数据下更稳健，也更适合多折交叉验证。

### 14.2 边界感知分割

通过 boundary band、boundary gate 和 boundary loss 显式优化息肉边界，适合内窥镜图像边界模糊的任务特点。

### 14.3 Exemplar 驱动的视觉提示

通过正样本、负样本和边界样本构建 prototype，再投影为 prompt token，使 SAM3 能够利用“相似病例”的视觉先验。

### 14.4 检索增强的空间-语义融合

检索结果不仅作为 prompt 输入，还能转化为 encoder memory bias、decoder feature bias 和 mask logit bias，实现更深层次的检索增强。

### 14.5 防泄漏实验工程

通过 preflight 和 PolypGen leakage check 保证实验严谨性，适合用于竞赛和论文展示。

## 15. 建议的消融实验设计

为了证明每个模块的贡献，建议设计如下消融实验：

| 实验编号 | 模型设置                                 | 目的                    |
| ---- | ------------------------------------ | --------------------- |
| A0   | 原始 SAM3                              | 基线性能                  |
| A1   | SAM3 + LoRA                          | 验证医学域参数高效适配效果         |
| A2   | SAM3 + LoRA + MedicalAdapter         | 验证医学纹理适配效果            |
| A3   | A2 + MultiScaleMedicalAdapter        | 验证多尺度上下文建模效果          |
| A4   | A3 + BoundaryAwareAdapter            | 验证边界增强效果              |
| A5   | A4 + ExemplarPromptAdapter           | 验证 exemplar prompt 效果 |
| A6   | A5 + RetrievalSpatialSemanticAdapter | 验证检索空间-语义融合效果         |
| A7   | A6 + Negative Prototype              | 验证负样本抑制假阳性的能力         |

重点观察指标：

* Dice / IoU：整体区域分割能力；
* Boundary F1 / HD95 / ASSD：边界质量；
* False Positive Rate：负样本抑制效果；
* False Negative Rate：小病灶漏检情况；
* PolypGen 外部测试结果：跨域泛化能力。

## 16. 当前模型改进总结

MedEx-SAM3 在原始 SAM3 基础上主要完成了以下改进：

1. 构建了官方 SAM3 的 tensor-native 训练与推理包装层；
2. 实现了分阶段 LoRA 参数高效微调；
3. 增加了医学图像特征适配器；
4. 增加了多尺度医学特征融合模块；
5. 增加了边界感知特征增强与边界辅助损失；
6. 增加了 mask logits 残差精修头；
7. 构建了 Exemplar Memory Bank；
8. 实现了正/负/边界 prototype 构建；
9. 实现了 Exemplar Prompt Adapter；
10. 实现了 Retrieval Spatial-Semantic Adapter；
11. 实现了不确定性和区域门控的 Gated Retrieval Fusion；
12. 设计了医学分割复合损失函数；
13. 加入了 preflight gate、数据泄漏检查和多指标评估。

整体来看，当前模型已经从“原始 SAM3 的简单医学微调”发展为一个具备医学特征适配、边界优化、样本库增强和检索先验融合能力的 MedEx-SAM3 框架。
