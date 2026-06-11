# MedEx-SAM3 模型改进说明文档

## 1. 文档目的

本文档说明本仓库在官方 SAM3 基础上进行医学息肉图像分割适配的真实代码结构与实验边界。本文只描述当前代码中可以被训练/验证入口支撑的内容，并将可选模块、系统扩展和未来实验设计与基础主线分开表述。

当前 MedEx-SAM3 不应被理解为“一次训练默认启用 LoRA、医学 adapter、边界 adapter、Exemplar Prompt 与 RSS-DA 的单一模型”。更准确的划分是：

1. 基础 LoRA 分支：`MedicalSAM3/scripts/train_lora_medical.py`。
2. Exemplar Prompt 分支：`MedicalSAM3/scripts/train_exemplar_prompt.py` 与 `MedicalSAM3/scripts/validate_medex_sam3.py`。
3. RSS-DA 检索增强分支：`MedicalSAM3/scripts/train_rssda.py` 与 `MedicalSAM3/scripts/validate_rssda.py`。
4. Agent 样本库闭环：`agent/` 下的工程扩展，不是当前 `MedicalSAM3/scripts` 主实验入口的必要依赖。

## 2. 官方 SAM3 基座与张量封装

官方 SAM3 作为通用分割基础模型，具备图像编码、文本/几何 prompt 编码、检测解码和 mask 生成能力。直接用于内窥镜息肉分割时，仍可能受到医学图像域差异、边界模糊、小目标、反光和黏液干扰等因素影响。

本项目新增 `MedicalSAM3/sam3_official/build_model.py` 与 `MedicalSAM3/sam3_official/tensor_forward.py`，用于将官方 SAM3 image model 包装成适合训练和验证脚本调用的 tensor-native 接口。`Sam3TensorForwardWrapper` 支持图像输入、文本 prompt、box prompt、point prompt、visual/exemplar prompt，以及检索先验注入。

在 Exemplar Prompt 分支中，`exemplar_prompt_tokens` 会被转换为 SAM3 可接收的 visual prompt embedding。在 RSS-DA 分支中，`retrieval_prior` 可以作用于 encoder memory、decoder feature bias 和 mask logits。

## 3. LoRA 参数高效微调

### 3.1 目标

SAM3 参数规模较大，直接全量微调会带来显存开销、训练成本和小样本过拟合风险。因此基础分支采用 LoRA 进行参数高效适配，在尽量保持 SAM3 通用能力的同时学习医学域差异。

### 3.2 实现

项目实现了 `LoRAConfig`、`LoRALinear`、`apply_lora_to_model`、`mark_only_lora_as_trainable`、`save_lora_weights`、`load_lora_weights` 和 `merge_lora_weights` 等工具。

默认 Stage-A 训练由 `train_lora_medical.py` 触发，目标 scope 是：

- `vision_encoder`
- `mask_decoder`

需要注意：`LoRAConfig` 中存在较宽的候选模块名列表，例如 `q_proj`、`k_proj`、`v_proj`、`out_proj`、`qkv`、`proj`、`fc1`、`fc2`、`linear1`、`linear2` 等；但 Stage-A 的实际替换范围会被进一步过滤。视觉编码器只在后 1/3 block 的注意力相关投影层注入 LoRA，mask decoder 也主要限定在注意力/投影类线性层。

### 3.3 可训练参数边界

`mark_only_lora_as_trainable` 针对的是 SAM3 基座模型，它只开放 LoRA A/B 参数。外层 `MedExSam3SegmentationModel` 还包含始终存在的 `refine_head`；如果显式启用 `MedicalImageAdapter` 或 `BoundaryAwareAdapter`，这些 adapter 参数也会参与训练。

因此，论文或报告中应写为：

> SAM3 基座参数被冻结，仅 LoRA 参数参与基座适配；外层医学分割包装器中的轻量 refine head 以及显式启用的 adapter 参数可参与训练。

## 4. 医学适配与边界精修

### 4.1 Residual Refine Head

`MedExSam3SegmentationModel` 始终包含一个 `refine_head`。它从 image embedding feature map 预测 residual mask delta，并以如下方式修正 SAM3 输出：

```text
final_mask_logits = sam3_mask_logits + 0.1 * refine_delta
```

这个模块是基础 LoRA 分支中默认存在的轻量修正头，用于对 mask logits 做小幅 residual refinement。

### 4.2 MedicalImageAdapter

`MedicalImageAdapter` 基于 bottleneck adapter 实现，并带有可选纹理卷积分支，用于增强医学图像局部纹理表达。它不是 `train_lora_medical.py` 的默认启用模块，当前入口通过 config 中的 `enable_medical_adapter` 控制。

### 4.3 BoundaryAwareAdapter

`BoundaryAwareAdapter` 使用边界编码、边界预测头、gate 和 bottleneck adapter 对 image feature 进行边界增强。训练阶段可由 GT mask 生成 boundary target 并返回 `boundary_loss`；推理阶段可由 coarse logits 或 feature contrast prior 估计边界区域。

它也不是基础 LoRA 的默认启用模块，需要通过 `--enable-boundary-adapter` 显式打开。只有启用该 adapter 且 forward 返回 `boundary_loss` 时，`train_lora_medical.py` 才会把该辅助损失加入总损失。

### 4.4 多尺度模块边界

当前主训练链路中没有名为 `MultiScaleMedicalAdapter` 的默认模块。仓库中存在其他多尺度相关实现，例如 Agent 工具里的 `MultiScaleFeatureAligner`，但它属于系统扩展/检索工作流，不应写成基础模型默认组件。

## 5. Exemplar Prompt

### 5.1 动机

单纯 text、box 或 point prompt 对医学图像中“相似病例形态”“负样本干扰区域”和“边界形态”的表达有限。因此 Exemplar Prompt 分支引入人工核验的 positive、negative 与 boundary exemplar memory bank。

### 5.2 原型构建

`PrototypeBuilder` 支持：

- mean prototype
- weighted prototype
- attention-fused prototype
- variance-aware clustered subprototypes

构建 prototype 时会综合 query-exemplar 相似度、quality score、boundary score、diversity score、difficulty score、uncertainty score 和 false positive risk。若 prototype variance 过高，模块可以退化为 clustered subprototypes，以降低单一原型不稳定带来的风险。

### 5.3 Prompt Token 注入

`ExemplarPromptAdapter` 接收 positive、negative、boundary prototypes 以及 query feature，并分别通过投影层生成 prompt tokens。当前实现会拼接 positive tokens、boundary tokens 和 negative tokens，三类 token 都会进入最终 visual prompt token 序列。

因此，不能再写成“negative tokens 不直接进入 prompt”。更准确的表述是：

> negative prototype 既会被投影为 prompt token 参与 visual prompt 序列，也可以通过 negative suppression loss 或 RSS-DA 中的 negative prior 抑制假阳性响应。

### 5.4 训练损失边界

`train_exemplar_prompt.py` 使用 `MedExLossComposer`，但 contrastive、negative suppression 和 consistency 是否实际参与训练取决于命令行开关和 forward 输入。报告实验设置时应明确是否启用：

- `--enable-contrastive-loss`
- `--enable-negative-suppression`
- `--enable-consistency-loss`

## 6. Exemplar Memory Bank

`ExemplarMemoryBank` 记录 exemplar 的路径、bbox、embedding、type、source dataset、fold、human verification、quality、boundary、diversity、difficulty、uncertainty 和 false positive risk 等字段。

严格外部评估协议下，memory bank 只能由训练 split 构建。PolypGen 作为 external test，不能进入 train split、val split、early stopping、memory bank 或 prototype building。

`MedicalSAM3/banks/README.md` 中的 `train_bank` 原则与该协议一致：positive/negative exemplars 只允许来自训练 split，`continual_bank` 不参与当前 strict external run。

## 7. RSS-DA 检索空间语义增强

Exemplar Prompt 主要作用于 prompt 层。RSS-DA 分支进一步将检索结果转化为空间和语义先验，注入到更深层的模型路径。

`RetrievalSpatialSemanticAdapter` 可以生成：

- spatial bias
- semantic prototype map
- decoder feature bias map
- mask logit bias map
- negative prompt mask logits
- fusion gate / uncertainty map / retrieval region mask 等辅助信号

`Sam3TensorForwardWrapper` 会接收 `retrieval_prior`，并把相关先验应用到 encoder memory 和 mask logits。该分支应单独报告为检索增强实验，不应与基础 LoRA 默认训练混写。

外部验证时，`validate_rssda.py` 会在 `bank_purpose=external-eval` 场景下检查 memory bank 是否包含 PolypGen 泄漏。

## 8. Bounding Box Prompt 协议

验证脚本支持三种 bbox source：

- `mask`：默认值，由真值 mask 派生 box，适合评估给定定位提示下的分割能力。
- `yolo`：使用 YOLO detector 或缓存框，更接近自动部署。
- `none`：无框提示对照。

所有外部测试结果必须报告 `--bbox-source`。如果命令没有显式指定 `--bbox-source yolo`，则默认不是自动检测框评估，而是 mask-derived GT box 评估。

这点对论文结果表非常重要，因为 mask-derived box、YOLO box 和 no-box protocol 的 Dice/IoU 不具备直接横向可比性。

## 9. Loss 与指标

`MedExLossComposer` 的基础分割损失为：

```text
L = w_bce * L_bce + w_dice * L_dice + w_boundary * L_boundary
```

其中：

- `L_bce`：像素级二分类监督。
- `L_dice`：缓解前景/背景类别不平衡。
- `L_boundary`：BoundaryBandDiceLoss，用于边界区域重合度。

基础 LoRA 训练在 `train_lora_medical.py` 中显式关闭 contrast、negative 和 consistency 权重，因此主要训练 BCE + Dice + BoundaryBandDiceLoss。Exemplar Prompt 与 RSS-DA 分支可根据命令和输入启用额外损失。

验证阶段统一计算：

- Dice
- IoU
- Precision
- Recall
- Boundary F1
- HD95
- ASSD
- False Positive Rate
- False Negative Rate

## 10. Preflight 与训练产物

`train_lora_medical.py` 包含 preflight 检查，覆盖：

1. split 文件是否存在。
2. train/val 是否为空。
3. PolypGen 是否泄漏到 train/val。
4. 官方 SAM3 是否构建成功。
5. LoRA 是否成功替换模块。
6. 可训练参数比例是否合理。
7. forward 是否成功。
8. backward 是否产生梯度。

训练过程会保存 `config_used.yaml`、`preflight_report.json`、`train_log.jsonl`、`val_metrics.json`、`best_lora.pt`、`best_adapter.pt` 和 `last.pt` 等产物，用于可复现和审计。

## 11. 与原始 SAM3 的差异

| 维度 | 原始 SAM3 | 当前 MedEx-SAM3 代码事实 |
|---|---|---|
| 模型定位 | 通用分割基础模型 | 面向内窥镜息肉分割的 SAM3 适配实验框架 |
| 参数更新 | 原始模型推理或全量微调 | 基础分支默认使用 Stage-A LoRA |
| 默认外层头 | 无专门医学 refine head | `refine_head` 始终存在 |
| 医学纹理适配 | 无显式医学 adapter | `MedicalImageAdapter` 已实现，但默认关闭 |
| 边界处理 | 通用 mask 输出 | `BoundaryAwareAdapter` 已实现，但默认关闭；BoundaryBandDiceLoss 在基础 loss 中使用 |
| Prompt 类型 | text / box / point / visual prompt | Exemplar Prompt 分支可注入 positive/negative/boundary tokens |
| 检索增强 | 无样本库机制 | RSS-DA 分支可注入 retrieval prior |
| 假阳性抑制 | 依赖模型自身判断 | negative exemplar、negative suppression loss 与 RSS-DA negative prior 可选 |
| 实验安全 | 无专门泄漏检查 | split、memory bank 和 external eval 均有 PolypGen leakage 检查 |

## 12. 当前技术亮点

### 12.1 参数高效医学域适配

默认 Stage-A LoRA 限制在视觉编码器后部注意力层和 mask decoder 投影层，有利于控制可训练参数规模并降低小样本过拟合风险。

### 12.2 边界敏感训练目标

基础 loss 包含 BoundaryBandDiceLoss；可选 `BoundaryAwareAdapter` 进一步提供边界 gate、boundary map 和辅助边界损失。

### 12.3 Exemplar 驱动视觉提示

人工核验样本库可以构建 positive、negative 与 boundary prototypes，并投影为 visual prompt tokens，使 SAM3 能利用相似病例和负样本区域先验。

### 12.4 检索增强的空间-语义融合

RSS-DA 将检索结果转化为 feature-level 和 logit-level prior，使增强不只停留在 prompt token 层。

### 12.5 防泄漏实验工程

数据划分、memory bank 和 external evaluation 均包含 PolypGen 泄漏检查，适合用于论文和竞赛场景中的实验审计。

## 13. 建议的消融实验设计

以下表格是建议实验设计，不代表当前仓库已经用同一协议完整跑完并证明所有模块贡献。正式论文表格必须统一 split、checkpoint、bbox source、evaluation script 和 output directory。

| 编号 | 设置 | 目的 | 当前状态说明 |
|---|---|---|---|
| A0 | official SAM3 | 原始基线 | 需明确 prompt 协议 |
| A1 | SAM3 + Stage-A LoRA + refine head | 验证医学域参数高效适配 | 当前基础主线 |
| A2 | A1 + MedicalImageAdapter | 验证医学纹理适配 | 可选模块，需显式启用 |
| A3 | A1 + BoundaryAwareAdapter | 验证边界增强 | 可选模块，需显式启用 |
| A4 | A1 + ExemplarPromptAdapter | 验证 exemplar prompt | 独立训练入口 |
| A5 | A4 + negative suppression | 验证负样本抑制 | 需显式开关 |
| A6 | A1 + RSS-DA joint retrieval | 验证检索空间语义增强 | 独立训练/验证入口 |
| A7 | A6 + hard-case gate | 验证检索门控策略 | 需统一 gate 策略与阈值 |

建议重点观察：

- Dice / IoU：整体区域分割能力。
- Boundary F1 / HD95 / ASSD：边界质量。
- False Positive Rate：负样本抑制效果。
- False Negative Rate：小病灶漏检情况。
- PolypGen external test：跨域泛化能力。

## 14. Agent 系统边界

`agent/` 下的 `ExemplarBankAgent`、质量控制、样本演化和 cross-attention reranking 等代码属于样本库管理和临床闭环工程扩展。它与主实验共享 positive/negative/boundary exemplar 概念，但不是当前 `MedicalSAM3/scripts` 训练和验证入口的必要依赖。

论文主实验应以 `MedicalSAM3` 训练/验证脚本为准；Agent 内容可放在系统扩展、临床交互闭环或未来工作章节。

## 15. 总结

当前仓库已经实现了一个围绕官方 SAM3 的医学息肉分割实验框架，包括 Stage-A LoRA、轻量 mask logits refine、可选医学/边界 adapter、Exemplar Prompt、RSS-DA 检索增强、数据泄漏检查和多指标评估。文档和论文撰写时最重要的是把“默认启用”“显式启用”“独立分支”和“系统扩展”分清楚，避免把所有模块合并成一个未经统一消融验证的默认模型。
