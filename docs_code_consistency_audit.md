# 医学图像分割代码与文档一致性审查报告

审查对象：本地仓库 `E:\BME`，重点覆盖 `MedicalSAM3/`、`agent/`、`Backend/` 以及根目录文档。  
审查目标：核对现有 README、方法说明、配置说明与实际训练/验证代码是否一致，并给出可直接替换到论文或项目文档中的中文表述。

## 0. 总体结论

当前仓库已经具备一条较完整的医学息肉分割实验链路：数据划分、官方 SAM3 张量封装、LoRA 注入、可选医学/边界适配器、Exemplar Prompt、RSS-DA 检索增强、外部 PolypGen 验证、YOLO/GT 框源切换、指标统计和若干结果产物均有代码或文件证据。

主要风险不是“代码不存在”，而是文档把多个不同阶段和不同入口的能力合并成了一条统一主路线，容易让读者误以为所有模块都在同一次训练中默认启用，并且已经完成完整消融证明。实际情况应表述为：

1. 基础训练主线是 `train_lora_medical.py`：官方 SAM3 + Stage-A LoRA + 轻量 residual refine head，医学适配器和边界适配器是可选项。
2. Exemplar Prompt 训练是独立入口 `train_exemplar_prompt.py`，依赖人工核验 memory bank，并可选启用 contrastive、negative suppression、consistency。
3. RSS-DA 检索增强是独立入口 `train_rssda.py` / `validate_rssda.py`，其检索先验可以注入 encoder memory、decoder feature 和 mask logit。
4. Agent 目录存在样本库质量控制、演化和检索 prior 的工程扩展，但它不是当前 `MedicalSAM3/scripts` 主训练脚本的直接依赖。
5. 外部验证默认框源需要特别说明：`--bbox-source mask` 会使用真值 mask 派生 box；只有显式指定 `--bbox-source yolo` 才是自动检测框评估。

建议文档级处理：保留现有代码与结果产物，但重写 `README_medex_sam3.md` 和 `doc.md` 中的主路线、模块默认状态、消融实验和评估协议表述；将 `fix_exemplar_prompt.md` 和 Agent 架构文档标注为开发记录/系统扩展，不作为论文主实验事实依据。

## 1. 关键事实核对

### 1.1 数据集与划分

代码事实：

- `MedicalSAM3/scripts/prepare_5fold_polyp.py` 将 Kvasir、CVC、KvasirCVC 作为训练/验证池，并将 PolypGen 作为外部测试集，见 `prepare_5fold_polyp.py:187-230`。
- 脚本写出 `external_polypgen_ids.txt`，并检查 PolypGen 是否泄漏进 train/val，见 `prepare_5fold_polyp.py:253-280`。
- 当前 `MedicalSAM3/outputs/medex_sam3/splits/split_summary.json` 显示 `train_val_count=2612`，其中 `kvasir_count=1000`、`kvasircvc_count=1612`、`external_polypgen_count=1537`，且 `leakage_check_passed=true`。
- `MedicalSAM3/utils/dataset.py` 中仍有旧式 `KvasirCVCDataset` / `PolypGenDataset` 与随机 85/15 split 工具，但当前主要训练入口使用的是 split 文件驱动的 `SplitSegmentationDataset`。

文档风险：

- 若论文或 README 只写“KvasirCVC 5-fold”会遗漏 Kvasir-SEG 与 CVC-ClinicDB 兼容扫描逻辑。
- 若引用 `最终结果/splits/split_summary.json`，需要注意该产物似乎对应另一次只含 KvasirCVC 训练池的运行；不能与 `MedicalSAM3/outputs/medex_sam3/splits` 混用。

推荐表述：

> 本研究将 Kvasir-SEG、CVC-ClinicDB 以及 KvasirCVC 规范化数据合并为训练/验证候选池，并使用固定随机种子生成五折交叉验证划分。PolypGen 仅作为外部测试集使用，划分脚本会额外写出外部测试 ID，并在生成完成后检查训练/验证集合中是否混入 PolypGen 样本。后续报告结果时需明确对应的 split 产物路径，避免将不同实验运行的划分统计混合比较。

### 1.2 LoRA 注入与可训练参数

代码事实：

- `train_lora_medical.py` 默认 stage 为 `stage_a`，默认目标范围是 `vision_encoder` 与 `mask_decoder`，见 `train_lora_medical.py:81-100`。
- Stage-A 的实际过滤策略比文档中的“默认线性层列表”更窄：视觉编码器只作用于后 1/3 block 的注意力相关投影，mask decoder 只作用于注意力/投影类线性层，见 `MedicalSAM3/adapters/lora.py:152-175`。
- `apply_lora_to_model` 负责按 scope、target module、stage 过滤并替换 `nn.Linear`，见 `lora.py:252-313`。
- `mark_only_lora_as_trainable` 只开放 SAM3 基座里的 LoRA A/B 参数，见 `lora.py:316-319`。
- 包装模型还会引入 `refine_head`，且启用医学/边界 adapter 时这些 adapter 也会训练，见 `common.py:548-560` 与 `common.py:588-590`。
- 现有 preflight 产物显示 fold 0 实际可训练参数比例约为 `0.000643`，见 `fold_0/preflight_report.json`。

文档风险：

- `doc.md` 中“只开放 LoRA 参数训练”只适用于 SAM3 基座参数，不适用于外层 `MedExSam3SegmentationModel` 的 `refine_head` 和可选 adapters。
- `doc.md` 中 LoRA 默认作用层列表容易被误读为 Stage-A 全部替换范围。

推荐表述：

> 本项目采用参数高效的 LoRA 微调方式对官方 SAM3 进行医学域适配。默认 Stage-A 仅在视觉编码器后部注意力层和 mask decoder 相关投影层注入低秩矩阵，从而限制可训练参数规模。需要注意的是，`mark_only_lora_as_trainable` 针对的是 SAM3 基座模型；外层医学分割包装器还包含一个轻量 `refine_head`，在显式启用医学适配器或边界适配器时，对应 adapter 参数也会参与训练。

### 1.3 医学适配器、边界适配器与 refine head

代码事实：

- `MedicalImageAdapter` 是 bottleneck adapter 加可选纹理卷积分支，见 `medical_adapter.py:44-65`。
- `BoundaryAwareAdapter` 使用边界编码、边界预测头、gate 与 bottleneck adapter，并在训练阶段可由 GT mask 生成边界监督，见 `boundary_adapter.py:33-90`。
- `MedExSam3SegmentationModel` 中 `MedicalImageAdapter` 默认关闭，只有 config 中 `enable_medical_adapter=true` 才启用；`BoundaryAwareAdapter` 默认关闭，需命令行 `--enable-boundary-adapter`，见 `train_lora_medical.py:179-180`。
- `refine_head` 总是存在，会从 image embedding feature map 预测 residual mask delta，并以 `0.1 * delta` 加到 SAM3 mask logits 上，见 `common.py:560` 与 `common.py:588-590`。

文档风险：

- `README_medex_sam3.md:3` 和 `doc.md:25` 把 medical/boundary adapters 写成主路线默认组件，和代码默认值不一致。
- `doc.md:118-122` 提到 `MultiScaleMedicalAdapter`，当前主训练链路没有该类；仓库中存在 `MultiScaleFeatureAdapter` 或 Agent 里的 `MultiScaleFeatureAligner`，但不是 `train_lora_medical.py` 默认组件。
- `doc.md:148` 写成训练总会加入 adapter `boundary_loss`，实际只有启用 `--enable-boundary-adapter` 且 adapter 返回 `boundary_loss` 时才加入。

推荐表述：

> 医学适配模块在当前代码中分为三类：第一，始终启用的轻量 residual refine head，用于对 SAM3 输出的 mask logits 做小幅修正；第二，可选的 `MedicalImageAdapter`，通过 bottleneck 与纹理卷积分支增强医学图像特征；第三，可选的 `BoundaryAwareAdapter`，在训练阶段可利用 GT mask 生成边界监督，在推理阶段则由粗分割结果或对比先验形成边界提示。除 refine head 外，医学适配器和边界适配器均不是默认启用模块，实验报告中应明确具体开关。

### 1.4 Exemplar Prompt 与原型库

代码事实：

- `ExemplarPromptAdapter` 将 positive、boundary、negative prototype 投影为 prompt tokens，并在 forward 中拼接三类 token，见 `exemplar_prompt_adapter.py:66-98`。
- `PrototypeBuilder` 支持 weighted prototype、attention-fused prototype、variance-aware clustered subprototypes，见 `prototype_builder.py:17-120`。
- `train_exemplar_prompt.py` 要求 batch size 为 1，并使用 memory bank 构造 positive/negative/boundary prototype；contrastive、negative suppression、consistency 是可选训练项，见 `train_exemplar_prompt.py:171`、`train_exemplar_prompt.py:227`、`train_exemplar_prompt.py:324-406`。
- `validate_medex_sam3.py` 会在有 memory bank 和 prompt checkpoint 时生成 exemplar prompt tokens，并比较 with/without exemplar 的指标差异，见 `validate_medex_sam3.py:101-135` 与 `validate_medex_sam3.py:225-270`。

文档风险：

- `doc.md:188` 说 negative tokens 不直接进入最终 prompt，这与当前代码不一致。当前实现会拼接 positive、boundary、negative tokens。
- `fix_exemplar_prompt.md` 是开发修复记录，不应作为最终论文方法描述引用。

推荐表述：

> Exemplar Prompt 分支通过人工核验的样本库构造 positive、negative 与 boundary prototypes，再由 `ExemplarPromptAdapter` 投影为视觉 prompt token 注入 SAM3 prompt 路径。当前实现中 negative token 会与 positive token、boundary token 一同进入最终 prompt token 序列；同时，训练脚本还可以利用 negative suppression 项约束负样本区域的响应。因此，论文中不应写成“negative prototype 只用于抑制、不进入 prompt”，而应区分 prompt token 注入与 loss 级负样本抑制两个作用位置。

### 1.5 RSS-DA 检索空间语义双适配

代码事实：

- `RetrievalSpatialSemanticAdapter` 生成 spatial bias、semantic map、negative prompt mask logits、fusion gate 等先验，见 `retrieval_spatial_semantic_adapter.py:52-198`。
- 官方 SAM3 tensor forward wrapper 可以接收 `exemplar_prompt_tokens` 和 `retrieval_prior`，将检索先验注入 encoder memory，并将 `mask_logit_bias_map` 加到 mask logits，见 `tensor_forward.py:371-410` 与 `tensor_forward.py:473-496`。
- `validate_rssda.py` 默认外部测试 split 为 `external_polypgen_ids.txt`，并在 external eval 时检查 memory bank 是否包含 PolypGen 泄漏，见 `validate_rssda.py:193-229` 与 `validate_rssda.py:300-301`。
- `train_rssda.py` 使用 `MedExLossComposer` 与 `CrossDomainConsistencyLoss`，并传入 `negative_prompt_mask_logits`，见 `train_rssda.py:286-287` 与 `train_rssda.py:379`。

文档风险：

- 可以把 RSS-DA 写成已实现的独立增强分支，但不能把它写成 `train_lora_medical.py` 默认训练路径。
- 需要区分 `agent/` 的 ExemplarBankAgent 检索 prior 与 `MedicalSAM3/scripts/retrieval_runtime.py` 的实验运行时。两者概念相关，但不是同一个调用链。

推荐表述：

> RSS-DA 分支在基础 SAM3-LoRA 模型之外引入检索空间语义先验。验证阶段首先从训练样本库检索 positive/negative/boundary exemplars，再由 `RetrievalSpatialSemanticAdapter` 生成空间 bias、语义 prototype map、decoder feature bias 与 mask logit bias，并通过 SAM3 tensor forward wrapper 注入 encoder、decoder 和 mask logits。该分支适合单独报告为检索增强实验，而不应与基础 LoRA 训练混写为同一默认模型。

### 1.6 边界框来源与外部验证协议

代码事实：

- `add_yolo_bbox_args` 的 `--bbox-source` 取值为 `none`、`mask`、`yolo`，默认是 `mask`，见 `yolo_adapter/cli.py:12-20`。
- 当 `bbox_source=mask` 时，box provider 返回 `None`，`SplitSegmentationDataset` 会回退到 `mask_to_box(mask)`，也就是由 GT mask 生成 box，见 `bbox_provider.py:154-161` 与 `common.py:490-500`。
- 当 `bbox_source=yolo` 时，才会使用 YOLO detector 或 cache；fallback 可选 `none`、`full`、`mask`、`error`，见 `bbox_provider.py:54-100`。

文档风险：

- 任何外部测试结果如果未显式传入 `--bbox-source yolo`，都应标注为“GT/mask-derived box prompt”协议，而不是自动检测框协议。
- 如果论文目标是端到端自动分割，需要单独报告 YOLO box 或 no-box protocol 的结果。

推荐表述：

> 当前验证脚本支持三种 box prompt 协议：`mask`、`yolo` 与 `none`。默认 `mask` 协议使用真值 mask 派生 bounding box，适合分析分割模型在给定定位提示下的上限性能；`yolo` 协议使用检测器或缓存框，更接近自动化部署；`none` 协议则用于无框提示对照。实验结果必须同时报告所用 `--bbox-source`，否则 Dice/IoU 等指标不可横向比较。

### 1.7 Loss 与指标

代码事实：

- `MedExLossComposer` 固定包含 BCE、Dice、BoundaryBandDiceLoss，并可选加入 contrastive、negative suppression、consistency，见 `losses.py:100-154`。
- `train_lora_medical.py` 显式关闭 contrast、negative、consistency 权重，所以基础 LoRA 训练主要使用 BCE + Dice + BoundaryBandDiceLoss，见 `train_lora_medical.py:431`。
- `train_exemplar_prompt.py` 默认构造完整 composer，但 contrastive、negative suppression、consistency 的输入由命令行开关控制，见 `train_exemplar_prompt.py:171` 与 `train_exemplar_prompt.py:406`。
- `compute_segmentation_metrics` 输出 Dice、IoU、Precision、Recall、Boundary F1、HD95、ASSD、FPR、FNR，见 `common.py:405-454`。

文档风险：

- 论文方法部分可以介绍可选 loss，但实验设置必须说明每个入口实际启用的 loss。
- 如果只报告 Dice/IoU 会遗漏边界质量指标；但如果强调 HD95/ASSD，也要说明空边界时当前实现返回 0 的处理策略。

推荐表述：

> 基础 LoRA 训练使用 BCE、Dice 与 BoundaryBandDiceLoss 组成的分割损失；Exemplar Prompt 与 RSS-DA 分支在此基础上可以进一步引入对比学习、负样本抑制和一致性约束。所有验证脚本统一计算区域重叠、边界质量和错误率指标，包括 Dice、IoU、Precision、Recall、Boundary F1、HD95、ASSD、FPR 与 FNR。

### 1.8 消融实验与现有结果产物

代码/文件事实：

- `MedicalSAM3/scripts/run_ablation.py` 当前生成的是 deterministic dummy metrics，不应作为真实性能结果引用。
- `MedicalSAM3/outputs/medex_sam3/ablation_table.md` 若由上述脚本生成，应标注为占位/流程验证产物。
- `最终结果/` 下存在多组外部验证 summary，例如 `polypgen_external_baseline_lora_only.json`、`polypgen_external_uncertainty_aware_retrieval.json`、`abl_curated_p2_n1_b1_gated/summary_metrics.json`、`yolo_ablation_eval/*/summary_metrics.json`。
- 其中 `polypgen_external_baseline_lora_only.json` Dice 约为 `0.855969`；`polypgen_external_uncertainty_aware_retrieval.json` Dice 约为 `0.852814`，相对 baseline Dice delta 约为 `-0.00315`；`abl_curated_p2_n1_b1_gated/summary_metrics.json` Dice 约为 `0.855069`，delta Dice 约为 `+0.000892`，hard-case gate 使用比例约为 `29.9%`。
- `detail.md` 的观察“样本库整体保守、总体 Dice 下降约 0.003、个别难样本提升”与上述部分检索结果产物大体一致。

文档风险：

- `doc.md:450-463` 的消融表应写成“建议实验设计”或“待补齐实验”，不能写成已完成证明。
- 不同结果目录可能对应不同 bbox 协议、split、checkpoint、检索策略和是否 hard-case gate，不能直接合并成同一张论文主表。

推荐表述：

> 当前仓库包含若干外部验证与检索增强结果产物，但结果来源、提示框协议和检索策略并不完全相同。初步产物显示，检索增强在部分难样本或 gated 场景下可能带来小幅收益，但在全量外部集上也可能出现约 0.003 Dice 的平均下降。因此，论文中应将 RSS-DA/Exemplar Prompt 结果表述为“难样本增强与保守分割倾向并存的初步发现”，并在最终表格中严格按同一 split、同一 bbox source、同一 checkpoint 和同一评价脚本重新生成可比结果。

### 1.9 Agent 与后端边界

代码事实：

- `agent/agents/exemplar_bank_agent.py` 确实实现了基于 HelloAgents 的样本库 ingest、retrieve_prior 与 feedback 更新，见 `exemplar_bank_agent.py:45-138`。
- `agent/tools/medical/exemplar_bank_retrieval.py` 实现了 `BoundaryEmbeddingHead`、`MultiScaleFeatureAligner`、`CrossAttentionReranker`、`RetrievalConfidenceEstimator` 与 `ExemplarRetrievalPipeline`，见该文件 `33-166`。
- 这些 Agent 工具与 `MedicalSAM3/scripts/retrieval_runtime.py` 的实验运行时是两个层次。前者偏系统/工作流扩展，后者偏论文实验/模型验证。
- `Backend/README.md` 是 FastAPI/Celery 应用说明，`segment-frame` 当前使用全图先验框作为无外部 prompt 的兜底策略，见 `Backend/README.md:72-74`。

文档风险：

- Agent 文档可以保留，但应避免把 Agent 反馈闭环、生命周期管理、质量演化机制写成当前主实验必经流程。
- 后端接口契约不应混入论文训练协议；它是应用层运行方式。

推荐表述：

> Agent 子系统用于管理医学 exemplar memory 的工程闭环，包括样本摄入、质量评分、生命周期演化、检索 prior 生成和医生反馈更新。该子系统与模型实验代码共享“positive/negative/boundary exemplar”概念，但不是当前 `MedicalSAM3/scripts` 训练与验证脚本的必要依赖。论文主实验应以 `MedicalSAM3` 下的训练/验证入口为准，Agent 可作为未来临床工作流或系统扩展章节描述。

## 2. 问题清单

| 风险 | 位置 | 问题 | 代码/文件证据 | 建议 |
|---|---|---|---|---|
| 高 | `README_medex_sam3.md:3` | 将 LoRA、medical/boundary adapters、exemplar memory、prototype fusion 写成统一主路线 | `train_lora_medical.py` 默认只启用 LoRA + refine head；medical adapter 依赖 config，boundary adapter 依赖命令行 | 改成“基础 LoRA、Exemplar Prompt、RSS-DA 检索增强三个入口” |
| 高 | `README_medex_sam3.md:5` | 引用不存在的旧入口 `MedicalSAM3/train_ext.py` | `rg --files` 未发现该文件 | 删除该句或改成“历史 wrapper 如存在仅作参考” |
| 高 | `doc.md:188` | 说 negative tokens 不进入最终 prompt | `ExemplarPromptAdapter.forward` 拼接 positive、boundary、negative tokens | 改为 negative token 同时进入 prompt，并可参与负样本抑制 |
| 高 | `doc.md:118-122` | `MultiScaleMedicalAdapter` 被写成当前方法模块 | 当前主链路未实现同名类；相关多尺度模块在 Agent 或扩展目录 | 移到未来工作/系统扩展，或改名并说明未进入主实验 |
| 高 | `README_medex_sam3.md:149-157` | 外部验证命令未说明默认 `bbox-source=mask` | `yolo_adapter/cli.py` 默认 `mask`；dataset 用 `mask_to_box(mask)` | 外部验证必须标注 GT-box / YOLO-box / no-box |
| 高 | `doc.md:450-463` | 消融表容易被读成已完成实验 | `run_ablation.py` 生成 dummy metrics；真实产物分散且协议不一 | 标为建议实验设计，真实结果需重跑统一协议 |
| 中 | `doc.md:95` | “只开放 LoRA 参数训练”表述过窄 | wrapper 中 `refine_head` 始终可训练，可选 adapter 也可训练 | 改为“基座仅 LoRA；外层轻量头/adapter 按配置训练” |
| 中 | `doc.md:148` | boundary loss 被写成默认加入 | `--enable-boundary-adapter` 默认关闭 | 改成“启用边界适配器时加入辅助边界损失” |
| 中 | `README_medex_sam3.md:79-97` | “完整 5-Fold”命令仍包含 `--max-train-steps 10` | 命令本身限制训练步数 | 改成 smoke 5-fold；完整训练命令去掉 step cap |
| 中 | `MedicalSAM3/configs/*.yaml` 与 `train_lora_medical.py` | lora config 中的 epochs/batch/lr 不会被 `train_lora_medical.py` 全量覆盖 CLI 默认值 | `train_lora_medical.py` 未调用 `apply_config_overrides`，只读取 seed/split_dir/adapter flag | 文档注明该脚本以 CLI 为准，或后续补齐 config override |
| 中 | `detail.md` | 只写结论，没有绑定结果文件、bbox 协议和检索策略 | `最终结果/` 下存在多个不同协议 summary | 扩展为实验观察表，列清结果路径 |
| 中 | `agent/EXEMPLAR_BANK_*.md` | Agent 能力容易与模型主实验混写 | Agent 代码存在，但主实验入口不依赖 Agent | 移入“系统扩展/临床闭环”章节 |
| 低 | `README.md`、`Backend/README.md` | 应用启动文档与论文训练文档边界不清 | Backend 是 FastAPI/Celery contract | 保留为工程部署说明，不作为模型实验依据 |

## 3. 删除、保留、迁移建议表

| 对象 | 处理建议 | 理由 | 具体动作 |
|---|---|---|---|
| `README_medex_sam3.md` | 保留并重写 | 是最接近实验入口的文档，但主路线和命令说明需修正 | 重写前 2 节、5-fold 命令、外部验证命令、常见问题 |
| `doc.md` | 保留为方法草稿，重写为“已实现/可选/设计中”三层结构 | 内容覆盖完整，但有默认启用、模块名称和消融状态混淆 | 删除或迁移 `MultiScaleMedicalAdapter` 当前主方法表述；修正 negative prompt；重写实验协议 |
| `fix_exemplar_prompt.md` | 移到开发记录 | 它是修复建议和变更记录，不是最终方法描述 | 放入 `docs/dev_notes/` 或在标题注明“历史修复记录” |
| `detail.md` | 保留并扩写 | 与现有结果产物中“检索保守、Dice 约下降 0.003”的观察相符 | 加入对应 JSON 路径、bbox source、retrieval mode、delta |
| `MedicalSAM3/banks/README.md` | 保留 | 与外部评估无泄漏原则一致 | 可补充 bank schema 与构建命令 |
| `agent/EXEMPLAR_BANK_WORKFLOW_SUMMARY.md` | 保留但迁移到系统扩展 | 代码存在，但不是主实验入口 | 标注“Agent workflow extension” |
| `agent/EXEMPLAR_BANK_AGENT_ARCHITECTURE.md` | 保留但迁移到系统扩展/未来工作 | 架构性强，容易被误读为主实验事实 | 在开头加边界说明 |
| `MedicalSAM3/outputs/medex_sam3/ablation_table.md` | 不删除，但禁止作为真实性能表引用 | 来源可能是 dummy ablation 脚本 | 标注 generated placeholder，或移入 `outputs/debug/` |
| `最终结果/` 下 summary JSON | 保留 | 是当前最有价值的结果证据 | 汇总前需按协议分组，避免混合比较 |
| `README.md` | 保留 | 工程启动说明 | 修复显示/编码问题时不要加入模型实验结论 |
| `Backend/README.md` | 保留 | 后端接口契约清楚 | 与论文实验文档分离 |

## 4. 可替换的专业中文文本

下面文本可作为 `README_medex_sam3.md` 和论文方法章节的替换基础。

### 4.1 方法总览

本项目围绕官方 SAM3 构建医学息肉图像分割实验框架。整体实现分为三个互相独立但可组合的实验分支：第一，基于 Stage-A LoRA 的医学域参数高效适配；第二，基于人工核验 exemplar memory 的 Exemplar Prompt 分支；第三，基于检索空间语义先验的 RSS-DA 分支。三者共享统一的数据划分、SAM3 tensor forward wrapper、mask 指标计算与外部 PolypGen 验证协议，但训练入口和默认启用模块不同。

基础 LoRA 分支以官方 SAM3 image model 为基座，冻结原始大模型参数，仅在选定注意力/投影层注入低秩可训练矩阵。为了适配医学分割输出，外层包装器还包含一个轻量 residual refine head，用于对 SAM3 mask logits 进行小幅修正。医学图像 adapter 与边界感知 adapter 是可选模块，实验报告中必须明确是否启用。

Exemplar Prompt 分支从人工核验样本库中检索 positive、negative 与 boundary exemplars，构造原型并投影为视觉 prompt tokens。当前实现中三类 token 均会进入最终 prompt token 序列；negative exemplar 同时还可通过负样本抑制损失约束假阳性响应。

RSS-DA 分支进一步将检索结果转化为空间和语义先验，生成 encoder memory bias、decoder feature bias 与 mask logit bias，并通过 SAM3 tensor forward wrapper 注入分割链路。该分支适合作为检索增强实验单独报告，而不应与基础 LoRA 默认训练混写。

### 4.2 数据集与验证协议

训练/验证划分由 `prepare_5fold_polyp.py` 生成。脚本会扫描 Kvasir-SEG、CVC-ClinicDB 和 KvasirCVC 规范化目录，将可用样本合并为五折交叉验证池；PolypGen 仅作为外部测试集，并写出独立的 `external_polypgen_ids.txt`。划分生成后会检查训练/验证集合是否包含 PolypGen 样本，以避免外部测试泄漏。

验证脚本支持三种 bounding-box prompt 协议：`mask`、`yolo` 与 `none`。默认 `mask` 协议由真值 mask 派生 box，适合评估给定定位提示下的分割能力；`yolo` 协议使用检测器或缓存框，适合评估更接近部署场景的自动提示；`none` 协议用于无框提示对照。所有实验结果必须同时报告 split 路径、checkpoint、bbox source、是否启用 exemplar/retrieval、以及是否使用 hard-case gate。

### 4.3 LoRA 与医学适配

Stage-A LoRA 默认作用于视觉编码器后部注意力层和 mask decoder 相关投影层。该策略避免在小规模医学数据上全量微调 SAM3，从而降低过拟合和显存开销。训练前的 preflight 会检查 split 完整性、PolypGen 泄漏、LoRA 替换数量、可训练参数比例以及一次前向/反向是否可运行。

医学适配部分由轻量 residual refine head、可选 `MedicalImageAdapter` 与可选 `BoundaryAwareAdapter` 组成。`refine_head` 始终存在，用于对 mask logits 增加小幅残差修正；`MedicalImageAdapter` 通过 bottleneck 和纹理卷积分支增强医学图像特征；`BoundaryAwareAdapter` 在训练阶段可使用 GT mask 生成边界监督，在推理阶段根据粗预测或对比先验形成边界增强。

### 4.4 Exemplar Prompt

Exemplar Prompt 分支使用人工核验的 positive、negative 与 boundary 样本库构建病例原型。原型构建支持加权均值、query-aware attention fusion 和方差感知的 clustered subprototypes。当检索到的 exemplar 分布过散时，系统可以退化为多个子原型以减少单一 prototype 的不稳定性。

`ExemplarPromptAdapter` 将三类原型分别投影为 prompt tokens，并拼接为 SAM3 可接收的 visual prompt embedding。该设计使模型不仅依赖当前图像的 text/box prompt，也能利用相似病灶、负样本区域和边界形态的先验信息。训练时可进一步加入对比损失、负样本抑制和一致性约束，但这些损失项应根据实际命令行开关报告。

### 4.5 RSS-DA 检索增强

RSS-DA 分支面向更深层的检索增强。它首先基于训练样本库检索 positive、negative 与 boundary exemplars，然后由 `RetrievalSpatialSemanticAdapter` 生成空间 bias、语义 prototype map、decoder feature bias、negative suppression map 与 mask logit bias。SAM3 tensor forward wrapper 会在 encoder memory 和最终 mask logits 处使用这些先验，从而实现 prompt-level、feature-level 与 logit-level 的联合增强。

外部 PolypGen 验证时，RSS-DA runtime 会检查 memory bank 是否包含 PolypGen 样本，以避免检索库泄漏。现有结果显示，检索增强可能改善部分难样本或 gated 子集，但也可能带来整体 Dice 的小幅下降。因此，论文表述应强调“检索增强对难样本具有潜在收益，同时需要通过 gate 和样本库质量控制抑制保守分割倾向”。

### 4.6 Loss 与指标

基础分割损失由 BCE、Dice loss 和 BoundaryBandDiceLoss 组成。基础 LoRA 训练默认关闭 contrastive、negative suppression 和 consistency 权重；Exemplar Prompt 与 RSS-DA 分支可根据实验设置启用这些额外项。验证阶段统一报告 Dice、IoU、Precision、Recall、Boundary F1、HD95、ASSD、FPR 和 FNR，以同时反映区域重叠、边界质量和错误率。

### 4.7 Agent 系统边界

Agent 子系统用于管理 exemplar memory 的工程闭环，包括样本摄入、质量评分、生命周期演化、检索 prior 生成和医生反馈更新。它与模型实验共享 positive/negative/boundary exemplar 的概念，但不是当前 `MedicalSAM3/scripts` 训练与验证入口的必要依赖。因此，论文主实验应以 `MedicalSAM3` 代码路径为准，Agent 相关内容可作为临床交互系统、持续学习或未来工作章节介绍。

## 5. 文档一致性检查清单

发布或提交论文前，请逐项检查：

- [ ] 每个结果表都标明 split 文件路径，例如 `MedicalSAM3/outputs/medex_sam3/splits/fold_0/val_ids.txt` 或 `external_polypgen_ids.txt`。
- [ ] 每个结果表都标明 `--bbox-source`，区分 `mask`、`yolo` 和 `none`。
- [ ] 基础 LoRA、Exemplar Prompt、RSS-DA 三个入口分开描述，不把它们写成同一默认模型。
- [ ] 写 LoRA 时区分“SAM3 基座仅 LoRA 可训练”和“外层 refine head/adapter 可训练”。
- [ ] 写 adapter 时明确 `MedicalImageAdapter` 与 `BoundaryAwareAdapter` 是可选模块。
- [ ] 不再使用 `MultiScaleMedicalAdapter` 作为当前主实验模块名称，除非补齐同名实现和调用链。
- [ ] 写 Exemplar Prompt 时说明 negative tokens 会进入当前 prompt token 序列。
- [ ] 写 loss 时按训练入口说明实际启用项，不能把可选 contrastive/negative/consistency 写成基础 LoRA 默认项。
- [ ] `run_ablation.py` 生成的 dummy metrics 不进入论文性能表。
- [ ] `最终结果/` 中不同协议的 summary 不混合比较；同一表格必须统一 checkpoint、split、bbox source 和 evaluation script。
- [ ] Agent 文档标注为系统扩展或临床闭环，不作为主实验必经流程。
- [ ] 后端 README 只描述应用接口和部署，不承担论文模型训练协议说明。
- [ ] 所有“完整 5-Fold”命令去掉 `--max-train-steps` / `--max-val-steps`，或改名为 smoke/短训命令。
- [ ] 所有 README 中引用的脚本路径通过 `rg --files` 或实际运行确认存在。
- [ ] 结果章节至少同时报告 Dice、IoU 和 Boundary F1；若报告 HD95/ASSD，说明当前空边界处理策略。

## 6. 建议的下一步

1. 先重写 `README_medex_sam3.md`：修正主路线、删除 `train_ext.py`、明确 bbox source、区分 smoke 与完整训练。
2. 再重写 `doc.md`：按“已实现并进入主实验 / 已实现但可选 / 系统扩展或未来工作”拆分。
3. 最后整理 `最终结果/`：用同一评价协议重跑或至少生成一张 provenance 表，列出每个 summary 的 split、checkpoint、bbox source、retrieval mode、gate、top-k 与指标。

