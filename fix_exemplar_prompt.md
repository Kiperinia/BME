# Exemplar Prompt 修复记录与实验边界

本文是 Exemplar Prompt 分支的历史修复记录，不是当前论文方法章节，也不是所有改动均已进入主训练路径的证明。正式方法描述请以 `doc.md`、`README_medex_sam3.md` 和实际脚本为准。

## 背景

`detail.md` 中记录过一个重要观察：

> 样本库质量不足、Exemplar Prompt 网络与训练不够稳定时，整体预测会出现保守分割倾向；个别难样本可能改善，但全量 Dice 可能下降约 0.003。

这个观察与当前结果产物中的部分检索增强实验趋势一致：检索/样本库增强并不天然保证全量指标提升，必须同时控制 memory bank 质量、负样本作用方式、gate 策略和验证协议。

## 已确认的代码事实

1. 当前 `ExemplarPromptAdapter` 会将 positive、boundary 和 negative prototypes 都投影为 prompt tokens，并拼接进最终 visual prompt token 序列。
2. negative exemplar 不只是“后处理抑制项”；它既可以进入 prompt token 序列，也可以通过 negative suppression loss 或 RSS-DA 中的 negative prior 抑制假阳性。
3. `train_exemplar_prompt.py` 是独立训练入口，不能写成基础 LoRA 训练默认启用的组件。
4. contrastive、negative suppression 和 consistency 等损失是否参与训练取决于命令行开关与 forward 输入。
5. 外部验证必须说明 `--bbox-source`。默认 `mask` 表示由 GT mask 派生 box，不是自动检测框协议。

## 文档写法要求

在 README 或论文中描述 Exemplar Prompt 时，推荐使用以下表述：

> Exemplar Prompt 分支使用人工核验的 positive、negative 与 boundary 样本库构建病例原型。三类原型会被投影为 visual prompt tokens 注入 SAM3 prompt 路径；其中 negative exemplar 还可通过负样本抑制损失或检索先验抑制假阳性响应。该分支是基础 LoRA 之外的独立训练/验证路径，实验结果需要单独报告 memory bank、top-k、loss 开关、bbox source 和 gate 策略。

## 不再作为当前事实引用的内容

旧版本文曾包含若干代码修改建议，例如替换 embedding extractor、重写评分逻辑、补充直方图分析等。这些内容应视为开发思路或后续实验任务，不能在论文中写成已经由当前主实验完整验证的事实。

如果需要继续推进该方向，建议新建 issue 或实验计划，明确：

- 使用哪一个 split 和 memory bank。
- 是否使用 SAM3 image embeddings 构建 exemplar embedding。
- 是否启用 negative suppression、contrastive loss 和 consistency loss。
- 使用 `mask`、`yolo` 还是 `none` bbox source。
- 评价是否基于同一 checkpoint 和同一 external test script。
