# Exemplar Bank Protocol

train_bank 仅用于 strict retrieval protocol。

- train_bank/positive 与 train_bank/negative 只允许来自 Kvasir/CVC train split 的代表性 exemplar。
- train_bank 内禁止放入 val split、PolypGen 或任何 external test image。
- continual_bank 只用于后续 continual adaptation，不参与当前 strict external run。

## 文档边界

该目录说明的是检索库的数据来源约束，不等同于训练脚本默认启用 Exemplar Prompt 或 RSS-DA。基础 LoRA 训练不依赖本目录；只有运行 Exemplar Prompt 或 RSS-DA 检索增强分支时，才需要加载经过人工核验且无 PolypGen 泄漏的 memory bank。

外部评估报告必须同时注明 memory bank 路径、bank purpose、split 路径和 `--bbox-source`。默认 `--bbox-source mask` 表示由 GT mask 派生 box，不是 YOLO 自动检测框。
