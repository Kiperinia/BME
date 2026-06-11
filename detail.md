# 当前实验观察备忘

本文只记录当前结果解读，不作为完整论文结果表。引用这些结论时，必须同时注明对应的结果文件、split、checkpoint、bbox source 和检索策略。

## 观察结论

1. Stage-A LoRA 是当前最稳定的基础分支。它对应 `train_lora_medical.py` 的基础路径，默认是 official SAM3 + Stage-A LoRA + residual refine head。
2. 边界相关改进对边界质量有潜在帮助，但要区分两类机制：基础 loss 中的 BoundaryBandDiceLoss，以及显式启用的 `BoundaryAwareAdapter`。后者不是默认开启。
3. Exemplar Prompt / RSS-DA 检索增强存在保守分割倾向。样本库质量、negative prototype、gate 策略和 prompt adapter 训练不足时，个别难样本可能改善，但全量 Dice 可能小幅下降，当前观察量级约为 0.003。

## 引用限制

- 不要把该观察写成最终结论，除非用同一评价脚本和同一协议重跑。
- 不要混合比较 `mask`、`yolo` 和 `none` 三种 bbox source。
- 不要把 `run_ablation.py` 的 dummy metrics 当作真实消融结果。
- 不要把 `agent/` 下的样本库闭环代码写成当前 `MedicalSAM3/scripts` 主训练链路的必经步骤。

## 推荐后续整理

正式写论文结果表前，建议为每个 summary 结果补一张 provenance 表，至少包含：

- output path
- split path
- checkpoint / LoRA checkpoint / adapter checkpoint / prompt checkpoint
- bbox source
- retrieval mode / retrieval policy
- top-k positive / negative / boundary
- hard-case gate 是否启用
- Dice / IoU / Boundary F1 / FPR / FNR
