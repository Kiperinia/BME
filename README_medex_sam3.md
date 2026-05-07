# MedEx-SAM3

MedEx-SAM3: Human-Verified Multi-Exemplar Prototype Prompting for Polyp Segmentation.

该实现将 MedEx-SAM3 主路线迁移到官方 SAM3 image model 的 tensor-level pipeline，并在仓库内保留原有 MedicalSAM3 扩展路线。旧脚本 MedicalSAM3/train_ext.py 仍然保留可导入，但不再是推荐主训练入口。

## Architecture

MedEx-SAM3 由以下模块串联组成：

1. SAM3 official image model
2. LoRA
3. Medical Adapter
4. Exemplar Prompt Adapter
5. Prototype Builder
6. Agent-curated Memory Bank

主路径位于以下目录：

- MedicalSAM3/sam3_official
- MedicalSAM3/adapters
- MedicalSAM3/exemplar
- MedicalSAM3/agents
- MedicalSAM3/scripts

## Data Preparation

训练与交叉验证数据：

- Kvasir-SEG
- CVC-ClinicDB

外部测试集：

- PolypGen

PolypGen 只允许作为 external test set，禁止参与 training、validation、hyperparameter tuning、prototype building、memory update 和 model selection。

## Repro Commands

准备 5-fold 划分：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\prepare_5fold_polyp.py --dummy
```

训练 LoRA 与医学适配：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\train_lora_medical.py --config MedicalSAM3\configs\medex_sam3_lora.yaml --fold 0 --dummy --enable-vision-lora --enable-mask-decoder-lora --enable-boundary-adapter --enable-msfa-adapter
```

构建 exemplar bank：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\build_exemplar_bank.py --split-file MedicalSAM3\outputs\medex_sam3\splits\fold_0\train_ids.txt --dummy
```

人工审核并更新 memory：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\update_memory_from_review.py --memory-bank-dir MedicalSAM3\outputs\medex_sam3\exemplar_bank --dummy
```

训练 exemplar prompt：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\train_exemplar_prompt.py --memory-bank MedicalSAM3\outputs\medex_sam3\exemplar_bank --dummy --enable-negative-suppression --enable-consistency-loss --enable-contrastive-loss
```

验证：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\validate_medex_sam3.py --mode fold --dummy --memory-bank MedicalSAM3\outputs\medex_sam3\exemplar_bank --prompt-checkpoint MedicalSAM3\outputs\medex_sam3\exemplar_prompt\prompt_adapter.pt
```

运行消融：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\run_ablation.py --config MedicalSAM3\configs\medex_sam3_ablation.yaml --dummy
```

汇总交叉验证与消融：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\summarize_cv_results.py
```

## Leakage Guardrails

- PolypGen 不参与任何训练或调参。
- MedicalSAM3/agents/leakage_checker.py 会检查 source_dataset、external_test_ids、fold 泄漏与重复样本。
- ExemplarMemoryBank 默认拒绝 PolypGen item。

## Ablation Matrix

run_ablation.py 支持以下配置：

1. SAM3 zero-shot
2. SAM3 + LoRA
3. SAM3 + LoRA + MedicalAdapter
4. SAM3 + LoRA + BoundaryAwareAdapter
5. SAM3 + positive single exemplar
6. SAM3 + positive Top-3 prototype
7. SAM3 + positive Top-5 weighted prototype
8. SAM3 + positive + negative prototype
9. SAM3 + positive + negative + boundary prototype
10. SAM3 + human-verified memory v1
11. SAM3 + human-verified memory v2

## FAQ

找不到 SAM3 checkpoint：

- 使用 dummy 模式做烟雾测试。
- 真实运行时通过 --checkpoint 指向官方 SAM3 image model checkpoint。
- 当前环境若无法访问 gated repo，会自动 fallback 到 dummy model，仅用于 smoke test。

CUDA OOM：

- 将 batch size 降到 1。
- 使用 fp16 或 bf16。
- 先只启用 LoRA，不启用额外 adapter。

module name 不匹配：

- 运行 python -m MedicalSAM3.sam3_official.module_inspector --device cpu --dtype fp32。
- 查看生成的 sam3_modules.txt 和 sam3_lora_targets.json。

memory bank 为空：

- 先运行 build_exemplar_bank.py。
- 再运行 update_memory_from_review.py 完成 human_verified 更新。

human review CSV 格式错误：

- 重新导出 review_queue.csv。
- 必填列至少包括 item_id、type、quality_score、notes、accept。
