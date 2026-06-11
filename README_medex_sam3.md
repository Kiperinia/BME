# MedEx-SAM3

MedEx-SAM3 是本仓库中围绕官方 SAM3 构建的医学息肉分割实验链路。当前代码不应被描述为“一条默认同时启用所有模块的主路线”，而应拆成三个可以组合但入口独立的实验分支：

1. 基础 LoRA 分支：`MedicalSAM3/scripts/train_lora_medical.py`，默认是 official SAM3 image model + Stage-A LoRA + residual refine head。
2. Exemplar Prompt 分支：`MedicalSAM3/scripts/train_exemplar_prompt.py`，依赖人工核验的 exemplar memory bank。
3. RSS-DA 检索增强分支：`MedicalSAM3/scripts/train_rssda.py` 和 `MedicalSAM3/scripts/validate_rssda.py`，用于检索空间语义先验注入。

`MedicalImageAdapter` 和 `BoundaryAwareAdapter` 是可选模块，不是基础 LoRA 训练的默认组件。外部验证结果必须同时说明 split、checkpoint、`--bbox-source`、是否启用 exemplar/retrieval、以及是否启用 hard-case gate。

## 训练前必须先跑 preflight

本机建议只执行低负载检查；真实 checkpoint preflight、单 fold 短训、完整 5-fold 和 external evaluation 建议在 Linux GPU 服务器上执行。服务器命令可以参考 `MedicalSAM3/outputs/medex_sam3/server_commands.md`。

Linux 服务器如果需要挂到 tmux 里执行安装，可以在仓库根目录运行：

```bash
chmod +x install-python-deps-linux.sh
RUN_IN_TMUX=1 TMUX_SESSION_NAME=medex-install TORCH_CHANNEL=cu126 ./install-python-deps-linux.sh
tmux attach -t medex-install
```

如果已经在 tmux 会话内部，直接执行安装脚本即可，不需要额外设置 `RUN_IN_TMUX`。

本机低负载预检查：

```powershell
.\.venv\Scripts\python.exe -m compileall MedicalSAM3
```

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\preflight_medex_sam3.py --fold 0 --image-size 128 --precision fp32 --device cpu --allow-dummy
```

preflight 默认只做检查；只有显式传入 `--run-short-train` 时，才会调用单 fold 短训。

真实 SAM3 preflight：

```bash
python MedicalSAM3/scripts/preflight_medex_sam3.py \
  --checkpoint /path/to/sam3.pt \
  --fold 0 \
  --image-size 512 \
  --precision fp16 \
  --device cuda \
  --require-official-sam3 \
  --min-lora-modules 1
```

## 数据划分

训练/验证划分由 `prepare_5fold_polyp.py` 生成。脚本会扫描 Kvasir-SEG、CVC-ClinicDB 和 KvasirCVC 规范化目录，并将可用样本合并为五折交叉验证池；PolypGen 仅作为外部测试集使用。

```bash
python MedicalSAM3/scripts/prepare_5fold_polyp.py \
  --data-root MedicalSAM3/data \
  --output-dir MedicalSAM3/outputs/medex_sam3/splits \
  --seed 42
```

生成后请检查：

- `MedicalSAM3/outputs/medex_sam3/splits/split_summary.json`
- `MedicalSAM3/outputs/medex_sam3/splits/fold_k/train_ids.txt`
- `MedicalSAM3/outputs/medex_sam3/splits/fold_k/val_ids.txt`
- `MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt`

PolypGen 不能进入 train split、val split、early stopping、memory bank 或 prototype building。若发现泄漏，应直接视为阻塞问题。

## Dummy Smoke

准备 dummy split：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\prepare_5fold_polyp.py --data-root MedicalSAM3/data --output-dir MedicalSAM3/outputs/medex_sam3/splits --dummy
```

运行 dummy smoke：

```powershell
.\.venv\Scripts\python.exe MedicalSAM3\scripts\train_lora_medical.py --fold 0 --dummy --allow-dummy --epochs 1 --batch-size 1 --image-size 128 --precision fp32 --device cpu --max-train-steps 2 --max-val-steps 2
```

## 单 Fold 短训

真实单 fold 短训只建议在服务器上运行。以下命令带有 `--max-train-steps` 和 `--max-val-steps`，只用于 smoke / short run，不代表完整训练。

```bash
python MedicalSAM3/scripts/train_lora_medical.py \
  --fold 0 \
  --checkpoint /path/to/sam3.pt \
  --epochs 1 \
  --batch-size 1 \
  --image-size 512 \
  --precision fp16 \
  --device cuda \
  --require-official-sam3 \
  --min-lora-modules 1 \
  --max-train-steps 10 \
  --max-val-steps 5
```

## 完整 5-Fold

只有 `MedicalSAM3/outputs/medex_sam3/preflight/readiness_checklist.json` 中 `ready_for_full_training=true` 时，才建议进入完整 5-fold。完整训练命令不应包含 step cap。

```bash
for FOLD in 0 1 2 3 4; do
  python MedicalSAM3/scripts/train_lora_medical.py \
    --fold ${FOLD} \
    --checkpoint /path/to/sam3.pt \
    --epochs 1 \
    --batch-size 1 \
    --image-size 512 \
    --precision fp16 \
    --device cuda \
    --require-official-sam3 \
    --min-lora-modules 1
done
```

基础 LoRA 分支默认不启用 `MedicalImageAdapter` 或 `BoundaryAwareAdapter`。如果需要报告 adapter 版本，请在命令和实验表中明确对应开关或 config。

## Exemplar Prompt 流程

构建 exemplar bank：

```bash
python MedicalSAM3/scripts/build_exemplar_bank.py \
  --split-file MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt \
  --output-dir MedicalSAM3/outputs/medex_sam3/exemplar_bank \
  --checkpoint /path/to/sam3.pt \
  --image-size 256
```

人工审核后更新 memory：

```bash
python MedicalSAM3/scripts/update_memory_from_review.py \
  --memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank/memory_v0.json \
  --review-csv MedicalSAM3/outputs/medex_sam3/exemplar_bank/review_queue.csv \
  --output-dir MedicalSAM3/outputs/medex_sam3/exemplar_bank
```

Exemplar Prompt preflight：

```bash
python MedicalSAM3/scripts/train_exemplar_prompt.py \
  --memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
  --checkpoint /path/to/sam3.pt \
  --split-file MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt \
  --prototype-mode weighted_mean \
  --preflight-only
```

Exemplar Prompt 训练：

```bash
python MedicalSAM3/scripts/train_exemplar_prompt.py \
  --memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
  --checkpoint /path/to/sam3.pt \
  --split-file MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt \
  --prototype-mode weighted_mean \
  --top-k-positive 3 \
  --top-k-negative 1 \
  --top-k-boundary 1 \
  --enable-negative-suppression
```

当前实现中 positive、boundary 和 negative prototypes 都会被投影为 prompt tokens 并进入最终 token 序列；negative exemplar 还可以通过 negative suppression loss 约束假阳性响应。

## PolypGen External Test

PolypGen 只能用于 external final evaluation，不能进入 train/val、early stopping、memory bank 或 prototype building。

验证脚本支持三种 box prompt 协议：

- `--bbox-source mask`：默认协议，由真值 mask 派生 box，适合评估给定定位提示下的分割上限。
- `--bbox-source yolo`：使用 YOLO detector 或缓存框，更接近自动化部署。
- `--bbox-source none`：无框提示对照。

使用默认 mask-derived box 的外部验证：

```bash
python MedicalSAM3/scripts/validate_medex_sam3.py \
  --external-test \
  --split-file MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt \
  --checkpoint /path/to/sam3.pt \
  --lora-checkpoint MedicalSAM3/outputs/medex_sam3/fold_0/best_lora.pt \
  --adapter-checkpoint MedicalSAM3/outputs/medex_sam3/fold_0/best_adapter.pt \
  --prompt-checkpoint MedicalSAM3/outputs/medex_sam3/exemplar_prompt/prompt_adapter.pt \
  --memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
  --bbox-source mask \
  --output-dir MedicalSAM3/outputs/medex_sam3/eval
```

使用 YOLO box 的外部验证需要显式指定：

```bash
python MedicalSAM3/scripts/validate_medex_sam3.py \
  --external-test \
  --split-file MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt \
  --checkpoint /path/to/sam3.pt \
  --lora-checkpoint MedicalSAM3/outputs/medex_sam3/fold_0/best_lora.pt \
  --bbox-source yolo \
  --yolo-weights /path/to/yolo.pt \
  --yolo-cache MedicalSAM3/outputs/medex_sam3/yolo_bbox_cache/validate_medex_sam3.json \
  --output-dir MedicalSAM3/outputs/medex_sam3/eval_yolo
```

## RSS-DA 检索增强

RSS-DA 是单独的检索增强分支，不是基础 LoRA 训练的默认路径。外部验证示例：

```bash
python MedicalSAM3/scripts/validate_rssda.py \
  --external-test \
  --split-file MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt \
  --checkpoint /path/to/sam3.pt \
  --lora-checkpoint MedicalSAM3/outputs/medex_sam3/fold_0/best_lora.pt \
  --memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
  --bank-purpose external-eval \
  --retrieval-policy uncertainty-aware \
  --mode joint \
  --bbox-source mask \
  --output-dir MedicalSAM3/outputs/medex_sam3/rssda_eval
```

外部评估时，检索库必须只包含训练 split 的样本，不能包含 PolypGen。

## 常见错误

fallback 到 dummy：

- 正式训练默认不允许 fallback。检查 `model_build_report.json` 和训练 `preflight_report.json` 中的 `used_dummy_fallback`。

LoRA replaced modules = 0：

- 查看 `MedicalSAM3/sam3_lora_targets.json`、`MedicalSAM3/outputs/medex_sam3/preflight/lora_injection_report.json`，以及官方模块扫描结果 `MedicalSAM3/sam3_modules.txt`。

hidden_dim mismatch：

- exemplar prompt token 最后一维必须等于 SAM3 hidden_dim；优先看 `model_build_report.json` 的 `hidden_dim` 字段。

split 为空：

- 先运行 `prepare_5fold_polyp.py`，检查 `split_summary.json`、`fold_k/train_ids.txt` 和 `fold_k/val_ids.txt`。

PolypGen leakage：

- PolypGen 只能存在于 `external_polypgen_ids.txt` 和 external final eval；如果进入 train/val 或 memory bank，应停止实验并重新生成 split/bank。

memory bank 为空：

- `train_exemplar_prompt.py` 只接受 `human_verified=True` 且正例数量 `>= 1` 的 memory bank。

CUDA OOM：

- 从 `batch-size 1`、`fp16`、`stage_a` LoRA 开始；必要时只启用 LoRA，不启用额外 adapter。

official SAM3 API 变化：

- 重新运行 `MedicalSAM3/scripts/preflight_medex_sam3.py` 和 module inspector，确认 `sam3_modules.txt`、`sam3_lora_targets.json`、`tensor_forward_report.json` 是否仍匹配当前安装的 SAM3。
