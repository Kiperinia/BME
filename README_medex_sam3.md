# MedEx-SAM3

MedEx-SAM3 当前主路线是：official SAM3 image model + LoRA + medical/boundary adapters + human-verified exemplar memory + prototype fusion。

当前主入口位于 MedicalSAM3/scripts/train_lora_medical.py 和 MedicalSAM3/scripts/train_exemplar_prompt.py。旧的 MedicalSAM3/train_ext.py 仍保留为旧 baseline / wrapper 路线，不是现在这条主训练链。

## 训练前必须先跑 preflight

本机只建议执行低负载检查；真实 checkpoint preflight、单 fold 短训、5-fold 和 external evaluation 请在 Linux 服务器上执行。服务器命令已写入 MedicalSAM3/outputs/medex_sam3/server_commands.md。

Linux 服务器如果需要挂到 tmux 里执行安装，可以直接在仓库根目录运行：

```bash
chmod +x install-python-deps-linux.sh
RUN_IN_TMUX=1 TMUX_SESSION_NAME=medex-install TORCH_CHANNEL=cu126 ./install-python-deps-linux.sh
tmux attach -t medex-install
```

如果已经在 tmux 会话内部，直接执行安装脚本即可，不需要额外设置 RUN_IN_TMUX。

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

真实单 fold 短训只建议在服务器上运行：

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

只有 MedicalSAM3/outputs/medex_sam3/preflight/readiness_checklist.json 中 ready_for_full_training=true 时，才允许进入完整 5-fold。

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
		--min-lora-modules 1 \
		--max-train-steps 10 \
		--max-val-steps 5
done
```

## Exemplar 流程

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

exemplar prompt preflight：

```bash
python MedicalSAM3/scripts/train_exemplar_prompt.py \
	--memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
	--checkpoint /path/to/sam3.pt \
	--split-file MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt \
	--prototype-mode weighted_mean \
	--preflight-only
```

exemplar prompt 训练：

```bash
python MedicalSAM3/scripts/train_exemplar_prompt.py \
	--memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
	--checkpoint /path/to/sam3.pt \
	--split-file MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt \
	--prototype-mode weighted_mean \
	--top-k-positive 3 \
	--enable-negative-suppression
```

## PolypGen External Test

PolypGen 只能用于 external final evaluation，不能进入 train/val、early stopping、memory bank 或 prototype building。

```bash
python MedicalSAM3/scripts/validate_medex_sam3.py \
	--external-test \
	--split-file MedicalSAM3/outputs/medex_sam3/splits/external_polypgen_ids.txt \
	--checkpoint /path/to/sam3.pt \
	--lora-checkpoint MedicalSAM3/outputs/medex_sam3/fold_0/best_lora.pt \
	--adapter-checkpoint MedicalSAM3/outputs/medex_sam3/fold_0/best_adapter.pt \
	--prompt-checkpoint MedicalSAM3/outputs/medex_sam3/exemplar_prompt/prompt_adapter.pt \
	--memory-bank MedicalSAM3/outputs/medex_sam3/exemplar_bank \
	--output-dir MedicalSAM3/outputs/medex_sam3/eval
```

## 常见错误

fallback 到 dummy：

- 正式训练默认不允许 fallback。检查 model_build_report.json 和训练 preflight_report.json 中 used_dummy_fallback。

LoRA replaced modules = 0：

- 查看 MedicalSAM3/sam3_lora_targets.json、MedicalSAM3/outputs/medex_sam3/preflight/lora_injection_report.json，以及官方模块扫描结果 MedicalSAM3/sam3_modules.txt。

hidden_dim mismatch：

- exemplar prompt token 最后一维必须等于 SAM3 hidden_dim；优先看 model_build_report.json 的 hidden_dim 字段。

split 为空：

- 先运行 prepare_5fold_polyp.py，检查 split_summary.json、fold_k/train_ids.txt 和 fold_k/val_ids.txt。

PolypGen leakage：

- PolypGen 只能存在于 external_polypgen_ids.txt 和 external final eval；如果进入 train/val 或 memory_v1.json，直接视为阻塞问题。

memory bank 为空：

- train_exemplar_prompt.py 只接受 human_verified=True 且正例数量 >= 1 的 memory bank。

CUDA OOM：

- 从 batch-size 1、fp16、stage_a LoRA 开始；必要时只启用 LoRA，不启用额外 adapter。

official SAM3 API 变化：

- 重新运行 MedicalSAM3/scripts/preflight_medex_sam3.py 和 module_inspector，确认 `sam3_modules.txt`、`sam3_lora_targets.json`、`tensor_forward_report.json` 是否仍匹配当前安装的 sam3。
