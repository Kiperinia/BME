# 闭环修正：Exemplar 训练总体 Dice 下降 0.003

## 问题定位

`detail.md` 指出："样本库质量不够 + Exemplar Prompt 的网络、训练做的不太好 → 保守分割，个别难样本提升但总体 Dice 下降 ~0.003"

代码根因三点：
1. `build_exemplar_bank.py` 用 2 层 CNN（ExemplarEncoder）提取嵌入，与 SAM3 查询嵌入特征空间不对齐，检索不可靠
2. 同文件所有分数硬编码（quality=0.8, diversity=0.5 等），`PrototypeBuilder._item_score` 的加权评分形同虚设
3. 训练 epoch=1，contrastive loss 未实际生效，prompt adapter 学不到有用映射

---

## 修改清单

### 修改 1：`MedicalSAM3/scripts/build_exemplar_bank.py`

**目标**：用 SAM3 的 image_embeddings 替代 ExemplarEncoder，计算数据驱动分数。

具体改动（共 3 处）：

**1a. 在 main() 开头构建 SAM3 模型并计算零样本分数（约第 127 行后插入）**

在 `records = read_records(args.split_file)` 之后、`if args.dummy and not records` 之前，插入：

```python
# --- SAM3-based embedding & scoring ---
from MedicalSAM3.scripts.common import MedExSam3SegmentationModel, compute_segmentation_metrics
_sam3_model = None
_sam3_wrapper = None
_device = "cpu"
if not args.dummy and args.checkpoint:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _base = build_official_sam3_image_model(args.checkpoint, device=_device, dtype="fp32", compile_model=False, allow_dummy_fallback=False)
    _sam3_wrapper = Sam3TensorForwardWrapper(model=_base, device=_device, dtype="fp32")
    _sam3_model = MedExSam3SegmentationModel(wrapper=_sam3_wrapper, enable_medical_adapter=False, enable_boundary_adapter=False, embed_dim=128).to(_device)
```

**1b. 在循环内替换嵌入提取和分数计算（约第 155-196 行）**

找到 `for index, record in enumerate(records):` 循环，将里面 _crop_tensor 调用之后的嵌入提取和 ExemplarItem 构建替换为：

```python
    # 计算 SAM3 零样本分数
    quality = 0.8  # fallback
    difficulty = 0.5
    if _sam3_model is not None:
        with torch.no_grad():
            _img, _msk = load_record_tensors(record, args.image_size, fallback_index=index)
            _out = _sam3_model(images=_img.unsqueeze(0).to(_device), boxes=_box.unsqueeze(0).to(_device))
            _metrics = compute_segmentation_metrics(_out["mask_logits"].detach().cpu(), _msk.unsqueeze(0))
            quality = float(_metrics.get("dice", 0.8))
            difficulty = 1.0 - quality

    # 用 SAM3 image_embeddings 做 masked average pooling 作为嵌入
    if _sam3_wrapper is not None:
        with torch.no_grad():
            _feat = _sam3_wrapper(images=_img.unsqueeze(0).to(_device))["image_embeddings"]  # [1, C, H, W]
            _mask_rs = F.interpolate(_msk.unsqueeze(0).to(_device), size=_feat.shape[-2:], mode="nearest")
            _embed = (_feat * _mask_rs).sum(dim=(-2, -1)) / _mask_rs.sum(dim=(-2, -1)).clamp_min(1e-6)
            _embed = F.normalize(_embed, dim=-1).squeeze(0)
        embedding_path = embeddings_dir / f"{item_id}.pt"
        torch.save(_embed.cpu(), embedding_path)
    else:
        # fallback: 原有 Encoder
        encoder = ExemplarEncoder(embed_dim=_infer_embed_dim(args.checkpoint, allow_dummy=args.dummy))
        with torch.no_grad():
            embeddings = encoder(crop_tensor.unsqueeze(0), mask_tensor.unsqueeze(0) if exemplar_type != "negative" else None)
        torch.save(embeddings, embedding_path)
        quality = 0.8 if exemplar_type != "negative" else 0.5
        difficulty = 0.5

    # 构建 items 时传入数据驱动分数
    item = ExemplarItem(
        item_id=item_id,
        ...
        quality_score=quality,
        difficulty_score=difficulty,
        diversity_score=0.5,  # 可在第二次遍历时更新
        uncertainty_score=0.2,
        ...
    )
```

**1c. 在保存 bank 前做一次 diversity 更新（加在 bank.save 之前）**

```python
# 第二次遍历：计算 diversity
_emb_list = []
for item in bank.items:
    if item.embedding_path:
        _e = torch.load(item.embedding_path, map_location="cpu", weights_only=False)
        if isinstance(_e, dict):
            _e = _e.get("foreground_embedding", _e.get("global_embedding", next(iter(_e.values()))))
        _emb_list.append((item, F.normalize(_e.squeeze(0) if _e.dim() > 1 else _e, dim=0)))
if len(_emb_list) > 1:
    _all_embs = torch.stack([e for _, e in _emb_list])
    _sims = _all_embs @ _all_embs.T
    for idx, (item, _) in enumerate(_emb_list):
        _sim = _sims[idx].clone()
        _sim[idx] = -1.0
        item.diversity_score = float(1.0 - _sim.max().item())
```

---

### 修改 2：`MedicalSAM3/configs/medex_sam3_exemplar.yaml`

将文件内容替换为：

```yaml
seed: 42
image_size: 128
batch_size: 1
epochs: 10
lr: 0.0001
weight_decay: 0.0001
precision: fp32
prototype_mode: weighted_mean
top_k_positive: 3
top_k_negative: 2
negative_lambda: 0.35
positive_weight: 1.0
negative_weight: 0.25
similarity_threshold: 0.6
confidence_scale: 8.0
similarity_weighting: soft
similarity_temperature: 0.125
retrieval_policy: uncertainty-aware
uncertainty_threshold: 0.35
uncertainty_scale: 10.0
policy_activation_threshold: 0.05
residual_strength: 0.5
retrieval_mode: joint
top_k_boundary: 1
enable_negative_suppression: true
enable_consistency_loss: true
enable_contrastive_loss: true
```

关键变更：`epochs: 10`、`top_k_positive: 3`、`top_k_negative: 2`、`similarity_threshold: 0.6`。

---

### 修改 3：`MedicalSAM3/scripts/train_exemplar_prompt.py`

**3a. 第 279 行：移除不必要的 detach（允许梯度回流）**

```python
# 改前
query_embedding = warmup_outputs["query_embedding"].detach()[0]
# 改后
query_embedding = warmup_outputs["query_embedding"][0]
```

**3b. 第 291-296 行：将 unsqueeze 统一化并传原始 query_feat**

```python
# 改前
prompt_tokens, prompt_aux = prompt_adapter(
    positive_proto=positive_proto,
    negative_proto=negative_proto,
    boundary_proto=boundary_proto,
    query_feat=warmup_outputs["query_embedding"],
)
# 改后（用同一个 detached query 避免冗余计算）
query_feat_for_adapter = query_embedding.detach().unsqueeze(0)
prompt_tokens, prompt_aux = prompt_adapter(
    positive_proto=positive_proto,
    negative_proto=negative_proto,
    boundary_proto=boundary_proto,
    query_feat=query_feat_for_adapter,
)
```

**3c. 在训练循环最后添加验证集评估（约第 360 行前）**

```python
# 每 epoch 结束后做一次验证
if epoch == args.epochs - 1 or (epoch > 0 and epoch % 3 == 0):
    model.eval()
    val_metrics_list = []
    with torch.no_grad():
        for val_batch in val_loader:  # 需要外部传入 val_loader
            v_images = val_batch["images"].to(device)
            v_masks = val_batch["masks"].to(device)
            v_boxes = val_batch["boxes"].to(device)
            v_out = model(images=v_images, boxes=v_boxes, text_prompt=val_batch["text_prompt"], gt_mask=v_masks)
            v_metrics = compute_segmentation_metrics(v_out["mask_logits"].detach(), v_masks.detach())
            val_metrics_list.append(v_metrics)
    avg_val_dice = float(torch.tensor([m["dice"] for m in val_metrics_list]).mean())
    print(f"Epoch {epoch} val_dice: {avg_val_dice:.4f}")
```

注意：需要在主函数参数中加 `--val-split-file` 并构建 val_loader。

---

### 修改 4（可选）：`MedicalSAM3/adapters/exemplar_prompt_adapter.py`

扩增 token 容量以提升表达力：

```python
# 第 48-50 行
num_pos_tokens: int = 4,      # 2→4
num_neg_tokens: int = 2,      # 1→2
num_boundary_tokens: int = 2, # 1→2
```

并在第 89 行的 cat 中拼接所有 token（含 negative）：

```python
# 改前
prompt_tokens = torch.cat([positive_tokens, boundary_tokens], dim=1)
# 改后
prompt_tokens = torch.cat([positive_tokens, boundary_tokens, negative_tokens], dim=1)
```

---

## 验证方案

### 验证 1：Bank 质量检查
```bash
python -c "
import json, torch
from pathlib import Path
bank = json.loads((Path('MedicalSAM3/outputs/medex_sam3/exemplar_bank/memory_v0.json').read_text()))
scores = [i['quality_score'] for i in bank['items']]
print(f'quality: min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f}')
divs = [i['diversity_score'] for i in bank['items'] if i.get('diversity_score',0)!=0.5]
print(f'diversity (non-default): {len(divs)} items, mean={sum(divs)/len(divs):.3f}' if divs else 'diversity: all default')
"
```

### 验证 2：delta-Dice 分布
在 `validate_medex_sam3.py` 中加逻辑：对每张验证图跑两次（with/without exemplar），输出 `delta_dice` 直方图。目标：mean(delta_dice) > 0。

### 验证 3：检索相似度检查
```python
# 随机抽 50 个验证样本，检索 top-3 positive，打印余弦相似度
```

---

## 回滚方案

如果上述修改后 Dice 仍下降：
1. 恢复 `exemplar_prompt_adapter.py` 的 token 数量
2. 在 yaml 中关掉 `enable_consistency_loss`（w_consistency=0.05 可能过强）
3. 将 `top_k_positive` 降回 1（最保守配置）
4. 如果 BANK 质量检查显示 quality 分布异常，回退到硬编码分数并只改 encoder
