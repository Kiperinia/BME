# visualization/

基于检索条件的 MedEx-SAM3 分析的可视化辅助工具包。提供检索原型面板、相似度热力图、掩码差异图和假阳覆盖图的生成功能。

## 文件说明

### `__init__.py`
模块入口，导出四个公开函数：`save_false_positive_overlay`、`save_mask_difference_visualization`、`save_retrieved_prototype_panel`、`save_similarity_heatmap_overlay`。

### `retrieval_vis.py`
检索影响诊断的可视化工具。

| 函数 | 说明 |
|---|---|
| `_to_uint8_rgb(image_tensor)` | 将图像张量转换为 uint8 RGB 数组 |
| `_to_uint8_gray(mask_tensor)` | 将掩码张量转换为 uint8 灰度数组 |
| `_normalize_map(values)` | 将张量值归一化到 [0, 1] |
| `_caption_tile(image, title, subtitle)` | 在图像底部添加标题栏 |
| `_load_tile_from_path(path, tile_size, title, subtitle)` | 从路径加载图像并添加标题，无效路径生成占位图 |
| `save_retrieved_prototype_panel(...)` | 保存检索原型面板图 |
| `save_similarity_heatmap_overlay(...)` | 保存相似度热力图叠加图 |
| `save_mask_difference_visualization(...)` | 保存掩码差异可视化图 |
| `save_false_positive_overlay(...)` | 保存假阳覆盖图 |

### `region_retrieval_vis.py`
区域感知检索可视化工具。

| 函数 | 说明 |
|---|---|
| `_to_rgb(image_tensor)` | 将图像张量转换为 RGB numpy 数组 |
| `_to_gray(mask_tensor, *, sigmoid)` | 将掩码张量转换为灰度数组，可选 sigmoid 激活 |
| `_resize_map(value, size, *, mode)` | 调整张量到指定尺寸 |
| `_heatmap_rgb(values)` | 将张量渲染为彩色热力图 |
| `_delta_rgb(delta_logits)` | 将差异 logits 渲染为红绿对比 RGB 图像 |
| `_change_map(baseline, corrected, gt_mask)` | 生成基线与校正掩码的变化图 |
| `_caption(image, title, subtitle)` | 在图像底部添加标题栏 |
| `_tile_from_path(path, tile_size, title)` | 从路径加载磁贴图，无效路径生成占位图 |
| `_first_entry_path(payload, polarity)` | 提取检索结果中第一个条目的裁剪路径 |
| `save_region_retrieval_panel(...)` | 保存区域检索可视化面板（10 视图布局） |
