# MedicalSAM3 通用工具模块

该目录提供医学图像分割训练和评估中使用的通用工具函数和类。

## 文件说明

### `__init__.py`
模块导出文件，导出 `CombinedSegLoss`、`DiceLoss`、`FocalLoss`、`dice_coefficient`、`iou_score`。

### `check_bank_leakage.py`
检索库泄漏检测工具，通过文件名、患者 ID、图像感知哈希和掩码哈希等多维度比对，检测评估样本与检索库之间是否存在泄漏。
- **SampleSignature** — 样本签名数据类。
- **run_bank_leakage_check** — 泄漏检测主函数。
- **main** — 命令行入口点。

### `dataset.py`
数据集加载器，提供通用医学分割数据集接口及 KvasirCVC / PolypGen 适配。
- **MedicalSegDataset** — 通用医学分割数据集类，支持图像-掩码对加载及边界框 Prompt。
- **NnUNetRawRGBDataset** — nnUNet 原始 RGB 格式适配，支持三通道分离文件加载。
- **KvasirCVCDataset** — KvasirCVC 数据集封装（继承 NnUNetRawRGBDataset）。
- **PolypGenDataset** — PolypGen 外部测试集封装（继承 NnUNetRawRGBDataset）。
- **TransformSubset** — 数据集子集变换包装器。
- **create_dataset** / **build_dataloaders** — 工厂函数和 DataLoader 构建函数。

### `losses.py`
损失函数集合，用于医学图像分割训练。
- **DiceLoss** — Dice Loss，区域相似性度量。
- **FocalLoss** — Focal Loss，聚焦难分类样本。
- **BoundaryLoss** — 边界加权损失，使用 Sobel 算子提取边界并赋予更高权重。
- **CombinedSegLoss** — 组合分割损失，加权融合 Dice、Focal 和 BCE 损失。

### `metrics.py`
评估指标函数，用于医学图像分割评估。
- **dice_coefficient** — Dice 系数计算。
- **iou_score** — IoU（交并比）计算。
- **precision_score** — 精确率计算。
- **recall_score** — 召回率计算。
- **compute_all_metrics** — 综合计算上述所有指标。

### `polypgen_site.py`
PolypGen 中心/站点 ID 解析辅助工具。
- **normalize_polypgen_site_id** — 标准化站点 ID 为 "C1"~"C6" 格式。
- **resolve_polypgen_site** — 从图像路径、元数据等多来源解析站点 ID。

### `transforms.py`
数据增强与预处理变换。
- **get_train_transforms** — 训练阶段增强管线（翻转、旋转、仿射、模糊、噪声等）。
- **get_val_transforms** — 验证阶段变换管线（仅缩放和归一化）。
- **ResizeNormalize** — albumentations 不可用时的回退实现。
- **mask_to_bbox** — 从掩码提取边界框。
- **jitter_bbox** — 边界框随机扰动。
