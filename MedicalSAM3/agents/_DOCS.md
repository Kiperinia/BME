# agents — Agent Workflows for MedEx-SAM3

本目录包含 MedEx-SAM3 的智能体工作流模块，提供分割推理、质量评估、失败挖掘、范例策管、泄漏检测、版本管理和人工审核等核心功能。

## 文件说明

### `__init__.py`
模块入口，导出所有公开类和函数。

### `segmentation_agent.py`
- **`SegmentationAgent`** — MedEx-SAM3 分割推理智能体。封装 SAM3 模型加载与推理流程，支持文本、框、点和范例提示。

### `quality_evaluator.py`
- **`_boundary_band`** — 计算掩码边界带状区域（膨胀减腐蚀）。
- **`QualityEvaluator`** — 分割质量与失败类型评估器。支持有/无真实掩码两种模式，输出 Dice、边界质量、不确定性等指标。

### `failure_miner.py`
- **`_bbox_from_mask`** — 从二值掩码计算边界框坐标。
- **`FailureMiner`** — 从失败分割案例中挖掘范例候选。自动保存裁剪的 ROI 区域并分析失败类型。

### `exemplar_curator.py`
- **`ExemplarCurator`** — 基于阈值的人体验证范例策管器。根据打分和人工验证标志决定入库或拒绝。

### `leakage_checker.py`
- **`_get_field`** — 安全地从数据类/字典/对象读取字段。
- **`LeakageChecker`** — 数据泄漏检测器。支持外部数据集泄漏、重复条目、图像折叠冲突检测。

### `memory_version_manager.py`
- **`MemoryVersionManager`** — 范例记忆库版本快照与回滚管理器。支持列表、保存和回滚版本。

### `human_review_queue.py`
- **`_resolve_bank`** — 解析并返回范例记忆库实例。
- **`export_review_queue`** — 将审核队列导出为 CSV 或 HTML。
- **`import_human_review`** — 导入人工审核 CSV 并更新记忆库状态。
