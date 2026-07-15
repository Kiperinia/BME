# exemplar_bank — RSS-DA Prototype Bank

本目录提供检索条件空间-语义适应（RSS-DA）原型库的核心数据结构与提取工具。

## 文件说明

### `__init__.py`
模块入口，导出 `PrototypeBankEntry`、`RSSDABank`、`PrototypeExtractor`、`masked_average_pool`。

### `bank.py`
- **`PrototypeBankEntry`** — 原型库条目数据类。存储特征路径、极性、数据集来源、质量分数和元数据。
- **`RSSDABank`** — RSS-DA 原型库。提供增删查改、`load`/`save` 序列化、`load_feature` 特征加载和 `stack_features` 特征堆叠。

### `extractor.py`
- **`masked_average_pool`** — 对特征图执行掩码加权平均池化并 L2 归一化。
- **`_resolve_feature_map`** — 将不同格式的特征（4D/3D）解析为统一 4D 特征图。
- **`PrototypeExtractor`** — SAM3 原型提取器。支持从特征图、模型输出或原始图像中提取原型特征，并提供 `save_prototype` 持久化到磁盘。
