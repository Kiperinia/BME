# retrieval/ — 检索银行库与融合工具

该目录提供了文件系统检索银行库的加载、管理和多库融合功能，支撑 MedEx-SAM3 的检索增强分割。

---

## 文件说明

### `__init__.py`
- **功能**：包声明文件，使用惰性导入暴露主要 API。
- **主要导出**：`load_retrieval_bank`、`fuse_multi_bank_retrieval`、`resolve_site_bank_paths`、`DirectoryBankLoader` 等。

---

### `bank_loader.py`
- **功能**：基于文件系统的正/负样本检索银行库加载器。支持两种银行库格式（元数据格式和目录格式），自动提取并缓存图像特征。
- **主要类/函数**：
  - `DirectoryBankLoader` — 目录结构银行库加载器，扫描 positive/negative 图像目录，提取并缓存特征。
    - `build_bank()` — 构建 RSSDABank 实例。
    - `retrieve()` — 根据查询特征检索 top-k 正/负样本原型。
  - `LoadedBankContext` — 已加载银行库的上下文数据类。
  - `load_retrieval_bank()` — 加载检索银行库的统一入口。
  - `resolve_protocol_bank_path()` — 按协议目的解析银行路径。

---

### `mask_prior.py`
- **功能**：检索掩码先验聚合，将检索条目的掩码按权重加权聚合生成先验，用于局部检索引导。
- **主要类/函数**：
  - `attach_retrieved_mask_priors()` — 将掩码先验附加到检索结果字典。
  - `_weighted_mask_prior()` — 权重聚合掩码先验的核心实现。

---

### `multi_bank_fusion.py`
- **功能**：自适应多库检索融合工具。支持训练库和站点库检索结果的按相似度加权融合。
- **主要类/函数**：
  - `fuse_multi_bank_retrieval()` — 融合训练库和站点库的检索结果（特征、原型、分数、权重）。
  - `annotate_single_bank_retrieval()` — 为单库检索结果添加融合注释信息。
  - `_bank_weights()` — 计算训练库和站点库的自适应融合权重。

---

### `region_gate.py`
- **功能**：区域感知门控机制，基于不确定性、边界和置信度确定需要检索修正的区域。
- **主要类/函数**：
  - `build_retrieval_region_mask()` — 构建检索区域掩码，定义需要检索干预的区域。

---

### `region_uncertainty.py`
- **功能**：区域级不确定性图计算，为检索区域门控提供熵、置信度、边界不确定性等输入。
- **主要类/函数**：
  - `build_region_uncertainty_maps()` — 构建完整的区域不确定性图集合。
  - `entropy_from_logits()` — 从 logits 计算熵图。
  - `confidence_from_logits()` — 从 logits 计算置信度图。
  - `boundary_uncertainty_from_logits()` — 计算边界不确定性图。
  - `low_confidence_lesion_from_logits()` — 计算低置信度病变区域图。

---

### `site_bank_resolver.py`
- **功能**：站点特定的检索银行库路由辅助工具。根据样本元数据解析站点 ID 并确定使用的银行库。
- **主要类/函数**：
  - `SiteBankResolution` — 站点银行解析结果的数据类。
  - `resolve_site_bank_paths()` — 根据样本元数据和模式解析站点银行路径，支持 train_only、site_only 和 train_plus_site 三种模式。
  - `_resolve_site_bank_path()` — 查找站点银行的实际路径。
