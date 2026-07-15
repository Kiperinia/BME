# evaluation/ — 检索条件化 MedEx-SAM3 评估工具

该目录提供了一套评估检索对分割模型影响的工具，涵盖提示敏感性分析、检索校准诊断、假阴性分析以及区域感知诊断。

---

## 文件说明

### `__init__.py`
- **功能**：包声明文件。
- **主要导出**：无。

---

### `retrieval_analysis.py`
- **功能**：分析检索是否显著改变分割输出。生成提示敏感性指标，比较不同检索变体（正例、反例、随机、空）对分割结果的影响。
- **主要类/函数**：
  - `main()` — 命令行入口，加载模型和银行库，对内部/外部数据运行分析并输出 JSON 报告。
  - `_build_prompt_variants()` — 构建多个提示变体的检索结果。
  - `_run_variant()` — 运行单个检索变体的前向传播。
  - `_prompt_sensitivity()` — 计算提示敏感性评分。
  - `_load_or_create_bank()` — 加载或创建虚拟银行库。

---

### `retrieval_calibration.py`
- **功能**：聚合检索校准诊断和假阴性分析。按病变大小和高负样本影响分组统计假阴性率。
- **主要类/函数**：
  - `summarize_retrieval_calibration()` — 汇总检索校准诊断（含检索诊断和假阴性分析）。
  - `summarize_false_negative_analysis()` — 假阴性分析，区分小/大病变和高负样本影响。
  - `write_retrieval_calibration_report()` — 生成校准报告 JSON 文件。
  - `main()` — 命令行入口。

---

### `retrieval_diagnostics.py`
- **功能**：结构化检索诊断导出。构建详细的检索诊断字典，包括 top-k 信息、相似度分布、门控诊断、多库融合信息等。
- **主要类/函数**：
  - `build_retrieval_diagnostics()` — 构建单个样本的完整检索诊断。
  - `summarize_retrieval_diagnostics()` — 汇总检索诊断统计。
  - `write_retrieval_diagnostics()` / `write_retrieval_diagnostics_summary()` — 写入诊断 JSON/JSONL 文件。

---

### `retrieval_influence.py`
- **功能**：量化检索是否改变分割行为和外部鲁棒性。通过多种检索变体的比较评估检索的影响并生成可视化。
- **主要类/函数**：
  - `main()` — 命令行入口，加载模型和银行库，运行影响分析并保存可视化。
  - `_run_variant()` — 运行带提示适配器诊断的检索变体。
  - `_visualize_image_case()` — 生成原型面板、热力图、掩码差异等可视化。
  - `_load_or_create_exemplar_memory_bank()` — 加载或创建样本记忆库。

---

### `region_retrieval_diagnostics.py`
- **功能**：结构化区域感知检索诊断。分析检索在病变区域、边界区域和高置信度区域的影响。
- **主要类/函数**：
  - `build_region_retrieval_diagnostics()` — 构建区域感知诊断字典。
  - `summarize_region_retrieval_diagnostics()` — 汇总区域诊断统计。
  - `write_region_retrieval_diagnostics()` — 写入区域诊断 JSONL 文件。
