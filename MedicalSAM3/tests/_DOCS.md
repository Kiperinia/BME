# MedicalSAM3 测试目录文档

## 目录概述

`tests/` 目录包含 MedicalSAM3 项目的全部单元测试，涵盖检索库管理、模型前向推理、适配器模块、损失函数、LoRA 注入、数据泄漏检测等核心功能。所有测试基于 `unittest` 框架。

---

## 文件清单

### test_select_bank_candidates.py
- **类**: `TestSelectBankCandidates`
- **功能**: 测试候选样本的选择与去重逻辑，验证正负样本平衡及输出结构。
- **辅助函数**: `_write_image` — 生成测试图像。
- **依赖**: `MedicalSAM3.scripts.select_bank_candidates`

### test_sam3_official_forward.py
- **类**: `TestSam3OfficialForward`
- **功能**: 测试官方 SAM3 模型的前向推理，包括基本 logits 输出、示例令牌和检索先验的输入。
- **依赖**: `MedicalSAM3.sam3_official.*`

### test_rssda_modules.py
- **类**: `TestRSSDAModules`
- **功能**: 综合测试 RSSDA 各模块：检索库往返、域推断、跨域检索、相似度适配器、门控融合、提示增强、提示敏感性、掩码差异率、原型提取器和目录加载器。
- **依赖**: `MedicalSAM3.adapters`, `MedicalSAM3.evaluation.*`, `MedicalSAM3.exemplar_bank`, `MedicalSAM3.models.*`, `MedicalSAM3.retrieval`, `MedicalSAM3.scripts.common`

### test_rssda_behavior_report.py
- **类**: `TestRSSDABehaviorReport`
- **功能**: 测试 RSSDA 行为报告工具的热力图摘要函数和指标差异分析函数。
- **依赖**: `MedicalSAM3.scripts.report_rssda_behavior`

### test_region_aware_retrieval.py
- **类**: `TestRegionAwareRetrieval`
- **功能**: 测试区域感知检索的区域门控掩码生成和基于区域的门控融合策略。
- **依赖**: `MedicalSAM3.models.prompt_adapter`, `MedicalSAM3.retrieval.region_gate`, `MedicalSAM3.retrieval.region_uncertainty`

### test_prototype_builder.py
- **类**: `TestPrototypeBuilder`
- **功能**: 测试原型构建器（均值/加权/注意力融合/聚类原型）及正负边界原型的构建。
- **依赖**: `MedicalSAM3.exemplar.memory_bank`, `MedicalSAM3.exemplar.prototype_builder`

### test_polypgen_site_routing.py
- **类**: `TestPolypGenSiteRouting`
- **功能**: 测试 PolypGen 站点 ID 解析（多种别名）及多库路由选择（训练库 + 站点库及其回退逻辑）。
- **依赖**: `MedicalSAM3.retrieval.site_bank_resolver`, `MedicalSAM3.utils.polypgen_site`

### test_polypgen_site.py
- **类**: `TestPolypGenSite`
- **功能**: 测试 PolypGen 站点 ID 的标准化和解析工具，包括路径/元数据模式匹配和失败处理。
- **依赖**: `MedicalSAM3.utils.polypgen_site`

### test_multi_bank_fusion.py
- **类**: `TestMultiBankFusion`
- **辅助类**: `_Entry` — 测试用检索条目。
- **辅助函数**: `_retrieval_fixture` — 生成检索结果夹具。
- **功能**: 测试训练库和站点库的多库融合，验证诊断信息分离和融合权重。
- **依赖**: `MedicalSAM3.retrieval.multi_bank_fusion`, `MedicalSAM3.retrieval.site_bank_resolver`

### test_mask_prior.py
- **类**: `TestMaskPrior`
- **辅助类**: `_Entry` — 测试用掩码路径条目。
- **功能**: 测试从检索条目掩码构建软先验掩码的功能。
- **依赖**: `MedicalSAM3.retrieval.mask_prior`

### test_loss_backward.py
- **类**: `TestLossBackward`
- **功能**: 测试所有损失函数（InfoNCE、负抑制、跨域一致性、一致性、原型方差、Dice、Hausdorff）的标量输出和反向传播。
- **依赖**: `MedicalSAM3.exemplar.losses`

### test_lora_injection.py
- **类**: `TestLoRAInjection`
- **辅助类**: `DummyAttentionBlock`, `DummyLoRAModel` — 模拟带 q_proj/v_proj 的注意力模型。
- **功能**: 测试 LoRA 注入替换目标模块、标记可训练参数及前向反向传播。
- **依赖**: `MedicalSAM3.adapters.lora`

### test_leakage_checker.py
- **类**: `TestLeakageChecker`
- **辅助函数**: `_item` — 创建 ExemplarItem 实例。
- **功能**: 测试泄漏检测器对外部数据集（PolypGen）、重复项和折泄漏的识别。
- **依赖**: `MedicalSAM3.agents.leakage_checker`, `MedicalSAM3.exemplar.memory_bank`

### test_exemplar_memory_bank.py
- **类**: `TestExemplarMemoryBank`
- **辅助函数**: `_make_item` — 创建 ExemplarItem 实例。
- **功能**: 测试示例记忆库的添加、拒绝外部数据集、保存和加载往返。
- **依赖**: `MedicalSAM3.exemplar.memory_bank`

### test_check_bank_leakage.py
- **类**: `TestCheckBankLeakage`
- **功能**: 测试银行泄漏检测工具对患者 ID 重叠、感知哈希重叠和掩码重叠的检测。
- **依赖**: `MedicalSAM3.utils.check_bank_leakage`
