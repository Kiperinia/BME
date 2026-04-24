# BRH demo 说明

这些 demo 对应前面讨论的四个步骤，用来说明一种更有研究点的边界精修设计。

1. `demo_error_targets.py`
从粗分割和 GT 的差异中构造监督，而不是只监督最终 mask。

2. `demo_polyp_shape_prior.py`
从低对比、平滑性和紧致性线索中构造轻量的息肉专用形状先验。

3. `demo_shape_aware_brh.py`
把误差置信度和形状先验合成为一个 refinement gate，用在 BRH 风格模块里。

4. `demo_polyp_eval_protocol.py`
评估该方法是否真正改善了低对比、边界模糊和小息肉子集，而不是只报一个全局 Dice。

建议阅读顺序：

1. run `python demo_error_targets.py`
2. run `python demo_polyp_shape_prior.py`
3. run `python demo_shape_aware_brh.py`
4. run `python demo_polyp_eval_protocol.py`

这些 demo 故意保持轻量，不会直接改动当前 MedicalSAM3 的训练流程。