# MedicalSAM3 训练策略模块

该目录实现了多种创新的训练策略，增强 Medical SAM3 在医学图像分割任务上的性能。

## 文件说明

### `__init__.py`
模块导出文件，不做具体功能实现。

### `contrastive.py`
对比学习增强策略，包含两个核心损失函数：
- **PixelContrastiveLoss** — 像素级对比损失，通过 InfoNCE 拉近前景像素、推远前景-背景像素特征，强化特征判别性。
- **PrototypeContrastiveLoss** — 原型对比损失，维护全局前景/背景原型，通过 EMA 更新并计算像素与原型之间的交叉熵损失。

### `ema_distillation.py`
EMA 教师-学生自蒸馏策略，包含三个核心组件：
- **EMATeacher** — EMA 教师模型，学生参数的指数移动平均，提供稳定的软标签。
- **SelfDistillationLoss** — 自蒸馏损失，计算学生与教师预测间的二元 KL 散度。
- **CurriculumScheduler** — 课程学习调度器，控制训练过程中样本难度的动态阈值。

### `ohem_loss.py`
难例挖掘损失（OHEM），包含：
- **OHEMLoss** — 结合 Focal Loss 和 Dice Loss，筛选 Top-K 最难像素进行反传，迫使网络关注难以分辨的边界和小目标。
