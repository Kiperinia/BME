"""
创新训练策略 — EMA 教师-学生自蒸馏 (Self-Distillation via EMA Teacher)

设计动机:
  在有限医学数据上微调大模型容易过拟合。
  EMA 教师模型可提供稳定的 "软标签"，作为正则化信号:
  - 学生模型 (在线更新) 预测 mask
  - 教师模型 (EMA 更新) 生成软目标
  - 学生同时学习 GT 硬标签和教师的软预测

额外收益:
  - 教师模型在推理时通常性能更优 (免费提升)
  - 天然的标签平滑 (Label Smoothing) 效果
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class EMATeacher:
    """EMA 教师模型，通过指数移动平均更新参数。

    教师模型参数为学生模型的指数移动平均，提供稳定的软标签用于自蒸馏。

    参数：
        - student_model: 学生模型实例
        - decay: EMA 衰减系数
    """

    def __init__(self, student_model: nn.Module, decay: float = 0.999):
        """初始化 EMA 教师模型。

        参数：
            - student_model: 学生模型实例
            - decay: EMA 衰减系数，默认 0.999
        """
        self.decay = decay
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval()
        # 冻结教师参数
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student_model: nn.Module) -> None:
        """用学生模型参数更新教师模型（EMA 加权平均）。

        参数：
            - student_model: 当前学生模型
        """
        for t_param, s_param in zip(self.teacher.parameters(),
                                     student_model.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1 - self.decay)

    @torch.no_grad()
    def predict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """使用教师模型进行预测（前向传播）。

        参数：
            - **kwargs: 传递给教师模型的关键字参数

        返回：
            - 教师模型的预测结果字典
        """
        self.teacher.eval()
        return self.teacher(**kwargs)


class SelfDistillationLoss(nn.Module):
    """自蒸馏损失，计算学生与教师模型预测之间的 KL 散度。

    通过软标签蒸馏使学生模型从教师模型的预测中学习。

    参数：
        - temperature: 蒸馏温度，控制软标签的平滑程度
        - weight: 蒸馏损失的权重系数
    """

    def __init__(self, temperature: float = 4.0, weight: float = 0.5):
        """初始化自蒸馏损失模块。

        参数：
            - temperature: 蒸馏温度，默认 4.0
            - weight: 蒸馏损失权重，默认 0.5
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        """计算学生和教师模型之间的二元 KL 散度蒸馏损失。

        参数：
            - student_logits: 学生模型的 logits 输出
            - teacher_logits: 教师模型的 logits 输出

        返回：
            - 标量蒸馏损失值
        """
        T = self.temperature

        # 像素级 KL 散度 (binary case)
        student_prob = torch.sigmoid(student_logits / T)
        teacher_prob = torch.sigmoid(teacher_logits / T).detach()

        # Binary KL divergence
        eps = 1e-7
        kl = teacher_prob * torch.log((teacher_prob + eps) / (student_prob + eps)) + \
             (1 - teacher_prob) * torch.log((1 - teacher_prob + eps) / (1 - student_prob + eps))

        return self.weight * kl.mean() * (T ** 2)


class CurriculumScheduler:
    """课程学习调度器，控制训练过程中样本难度的动态阈值。

    在预热阶段线性增加难度阈值，使模型从简单样本逐步过渡到困难样本。

    参数：
        - total_epochs: 总训练轮数
        - warmup_ratio: 预热阶段占总轮数的比例
    """

    def __init__(self, total_epochs: int, warmup_ratio: float = 0.3):
        """初始化课程学习调度器。

        参数：
            - total_epochs: 总训练轮数
            - warmup_ratio: 预热比例，默认 0.3
        """
        self.total_epochs = total_epochs
        self.warmup_ratio = warmup_ratio

    def get_difficulty_threshold(self, epoch: int) -> float:
        """获取当前 epoch 的难度阈值。

        参数：
            - epoch: 当前训练轮次

        返回：
            - 0.3 到 1.0 之间的难度阈值
        """
        progress = epoch / max(self.total_epochs, 1)
        if progress < self.warmup_ratio:
            # 线性增长
            return 0.3 + 0.7 * (progress / self.warmup_ratio)
        return 1.0

    @staticmethod
    def compute_sample_difficulty(mask: torch.Tensor) -> float:
        """计算单个样本的难度分数。

        基于前景区域占比评估样本难度，前景占比越小难度越大。

        参数：
            - mask: 二值掩码张量

        返回：
            - 0 到 1 之间的难度分数
        """
        area_ratio = mask.float().mean().item()
        # 面积比越小，难度越大
        difficulty = 1.0 - min(area_ratio * 10, 1.0)  # 面积 < 10% 为难例
        return difficulty
