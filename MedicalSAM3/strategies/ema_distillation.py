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
    """
    Exponential Moving Average 教师模型管理器
    """

    def __init__(self, student_model: nn.Module, decay: float = 0.999):
        """
        Args:
            student_model: 学生模型
            decay: EMA 衰减系数 (越大越稳定)
        """
        self.decay = decay
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval()
        # 冻结教师参数
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student_model: nn.Module) -> None:
        """使用学生模型的参数 EMA 更新教师"""
        for t_param, s_param in zip(self.teacher.parameters(),
                                     student_model.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1 - self.decay)

    @torch.no_grad()
    def predict(self, **kwargs) -> Dict[str, torch.Tensor]:
        """教师模型推理"""
        self.teacher.eval()
        return self.teacher(**kwargs)


class SelfDistillationLoss(nn.Module):
    """
    自蒸馏损失: 学生预测与教师软目标之间的一致性损失

    L_distill = KL(σ(student/τ) || σ(teacher/τ))
    """

    def __init__(self, temperature: float = 4.0, weight: float = 0.5):
        """
        Args:
            temperature: 蒸馏温度 (越高越平滑)
            weight: 蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: (B, 1, H, W)
            teacher_logits: (B, 1, H, W)
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
    """
    课程学习调度器

    控制训练过程中样本难度的渐进式提升:
    - 前期: 使用简单样本 (大 bbox, 高对比度目标)
    - 中期: 引入中等难度
    - 后期: 全部样本 (含小目标、模糊边界)

    样本难度的度量:
    - 目标面积比 (小面积 = 更难)
    - 边界复杂度 (周长/面积比)
    """

    def __init__(self, total_epochs: int, warmup_ratio: float = 0.3):
        self.total_epochs = total_epochs
        self.warmup_ratio = warmup_ratio

    def get_difficulty_threshold(self, epoch: int) -> float:
        """
        获取当前 epoch 允许的最大难度阈值 [0, 1]
        0 = 仅最简单样本, 1 = 所有样本
        """
        progress = epoch / max(self.total_epochs, 1)
        if progress < self.warmup_ratio:
            # 线性增长
            return 0.3 + 0.7 * (progress / self.warmup_ratio)
        return 1.0

    @staticmethod
    def compute_sample_difficulty(mask: torch.Tensor) -> float:
        """
        计算单个样本的难度 [0, 1]
        基于目标区域面积占比 (越小越难)
        """
        area_ratio = mask.float().mean().item()
        # 面积比越小，难度越大
        difficulty = 1.0 - min(area_ratio * 10, 1.0)  # 面积 < 10% 为难例
        return difficulty
