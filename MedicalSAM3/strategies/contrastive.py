"""
创新训练策略 — 对比学习增强 (Contrastive Learning Enhancement)

设计动机:
  Medical SAM3 的关键在于文本 / 视觉语义对齐。
  通过像素级对比学习，强化前景特征的聚类性和前景-背景的判别性。
  这有助于模型在域迁移场景下更鲁棒地定位目标。

实现:
  1. 像素级对比损失 (Pixel-wise Contrastive Loss)
     - 同一样本内: 前景像素特征互相拉近，前景-背景互相推远
  2. 特征原型对比 (Prototype Contrastive Loss)
     - 跨样本: 维护前景/背景类的全局原型，拉近同类、推远异类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PixelContrastiveLoss(nn.Module):
    """
    像素级监督对比损失

    对编码器输出的特征图，按前景/背景 mask 采样锚点，
    构建正/负对，计算 InfoNCE 损失。
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_anchor: int = 256,
        num_negative: int = 512,
        proj_dim: int = 128,
        feat_dim: int = 256,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_anchor = num_anchor
        self.num_negative = num_negative

        # 投影头
        self.projector = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, proj_dim, kernel_size=1),
        )

    def _sample_pixels(self, mask: torch.Tensor, n: int,
                       fg: bool = True) -> torch.Tensor:
        """
        从 mask 中随机采样 n 个前景/背景像素索引
        Args:
            mask: (H, W) 二值 mask
            n: 采样数量
            fg: True=前景, False=背景
        Returns:
            indices: (n,) 展平后的像素索引
        """
        flat = mask.flatten()
        if fg:
            candidates = (flat > 0.5).nonzero(as_tuple=False).squeeze(-1)
        else:
            candidates = (flat <= 0.5).nonzero(as_tuple=False).squeeze(-1)

        if len(candidates) == 0:
            return torch.zeros(n, dtype=torch.long, device=mask.device)

        # 随机采样 (有放回)
        idx = torch.randint(0, len(candidates), (n,), device=mask.device)
        return candidates[idx]

    def forward(self, features: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) 编码器特征
            masks: (B, 1, H, W) GT mask (需下采样到特征尺寸)
        Returns:
            对比损失 (标量)
        """
        # 投影
        proj = self.projector(features)  # (B, proj_dim, H, W)
        proj = F.normalize(proj, dim=1)

        # 下采样 mask 到特征尺寸
        B, C, H, W = proj.shape
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode="nearest")

        loss = torch.tensor(0.0, device=features.device)
        count = 0

        for b in range(B):
            feat = proj[b]         # (C, H, W)
            msk = masks[b, 0]     # (H, W)
            flat_feat = feat.flatten(1).T  # (H*W, C)

            # 采样锚点 (前景)
            anchor_idx = self._sample_pixels(msk, self.num_anchor, fg=True)
            if anchor_idx.sum() == 0:
                continue
            anchors = flat_feat[anchor_idx]  # (K, C)

            # 正例: 其他前景像素
            pos_idx = self._sample_pixels(msk, self.num_anchor, fg=True)
            positives = flat_feat[pos_idx]

            # 负例: 背景像素
            neg_idx = self._sample_pixels(msk, self.num_negative, fg=False)
            negatives = flat_feat[neg_idx]

            # InfoNCE
            pos_sim = (anchors * positives).sum(dim=1) / self.temperature  # (K,)
            neg_sim = (anchors @ negatives.T) / self.temperature           # (K, N_neg)

            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)     # (K, 1+N_neg)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
            loss = loss + F.cross_entropy(logits, labels)
            count += 1

        return loss / max(count, 1)


class PrototypeContrastiveLoss(nn.Module):
    """
    原型对比损失

    维护前景/背景的特征原型 (EMA 更新)，
    鼓励每个像素特征与同类原型接近、与异类原型远离。
    """

    def __init__(self, feat_dim: int = 256, temperature: float = 0.1,
                 momentum: float = 0.999):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        # 全局原型 (前景 / 背景)
        self.register_buffer("fg_prototype", torch.randn(1, feat_dim))
        self.register_buffer("bg_prototype", torch.randn(1, feat_dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor,
                          masks: torch.Tensor) -> None:
        """EMA 更新原型"""
        B, C, H, W = features.shape
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode="nearest")

        flat_feat = features.flatten(2).permute(0, 2, 1)       # (B, N, C)
        flat_mask = masks.flatten(2).permute(0, 2, 1)           # (B, N, 1)

        fg_mask = flat_mask.squeeze(-1) > 0.5
        bg_mask = ~fg_mask

        fg_feats = flat_feat[fg_mask]
        bg_feats = flat_feat[bg_mask]

        if fg_feats.shape[0] > 0:
            new_fg = F.normalize(fg_feats.mean(dim=0, keepdim=True), dim=1)
            if not self.initialized:
                self.fg_prototype.copy_(new_fg)
            else:
                self.fg_prototype.copy_(
                    self.momentum * self.fg_prototype + (1 - self.momentum) * new_fg
                )

        if bg_feats.shape[0] > 0:
            new_bg = F.normalize(bg_feats.mean(dim=0, keepdim=True), dim=1)
            if not self.initialized:
                self.bg_prototype.copy_(new_bg)
            else:
                self.bg_prototype.copy_(
                    self.momentum * self.bg_prototype + (1 - self.momentum) * new_bg
                )

        self.initialized.fill_(True)

    def forward(self, features: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        计算原型对比损失
        """
        B, C, H, W = features.shape
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode="nearest")

        # 更新原型
        self.update_prototypes(features.detach(), masks.detach())

        # 归一化特征
        features_norm = F.normalize(features, dim=1)  # (B, C, H, W)
        flat_feat = features_norm.flatten(2).permute(0, 2, 1)  # (B, N, C)
        flat_mask = masks.flatten(2).squeeze(1)  # (B, N)

        # 与原型的相似度
        fg_proto = F.normalize(self.fg_prototype, dim=1)  # (1, C)
        bg_proto = F.normalize(self.bg_prototype, dim=1)

        fg_sim = (flat_feat @ fg_proto.T).squeeze(-1) / self.temperature  # (B, N)
        bg_sim = (flat_feat @ bg_proto.T).squeeze(-1) / self.temperature

        # 前景像素应与 fg_proto 相近
        # 背景像素应与 bg_proto 相近
        logits = torch.stack([fg_sim, bg_sim], dim=-1)  # (B, N, 2)
        labels = (flat_mask < 0.5).long()  # 0=前景, 1=背景

        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        return loss
