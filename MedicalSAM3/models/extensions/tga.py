"""Text-Guided Attention 扩展模块。

该模块用文本嵌入调制图像特征，在扩展版 MedicalSAM3 中承担轻量跨模态对齐职责。
"""

import torch
import torch.nn as nn


class TextGuidedAttention(nn.Module):
    """文本引导注意力模块，用文本嵌入调制图像特征实现跨模态对齐。

    参数：
        - embed_dim: 嵌入维度。
        - num_heads: 多头注意力的头数。

    返回：
        - 构建可用的文本引导注意力模块实例。
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        """初始化文本查询投影、交叉注意力、门控与通道注意力子模块。

        参数：
            - embed_dim: 嵌入维度。
            - num_heads: 多头注意力的头数。

        返回：
            - 无返回值，完成各子模块的构建。
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.text_to_query = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        self.channel_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, image_feat: torch.Tensor,
                text_embed: torch.Tensor) -> torch.Tensor:
        """用文本嵌入通过交叉注意力与门控调制图像特征。

        参数：
            - image_feat: 形状为 [B, C, H, W] 的图像特征张量。
            - text_embed: 形状为 [B, C] 的文本嵌入张量。

        返回：
            - 与 image_feat 同形状的文本调制后特征张量。
        """

        residual = image_feat
        img_seq = image_feat.flatten(2).transpose(1, 2)

        text_q = self.text_to_query(text_embed).unsqueeze(1)

        text_q_norm = self.norm_q(text_q)
        img_kv_norm = self.norm_kv(img_seq)
        attn_out, _ = self.cross_attn(
            text_q_norm, img_kv_norm, img_kv_norm
        )

        fused_text = torch.cat([text_embed.unsqueeze(1), attn_out], dim=-1)
        gate_val = self.gate(fused_text)

        ch_attn = self.channel_attn(gate_val.squeeze(1))
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)

        out = residual * ch_attn + residual
        return out
