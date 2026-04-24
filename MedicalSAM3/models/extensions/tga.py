import torch
import torch.nn as nn


class TextGuidedAttention(nn.Module):
    """
    Text-Guided Attention (TGA)

    将文本 encoding 引入图像特征，通过跨模态 attention 调制图像特征。
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
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