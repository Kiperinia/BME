"""
Medical SAM3 — 网络结构扩展模块 (Innovation)

通过模块化设计实现四种创新扩展，与 baseline 完全解耦:

1. MultiScaleFeatureAdapter (MSFA)
   — 多尺度特征适配器，在 Image Encoder 输出后引入多尺度上下文聚合
   — 灵感来源: ASPP (DeepLab) + FPN 思想

2. AdaptivePromptGenerator (APG)
   — 自适应 Prompt 生成器，从图像特征自动生成伪 bbox / 伪点提示
   — 减少对人工标注 bbox 的依赖

3. BoundaryRefinementHead (BRH)
   — 边界精细化头部，在 Mask Decoder 后追加边界感知的精修模块
   — 提升分割边界精度，对小目标和模糊边界特别有效

4. TextGuidedAttention (TGA)
   — 文本引导注意力模块，将文本 embedding 与图像特征做跨模态融合
   — 实现文本驱动的分割，类似 CLIP-driven segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


# ═══════════════════════════════════════════════════════
#  创新 1: 多尺度特征适配器 (MSFA)
# ═══════════════════════════════════════════════════════

class MultiScaleFeatureAdapter(nn.Module):
    """
    Multi-Scale Feature Adapter (MSFA)

    对编码器输出的特征图进行多尺度空洞卷积 + 通道注意力融合。
    类似 ASPP 但更轻量化，适合嵌入到 SAM 架构中。

    设计动机:
      - 医学图像中的病灶尺度变化大 (如息肉从几十像素到数百像素)
      - 单尺度编码器特征可能忽略小目标细节或大目标全局上下文
      - MSFA 通过并行多尺度分支捕获不同感受野的特征
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256,
                 dilations: tuple = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(dilations),
                          kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels // len(dilations)),
                nn.GELU(),
            ))

        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // len(dilations),
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // len(dilations)),
            nn.GELU(),
        )

        # 融合卷积
        fused_ch = out_channels // len(dilations) * (len(dilations) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # 通道注意力 (SE Block)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 编码器输出特征
        Returns:
            (B, C, H, W) 增强后的特征
        """
        residual = x
        branch_outs = [branch(x) for branch in self.branches]

        # 全局上下文
        gp = self.global_pool(x)
        gp = gp.expand(-1, -1, x.shape[2], x.shape[3])
        branch_outs.append(gp)

        # 拼接 + 融合
        fused = self.fuse(torch.cat(branch_outs, dim=1))

        # 通道注意力加权
        attn = self.channel_attn(fused).unsqueeze(-1).unsqueeze(-1)
        fused = fused * attn

        return fused + residual


# ═══════════════════════════════════════════════════════
#  创新 2: 自适应 Prompt 生成器 (APG)
# ═══════════════════════════════════════════════════════

class AdaptivePromptGenerator(nn.Module):
    """
    Adaptive Prompt Generator (APG)

    从图像特征自动推断候选 bounding box 和关键点，减少对外部 prompt 的依赖。
    在训练时与 GT bbox 做监督，推理时可自主预测。

    设计动机:
      - Medical SAM3 论文指出: 许多previous methods 依赖 GT 派生的 bbox
      - APG 通过轻量检测头自动生成候选区域
      - 训练时可用 GT bbox 监督 APG，推理时 APG 做自动定位
    """

    def __init__(self, in_channels: int = 256, num_queries: int = 8,
                 embed_dim: int = 256):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # 类别无关的 Region Proposal Network
        self.rpn_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # 前景/背景分类
        self.cls_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        # BBox 回归 (4 offsets)
        self.bbox_head = nn.Conv2d(in_channels, 4, kernel_size=1)

        # 关键点生成 (可输出 Top-K 前景点)
        self.point_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

        # 温度参数: 控制 soft-argmax 的锐度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor,
                gt_bbox: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C, H, W) 编码器特征
            gt_bbox: (B, 4) 训练时的 GT bbox (用于额外监督)
        Returns:
            dict:
              - pred_bbox: (B, 4) 预测的 bbox [x1,y1,x2,y2] 归一化到 [0,1]
              - pred_points: (B, K, 2) 预测的关键点
              - point_scores: (B, K) 关键点置信度
              - cls_map: (B, 1, H, W) 前景概率图
              - bbox_loss: 标量 (仅在 gt_bbox 提供时有值)
        """
        B, C, H, W = features.shape
        rpn_feat = self.rpn_conv(features)

        # 前景概率图
        cls_map = self.cls_head(rpn_feat).sigmoid()  # (B, 1, H, W)

        # BBox 预测: 从前景概率最大区域导出
        bbox_map = self.bbox_head(rpn_feat).sigmoid()  # (B, 4, H, W)

        # 软 argmax 获取质心位置
        flat_cls = cls_map.flatten(2)  # (B, 1, H*W)
        soft_weights = F.softmax(flat_cls / self.temperature.clamp(min=0.01), dim=-1)

        # 生成坐标网格
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, H, device=features.device),
            torch.linspace(0, 1, W, device=features.device),
            indexing="ij",
        )
        coords = torch.stack([gx, gy], dim=-1).flatten(0, 1)  # (H*W, 2)
        centroid = (soft_weights @ coords.unsqueeze(0).expand(B, -1, -1)).squeeze(1)  # (B, 2)

        # 从 bbox_map 导出 bbox (cx, cy, w, h -> x1, y1, x2, y2)
        bbox_params = (soft_weights.unsqueeze(-1) * bbox_map.flatten(2).permute(0, 2, 1).unsqueeze(1)).sum(2).squeeze(1)
        # 简化: 直接用 centroid + 偏移
        cx, cy = centroid[:, 0], centroid[:, 1]
        bw = bbox_params[:, 2].clamp(0.05, 0.95)
        bh = bbox_params[:, 3].clamp(0.05, 0.95)
        pred_bbox = torch.stack([
            (cx - bw / 2).clamp(0, 1),
            (cy - bh / 2).clamp(0, 1),
            (cx + bw / 2).clamp(0, 1),
            (cy + bh / 2).clamp(0, 1),
        ], dim=1)  # (B, 4)

        # 关键点: 取 Top-K 前景点
        point_map = self.point_head(rpn_feat)  # (B, 1, H, W)
        flat_pts = point_map.flatten(2)  # (B, 1, H*W)
        topk_vals, topk_idx = flat_pts.squeeze(1).topk(self.num_queries, dim=-1)

        # 转换为坐标
        topk_y = topk_idx // W
        topk_x = topk_idx % W
        pred_points = torch.stack([
            topk_x.float() / W,
            topk_y.float() / H,
        ], dim=-1)  # (B, K, 2)
        point_scores = topk_vals.sigmoid()

        result = {
            "pred_bbox": pred_bbox,
            "pred_points": pred_points,
            "point_scores": point_scores,
            "cls_map": cls_map,
        }

        # 训练时的 bbox 回归损失
        if gt_bbox is not None:
            # 归一化 gt_bbox
            gt_norm = gt_bbox.clone()
            bbox_loss = F.l1_loss(pred_bbox, gt_norm)
            result["bbox_loss"] = bbox_loss

        return result


# ═══════════════════════════════════════════════════════
#  创新 3: 边界精细化头部 (BRH)
# ═══════════════════════════════════════════════════════

class BoundaryRefinementHead(nn.Module):
    """
    Boundary Refinement Head (BRH)

    在 Mask Decoder 之后追加的边界精修模块:
      1. 从粗 mask 提取边界区域
      2. 在边界区域内进行局部精细预测
      3. 融合粗 mask 和精细预测

    设计动机:
      - 医学图像中的分割精度高度依赖边界准确性
      - SAM 系列的 mask decoder 输出分辨率有限，边界可能模糊
      - BRH 专注于边界区域的局部精修
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 num_refine_layers: int = 3, dilation_base: int = 2):
        super().__init__()
        layers = []
        ch = in_channels + 3  # mask logit + 原始图像 (3 通道)

        for i in range(num_refine_layers):
            out_ch = hidden_dim if i < num_refine_layers - 1 else 1
            d = dilation_base ** i
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=d,
                                    dilation=d, bias=False))
            if i < num_refine_layers - 1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.GELU())
            ch = out_ch

        self.refine_net = nn.Sequential(*layers)

        # 边界检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        # 边界区域膨胀核 (用于扩大边界 ROI)
        self.boundary_dilation = 5

    def _get_boundary_mask(self, coarse_mask: torch.Tensor) -> torch.Tensor:
        """提取粗 mask 的边界区域"""
        prob = coarse_mask.sigmoid()
        gx = F.conv2d(prob, self.sobel_x, padding=1)
        gy = F.conv2d(prob, self.sobel_y, padding=1)
        boundary = (gx.abs() + gy.abs()).clamp(0, 1)
        # 膨胀边界区域
        kernel_size = self.boundary_dilation * 2 + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size,
                                stride=1, padding=self.boundary_dilation)
        return boundary

    def forward(self, coarse_mask: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_mask: (B, 1, H, W) mask decoder 输出的 logits
            image: (B, 3, H, W) 原始图像
        Returns:
            refined_mask: (B, 1, H, W) 精修后的 mask logits
        """
        # 确保尺寸对齐
        if image.shape[-2:] != coarse_mask.shape[-2:]:
            image = F.interpolate(image, size=coarse_mask.shape[-2:],
                                  mode="bilinear", align_corners=False)

        # 拼接 mask + 图像
        x = torch.cat([coarse_mask, image], dim=1)  # (B, 4, H, W)

        # 精修
        delta = self.refine_net(x)  # (B, 1, H, W)

        # 仅在边界区域应用修正
        boundary = self._get_boundary_mask(coarse_mask)
        refined = coarse_mask + delta * boundary

        return refined


# ═══════════════════════════════════════════════════════
#  创新 4: 文本引导注意力 (TGA)
# ═══════════════════════════════════════════════════════

class TextGuidedAttention(nn.Module):
    """
    Text-Guided Attention (TGA)

    将文本 encoding 引入图像特征，通过跨模态 cross-attention 来
    “用文本告诉模型要分割什么”。

    设计动机:
      - 医学图像中同一张图可能包含多种结构 (息肉、血管、淋巴结 等)
      - 文本 prompt (如 "polyp") 可指定目标类别，实现类别感知分割
      - 通过 cross-attention, 文本特征作为 query 对图像特征做注意力加权，
        突出与文本语义相关的区域

    模块流程:
      1. 文本 embedding -> expand 为 spatial query
      2. Cross-attention: text (Q) x image_feat (K, V)
      3. 通道加权融合 + 残差连接
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # 文本 -> spatial query 投影
        self.text_to_query = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # Cross-attention: text-guided image attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # 融合门控
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # 通道注意力 (text-conditioned SE)
        self.channel_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, image_feat: torch.Tensor,
                text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_feat: (B, C, H, W) 图像特征
            text_embed: (B, C) 文本全局表示
        Returns:
            (B, C, H, W) 文本引导后的图像特征
        """
        B, C, H, W = image_feat.shape
        residual = image_feat

        # image_feat -> (B, H*W, C)
        img_seq = image_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # text query: 扩展为序列
        text_q = self.text_to_query(text_embed).unsqueeze(1)  # (B, 1, C)

        # Cross-attention: text queries image
        text_q_norm = self.norm_q(text_q)
        img_kv_norm = self.norm_kv(img_seq)
        attn_out, attn_weights = self.cross_attn(
            text_q_norm, img_kv_norm, img_kv_norm
        )  # attn_out: (B, 1, C)

        # 融合门控: 结合原始文本和 attention 输出
        fused_text = torch.cat([text_embed.unsqueeze(1), attn_out], dim=-1)  # (B, 1, 2C)
        gate_val = self.gate(fused_text)  # (B, 1, C)

        # 通道注意力: 用文本语义调制图像特征的通道权重
        ch_attn = self.channel_attn(gate_val.squeeze(1))  # (B, C)
        ch_attn = ch_attn.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # 加权图像特征 + 残差
        out = residual * ch_attn + residual
        return out


# ═══════════════════════════════════════════════════════
#  扩展模型: 整合所有创新模块
# ═══════════════════════════════════════════════════════

class MedSAM3Extended(nn.Module):
    """
    扩展版 Medical SAM3，整合四个创新模块。
    可按需启用/禁用各模块。
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_msfa: bool = True,
        use_apg: bool = True,
        use_brh: bool = True,
        use_tga: bool = True,
        image_size: int = 1024,
    ):
        super().__init__()
        self.base = base_model
        self.image_size = image_size
        self.use_msfa = use_msfa
        self.use_apg = use_apg
        self.use_brh = use_brh
        self.use_tga = use_tga

        if use_msfa:
            self.msfa = MultiScaleFeatureAdapter(in_channels=256, out_channels=256)
        if use_apg:
            self.apg = AdaptivePromptGenerator(in_channels=256)
        if use_brh:
            self.brh = BoundaryRefinementHead()
        if use_tga:
            self.tga = TextGuidedAttention(embed_dim=256)

    def forward(
        self,
        images: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        text_prompt: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        扩展的 forward 流程:
        1. Image Encoder -> [MSFA 增强] -> 特征
        2. [TGA 文本引导注意力] (可选)
        3. [APG 生成候选 prompt] (可选)
        4. Text Encoder + Prompt Encoder + Mask Decoder
        5. [BRH 边界精修] (可选)
        """
        results = {}

        # 1) 图像编码
        image_embed = self.base.image_encoder(images)

        # 1.5) MSFA 多尺度增强
        if self.use_msfa:
            image_embed = self.msfa(image_embed)

        # 1.7) 文本编码
        text_embed = None
        if text_prompt is not None:
            token_ids = self.base.tokenizer.batch_encode(text_prompt).to(images.device)
            text_embed = self.base.text_encoder(token_ids)  # (B, embed_dim)

        # 1.8) TGA 文本引导注意力
        if self.use_tga and text_embed is not None:
            image_embed = self.tga(image_embed, text_embed)

        # 2) APG 自适应 prompt
        if self.use_apg:
            apg_out = self.apg(image_embed, gt_bbox=bboxes)
            results["apg_output"] = apg_out
            # 如果没有外部 bbox，使用 APG 预测的
            if bboxes is None:
                bboxes = apg_out["pred_bbox"] * self.image_size

        # 3) Prompt 编码 + Mask 解码
        sparse_embed, dense_embed = self.base.prompt_encoder(
            bboxes=bboxes, points=points, point_labels=point_labels,
            text_embed=text_embed,
        )
        low_res_masks, iou_pred = self.base.mask_decoder(
            image_embed, sparse_embed, dense_embed, multimask_output=False
        )

        # 上采样
        masks = F.interpolate(
            low_res_masks, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        )

        # 4) BRH 边界精修
        if self.use_brh:
            masks = self.brh(masks, images)

        results["masks"] = masks
        results["iou_predictions"] = iou_pred

        return results


def build_medsam3_extended(
    base_model: nn.Module,
    use_msfa: bool = True,
    use_apg: bool = True,
    use_brh: bool = True,
    use_tga: bool = True,
    image_size: int = 1024,
) -> MedSAM3Extended:
    """构建扩展版模型"""
    return MedSAM3Extended(
        base_model=base_model,
        use_msfa=use_msfa,
        use_apg=use_apg,
        use_brh=use_brh,
        use_tga=use_tga,
        image_size=image_size,
    )
