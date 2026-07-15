"""Boundary Refinement Head 扩展模块。

负责根据粗分割结果、图像上下文和形状先验进行边界细化，并在训练时生成辅助监督。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_logits(mask: torch.Tensor) -> torch.Tensor:
    """将掩码转换为 logits，若已处于 logit 范围则原样返回。

    参数：
        - mask: 输入掩码张量，值域在 [0,1] 时会被 logit 化。

    返回：
        - 转换后的 logits 张量。
    """
    if mask.min() < 0 or mask.max() > 1:
        return mask
    return torch.logit(mask.clamp(1e-4, 1 - 1e-4))


def _boundary_from_binary_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    """用 Sobel 算子从二值掩码提取边界，并可选最大池化膨胀。

    参数：
        - mask: 输入二值掩码张量。
        - dilation: 边界膨胀强度，0 表示不膨胀。

    返回：
        - 边界响应张量，值域 [0,1]。
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    boundary = (grad_x.abs() + grad_y.abs()).clamp(0, 1)
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=dilation)
    return boundary


def _local_contrast_map(image: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """计算图像的局部对比度图，低对比度区域得到更高响应。

    参数：
        - image: 输入图像张量。
        - kernel_size: 局部均值池化的核大小。

    返回：
        - 归一化到 [0,1] 的局部对比度图。
    """
    gray = image.mean(dim=1, keepdim=True)
    avg = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    contrast = (gray - avg).abs()
    normalized = 1.0 - contrast / (contrast.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return normalized.clamp(0, 1)


def _smoothness_prior(prob_mask: torch.Tensor) -> torch.Tensor:
    """根据概率掩码与局部均值的差异计算平滑先验。

    参数：
        - prob_mask: 概率掩码张量，值域 [0,1]。

    返回：
        - 平滑先验图，值域 [0,1]。
    """
    avg = F.avg_pool2d(prob_mask, kernel_size=5, stride=1, padding=2)
    return (1.0 - (prob_mask - avg).abs()).clamp(0, 1)


def _compactness_proxy(prob_mask: torch.Tensor) -> torch.Tensor:
    """用大核均值池化近似估计掩码的紧致度先验。

    参数：
        - prob_mask: 概率掩码张量，值域 [0,1]。

    返回：
        - 紧致度先验图，值域 [0,1]。
    """
    pooled = F.avg_pool2d(prob_mask, kernel_size=11, stride=1, padding=5)
    center_bias = pooled / (prob_mask.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return center_bias.clamp(0, 1)


def build_polyp_shape_prior(image: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
    """融合局部对比度、平滑度与紧致度先验构建息肉形状先验。

    参数：
        - image: 输入图像张量。
        - coarse_logits: 粗分割 logits 张量。

    返回：
        - 形状先验图，值域 [0,1]。
    """
    coarse_prob = coarse_logits.sigmoid()
    ambiguity = _local_contrast_map(image)
    smooth = _smoothness_prior(coarse_prob)
    compact = _compactness_proxy(coarse_prob)
    shape_prior = 0.45 * ambiguity + 0.35 * smooth + 0.20 * compact
    return shape_prior.clamp(0, 1)


def build_error_targets(
    coarse_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    dilation: int = 2,
) -> dict[str, torch.Tensor]:
    """根据粗分割与真值掩码构建误差区域等辅助训练目标。

    参数：
        - coarse_logits: 粗分割 logits 张量。
        - gt_mask: 真值二值掩码张量。
        - dilation: 边界膨胀强度。

    返回：
        - 包含 coarse_prob、coarse_boundary、gt_boundary、error_region、signed_error 的字典。
    """
    coarse_prob = coarse_logits.sigmoid()
    coarse_boundary = _boundary_from_binary_mask((coarse_prob > 0.5).float(), dilation=dilation)
    gt_boundary = _boundary_from_binary_mask(gt_mask.float(), dilation=dilation)
    disagreement = (coarse_prob - gt_mask).abs()
    boundary_union = torch.maximum(coarse_boundary, gt_boundary)
    error_region = disagreement * boundary_union
    signed_error = (gt_mask - coarse_prob) * boundary_union
    return {
        "coarse_prob": coarse_prob,
        "coarse_boundary": coarse_boundary,
        "gt_boundary": gt_boundary,
        "error_region": error_region,
        "signed_error": signed_error,
    }


class BoundaryRefinementHead(nn.Module):
    """边界细化头，结合粗分割、图像上下文与形状先验对边界进行门控细化。

    参数：
        - in_channels: 粗掩码输入通道数。
        - hidden_dim: 主干隐藏通道数。
        - num_refine_layers: 级联细化卷积层数。
        - dilation_base: 各层膨胀率的指数底数。

    返回：
        - 构建可用的边界细化头模块实例。
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 num_refine_layers: int = 3, dilation_base: int = 2):
        """初始化级联膨胀卷积主干、delta/error 输出头与 Sobel 边界缓冲。

        参数：
            - in_channels: 粗掩码输入通道数。
            - hidden_dim: 主干隐藏通道数。
            - num_refine_layers: 级联细化卷积层数。
            - dilation_base: 各层膨胀率的指数底数。

        返回：
            - 无返回值，完成子模块与缓冲的构建。
        """
        super().__init__()
        layers = []
        ch = in_channels + 3

        for i in range(num_refine_layers):
            d = dilation_base ** i
            layers.append(nn.Conv2d(ch, hidden_dim, kernel_size=3, padding=d,
                                    dilation=d, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            ch = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.error_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        self.boundary_dilation = 5

    def _get_boundary_mask(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        """从粗分割 logits 经 Sobel 与最大池化膨胀得到边界掩码。

        参数：
            - coarse_logits: 粗分割 logits 张量。

        返回：
            - 膨胀后的边界掩码张量，值域 [0,1]。
        """

        prob = coarse_logits.sigmoid()
        gx = F.conv2d(prob, self.sobel_x, padding=1)
        gy = F.conv2d(prob, self.sobel_y, padding=1)
        boundary = (gx.abs() + gy.abs()).clamp(0, 1)
        kernel_size = self.boundary_dilation * 2 + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size,
                                stride=1, padding=self.boundary_dilation)
        return boundary

    def build_training_targets(
        self,
        coarse_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """对齐粗掩码与真值掩码尺寸后构建误差训练目标。

        参数：
            - coarse_mask: 粗分割掩码（概率或 logits）。
            - gt_mask: 真值二值掩码张量。

        返回：
            - 由 build_error_targets 生成的训练目标字典。
        """
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        if gt_mask.shape[-2:] != coarse_mask.shape[-2:]:
            gt_mask = F.interpolate(gt_mask.float(), size=coarse_mask.shape[-2:], mode="nearest")
        coarse_logits = _to_logits(coarse_mask)
        return build_error_targets(coarse_logits, gt_mask.float(), dilation=2)

    def forward(
        self,
        coarse_mask: torch.Tensor,
        image: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """对粗掩码做边界门控细化，可选返回辅助中间结果。

        参数：
            - coarse_mask: 粗分割掩码（概率或 logits）。
            - image: 输入图像张量，用于形状先验。
            - gt_mask: 可选真值掩码，提供时附带训练目标。
            - return_aux: 是否返回辅助字典。

        返回：
            - return_aux 为假时返回细化后概率掩码；为真时返回包含
              mask_logits、masks、delta、refinement_gate 等字段的字典。
        """

        coarse_logits = _to_logits(coarse_mask.float())

        if image.shape[-2:] != coarse_logits.shape[-2:]:
            image = F.interpolate(image, size=coarse_logits.shape[-2:],
                                  mode="bilinear", align_corners=False)

        x = torch.cat([coarse_logits, image], dim=1)
        feat = self.trunk(x)
        delta = self.delta_head(feat)
        error_confidence = self.error_head(feat).sigmoid()
        boundary = self._get_boundary_mask(coarse_logits)
        shape_prior = build_polyp_shape_prior(image, coarse_logits)
        refinement_gate = boundary * error_confidence * shape_prior
        refined_logits = coarse_logits + delta * refinement_gate

        if not return_aux:
            return refined_logits.sigmoid()

        result = {
            "mask_logits": refined_logits,
            "masks": refined_logits.sigmoid(),
            "delta": delta,
            "error_confidence": error_confidence,
            "boundary_mask": boundary,
            "shape_prior": shape_prior,
            "refinement_gate": refinement_gate,
        }
        if gt_mask is not None:
            result["training_targets"] = self.build_training_targets(coarse_logits, gt_mask)
        return result
