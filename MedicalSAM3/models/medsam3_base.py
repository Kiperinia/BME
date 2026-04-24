"""
Medical SAM3 — 模型入口 (SAM3 后端)

本模块提供 MedSAM3 的构建函数，后端固定使用 SAM3 包装器。
自包含轻量模型已移除；SAM3 包装器实现见 medsam3_wrapper.py。

依赖: sam3 包 (Linux / macOS 环境，需 triton)。
"""

import os
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─────────────────── SAM3 可用性检测 ───────────────────
try:
    from sam3 import build_sam3_image_model
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    logger.warning(
        "sam3 库不可用 (Windows 不支持 sam3 的 triton 依赖)。"
        "请在 Linux / macOS 环境下安装 sam3。"
    )

from .medsam3_wrapper import MedSAM3Wrapper  # noqa: E402


# ─────────────────── 医学数据集文本提示词映射 ───────────────────
DATASET_TEXT_PROMPTS = {
    "kvasir": "polyp",

}


def build_medsam3(
#配置
    checkpoint_path: Optional[str] = None,
    image_size: int = 1024,
    device: str = "cuda",
    load_from_hf: bool = True,
) -> nn.Module:
    """
    构建 MedSAM3 模型 (SAM3 后端) 并加载权重。

    Args:
        checkpoint_path: MedSAM3.pt 权重路径；为 None 时从 HuggingFace 加载默认权重。
        image_size:      输入图像尺寸 (保留参数，SAM3 内部处理分辨率)。
        device:          目标设备，如 "cuda" 或 "cpu"。
        load_from_hf:    为 True 且未提供 checkpoint 时，从 HuggingFace 拉取 SAM3 默认权重。

    Returns:
        MedSAM3Wrapper 实例，已移至 device。

    Raises:
        RuntimeError: 若 sam3 包不可用。
    """
    if not HAS_SAM3:
        raise RuntimeError(
            "sam3 包不可用，无法构建 MedSAM3。"
        )

    logger.info("Building MedSAM3 with SAM3 backend")
    sam3_model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=load_from_hf,
        device=device,
    )
    model = MedSAM3Wrapper(sam3_model)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_custom_checkpoint(checkpoint_path)
        logger.info(f"Loaded MedSAM3 checkpoint: {checkpoint_path}")

    return model.to(device)
