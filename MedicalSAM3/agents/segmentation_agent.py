"""Segmentation inference agent for MedEx-SAM3."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch

from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper


class SegmentationAgent:
    """MedEx-SAM3 分割推理智能体。

    封装 SAM3 模型加载与推理流程，支持文本提示、框提示、点提示和范例提示。
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "fp32",
    ) -> None:
        """初始化分割智能体。

        参数：
            - checkpoint_path: 模型检查点路径（可选）
            - device: 运行设备（默认 "cpu"）
            - dtype: 精度模式（默认 "fp32"）

        返回：
            - None
        """
        self.wrapper = Sam3TensorForwardWrapper(
            model=build_official_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=device,
                dtype=dtype,
                compile_model=False,
            ),
            device=device,
            dtype=dtype,
        )

    def _load_image(self, image_path: str | Path) -> torch.Tensor:
        """加载并预处理图像为模型输入张量。

        参数：
            - image_path: 图像文件路径

        返回：
            - 形状为 (1, 3, H, W) 的归一化张量
        """
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    def predict(
        self,
        image_path: str | Path,
        text_prompt: Optional[list[str]] = None,
        box_prompt: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        exemplar_prompt_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        """执行分割推理并返回结果。

        支持多种提示方式组合，返回掩码、分数和中间特征。

        参数：
            - image_path: 图像文件路径
            - text_prompt: 文本提示列表（可选）
            - box_prompt: 边界框提示张量（可选）
            - points: 点提示坐标张量（可选）
            - point_labels: 点提示标签张量（可选）
            - exemplar_prompt_tokens: 范例提示 token 张量（可选）

        返回：
            - 包含 mask、score 和 intermediate_metadata 的字典
        """
        image = self._load_image(image_path)
        outputs = self.wrapper(
            images=image,
            text_prompt=text_prompt,
            boxes=box_prompt,
            points=points,
            point_labels=point_labels,
            exemplar_prompt_tokens=exemplar_prompt_tokens,
        )
        return {
            "mask": outputs["masks"],
            "score": outputs["scores"],
            "intermediate_metadata": outputs["intermediate_features"],
        }
