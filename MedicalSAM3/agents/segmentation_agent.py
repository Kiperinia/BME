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
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "fp32",
    ) -> None:
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
