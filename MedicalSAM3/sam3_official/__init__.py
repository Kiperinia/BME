"""官方 SAM3 图像模型集成包，为 MedEx-SAM3 提供统一接口。

本包对外暴露官方 SAM3 图像模型的构建、冻结/解冻、参数统计工具，
以及张量化的前向推理包装器 Sam3TensorForwardWrapper。
"""

from .build_model import (
    build_official_sam3_image_model,
    count_trainable_parameters,
    freeze_model,
    print_trainable_parameters,
    unfreeze_by_keywords,
)
from .tensor_forward import Sam3TensorForwardWrapper

__all__ = [
    "Sam3TensorForwardWrapper",
    "build_official_sam3_image_model",
    "count_trainable_parameters",
    "freeze_model",
    "print_trainable_parameters",
    "unfreeze_by_keywords",
]
