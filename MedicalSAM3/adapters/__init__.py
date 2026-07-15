"""MedEx-SAM3 适配器模块集合。

本包聚合了 MedEx-SAM3 用到的各类适配器，包括边界感知适配器、示例提示适配器、
检索空间语义适配器、LoRA 低秩注入工具以及医学图像特征适配器，供上层模型按需导入。
"""

from .boundary_adapter import BoundaryAwareAdapter
from .exemplar_prompt_adapter import ExemplarPromptAdapter
from .retrieval_spatial_semantic_adapter import RetrievalSpatialSemanticAdapter
from .lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora_to_model,
    get_lora_state_dict,
    is_target_module,
    load_lora_weights,
    mark_only_lora_as_trainable,
    merge_lora_weights,
    replace_linear_with_lora,
    save_lora_weights,
)
from .medical_adapter import BottleneckAdapter, MedicalImageAdapter

__all__ = [
    "BottleneckAdapter",
    "BoundaryAwareAdapter",
    "ExemplarPromptAdapter",
    "LoRAConfig",
    "LoRALinear",
    "MedicalImageAdapter",
    "RetrievalSpatialSemanticAdapter",
    "apply_lora_to_model",
    "get_lora_state_dict",
    "is_target_module",
    "load_lora_weights",
    "mark_only_lora_as_trainable",
    "merge_lora_weights",
    "replace_linear_with_lora",
    "save_lora_weights",
]
