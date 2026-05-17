"""Adapter modules for MedEx-SAM3."""

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
from .medical_adapter import BottleneckAdapter, MedicalImageAdapter, MultiScaleMedicalAdapter

__all__ = [
    "BottleneckAdapter",
    "BoundaryAwareAdapter",
    "ExemplarPromptAdapter",
    "LoRAConfig",
    "LoRALinear",
    "MedicalImageAdapter",
    "MultiScaleMedicalAdapter",
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
