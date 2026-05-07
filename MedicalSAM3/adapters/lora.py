"""LoRA injection helpers for MedEx-SAM3."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _infer_scope_from_name(name: str) -> str:
    lowered = name.lower()
    if "vision_encoder" in lowered or "image_encoder" in lowered:
        return "vision_encoder"
    if "detector_encoder" in lowered:
        return "detector_encoder"
    if "detector_decoder" in lowered:
        return "detector_decoder"
    if "mask_decoder" in lowered:
        return "mask_decoder"
    if "prompt" in lowered:
        return "prompt_encoder"
    if "exemplar" in lowered:
        return "exemplar_projection"
    return "unknown"


def _parse_block_index(name: str) -> Optional[int]:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def _collect_scope_block_depths(model: nn.Module) -> dict[str, int]:
    depths: dict[str, int] = {}
    for name, _ in model.named_modules():
        block_index = _parse_block_index(name)
        if block_index is None:
            continue
        scope = _infer_scope_from_name(name)
        depths[scope] = max(depths.get(scope, -1), block_index + 1)
    return depths


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    target_scopes: list[str] = field(
        default_factory=lambda: ["vision_encoder", "detector_decoder", "mask_decoder"]
    )
    exclude_keywords: list[str] = field(default_factory=lambda: ["text_encoder"])
    train_bias: bool = False


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.rank = config.rank
        self.alpha = config.alpha
        self.scale = config.alpha / max(config.rank, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.train_bias = config.train_bias
        self.merged = False

        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = config.train_bias

        self.lora_A = nn.Linear(base_linear.in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        return self.base_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if self.merged:
            return base
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base + update

    def merge(self) -> None:
        if self.merged:
            return
        delta = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scale
        self.base_linear.weight.data = self.base_linear.weight.data + delta.to(self.base_linear.weight.data.dtype)
        self.merged = True


def is_target_module(name: str, module: nn.Module, config: LoRAConfig) -> bool:
    if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
        return False
    lowered = name.lower()
    if any(keyword.lower() in lowered for keyword in config.exclude_keywords):
        return False
    scope = _infer_scope_from_name(name)
    if scope not in config.target_scopes:
        return False
    target_hits = [keyword for keyword in config.target_modules if keyword.lower() in lowered]
    return bool(target_hits)


def _should_apply_default_stage_rule(
    module_name: str,
    scope_depths: dict[str, int],
) -> bool:
    scope = _infer_scope_from_name(module_name)
    lowered = module_name.lower()

    if scope == "vision_encoder":
        block_index = _parse_block_index(module_name)
        total_depth = scope_depths.get(scope, 0)
        if block_index is None or total_depth <= 0:
            return True
        is_late_block = block_index >= max((2 * total_depth) // 3, 0)
        return is_late_block and any(token in lowered for token in ["q_proj", "v_proj"])

    if scope == "mask_decoder":
        return any(token in lowered for token in ["q_proj", "k_proj", "v_proj", "out_proj"])

    return True


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_with_lora(model: nn.Module, module_name: str, config: LoRAConfig) -> LoRALinear:
    parent, child_name = _get_parent_module(model, module_name)
    current_module = getattr(parent, child_name)
    if not isinstance(current_module, nn.Linear):
        raise TypeError(f"Target module is not nn.Linear: {module_name}")
    lora_module = LoRALinear(current_module, config)
    setattr(parent, child_name, lora_module)
    return lora_module


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> list[str]:
    scope_depths = _collect_scope_block_depths(model)
    replaced: list[str] = []
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not is_target_module(name, module, config):
            continue
        if not _should_apply_default_stage_rule(name, scope_depths):
            continue
        replace_linear_with_lora(model, name, config)
        replaced.append(name)

    if not replaced:
        for name, module in named_modules:
            if isinstance(module, nn.Linear) and _infer_scope_from_name(name) in config.target_scopes:
                if any(keyword.lower() in name.lower() for keyword in config.exclude_keywords):
                    continue
                replace_linear_with_lora(model, name, config)
                replaced.append(name)

    logger.info("Applied LoRA to %s modules.", len(replaced))
    if replaced:
        print("LoRA replaced modules:")
        for name in replaced:
            print(f"- {name}")
    else:
        print("LoRA replaced modules: 0")
    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
    return model


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_lora_state_dict(model), destination)


def load_lora_weights(model: nn.Module, path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model
