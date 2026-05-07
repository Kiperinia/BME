"""LoRA injection helpers for MedEx-SAM3."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import warnings

import torch
import torch.nn as nn

from MedicalSAM3.sam3_official.build_model import count_trainable_parameters
from MedicalSAM3.sam3_official.module_inspector import (
    DEFAULT_PREFLIGHT_TARGETS_JSON,
    DEFAULT_SCOPE_ALIASES,
    DEFAULT_TARGETS_JSON,
    classify_scope,
    suggest_lora_targets,
)

logger = logging.getLogger(__name__)
DEFAULT_LORA_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "medex_sam3" / "preflight" / "lora_injection_report.json"


def _infer_scope_from_name(name: str, scope_aliases: dict[str, list[str]] | None = None) -> str:
    return classify_scope(name, scope_aliases=scope_aliases)


def _parse_block_index(name: str) -> Optional[int]:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[index + 1].isdigit():
            return int(parts[index + 1])
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])
        if part == "resblocks" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def _collect_scope_block_depths(model: nn.Module, scope_aliases: dict[str, list[str]] | None = None) -> dict[str, int]:
    depths: dict[str, int] = {}
    for name, _ in model.named_modules():
        block_index = _parse_block_index(name)
        if block_index is None:
            continue
        scope = _infer_scope_from_name(name, scope_aliases=scope_aliases)
        depths[scope] = max(depths.get(scope, -1), block_index + 1)
    return depths


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "qkv",
            "proj",
            "fc1",
            "fc2",
            "linear1",
            "linear2",
            "c_fc",
            "c_proj",
        ]
    )
    target_scopes: list[str] = field(
        default_factory=lambda: ["vision_encoder", "detector_decoder", "mask_decoder"]
    )
    exclude_keywords: list[str] = field(default_factory=lambda: ["text_encoder", "language_backbone"])
    train_bias: bool = False
    scope_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {key: list(value) for key, value in DEFAULT_SCOPE_ALIASES.items()}
    )
    min_replaced_modules: int = 1
    allow_unknown_scope: bool = False
    stage: str = "stage_a"


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
    scope = _infer_scope_from_name(name, scope_aliases=config.scope_aliases)
    if scope == "text_encoder":
        return False
    if scope == "unknown" and not config.allow_unknown_scope:
        return False
    if scope not in config.target_scopes:
        return False
    target_hits = [keyword for keyword in config.target_modules if keyword.lower() in lowered]
    return bool(target_hits)


def _matches_stage_rule(module_name: str, config: LoRAConfig, scope_depths: dict[str, int]) -> bool:
    scope = _infer_scope_from_name(module_name, scope_aliases=config.scope_aliases)
    lowered = module_name.lower()
    stage = config.stage.lower()

    if stage == "stage_a":
        if scope == "vision_encoder":
            block_index = _parse_block_index(module_name)
            total_depth = scope_depths.get(scope, 0)
            late_start = (2 * total_depth) // 3 if total_depth > 0 else 0
            if not any(token in lowered for token in ["q_proj", "v_proj", "qkv", "attn.proj", "out_proj", ".proj"]):
                return False
            if block_index is None:
                warnings.warn(
                    f"Vision encoder block index not found for LoRA target {module_name}; allowing projection fallback.",
                    stacklevel=2,
                )
                return True
            return block_index >= late_start

        if scope == "mask_decoder":
            return any(token in lowered for token in ["q_proj", "k_proj", "v_proj", "qkv", "out_proj", ".proj"])

        return False

    if stage == "stage_b":
        if scope == "detector_decoder":
            return any(token in lowered for token in ["cross_attn", "self_attn", "ca_text", "q_proj", "k_proj", "v_proj", "qkv", "out_proj", ".proj"])
        if scope in {"prompt_encoder", "exemplar_projection"}:
            return any(token in lowered for token in ["proj", "project", "prompt_mlp"])
        return False

    if stage == "stage_c":
        if scope in {"detector_encoder", "detector_decoder"}:
            return any(token in lowered for token in ["fc1", "fc2", "linear1", "linear2", "c_fc", "c_proj", "mlp"])
        return False

    raise ValueError(f"Unsupported LoRA stage: {config.stage}")


def _scope_allowed(scope: str, config: LoRAConfig) -> bool:
    if scope == "text_encoder":
        return False
    if scope == "unknown" and not config.allow_unknown_scope:
        return False
    return scope in config.target_scopes


def _load_target_catalog(model: nn.Module, config: LoRAConfig) -> tuple[list[dict[str, Any]], str, bool]:
    current_linear_names = sorted(name for name, module in model.named_modules() if isinstance(module, nn.Linear))
    target_path = DEFAULT_TARGETS_JSON
    stale = True
    source = "generated"
    catalog: list[dict[str, Any]] = []

    if target_path.exists():
        try:
            loaded = json.loads(target_path.read_text(encoding="utf-8"))
            loaded_names = sorted(item.get("name") for item in loaded if isinstance(item, dict) and item.get("name"))
            if loaded_names == current_linear_names:
                catalog = loaded
                stale = False
                source = "catalog"
        except Exception:
            stale = True

    if stale:
        catalog = suggest_lora_targets(model, scope_aliases=config.scope_aliases)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_PREFLIGHT_TARGETS_JSON.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        DEFAULT_PREFLIGHT_TARGETS_JSON.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    return catalog, source, stale


def _write_lora_report(report: dict[str, Any]) -> Path:
    DEFAULT_LORA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LORA_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return DEFAULT_LORA_REPORT


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
    scope_depths = _collect_scope_block_depths(model, scope_aliases=config.scope_aliases)
    replaced: list[str] = []
    catalog, catalog_source, catalog_stale = _load_target_catalog(model, config)
    current_modules = dict(model.named_modules())
    total_linear_layers = sum(1 for module in current_modules.values() if isinstance(module, nn.Linear))
    candidate_names: list[str] = []

    for item in catalog:
        name = str(item.get("name", ""))
        module = current_modules.get(name)
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        scope = str(item.get("scope") or item.get("scope_guess") or _infer_scope_from_name(name, config.scope_aliases))
        if not _scope_allowed(scope, config):
            continue
        if any(keyword.lower() in name.lower() for keyword in config.exclude_keywords):
            continue
        if config.target_modules and not any(keyword.lower() in name.lower() for keyword in config.target_modules):
            continue
        if not _matches_stage_rule(name, config, scope_depths):
            continue
        candidate_names.append(name)

    for name in candidate_names:
        replace_linear_with_lora(model, name, config)
        replaced.append(name)

    trainable, total, ratio = count_trainable_parameters(model)
    report = {
        "stage": config.stage,
        "total_linear_layers": total_linear_layers,
        "lora_candidate_count": len(candidate_names),
        "replaced_module_count": len(replaced),
        "replaced_modules": replaced,
        "target_catalog_source": catalog_source,
        "target_catalog_stale": catalog_stale,
        "min_replaced_modules": config.min_replaced_modules,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_ratio": ratio,
    }
    _write_lora_report(report)

    logger.info("Applied LoRA to %s modules.", len(replaced))
    print(f"total linear layers: {total_linear_layers}")
    print(f"lora candidate count: {len(candidate_names)}")
    print(f"replaced module count: {len(replaced)}")
    if replaced:
        print("replaced module names:")
        for name in replaced:
            print(f"- {name}")
    else:
        print("replaced module names:")
        print("- <none>")
    print(f"trainable parameter ratio: {ratio:.6f}")

    if len(replaced) < config.min_replaced_modules:
        raise RuntimeError(
            f"LoRA replaced modules {len(replaced)} is below required minimum {config.min_replaced_modules}."
        )
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
