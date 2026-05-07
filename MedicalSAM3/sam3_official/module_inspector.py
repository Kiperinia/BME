"""Inspect official or fallback SAM3 modules and suggest LoRA insertion points."""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import torch.nn as nn

from .build_model import build_official_sam3_image_model

logger = logging.getLogger(__name__)

MEDICALSAM3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODULES_TXT = MEDICALSAM3_ROOT / "sam3_modules.txt"
DEFAULT_TARGETS_JSON = MEDICALSAM3_ROOT / "sam3_lora_targets.json"
DEFAULT_PREFLIGHT_DIR = MEDICALSAM3_ROOT / "outputs" / "medex_sam3" / "preflight"
DEFAULT_PREFLIGHT_MODULES_JSON = DEFAULT_PREFLIGHT_DIR / "sam3_modules.json"
DEFAULT_PREFLIGHT_TARGETS_JSON = DEFAULT_PREFLIGHT_DIR / "sam3_lora_targets.json"

DEFAULT_SCOPE_ALIASES = {
    "vision_encoder": ["vision_backbone", "image_encoder", "trunk.blocks", "backbone.vision"],
    "detector_encoder": ["transformer.encoder", ".encoder.layers."],
    "detector_decoder": ["decoder", "transformer.decoder"],
    "mask_decoder": ["mask", "segmentation_head", "segmentation"],
    "prompt_encoder": ["prompt", "geometry_encoder"],
    "exemplar_projection": ["exemplar", "visual_prompt"],
    "text_encoder": ["text_encoder", ".text.", "language_backbone"],
}

DEFAULT_KEYWORDS = [
    "image_encoder",
    "vision_backbone",
    "vision_encoder",
    "detector",
    "decoder",
    "mask_decoder",
    "segmentation_head",
    "prompt",
    "exemplar",
    "geometry_encoder",
    "cross_attn",
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv",
    "proj",
    "out_proj",
    "fc1",
    "fc2",
    "linear1",
    "linear2",
    "mlp",
]


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _parse_block_index(name: str) -> int | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[index + 1].isdigit():
            return int(parts[index + 1])
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])
        if part == "resblocks" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def _collect_scope_depths(model: nn.Module, scope_aliases: dict[str, list[str]] | None = None) -> dict[str, int]:
    depths: dict[str, int] = {}
    for name, _ in model.named_modules():
        block_index = _parse_block_index(name)
        if block_index is None:
            continue
        scope = classify_scope(name, scope_aliases=scope_aliases)
        depths[scope] = max(depths.get(scope, -1), block_index + 1)
    return depths


def classify_scope(name: str, scope_aliases: dict[str, list[str]] | None = None) -> str:
    aliases = scope_aliases or DEFAULT_SCOPE_ALIASES
    lowered = name.lower()

    if any(alias in lowered for alias in aliases.get("text_encoder", [])) and "transformer.decoder" not in lowered:
        return "text_encoder"
    if any(alias in lowered for alias in aliases.get("prompt_encoder", [])):
        return "prompt_encoder"
    if any(alias in lowered for alias in aliases.get("exemplar_projection", [])):
        return "exemplar_projection"
    if any(alias in lowered for alias in aliases.get("vision_encoder", [])):
        return "vision_encoder"
    if any(alias in lowered for alias in aliases.get("mask_decoder", [])):
        return "mask_decoder"
    if any(alias in lowered for alias in aliases.get("detector_decoder", [])):
        return "detector_decoder"
    if any(alias in lowered for alias in aliases.get("detector_encoder", [])) and "text_encoder" not in lowered:
        return "detector_encoder"
    return "unknown"


def _candidate_kind(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ["q_proj", "k_proj", "v_proj", "qkv", "out_proj", ".proj"]):
        return "attention_projection"
    if any(token in lowered for token in ["fc1", "fc2", "linear1", "linear2", "c_fc", "c_proj"]):
        return "mlp"
    if any(token in lowered for token in ["prompt", "geometry", "project", "prompt_mlp"]):
        return "prompt_projection"
    return "linear"


def _select_default_stages(name: str, scope: str, block_index: int | None, scope_depths: dict[str, int]) -> list[str]:
    lowered = name.lower()
    stages: list[str] = []

    if scope == "vision_encoder" and any(token in lowered for token in ["q_proj", "v_proj", "qkv", "attn.proj", "out_proj", ".proj"]):
        total_depth = scope_depths.get(scope, 0)
        late_start = int(math.floor((2.0 * total_depth) / 3.0)) if total_depth > 0 else 0
        if block_index is None or block_index >= late_start:
            stages.append("stage_a")

    if scope == "mask_decoder" and any(token in lowered for token in ["q_proj", "k_proj", "v_proj", "qkv", "out_proj", ".proj"]):
        stages.append("stage_a")

    if scope == "detector_decoder" and any(token in lowered for token in ["cross_attn", "self_attn", "ca_text", "q_proj", "k_proj", "v_proj", "qkv", "out_proj", ".proj"]):
        stages.append("stage_b")

    if scope in {"prompt_encoder", "exemplar_projection"} and any(token in lowered for token in ["proj", "project", "prompt_mlp"]):
        stages.append("stage_b")

    if scope in {"detector_encoder", "detector_decoder"} and any(token in lowered for token in ["fc1", "fc2", "linear1", "linear2", "c_fc", "c_proj"]):
        stages.append("stage_c")

    return stages


def list_named_modules(
    model: nn.Module,
    save_path: str | Path | None = None,
    scope_aliases: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    modules = []
    for name, module in model.named_modules():
        modules.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "parameters": _module_parameter_count(module),
                "scope_guess": classify_scope(name, scope_aliases=scope_aliases),
            }
        )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json_path = path.with_suffix(".json")
        txt_path = path.with_suffix(".txt")
        json_path.write_text(json.dumps(modules, indent=2), encoding="utf-8")
        txt_lines = [f"{item['name'] or '<root>'}\t{item['type']}\t{item['parameters']}" for item in modules]
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    return modules


def find_modules_by_keywords(model: nn.Module, keywords: list[str]) -> dict[str, list[str]]:
    all_names = [name for name, _ in model.named_modules()]
    results: dict[str, list[str]] = {}
    for keyword in keywords:
        matches = [name for name in all_names if keyword.lower() in name.lower()]
        results[keyword] = matches
        if not matches:
            warnings.warn(f"No modules matched keyword '{keyword}'.", stacklevel=2)
    return results


def suggest_lora_targets(
    model: nn.Module,
    scope_aliases: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    scope_depths = _collect_scope_depths(model, scope_aliases=scope_aliases)
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)
            scope = classify_scope(name, scope_aliases=scope_aliases)
            block_index = _parse_block_index(name)
            suggestions.append(
                {
                    "name": name,
                    "scope": scope,
                    "scope_guess": scope,
                    "block_index": block_index,
                    "candidate_kind": _candidate_kind(name),
                    "default_stages": _select_default_stages(name, scope, block_index, scope_depths),
                }
            )

    if not suggestions:
        warnings.warn("No standard LoRA target names found; falling back to all Linear layers.", stacklevel=2)
        suggestions = [
            {
                "name": name,
                "scope": classify_scope(name, scope_aliases=scope_aliases),
                "scope_guess": classify_scope(name, scope_aliases=scope_aliases),
                "block_index": _parse_block_index(name),
                "candidate_kind": _candidate_kind(name),
                "default_stages": [],
            }
            for name in linear_names
        ]
    return suggestions


def write_inspection_outputs(
    modules: list[dict[str, Any]],
    lora_targets: list[dict[str, Any]],
) -> None:
    DEFAULT_PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODULES_TXT.write_text(
        "\n".join(
            f"{item['name'] or '<root>'}\t{item['type']}\t{item['parameters']}\t{item['scope_guess']}" for item in modules
        ),
        encoding="utf-8",
    )
    DEFAULT_TARGETS_JSON.write_text(json.dumps(lora_targets, indent=2), encoding="utf-8")
    DEFAULT_PREFLIGHT_MODULES_JSON.write_text(json.dumps(modules, indent=2), encoding="utf-8")
    DEFAULT_PREFLIGHT_TARGETS_JSON.write_text(json.dumps(lora_targets, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect official SAM3 module names.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--allow-dummy", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = build_official_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        compile_model=False,
        allow_dummy_fallback=args.allow_dummy,
    )
    modules = list_named_modules(model)
    keyword_hits = find_modules_by_keywords(model, DEFAULT_KEYWORDS)
    lora_targets = suggest_lora_targets(model)
    write_inspection_outputs(modules, lora_targets)

    matched_keywords = {key: value for key, value in keyword_hits.items() if value}
    logger.info("Matched %s keyword groups and suggested %s LoRA targets.", len(matched_keywords), len(lora_targets))
    print(json.dumps({"keywords": matched_keywords, "lora_targets": lora_targets[:20]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
