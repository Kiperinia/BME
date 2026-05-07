"""Inspect official or fallback SAM3 modules and suggest LoRA insertion points."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch.nn as nn

from .build_model import build_official_sam3_image_model

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS = [
    "image_encoder",
    "vision_encoder",
    "detector",
    "decoder",
    "mask_decoder",
    "prompt",
    "exemplar",
    "cross_attn",
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
    "mlp",
]


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _classify_scope(name: str) -> str:
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


def list_named_modules(model: nn.Module, save_path: str | Path | None = None) -> list[dict[str, Any]]:
    modules = []
    for name, module in model.named_modules():
        modules.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "parameters": _module_parameter_count(module),
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


def suggest_lora_targets(model: nn.Module) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)
            lowered = name.lower()
            if any(token in lowered for token in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]):
                suggestions.append({"name": name, "scope": _classify_scope(name)})

    if not suggestions:
        warnings.warn("No standard LoRA target names found; falling back to all Linear layers.", stacklevel=2)
        suggestions = [{"name": name, "scope": _classify_scope(name)} for name in linear_names]
    return suggestions


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect official SAM3 module names.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--output", default="sam3_modules")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model = build_official_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        compile_model=False,
    )
    modules = list_named_modules(model, save_path=args.output)
    keyword_hits = find_modules_by_keywords(model, DEFAULT_KEYWORDS)
    lora_targets = suggest_lora_targets(model)

    Path("sam3_modules.txt").write_text(
        "\n".join(f"{item['name'] or '<root>'}\t{item['type']}\t{item['parameters']}" for item in modules),
        encoding="utf-8",
    )
    Path("sam3_lora_targets.json").write_text(json.dumps(lora_targets, indent=2), encoding="utf-8")

    matched_keywords = {key: value for key, value in keyword_hits.items() if value}
    logger.info("Matched %s keyword groups and suggested %s LoRA targets.", len(matched_keywords), len(lora_targets))
    print(json.dumps({"keywords": matched_keywords, "lora_targets": lora_targets[:20]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
