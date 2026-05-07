"""Forward hook helpers for extracting intermediate SAM3 features."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

import torch
import torch.nn as nn


def _detach_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach()


def _sanitize_output(output: object) -> object:
    if isinstance(output, torch.Tensor):
        return _detach_tensor(output)
    if isinstance(output, (list, tuple)):
        return type(output)(_sanitize_output(item) for item in output)
    if isinstance(output, dict):
        return {key: _sanitize_output(value) for key, value in output.items()}
    return output


class FeatureHookManager:
    def __init__(self) -> None:
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.features: OrderedDict[str, object] = OrderedDict()

    def add(self, model: nn.Module, module_name: str) -> None:
        module = dict(model.named_modules()).get(module_name)
        if module is None:
            return

        def _hook(_: nn.Module, __: tuple[object, ...], output: object) -> None:
            self.features[module_name] = _sanitize_output(output)

        self.handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        self.features.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def register_feature_hooks(
    model: nn.Module,
    keywords: Optional[Iterable[str]] = None,
    max_hooks: Optional[int] = None,
) -> FeatureHookManager:
    manager = FeatureHookManager()
    lowered = [keyword.lower() for keyword in (keywords or [])]
    matched = 0
    for name, _ in model.named_modules():
        if not name:
            continue
        if lowered and not any(keyword in name.lower() for keyword in lowered):
            continue
        manager.add(model, name)
        matched += 1
        if max_hooks is not None and matched >= max_hooks:
            break
    return manager
