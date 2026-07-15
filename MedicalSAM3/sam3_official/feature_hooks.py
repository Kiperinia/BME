"""提取 SAM3 中间特征的前向钩子辅助工具。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

import torch
import torch.nn as nn


def _detach_tensor(value: torch.Tensor) -> torch.Tensor:
    """将张量从计算图中分离并返回。

    参数：
        - value: 需要分离的张量。

    返回：
        - 分离后的张量（不再保留梯度）。
    """
    return value.detach()


def _sanitize_output(output: object) -> object:
    """递归地对输出对象中的张量进行分离处理。

    参数：
        - output: 模型模块的输出，可能是张量、列表、元组或字典。

    返回：
        - 结构与输入相同但所有嵌套张量均已分离的对象。
    """
    if isinstance(output, torch.Tensor):
        return _detach_tensor(output)
    if isinstance(output, (list, tuple)):
        return type(output)(_sanitize_output(item) for item in output)
    if isinstance(output, dict):
        return {key: _sanitize_output(value) for key, value in output.items()}
    return output


class FeatureHookManager:
    """管理 SAM3 模块前向钩子的注册与特征收集。

    该类负责在指定子模块上注册前向钩子，并将每次前向传播捕获的输出
    存储到有序字典中，便于后续分析中间特征。
    """

    def __init__(self) -> None:
        """初始化钩子管理器，创建空的句柄列表与特征字典。"""
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.features: OrderedDict[str, object] = OrderedDict()

    def add(self, model: nn.Module, module_name: str) -> None:
        """在指定名称的子模块上注册前向钩子。

        参数：
            - model: 被挂载钩子的模型。
            - module_name: 目标子模块的完整限定名；若不存在则跳过。
        """
        module = dict(model.named_modules()).get(module_name)
        if module is None:
            return

        def _hook(_: nn.Module, __: tuple[object, ...], output: object) -> None:
            """前向钩子回调，捕获并存储模块输出。

            参数：
                - _: 触发钩子的模块（未使用）。
                - __: 模块的输入元组（未使用）。
                - output: 模块的前向输出。
            """
            self.features[module_name] = _sanitize_output(output)

        self.handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        """清空已收集的特征字典，但保留已注册的钩子。"""
        self.features.clear()

    def remove(self) -> None:
        """移除所有已注册的钩子并清空特征字典。"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def register_feature_hooks(
    model: nn.Module,
    keywords: Optional[Iterable[str]] = None,
    max_hooks: Optional[int] = None,
) -> FeatureHookManager:
    """按关键词为模型子模块批量注册前向特征钩子。

    参数：
        - model: 待注册钩子的模型。
        - keywords: 用于匹配子模块名称的关键词列表；为空时匹配全部命名模块。
        - max_hooks: 最多注册的钩子数量；为 None 时不限制。

    返回：
        - 已注册钩子的 FeatureHookManager 实例。
    """
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
