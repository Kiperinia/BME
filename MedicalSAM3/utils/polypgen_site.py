"""PolypGen 中心/站点 ID 解析辅助工具。"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional


POLYPGEN_SITE_IDS = tuple(f"C{index}" for index in range(1, 7))


def normalize_polypgen_site_id(value: str | None) -> str | None:
    """标准化 PolypGen 站点 ID 为 "C1"~"C6" 格式。

    参数：
        - value: 原始站点 ID 字符串

    返回：
        - 标准化后的站点 ID，无法解析则返回 None
    """
    if value is None:
        return None
    match = re.search(r"([1-6])", str(value))
    if match is None:
        return None
    return f"C{match.group(1)}"


def _candidate_strings(
    *,
    image_path: str | Path | None,
    metadata: dict[str, Any] | None,
    sample_id: str | None,
    dataset_name: str | None,
) -> Iterable[str]:
    """生成可能包含站点 ID 的候选字符串。

    参数：
        - image_path: 图像路径
        - metadata: 元数据字典
        - sample_id: 样本 ID
        - dataset_name: 数据集名称

    返回：
        - 候选字符串的迭代器
    """
    if image_path:
        path_text = str(image_path)
        yield path_text
        yield Path(path_text).name
        yield Path(path_text).stem
    if sample_id:
        yield str(sample_id)
    if dataset_name:
        yield str(dataset_name)
    if not metadata:
        return
    for key in (
        "site_id",
        "center",
        "center_id",
        "site",
        "dataset_name",
        "sample_id",
        "image_id",
        "image_path",
        "mask_path",
        "source_dataset",
    ):
        value = metadata.get(key)
        if value:
            yield str(value)


def _extract_site_id(text: str) -> str | None:
    """从文本中提取 PolypGen 站点 ID。

    参数：
        - text: 待解析的文本

    返回：
        - 站点 ID "C1"~"C6"，未找到则返回 None
    """
    normalized = text.strip().lower()
    if not normalized:
        return None

    patterns = (
        r"(?:^|[^a-z0-9])c([1-6])(?:$|[^a-z0-9])",
        r"center[_\-\s]?([1-6])",
        r"polypgen[_\-\s]?c([1-6])",
        r"site[_\-\s]?([1-6])",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is not None:
            return f"C{match.group(1)}"

    compact = re.sub(r"[^a-z0-9]", "", normalized)
    compact_patterns = (
        r"(?:^|.*)center([1-6])(?:.*|$)",
        r"(?:^|.*)polypgenc([1-6])(?:.*|$)",
        r"(?:^|.*)site([1-6])(?:.*|$)",
    )
    for pattern in compact_patterns:
        match = re.search(pattern, compact)
        if match is not None:
            return f"C{match.group(1)}"
    return None


def resolve_polypgen_site(
    *,
    image_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    sample_id: str | None = None,
    dataset_name: str | None = None,
    warn: bool = True,
) -> Optional[str]:
    """从多个来源解析 PolypGen 站点 ID。

    依次检查图像路径、元数据、样本 ID 和数据集名称中的站点信息。

    参数：
        - image_path: 图像路径
        - metadata: 元数据字典
        - sample_id: 样本 ID
        - dataset_name: 数据集名称
        - warn: 无法解析时是否发出警告

    返回：
        - 解析到的站点 ID，未找到则返回 None
    """
    for candidate in _candidate_strings(
        image_path=image_path,
        metadata=metadata,
        sample_id=sample_id,
        dataset_name=dataset_name,
    ):
        resolved = _extract_site_id(candidate)
        if resolved in POLYPGEN_SITE_IDS:
            return resolved
    if warn:
        warnings.warn(
            "Unable to resolve PolypGen site id from sample metadata; continuing without site-specific retrieval.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


__all__ = ["POLYPGEN_SITE_IDS", "normalize_polypgen_site_id", "resolve_polypgen_site"]
