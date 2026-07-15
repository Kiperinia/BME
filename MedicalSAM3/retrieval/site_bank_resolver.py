"""站点特定的检索库路由辅助工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from MedicalSAM3.utils.polypgen_site import resolve_polypgen_site


SUPPORTED_SITE_BANK_MODES = {"train_only", "site_only", "train_plus_site"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SiteBankResolution:
    """站点银行解析结果的数据类，包含模式、站点 ID、银行路径和回退信息。

    参数：
        - 无。

    返回：
        - 用于下游工作流的站点银行解析实例。
    """
    mode: str
    site_id: Optional[str]
    train_bank_path: Path
    continual_bank_root: Path
    site_bank_path: Optional[Path]
    expected_site_bank: Optional[Path]
    selected_bank_paths: list[Path]
    fallback_to_train_bank: bool = False
    fallback_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def _resolve_site_bank_path(continual_bank_root: Path, site_id: str | None) -> Path | None:
    """解析指定站点的银行路径。

    参数：
        - continual_bank_root: 持续学习银行根目录。
        - site_id: 站点标识符。

    返回：
        - 站点银行路径，若不存在则返回 None。
    """
    if not site_id:
        return None
    candidate = continual_bank_root / site_id
    if candidate.exists():
        return candidate
    return None


def _scan_bank_images(bank_path: Path) -> list[Path]:
    """扫描银行目录中的正/负样本图像路径。

    参数：
        - bank_path: 银行根目录路径。

    返回：
        - 找到的图像文件路径列表。
    """
    image_paths: list[Path] = []
    for polarity in ("positive", "negative"):
        polarity_root = bank_path / polarity
        structured_root = polarity_root / "images"
        if structured_root.exists():
            image_paths.extend(
                path for path in structured_root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            continue
        image_paths.extend(
            path for path in polarity_root.glob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    return image_paths


def _bank_has_entries(bank_path: Path | None) -> bool:
    """检查银行目录是否包含有效的图像条目。

    参数：
        - bank_path: 银行目录路径。

    返回：
        - 是否存在有效条目。
    """
    if bank_path is None or not bank_path.exists():
        return False
    return bool(_scan_bank_images(bank_path))


def resolve_site_bank_paths(
    *,
    sample_metadata: dict[str, Any] | None,
    train_bank: str | Path,
    continual_bank_root: str | Path,
    mode: str = "train_plus_site",
) -> SiteBankResolution:
    """解析站点银行路径，根据样本元数据确定使用的银行库和回退策略。

    参数：
        - sample_metadata: 样本元数据，包含图像路径等信息。
        - train_bank: 训练银行路径。
        - continual_bank_root: 持续学习银行根目录。
        - mode: 银行模式，支持 "train_only"、"site_only" 和 "train_plus_site"。

    返回：
        - SiteBankResolution 实例，包含解析后的路径和回退信息。
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SUPPORTED_SITE_BANK_MODES:
        raise ValueError(f"Unsupported site bank mode: {mode}")

    train_bank_path = Path(train_bank)
    continual_root_path = Path(continual_bank_root)
    metadata = sample_metadata or {}
    site_id = resolve_polypgen_site(
        image_path=metadata.get("image_path"),
        metadata=metadata,
        sample_id=str(metadata.get("sample_id") or metadata.get("image_id") or "") or None,
        dataset_name=str(metadata.get("dataset_name") or metadata.get("source_dataset") or "") or None,
        warn=False,
    )
    expected_site_bank = continual_root_path / site_id if site_id else None
    site_bank_path = _resolve_site_bank_path(continual_root_path, site_id)

    warnings_list: list[str] = []
    selected_bank_paths: list[Path] = []
    fallback_to_train_bank = False
    fallback_reason: Optional[str] = None
    train_bank_has_entries = _bank_has_entries(train_bank_path)
    site_bank_has_entries = _bank_has_entries(site_bank_path)

    if normalized_mode != "train_only" and not site_id:
        fallback_reason = "site_id_unresolved"
    elif normalized_mode != "train_only" and site_bank_path is None:
        fallback_reason = "site_bank_missing"
    elif normalized_mode != "train_only" and not site_bank_has_entries:
        fallback_reason = "site_bank_empty"

    if normalized_mode == "train_only":
        selected_bank_paths = [train_bank_path]
    elif normalized_mode == "site_only":
        if site_bank_path is not None and site_bank_has_entries:
            selected_bank_paths = [site_bank_path]
        else:
            fallback_to_train_bank = True
            selected_bank_paths = [train_bank_path]
            warnings_list.append(
                f"Site-specific continual bank unavailable for {site_id or 'unknown site'} ({fallback_reason or 'unknown'}); falling back to train_bank."
            )
    else:
        selected_bank_paths = [train_bank_path]
        if site_bank_path is not None and site_bank_has_entries:
            selected_bank_paths.append(site_bank_path)
        else:
            fallback_to_train_bank = True
            warnings_list.append(
                f"Site-specific continual bank unavailable for {site_id or 'unknown site'} ({fallback_reason or 'unknown'}); using train_bank only."
            )

    if not train_bank_has_entries:
        warnings_list.append(f"Train bank appears empty: {train_bank_path}")

    return SiteBankResolution(
        mode=normalized_mode,
        site_id=site_id,
        train_bank_path=train_bank_path,
        continual_bank_root=continual_root_path,
        site_bank_path=site_bank_path,
        expected_site_bank=expected_site_bank,
        selected_bank_paths=selected_bank_paths,
        fallback_to_train_bank=fallback_to_train_bank,
        fallback_reason=fallback_reason,
        warnings=warnings_list,
    )


__all__ = ["SUPPORTED_SITE_BANK_MODES", "SiteBankResolution", "resolve_site_bank_paths"]
