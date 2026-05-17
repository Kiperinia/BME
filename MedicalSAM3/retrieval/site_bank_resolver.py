"""Site-specific retrieval bank routing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from MedicalSAM3.utils.polypgen_site import resolve_polypgen_site


SUPPORTED_SITE_BANK_MODES = {"train_only", "site_only", "train_plus_site"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SiteBankResolution:
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
    if not site_id:
        return None
    candidate = continual_bank_root / site_id
    if candidate.exists():
        return candidate
    return None


def _scan_bank_images(bank_path: Path) -> list[Path]:
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