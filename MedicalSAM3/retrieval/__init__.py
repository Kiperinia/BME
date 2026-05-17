"""Filesystem-backed retrieval bank helpers."""

from __future__ import annotations

from importlib import import_module

_LAZY_IMPORTS = {
    "DirectoryBankLoader": ("MedicalSAM3.retrieval.bank_loader", "DirectoryBankLoader"),
    "LoadedBankContext": ("MedicalSAM3.retrieval.bank_loader", "LoadedBankContext"),
    "load_retrieval_bank": ("MedicalSAM3.retrieval.bank_loader", "load_retrieval_bank"),
    "resolve_protocol_bank_path": ("MedicalSAM3.retrieval.bank_loader", "resolve_protocol_bank_path"),
    "annotate_single_bank_retrieval": ("MedicalSAM3.retrieval.multi_bank_fusion", "annotate_single_bank_retrieval"),
    "fuse_multi_bank_retrieval": ("MedicalSAM3.retrieval.multi_bank_fusion", "fuse_multi_bank_retrieval"),
    "SUPPORTED_SITE_BANK_MODES": ("MedicalSAM3.retrieval.site_bank_resolver", "SUPPORTED_SITE_BANK_MODES"),
    "SiteBankResolution": ("MedicalSAM3.retrieval.site_bank_resolver", "SiteBankResolution"),
    "resolve_site_bank_paths": ("MedicalSAM3.retrieval.site_bank_resolver", "resolve_site_bank_paths"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))