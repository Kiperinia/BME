"""基于文件系统的检索库辅助工具。"""

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
    """惰性导入模块属性。

    参数：
        - name: 要导入的属性名称。

    返回：
        - 导入后的模块属性。
    """
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """列出模块的所有公开属性，包括惰性导入的名称。

    参数：
        - 无。

    返回：
        - 排序后的属性名称列表。
    """
    return sorted(list(globals().keys()) + list(__all__))
