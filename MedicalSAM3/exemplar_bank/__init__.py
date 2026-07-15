"""MedEx-SAM3 的 RSS-DA 原型样本库工具模块。提供原型样本库条目、RSS-DA 样本库的加载/保存/管理，以及基于特征图的原型提取功能。"""

from .bank import PrototypeBankEntry, RSSDABank
from .extractor import PrototypeExtractor, masked_average_pool

__all__ = [
    "PrototypeBankEntry",
    "PrototypeExtractor",
    "RSSDABank",
    "masked_average_pool",
]