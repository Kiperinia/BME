"""RSS-DA prototype bank utilities for MedEx-SAM3."""

from .bank import PrototypeBankEntry, RSSDABank
from .extractor import PrototypeExtractor, masked_average_pool

__all__ = [
    "PrototypeBankEntry",
    "PrototypeExtractor",
    "RSSDABank",
    "masked_average_pool",
]