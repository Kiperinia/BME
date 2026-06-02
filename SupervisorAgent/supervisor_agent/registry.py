from __future__ import annotations

from typing import Callable, Dict, Iterable, List, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        if name in self._items:
            raise KeyError(f"Duplicate registry entry: {name}")
        self._items[name] = item

    def get(self, name: str) -> T:
        return self._items[name]

    def list_names(self) -> List[str]:
        return list(self._items.keys())

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

    def build(self, name: str, factory: Callable[[], T]) -> T:
        if name not in self._items:
            self.register(name, factory())
        return self._items[name]
