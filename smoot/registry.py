from __future__ import annotations

from .smoot import get_registry_size, get_all_registry_keys
from .quantity import Unit


class UnitRegistry:
    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, Unit] = {}

    def __len__(self) -> int:
        return get_registry_size()

    def __dir__(self) -> list[str]:
        return get_all_registry_keys()

    def __getitem__(self, expression: str) -> Unit:
        if (res := self._cache.get(expression)) is not None:
            return res

        res = Unit.parse(expression)
        self._cache[expression] = res
        return res

    def __getattribute__(self, name: str) -> Unit:
        if name in UnitRegistry.__slots__:
            return object.__getattribute__(self, name)

        return self[name]
