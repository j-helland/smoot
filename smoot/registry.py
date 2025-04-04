from __future__ import annotations
from typing import Any

from .smoot import F64Unit, get_registry_size, get_all_registry_keys


class UnitRegistry:
    __slots__ = "_cache"

    def __init__(self) -> None:
        self._cache: dict[str, F64Unit] = {}

    def __len__(self) -> int:
        return get_registry_size()

    def __dir__(self) -> list[str]:
        return get_all_registry_keys() 

    def __getitem__(self, expression: str) -> F64Unit:
        if (res := self._cache.get(expression)) is not None:
            return res

        res = F64Unit.parse(expression)
        self._cache[expression] = res
        return res

    def __getattribute__(self, name: str) -> Any:
        if name in UnitRegistry.__slots__:
            return object.__getattribute__(self, name)

        return self[name]
