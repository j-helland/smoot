from __future__ import annotations
import sys
from pathlib import Path
from typing import ClassVar, Final

import smoot
from .smoot import UnitRegistry as InnerUnitRegistry, Unit as InnerUnit


if sys.version_info >= (3, 9):
    from importlib.resources import files

    _DEFAULT_UNIT_DEFINITIONS_PATH: Final[Path] = files("smoot.data").joinpath(
        "default_en.txt"
    )
    _UNIT_CACHE_PATH: Final[Path] = files("smoot.data").joinpath(
        ".registry_cache.smoot"
    )
else:
    # TODO(jwh): Remove once support for Python 3.8 is dropped.
    _DEFAULT_UNIT_DEFINITIONS_PATH: Final[Path] = (
        Path(smoot.__file__).parent / "data" / "default_en.txt"
    )
    _UNIT_CACHE_PATH: Final[Path] = (
        Path(smoot.__file__).parent / "data" / ".registry_cache.smoot"
    )


class ApplicationRegistry:
    _registry: ClassVar[UnitRegistry | None] = None

    @classmethod
    def get(cls) -> UnitRegistry:
        if cls._registry is None:
            # Use the default instantiation
            cls._registry = UnitRegistry()
        return cls._registry

    @classmethod
    def set(cls, registry: UnitRegistry) -> None:
        cls._registry = registry


class UnitRegistry:
    __slots__ = (
        "__class__",
        "_UnitRegistry__inner",
        "Quantity",
        "Unit",
    )
    __functions__ = (
        "load_definitions",
        "_from",
        # pickle routines
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
    )

    def __init__(
        self,
        *,
        path: Path | None = None,
        data: str | None = None,
        empty: bool = False,
    ) -> None:
        if path is not None and data is not None:
            msg = (
                f"Expected either path or data, not both. Got path={path}, data={data}"
            )
            raise ValueError(msg)

        if path:
            self._UnitRegistry__inner = InnerUnitRegistry.new_from_file(str(path))
        elif data:
            self._UnitRegistry__inner = InnerUnitRegistry.new_from_str(data)
        elif empty:
            self._UnitRegistry__inner = InnerUnitRegistry()
        else:
            self._UnitRegistry__inner = InnerUnitRegistry.new_from_cache_or_file(
                cache_path=str(_UNIT_CACHE_PATH),
                file_path=str(_DEFAULT_UNIT_DEFINITIONS_PATH),
            )

        # Create a new Quantity class with a reference to this UnitRegistry instance.
        # TODO(jwh): Mark as a type alias after support for Python 3.8 is dropped
        self.Quantity: type[smoot.Quantity] = type(
            "Quantity",
            (smoot.Quantity,),
            {
                "__module__": __name__,  # for pickle support
                "_Quantity__registry": self,
            },
        )
        self.Unit: type[smoot.Unit] = type(
            "Unit",
            (smoot.Unit,),
            {
                "__module__": __name__,  # for pickle support
                "_Unit__registry": self,
            },
        )

    def _from(self, u: InnerUnit) -> smoot.Unit:
        new = object.__new__(self.Unit)
        new._Unit__inner = u
        new._Unit__registry = self
        return new

    def load_definitions(self, data: str) -> None:
        self._UnitRegistry__inner.extend(data)

    def __len__(self) -> int:
        return self._UnitRegistry__inner.get_registry_size()

    def __dir__(self) -> list[str]:
        return self._UnitRegistry__inner.get_all_registry_keys()

    def __getitem__(self, expression: str) -> smoot.Unit:
        res = self.Unit.parse(expression)
        return res

    def __getattribute__(self, name: str) -> smoot.Unit:
        if name in UnitRegistry.__slots__ or name in UnitRegistry.__functions__:
            return object.__getattribute__(self, name)

        return self[name]
