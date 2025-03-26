from __future__ import annotations

import functools
from typing import Generic, TypeAlias, TypeVar, Union

from .smoot import F64Unit, F64Quantity

T = TypeVar("T")
UnitsLike: TypeAlias = Union[str, F64Unit]
ValueLike: TypeAlias = Union[str, T]

class Quantity(Generic[T]):
    __slots__ = ("__inner",)

    def __init__(self, value: ValueLike[T], units: UnitsLike | None = None) -> None:
        t = type(value)
        if t is str:
            if units is not None:
                msg = f"Cannot pass a string to parse with separate units {units}"
                raise ValueError(msg)
            quantity = F64Quantity.parse(value)
        elif t in (int, float):
            _units = self._get_units(units) if units is not None else None
            quantity = F64Quantity(value=value, units=_units)
        else:
            msg = f"Unsupported type {t}"
            raise NotImplementedError(msg)

        self.__inner = quantity

    @property
    def m(self) -> T:
        return self.__inner.m

    def m_as(self, units: UnitsLike) -> T:
        """Convert the quantity to the specified units and return its magnitude.
        
        Parameters
        ----------
        units : UnitsLike
            The units to convert to. Can be a string or an existing unit instance.
        """
        return self.__inner.m_as(self._get_units(units))

    def __add__(self, other: Quantity[T]) -> Quantity[T]:
        return self.__inner + self._get_other(other).__inner

    @staticmethod
    def _get_other(other) -> Quantity[T]:
        return other if isinstance(other, Quantity) else Quantity[T](other)

    @staticmethod
    def _get_units(units: UnitsLike) -> F64Unit:
        return F64Unit.parse(units) if type(units) is str else units
