from __future__ import annotations

from typing import Generic, TypeAlias, TypeVar, Union

import numpy as np

from .smoot import (
    F64Unit,
    F64Quantity,
    I64Quantity,
    ArrayF64Quantity,
    ArrayI64Quantity,
)

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
        elif t is float:
            _units = self._get_units(units) if units is not None else None
            quantity = F64Quantity(value=value, units=_units)
        elif t is int:
            _units = self._get_units(units) if units is not None else None
            quantity = I64Quantity(value=value, units=_units)
        elif t in (list, tuple, np.ndarray):
            _units = self._get_units(units) if units is not None else None
            arr = np.array(value)
            QType = ArrayI64Quantity if (arr.dtype == np.int64) else ArrayF64Quantity
            quantity = QType(value=arr, units=_units)
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

    def __str__(self) -> str:
        return str(self.__inner)

    def __eq__(self, other: Quantity[T]) -> bool:
        return self.__inner == other.__inner

    def __add__(self, other: Quantity[T]) -> Quantity[T]:
        return self.__inner + self._get_other(other).__inner

    @staticmethod
    def _get_other(other) -> Quantity[T]:
        return other if isinstance(other, Quantity) else Quantity[T](other)

    @staticmethod
    def _get_units(units: UnitsLike) -> F64Unit:
        return F64Unit.parse(units) if type(units) is str else units
