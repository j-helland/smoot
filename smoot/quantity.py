from __future__ import annotations

import functools
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
    def magnitude(self) -> T:
        """Return the unitless value of this quantity."""
        return self.__inner.m

    @property
    def m(self) -> T:
        """Return the unitless value of this quantity."""
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

    # ==================================================
    # Operators
    # ==================================================
    def __eq__(self, other: Quantity[T]) -> bool:
        return self.__inner == other.__inner

    def __neg__(self) -> Quantity[T]:
        new = object.__new__(Quantity)
        new.__inner = -new.__inner
        return new

    def __add__(self, other: Quantity[T]) -> Quantity[T]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner + self._get_inner(other)
        return new

    def __iadd__(self, other: Quantity[T]) -> Quantity[T]:
        self.__inner += self._get_inner(other)
        return self

    def __sub__(self, other: Quantity[T]) -> Quantity[T]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner - self._get_inner(other)
        return new

    def __isub__(self, other: Quantity[T]) -> Quantity[T]:
        self.__inner -= self._get_inner(other)
        return self

    def __mul__(self, other: Quantity[T]) -> Quantity[T]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner * self._get_inner(other)
        return new

    def __imul__(self, other: Quantity[T]) -> Quantity[T]:
        self.__inner *= self._get_inner(other)
        return self

    def __truediv__(self, other: Quantity[T]) -> Quantity[T]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner / self._get_inner(other)
        return new

    def __itruediv__(self, other: Quantity[T]) -> Quantity[T]:
        self.__inner /= other.__inner
        return self

    def __pow__(self, other: Quantity[T] | T) -> Quantity[T]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner.__pow__(self._get_magnitude(other))
        return new

    @staticmethod
    def _get_inner(other):
        return (
            Quantity(other).__inner
            if not isinstance(other, Quantity)
            else other.__inner
        )

    @staticmethod
    def _get_units(units: UnitsLike) -> F64Unit:
        return F64Unit.parse(units) if type(units) is str else units

    @staticmethod
    def _get_magnitude(other):
        return (
            other
            if not isinstance(other, Quantity)
            else other.magnitude
        )
