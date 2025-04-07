from __future__ import annotations

from numbers import Real
from types import NotImplementedType
from typing import Any, Generic, TypeVar, Union
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

from .smoot import (
    F64Unit,
    F64Quantity,
    I64Quantity,
    ArrayF64Quantity,
    ArrayI64Quantity,
)

T = TypeVar(
    "T",
    int,
    float,
    list[int],
    list[float],
    tuple[int, ...],
    tuple[float, ...],
    NDArray[np.float64],
    NDArray[np.int64],
)
R = TypeVar("R", int, float, NDArray[np.float64], NDArray[np.int64])
ValueLike = Union[str, T]
UnitsLike = Union[str, F64Unit]


class Quantity(Generic[T, R]):
    __slots__ = ("__inner",)

    def __init__(self, value: ValueLike[T], units: UnitsLike | None = None) -> None:
        t = type(value)
        quantity: F64Quantity | I64Quantity | ArrayF64Quantity | ArrayI64Quantity
        if t is str:
            # parsable
            if units is not None:
                msg = f"Cannot pass a string to parse with separate units {units}"
                raise ValueError(msg)
            quantity = F64Quantity.parse(value)  # type: ignore[arg-type]
        elif t is float:
            _units = self._get_units(units) if units is not None else None
            quantity = F64Quantity(value=value, units=_units)  # type: ignore[arg-type]
        elif t is int:
            _units = self._get_units(units) if units is not None else None
            quantity = I64Quantity(value=value, units=_units)  # type: ignore[arg-type]
        elif t in (list, tuple, np.ndarray):
            # arraylike
            _units = self._get_units(units) if units is not None else None
            arr = np.array(value)
            QType = ArrayI64Quantity if (arr.dtype == np.int64) else ArrayF64Quantity
            quantity = QType(value=arr, units=_units)
        else:
            msg = f"Unsupported type {t}"
            raise NotImplementedError(msg)

        self.__inner: (
            F64Quantity | I64Quantity | ArrayF64Quantity | ArrayI64Quantity
        ) = quantity

    @property
    def magnitude(self) -> R:
        """Return the unitless value of this quantity."""
        return self.__inner.m  # type: ignore[return-value]

    @property
    def m(self) -> R:
        """Return the unitless value of this quantity."""
        return self.__inner.m  # type: ignore[return-value]

    def m_as(self, units: UnitsLike) -> R:
        """Convert the quantity to the specified units and return its magnitude.

        Parameters
        ----------
        units : UnitsLike
            The units to convert to. Can be a string or an existing unit instance.
        """
        return self.__inner.m_as(self._get_units(units))  # type: ignore[return-value]

    def to(self, units: str | F64Unit) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner.to(self._get_units(units))
        return new

    def ito(self, units: str | F64Unit) -> Quantity[T, R]:
        self.__inner.ito(self._get_units(units))
        return self

    # ==================================================
    # Standard dunder methods
    # ==================================================
    def __str__(self) -> str:
        return str(self.__inner)

    # ==================================================
    # Operators
    # ==================================================
    def __eq__(self, other: Any) -> bool:
        return self.__inner == self._get_inner(other)

    def __neg__(self) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = -self.__inner
        return new

    def __add__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner + self._get_inner(other)  # type: ignore[operator]
        return new

    def __radd__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self + other

    def __iadd__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        self.__inner += self._get_inner(other)  # type: ignore[operator]
        return self

    def __sub__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner - self._get_inner(other)  # type: ignore[operator]
        return new

    def __rsub__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) - self

    def __isub__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        self.__inner -= self._get_inner(other)  # type: ignore[operator]
        return self

    def __mul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner * self._get_inner(other)  # type: ignore[operator]
        return new

    def __rmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self * other

    def __imul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        self.__inner *= self._get_inner(other)  # type: ignore[operator]
        return self

    def __matmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner @ self._get_inner(other)  # type: ignore[operator]
        return new

    def __rmatmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) @ self

    def __imatmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        self.__inner @= self._get_inner(other)  # type: ignore[operator]
        return self

    def __truediv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner / self._get_inner(other)  # type: ignore[operator]
        return new

    def __rtruediv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) / self

    def __itruediv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        self.__inner /= self._get_inner(other)  # type: ignore[operator]
        return self

    def __floordiv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner // self._get_inner(other)  # type: ignore[operator]
        return new

    def __rfloordiv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) // self

    def __ifloordiv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        self.__inner //= self._get_inner(other)  # type: ignore[operator]
        return self

    def __pow__(
        self, other: Quantity[T, R] | T, modulo: Real | None = None
    ) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner.__pow__(self._get_magnitude(other), modulo)  # type: ignore[arg-type, operator]
        return new

    def __rpow__(
        self, other: Quantity[T, R] | T, modulo: Real | None = None
    ) -> Quantity[T, R]:
        return self._get_quantity(other).__pow__(self, modulo)

    def __ipow__(
        self, other: Quantity[T, R] | T, modulo: Real | None = None
    ) -> Quantity[T, R]:
        self.__inner = self.__inner.__pow__(self._get_magnitude(other), modulo)  # type: ignore[arg-type, operator]
        return self

    def __floor__(self) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner.__floor__()
        return new

    def __ceil__(self) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner.__ceil__()
        return new

    def __round__(self, ndigits: int | None = None) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = round(self.__inner, ndigits=ndigits)
        return new

    def __abs__(self) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = self.__inner.__abs__()
        return new

    def __trunc__(self) -> int:
        return self.__inner.__trunc__()

    def __float__(self) -> float:
        return float(self.__inner)

    def __int__(self) -> int:
        return int(self.__inner)

    # ==================================================
    # numpy ufunc support
    # ==================================================
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Self,
        **kwargs: Any,
    ) -> None | NotImplementedType | Quantity[T, R]:
        if method != "__call__":
            return NotImplemented

        # Extract the numpy array and invoke the ufunc on the Python side. This results in 
        # two "unnecessary" copies of the underlying array, not suitable for large arrays.
        # 
        # One benefit is that this handles type conversion (e.g. int -> float) seamlessly.
        return Quantity(
            value=ufunc(*(q.magnitude for q in inputs), **kwargs),
            units=self.__inner.units,
        )

    # ==================================================
    # utils
    # ==================================================
    @staticmethod
    def _get_quantity(other: Any) -> Quantity[T, R]:
        return other if isinstance(other, Quantity) else Quantity(other)

    @staticmethod
    def _get_inner(
        other: Any,
    ) -> F64Quantity | I64Quantity | ArrayF64Quantity | ArrayI64Quantity:
        return Quantity._get_quantity(other).__inner

    @staticmethod
    def _get_units(units: UnitsLike) -> F64Unit:
        return F64Unit.parse(units) if type(units) is str else units  # type: ignore[return-value]

    @staticmethod
    def _get_magnitude(other: Any) -> R:
        return other if not isinstance(other, Quantity) else other.magnitude
