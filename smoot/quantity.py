# mypy: ignore-errors
from __future__ import annotations

from numbers import Real
from typing import Any, Generic, Iterable, TypeVar, Union
import typing
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

import smoot

from .smoot import (
    Unit as InnerUnit,
    F64Quantity,
    # I64Quantity,
    ArrayF64Quantity,
    # ArrayI64Quantity,
    # array_i64_to_f64_quantity,
    # i64_to_f64_quantity,
)


E = TypeVar("E", int, float, np.float64, np.float32, np.int64, np.int32)
_BaseArrayLike = Union[E, Iterable[E]]
ArrayLike = Union[_BaseArrayLike[E], Iterable[_BaseArrayLike[E]]]

T = TypeVar(
    "T",
    int,
    float,
    ArrayLike[int],
    ArrayLike[float],
    ArrayLike[np.float64],
    ArrayLike[np.float32],
    ArrayLike[np.int64],
    ArrayLike[np.int32],
    NDArray[np.float64],
    NDArray[np.int64],
)
R = TypeVar("R", int, float, NDArray[np.float64], NDArray[np.int64])
ValueLike = Union[str, T]
UnitsLike = Union[str, smoot.Unit]


class Quantity(Generic[T, R]):
    __slots__ = ("__inner", "__registry")

    def __init__(self, value: ValueLike[T], units: UnitsLike | None = None) -> None:
        try:
            registry = self._Quantity__registry
        except AttributeError:
            msg = (
                "Attempted to instantiate an abstract Quantity. "
                "Please use a Quantity via UnitRegistry e.g. `UnitRegistry().Quantity`."
            )
            raise TypeError(msg)

        t = type(value)
        quantity: F64Quantity | ArrayF64Quantity
        if t is str:
            # parsable
            if units is not None:
                msg = f"Cannot pass a string to parse with separate units {units}"
                raise ValueError(msg)
            quantity = F64Quantity.parse(value, registry=registry)
        elif t in (int, float, np.int64, np.int32, np.float64, np.float32):
            factor, _units = (
                self._get_units(units) if units is not None else (None, None)
            )
            quantity = F64Quantity(value=value, units=_units, factor=factor)
        # elif t in (int, np.int64, np.int32):
        #     factor, _units = (
        #         self._get_units(units) if units is not None else (None, None)
        #     )
        #     quantity = I64Quantity(value=value, units=_units, factor=factor)
        elif t in (list, tuple, np.ndarray):
            # arraylike
            factor, _units = (
                self._get_units(units) if units is not None else (None, None)
            )
            arr = np.array(value, dtype=np.float64)
            # QType = ArrayI64Quantity if (arr.dtype == np.int64) else ArrayF64Quantity
            # quantity = QType(value=arr, units=_units, factor=factor)
            quantity = ArrayF64Quantity(value=arr, units=_units, factor=factor)
        else:
            msg = f"Unsupported type {t}"
            raise NotImplementedError(msg)

        self.__inner: F64Quantity | ArrayF64Quantity = quantity

    @property
    def dimensionless(self) -> bool:
        """True if this quantity is dimensionless.

        Examples
        --------
        ```python
        assert Quantity(1).dimensionless
        assert not Quantity(1, "meter").dimensionless
        ```
        """
        return self.__inner.dimensionless

    @property
    def dimensionality(self) -> dict[str, float] | None:
        """Returns the dimensionality of this quantity.

        Examples
        --------
        ```python
        assert Quantity(1).dimensionality is None
        assert Quantity(1, "meter").dimensionality == {"[length]": 1.0}
        assert Quantity(1, "newton").dimensionality == {"[length]": 1.0, "[mass]": 1.0, "[time]": -2.0}
        ```
        """
        return self.__inner.dimensionality(self.__registry)

    @property
    def unitless(self) -> bool:
        """True if this quantity is dimensionless.

        Examples
        --------
        ```python
        assert Quantity(1).unitless
        assert not Quantity(1, "meter").unitless
        ```
        """
        return self.__inner.unitless

    @property
    def magnitude(self) -> R:
        """Return the unitless value of this quantity.

        Examples
        --------
        ```python
        assert Quantity("1 meter").magnitude == 1
        ```
        """
        return self.__inner.m

    @property
    def m(self) -> R:
        """Return the unitless value of this quantity.

        Examples
        --------
        ```python
        assert Quantity("1 meter").m == 1
        ```
        """
        return self.__inner.m

    def m_as(self, units: UnitsLike) -> R:
        """Convert the quantity to the specified units and return its magnitude.

        Parameters
        ----------
        units : UnitsLike
            The units to convert to. Can be a string or an existing unit instance.

        Examples
        --------
        ```python
        assert Quantity("1000 meter").m_as("km") == 1
        ```
        """
        factor, _units = self._get_units(units)
        return self.__inner.m_as(_units, factor=factor)

    @property
    def units(self) -> smoot.Unit:
        """Return the units of this quantity.

        Examples
        --------
        ```python
        assert Quantity("1 meter").units == units.meter
        ```
        """
        return smoot.Unit._from(self.__inner.units)

    @property
    def u(self) -> smoot.Unit:
        """Return the units of this quantity.

        Examples
        --------
        ```python
        assert Quantity("1 meter").u == units.meter
        ```
        """
        return smoot.Unit._from(self.__inner.u)

    def to(self, units: str | smoot.Unit) -> Quantity[T, R]:
        """Return a copy of this quantity converted to the target units.

        Examples
        --------
        ```python
        assert Quantity("1000 meter").to("km") == Quantity("1 km")
        ```
        """
        factor, _units = self._get_units(units)
        new = object.__new__(Quantity)
        new.__inner = self.__inner.to(_units, factor=factor)
        return new

    def ito(self, units: str | smoot.Unit) -> Quantity[T, R]:
        """In-place convert this quantity to the target units.

        Examples
        --------
        ```python
        q = Quantity("1000 meter")
        q.ito("km")
        assert q == Quantity("1 km")
        ```
        """
        factor, _units = self._get_units(units)
        self.__inner.ito(_units, factor=factor)
        return self

    def to_root_units(self) -> Quantity[T, R]:
        """Return a copy of this quantity converted to root units.

        Examples
        --------
        ```python
        assert Quantity("1 kilometer / hour").to_root_units() == Quantity("3.6 meter / second")
        ```
        """
        new = object.__new__(Quantity)
        new.__inner = self.__inner.to_root_units(self.__registry)
        return new

    def ito_root_units(self) -> None:
        """In-place convert this quantity to root units.

        Examples
        --------
        ```python
        q = Quantity("1 kilometer / hour")
        q.ito_root_units()
        assert q == Quantity("3.6 meter / second")
        ```
        """
        self.__inner.ito_root_units(self.__registry)

    # ==================================================
    # Standard dunder methods
    # ==================================================
    def __str__(self) -> str:
        return self.__inner.__str__()

    def __repr__(self) -> str:
        return f"<Quantity('{self.__inner}')>"

    def __hash__(self) -> int:
        return hash(self.__inner)

    # ==================================================
    # Pickle support
    # ==================================================
    @staticmethod
    def _load(inner: F64Quantity | ArrayF64Quantity) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = inner

        # Use the currently set application registry.
        # WARNING: This may not be the same as the registry the pickled quantity was created with.
        new.__registry = smoot.ApplicationRegistry.get()._UnitRegistry__inner

        return new

    def __reduce__(self):
        return (Quantity._load, (self.__inner,))

    # ==================================================
    # Operators
    # ==================================================
    def __eq__(self, other: Any) -> Any:
        return self.__inner == self._get_inner(other)

    def __neg__(self) -> Quantity[T, R]:
        new = object.__new__(Quantity)
        new.__inner = -self.__inner
        return new

    def __add__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(Quantity)
        new.__inner = q1.__inner + q2.__inner
        return new

    def __radd__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self + other

    def __iadd__(self, other: Quantity[T, R] | T) -> Self:
        q1, q2 = self._upcast(self._get_quantity(other))
        self.__inner = q1.__inner + q2.__inner
        return self

    def __sub__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(Quantity)
        new.__inner = q1.__inner - q2.__inner
        return new

    def __rsub__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) - self

    def __isub__(self, other: Quantity[T, R] | T) -> Self:
        q1, q2 = self._upcast(self._get_quantity(other))
        self.__inner = q1.__inner - q2.__inner
        return self

    def __mul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(Quantity)
        new.__inner = q1.__inner * q2.__inner
        return new

    def __rmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self * other

    def __imul__(self, other: Quantity[T, R] | T) -> Self:
        q1, q2 = self._upcast(self._get_quantity(other))
        self.__inner = q1.__inner * q2.__inner
        return self

    def __matmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        inner = self.__inner @ self._get_inner(other)

        # matmul may have produced a scalar
        magnitude = typing.cast(np.ndarray, inner.magnitude)
        if magnitude.shape == (1,):
            return self.__class__(magnitude[0], smoot.Unit._from(inner.units))

        new = object.__new__(Quantity)
        new.__inner = inner
        return new

    def __rmatmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) @ self

    def __imatmul__(self, other: Quantity[T, R] | T) -> Self:
        self.__inner @= self._get_inner(other)
        return self

    def __truediv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(Quantity)
        new.__inner = q1.__inner / q2.__inner
        return new

    def __rtruediv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) / self

    def __itruediv__(self, other: Quantity[T, R] | T) -> Self:
        q1, q2 = self._upcast(self._get_quantity(other))
        self.__inner = q1.__inner / q2.__inner
        return self

    def __floordiv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(Quantity)
        new.__inner = q1.__inner // q2.__inner
        return new

    def __rfloordiv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) // self

    def __ifloordiv__(self, other: Quantity[T, R] | T) -> Self:
        q1, q2 = self._upcast(self._get_quantity(other))
        self.__inner = q1.__inner // q2.__inner
        return self

    def __mod__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(Quantity)
        new.__inner = q1.__inner % q2.__inner
        return new

    def __rmod__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) % self

    def __imod__(self, other: Quantity[T, R] | T) -> Self:
        q1, q2 = self._upcast(self._get_quantity(other))
        self.__inner = q1.__inner % q2.__inner
        return self

    def __pow__(
        self, other: Quantity[T, R] | T, modulo: Real | None = None
    ) -> Quantity[T, R]:
        other_inner = self._get_inner(other)
        if not other_inner.units.dimensionless:
            msg = f"Expected dimensionless exponent but got: {other_inner.units}"
            raise ValueError(msg)

        new = object.__new__(Quantity)
        if type(self.__inner) is ArrayF64Quantity:
            new.__inner = self.__inner.arr_pow(other_inner)
        else:
            new.__inner = self.__inner.__pow__(other_inner.magnitude, modulo)
        return new

    def __rpow__(
        self, other: Quantity[T, R] | T, modulo: Real | None = None
    ) -> Quantity[T, R]:
        return self._get_quantity(other).__pow__(self, modulo)

    def __ipow__(self, other: Quantity[T, R] | T, modulo: Real | None = None) -> Self:
        if isinstance(other, Quantity) and not other.units.dimensionless:
            msg = f"Expected dimensionless exponent but got: {other.units}"
            raise ValueError(msg)

        self.__inner = self.__inner.__pow__(self._get_magnitude(other), modulo)
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
    ) -> None | type(NotImplemented) | Quantity[T, R]:
        if method != "__call__":
            return NotImplemented

        # Extract the numpy array and invoke the ufunc on the Python side. This results in
        # two "unnecessary" copies of the underlying array, not suitable for large arrays.
        #
        # One benefit is that this handles type conversion (e.g. int -> float) seamlessly.
        return self.__class__(
            value=ufunc(*(q.magnitude for q in inputs), **kwargs),
            units=smoot.Unit._from(self.__inner.units),
        )

    # ==================================================
    # private utils
    # ==================================================
    def _get_quantity(self, other: Any) -> Quantity:
        if isinstance(other, Quantity):
            return other

        # We might be an array type, in which case other needs to be wrapped as an array
        # for compatible operators.
        is_array = type(self.__inner) is ArrayF64Quantity
        if is_array and type(other) not in (list, tuple, np.ndarray):
            return self.__class__([other])

        return self.__class__(other)

    def _get_inner(
        self,
        other: Any,
    ) -> F64Quantity | ArrayF64Quantity:
        return self._get_quantity(other).__inner

    def _upcast(self, other: Quantity):
        t1 = type(self.__inner)
        t2 = type(other.__inner)
        if t1 is t2:
            return self, other

        if t1 is F64Quantity:
            new = object.__new__(Quantity)
            new.__inner = ArrayF64Quantity(np.array([self.m]), self.__inner.units)
            return new, other
        if t2 is F64Quantity:
            new = object.__new__(Quantity)
            new.__inner = ArrayF64Quantity(np.array([other.m]), other.__inner.units)
            return self, new

        msg = f"No conversion exists between {t1} and {t2}"
        raise NotImplementedError(msg)

    @classmethod
    def _get_units(cls, units: UnitsLike | InnerUnit) -> tuple[float, InnerUnit]:
        t = type(units)
        if t is str:
            return InnerUnit.parse(units, cls.__registry)
        if t is InnerUnit:
            return (1.0, units)
        return (1.0, units._Unit__inner)

    @staticmethod
    def _get_magnitude(other: Any) -> R:
        if isinstance(other, Quantity):
            return other.magnitude
        return other
