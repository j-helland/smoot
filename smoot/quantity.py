# mypy: ignore-errors
from __future__ import annotations

from numbers import Real
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar, Union
import typing
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray

import smoot
from smoot.numpy_functions import NP_HANDLED_FUNCTIONS, NP_HANDLED_UFUNCS
from smoot.utils import warn_for_large_arrays

from .smoot import (
    Unit as InnerUnit,
    F64Quantity,
    # I64Quantity,
    ArrayF64Quantity,
    # ArrayI64Quantity,
    # array_i64_to_f64_quantity,
    # i64_to_f64_quantity,
    UnitRegistry,
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

    def __init__(
        self,
        value: ValueLike[T] | Quantity[T, R],
        units: UnitsLike | Quantity[T, R] | None = None,
        *,
        registry: UnitRegistry | None = None,
    ) -> None:
        if isinstance(value, Quantity):
            if units is None:
                self.__inner = value.__inner
            else:
                self.__inner = value.to(units).__inner
            self.__registry = value.__registry
            return

        # Retrieve the registry.
        # If it doesn't exist, this means that the user tried to instantiate this class
        # without an associated UnitRegistry, which is a hard error.
        if registry is None:
            try:
                registry = self.__registry._UnitRegistry__inner
            except AttributeError:
                msg = (
                    "Attempted to instantiate an abstract Quantity. "
                    "Please use a Quantity via UnitRegistry e.g. `UnitRegistry().Quantity`."
                )
                raise TypeError(msg)

        quantity: F64Quantity | ArrayF64Quantity
        if isinstance(value, str):
            # String containing a quantity expression e.g. '1 meter'
            if units is not None:
                msg = f"Cannot pass a string to parse with separate units {units}"
                raise ValueError(msg)
            quantity = F64Quantity.parse(value, registry=registry)

        elif isinstance(
            value, (int, float, np.int64, np.int32, np.float64, np.float32)
        ):
            # Numeric value and a unit.
            # The unit itself may be a string expression.
            factor, _units = (
                self._get_units(units) if units is not None else (None, None)
            )
            quantity = F64Quantity(value=value, units=_units, factor=factor)

        elif isinstance(value, (list, tuple, np.ndarray)):
            # Array value and a unit.
            # The unit itself may be a string expression.
            factor, _units = (
                self._get_units(units) if units is not None else (None, None)
            )
            arr = np.array(value, dtype=np.float64)

            # TODO(jwh): Remove this once Smoot has a better story for handling large arrays
            #            on the Rust side.
            warn_for_large_arrays(arr)

            quantity = ArrayF64Quantity(value=arr, units=_units, factor=factor)

        else:
            msg = f"Unsupported type {type(value)}"
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
        return self.__inner.dimensionality(self.__registry._UnitRegistry__inner)

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

    def m_as(self, units: UnitsLike | Quantity) -> R:
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
        return self.__registry._from(self.__inner.units)

    @property
    def u(self) -> smoot.Unit:
        """Return the units of this quantity.

        Examples
        --------
        ```python
        assert Quantity("1 meter").u == units.meter
        ```
        """
        return self.__registry._from(self.__inner.u)

    def is_compatible_with(self, units: str | smoot.Unit | Quantity) -> bool:
        """Return True if this unit is compatible with the other.

        Examples
        --------
        ```python
        assert Quantity("1 meter").is_compatible_with(units.kilometer)
        assert Quantity("1 meter").is_compatible_with(Quantity("1 kilometer"))
        assert not Quantity("1 meter").is_compatible_with(units.second)
        ```
        """
        return self.units.is_compatible_with(units)

    def to(self, units: str | smoot.Unit | Quantity) -> Quantity[T, R]:
        """Return a copy of this quantity converted to the target units.

        Examples
        --------
        ```python
        assert Quantity("1000 meter").to("km") == Quantity("1 km")
        ```
        """
        factor, _units = self._get_units(units)
        new = object.__new__(self.__class__)
        new.__inner = self.__inner.to(_units, factor=factor)
        return new

    def ito(self, units: str | smoot.Unit | Quantity) -> Quantity[T, R]:
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
        new = object.__new__(self.__class__)
        new.__inner = self.__inner.to_root_units(self.__registry._UnitRegistry__inner)
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
        self.__inner.ito_root_units(self.__registry._UnitRegistry__inner)

    # ==================================================
    # Standard dunder methods
    # ==================================================
    def __len__(self) -> int:
        try:
            return self.__inner.__len__()
        except AttributeError:
            msg = f"object of type '{Quantity.__name__}' has not len()"
            raise TypeError(msg)

    def __str__(self) -> str:
        return self.__inner.__str__()

    def __repr__(self) -> str:
        return f"<Quantity('{self.__inner}')>"

    def __hash__(self) -> int:
        return hash(self.__inner)

    def __iter__(self) -> Iterator[Quantity]:
        it_inner = iter(self.__inner)

        def it_outer():
            for nx in it_inner:
                new = object.__new__(self.__class__)
                new.__inner = nx
                yield new

        return it_outer()

    # ==================================================
    # Pickle support
    # ==================================================
    @staticmethod
    def _load(inner: F64Quantity | ArrayF64Quantity) -> Quantity[T, R]:
        # Use the currently set application registry.
        # WARNING: This may not be the same as the registry the pickled quantity was created with.
        new = object.__new__(smoot.ApplicationRegistry.get().Quantity)
        new.__inner = inner
        return new

    def __reduce__(self):
        return (Quantity._load, (self.__inner,))

    # ==================================================
    # Operators
    # ==================================================
    def __eq__(self, other: Any) -> Any:
        return self.__inner == self._get_inner(other)

    def __neg__(self) -> Quantity[T, R]:
        new = object.__new__(self.__class__)
        new.__inner = -self.__inner
        return new

    def __add__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(self.__class__)
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
        new = object.__new__(self.__class__)
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
        new = object.__new__(self.__class__)
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
            return self.__class__(magnitude[0], self.__registry._from(inner.units))

        new = object.__new__(self.__class__)
        new.__inner = inner
        return new

    def __rmatmul__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        return self._get_quantity(other) @ self

    def __imatmul__(self, other: Quantity[T, R] | T) -> Self:
        self.__inner @= self._get_inner(other)
        return self

    def __truediv__(self, other: Quantity[T, R] | T) -> Quantity[T, R]:
        q1, q2 = self._upcast(self._get_quantity(other))
        new = object.__new__(self.__class__)
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
        new = object.__new__(self.__class__)
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
        new = object.__new__(self.__class__)
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

        new = object.__new__(self.__class__)
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
        new = object.__new__(self.__class__)
        new.__inner = self.__inner.__floor__()
        return new

    def __ceil__(self) -> Quantity[T, R]:
        new = object.__new__(self.__class__)
        new.__inner = self.__inner.__ceil__()
        return new

    def __round__(self, ndigits: int | None = None) -> Quantity[T, R]:
        new = object.__new__(self.__class__)
        new.__inner = round(self.__inner, ndigits=ndigits)
        return new

    def __abs__(self) -> Quantity[T, R]:
        new = object.__new__(self.__class__)
        new.__inner = self.__inner.__abs__()
        return new

    def __trunc__(self) -> int:
        return self.__inner.__trunc__()

    def __float__(self) -> float:
        return float(self.__inner)

    def __int__(self) -> int:
        return int(self.__inner)

    # ==================================================
    # numpy support
    # ==================================================
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Self,
        **kwargs: Any,
    ) -> Quantity[T, R] | type(NotImplemented):
        if (func := NP_HANDLED_UFUNCS.get(ufunc.__name__)) is None:
            return NotImplemented
        return func(*inputs, **kwargs)

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Quantity[T, R] | type(NotImplemented):
        func_name = ".".join(func.__module__.split(".")[1:] + [func.__name__])
        if (func := NP_HANDLED_FUNCTIONS.get(func_name)) is None:
            return NotImplemented
        return func(*args, **kwargs)

    # ==================================================
    # private utils
    # ==================================================
    def _get_quantity(self, other: Any) -> Quantity:
        if isinstance(other, Quantity):
            return other
        if isinstance(other, smoot.Unit):
            return self.__class__(1, other)

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
            new = object.__new__(self.__class__)
            new.__inner = ArrayF64Quantity(np.array([self.m]), self.__inner.units)
            return new, other
        if t2 is F64Quantity:
            new = object.__new__(self.__class__)
            new.__inner = ArrayF64Quantity(np.array([other.m]), other.__inner.units)
            return self, new

        msg = f"No conversion exists between {t1} and {t2}"
        raise NotImplementedError(msg)

    def _get_units(
        self,
        units: UnitsLike | InnerUnit | Quantity[T, R],
    ) -> tuple[float, InnerUnit]:
        t = type(units)
        if t is str:
            return InnerUnit.parse(units, self.__registry._UnitRegistry__inner)
        if t is InnerUnit:
            return (1.0, units)
        if isinstance(units, Quantity):
            return (1.0, units._Quantity__inner.units)
        return (1.0, units._Unit__inner)

    @staticmethod
    def _get_magnitude(other: Any) -> R:
        if isinstance(other, Quantity):
            return other.magnitude
        return other
