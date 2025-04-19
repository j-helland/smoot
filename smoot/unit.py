# mypy: ignore-errors
from __future__ import annotations
from typing import Any, Iterable, Union
import numpy as np
from typing_extensions import Self

import smoot
from .smoot import (
    Unit as InnerUnit,
    div_unit,
    mul_unit,
    rdiv_unit,
    arr_mul_unit,
    arr_div_unit,
    arr_rdiv_unit,
)

OperatorUnitLike = Union["Unit", float, int, Iterable[float], Iterable[int]]


class Unit:
    __slots__ = ("__inner", "__registry")

    def __init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def parse(cls, s: str) -> Unit:
        """Parse the given expression into a unit.

        Examples
        --------
        ```python
        assert Unit.parse("meter") == units.meter
        assert Unit.parser("1 / meter") == (1 / units.meter)
        ```
        """
        try:
            registry = cls.__registry  # type: ignore[attr-defined]
        except AttributeError:
            msg = (
                "Attempted to instantiate an abstract Unit. "
                "Please use a Unit via UnitRegistry e.g. `UnitRegistry().Unit`."
            )
            raise TypeError(msg)

        new = object.__new__(Unit)
        # Ignore the reduction factor when directly returning units to users
        # since the factor is only used internally.
        _, new.__inner = InnerUnit.parse(s, registry=registry)
        new.__registry = registry
        return new

    @classmethod
    def _from(cls, u: InnerUnit) -> Unit:
        new = object.__new__(Unit)
        new.__inner = u
        new.__registry = cls.__registry
        return new

    def is_compatible_with(self, other: Unit) -> bool:
        """Return True if this unit is compatible with the other.

        Examples
        --------
        ```python
        assert units.meter.is_compatible_with(units.kilometer)
        assert not units.meter.is_compatible_with(units.second)
        ```
        """
        return self.__inner.is_compatible_with(other.__inner)

    @property
    def dimensionless(self) -> bool:
        """Return True if this unit is dimensionless.

        Examples
        --------
        ```python
        assert units.dimensionless.dimensionless
        assert not units.meter.dimensionless
        assert (units.meter / units.meter).dimensionless
        ```
        """
        return self.__inner.dimensionless

    @property
    def dimensionality(self) -> dict[str, float] | None:
        """Returns the dimensionality of this unit.

        Examples
        --------
        ```python
        assert units.dimensionless.dimensionality is None
        assert units.meter.dimensionality == {"[length]": 1.0}
        assert units.newton.dimensionality == {"[length]": 1.0, "[mass]": 1.0, "[time]": -2.0}
        ```
        """
        return self.__inner.dimensionality(self.__registry)

    def to_root_units(self) -> Unit:
        """Return a simplified version of this unit with the minimum number of base units.

        Examples
        --------
        ```python
        assert str(units.meter.to_root_units()) == "meter"
        assert str(units.kilometer.to_root_units()) == "meter
        assert str(units.newton.to_root_units()) == "(gram * meter) / second ** 2"
        ```
        """
        new = object.__new__(Unit)
        new.__inner = self.__inner.to_root_units(self.__registry)
        new.__registry = self.__registry
        return new

    def ito_root_units(self) -> Self:
        self.__inner.ito_root_units(self.__registry)
        return self

    def __str__(self) -> str:
        return self.__inner.__str__()

    def __repr__(self) -> str:
        return f"<Unit('{self}')>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Unit):
            return False
        return self.__inner == other.__inner

    def __mul__(self, other: OperatorUnitLike) -> Unit | smoot.Quantity:
        if type(other) in (int, float):
            new = object.__new__(smoot.Quantity)
            new._Quantity__inner = mul_unit(num=other, unit=self.__inner)
            new._Quantity__registry = self.__registry
        elif isinstance(other, Iterable):
            new = object.__new__(smoot.Quantity)
            arr = np.array(other, dtype=np.float64)
            new._Quantity__inner = arr_mul_unit(arr=arr, unit=self.__inner)
            new._Quantity__registry = self.__registry
        else:
            new = object.__new__(Unit)
            new.__inner = self.__inner * other.__inner
            new.__registry = self.__registry
        return new

    def __rmul__(self, other: OperatorUnitLike) -> Unit | smoot.Quantity:
        return self * other

    def __imul__(self, other: Unit) -> Self:
        self.__inner *= other.__inner
        return self

    def __truediv__(self, other: OperatorUnitLike) -> Unit | smoot.Quantity:
        if type(other) in (int, float):
            new = object.__new__(smoot.Quantity)
            new._Quantity__inner = div_unit(unit=self.__inner, num=other)
            new._Quantity__registry = self.__registry
        elif isinstance(other, Iterable):
            new = object.__new__(smoot.Quantity)
            arr = np.array(other, dtype=np.float64)
            new._Quantity__inner = arr_div_unit(unit=self.__inner, arr=arr)
            new._Quantity__registry = self.__registry
        else:
            new = object.__new__(Unit)
            new.__inner = self.__inner / other.__inner
            new.__registry = self.__registry
        return new

    def __itruediv__(self, other: Unit) -> Self:
        self.__inner /= other.__inner
        return self

    def __rtruediv__(self, other: OperatorUnitLike) -> Unit | smoot.Quantity:
        if type(other) in (int, float):
            new = object.__new__(smoot.Quantity)
            new._Quantity__inner = rdiv_unit(num=other, unit=self.__inner)
            new._Quantity__registry = self.__registry
        elif isinstance(other, Iterable):
            new = object.__new__(smoot.Quantity)
            arr = np.array(other, dtype=np.float64)
            new._Quantity__inner = arr_rdiv_unit(arr=arr, unit=self.__inner)
            new._Quantity__registry = self.__registry
        else:
            new = object.__new__(Unit)
            new.__inner = other.__inner__ / self.__inner
            new.__registry = self.__registry
        return new

    def __pow__(self, p: int | float, modulo: int | float | None = None) -> Unit:
        new = object.__new__(Unit)
        new.__inner = self.__inner.__pow__(p, modulo)
        new.__registry = self.__registry
        return new

    def __ipow__(self, p: int | float, modulo: int | float | None = None) -> Self:
        self.__inner.__ipow__(p, modulo)
        return self
