"""Bespoke implementations of numpy functions using numpy's __array_ufunc__ and __array_function__ mechanisms."""

from __future__ import annotations
from enum import Enum, auto
from typing import Callable, Final

import numpy as np
from numpy.typing import NDArray

import smoot

NP_HANDLED_UFUNCS: Final[dict[str, Callable]] = {}
NP_HANDLED_FUNCTIONS: Final[dict[str, Callable]] = {}


class _FunctionType(Enum):
    """Supported numpy function types."""

    ufunc = auto()
    function = auto()


def _implements(func_name: str, func_type: _FunctionType) -> Callable:
    """Register the decorated function as an implementation of an existing numpy function."""

    def decorator(func: Callable) -> Callable:
        if func_type == _FunctionType.function:
            NP_HANDLED_FUNCTIONS[func_name] = func
        elif func_type == _FunctionType.ufunc:
            NP_HANDLED_UFUNCS[func_name] = func
        else:
            raise NotImplementedError(func_type)
        return func

    return decorator


# ==================================================
# Automated function implementations
# ==================================================
def _unary_unchanged_units(func_name: str, func_type: _FunctionType) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
        return x.__class__(
            value=func(x.magnitude, *args, **kwargs),
            units=x.units,
            registry=x._Quantity__registry,
        )


def _unary_requires_unit(
    func_name: str,
    func_type: _FunctionType,
    in_units: str,
    out_units: str | None,
) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
        return x.__class__(
            value=func(x.m_as(in_units), *args, **kwargs),
            units=out_units,
            registry=x._Quantity__registry,
        )


def _unary_boolean_output(func_name: str, func_type: _FunctionType) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x: smoot.Quantity, *args, **kwargs) -> NDArray[np.bool]:
        return func(x.magnitude, *args, **kwargs)


def _binary_unchanged_units(
    func_name: str,
    func_type: _FunctionType,
    requires_compatible_units: bool,
) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
        m2 = x2.m_as(x1) if requires_compatible_units else x2.magnitude

        return x1.__class__(
            value=func(x1.magnitude, m2, *args, **kwargs),
            units=x1.units,
            registry=x1._Quantity__registry,
        )


def _binary_requires_units(
    func_name: str,
    func_type: _FunctionType,
    in_unit: str,
    out_unit: str | None,
) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
        return x1.__class__(
            value=func(x1.m_as(in_unit), x2.m_as(in_unit), *args, **kwargs),
            units=out_unit,
            registry=x1._Quantity__registry,
        )


def _binary_boolean_output(func_name: str, func_type: _FunctionType) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(
        x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
    ) -> NDArray[np.bool]:
        return func(x1.magnitude, x2.m_as(x1), *args, **kwargs)


# Functions for which the output units match the input units.
for func_name, func_type in (
    # ufunc
    ("negative", _FunctionType.ufunc),
    ("positive", _FunctionType.ufunc),
    ("absolute", _FunctionType.ufunc),
    ("fabs", _FunctionType.ufunc),
    ("rint", _FunctionType.ufunc),
    ("sign", _FunctionType.ufunc),
    ("floor", _FunctionType.ufunc),
    ("ceil", _FunctionType.ufunc),
    ("trunc", _FunctionType.ufunc),
    # function
    ("sum", _FunctionType.function),
):
    _unary_unchanged_units(func_name, func_type)

# Functions that require dimensionless quantities as inputs.
for func_name, func_type, in_units, out_units in (
    ("exp", _FunctionType.ufunc, "dimensionless", None),
    ("exp2", _FunctionType.ufunc, "dimensionless", None),
    ("log", _FunctionType.ufunc, "dimensionless", None),
    ("log2", _FunctionType.ufunc, "dimensionless", None),
    ("log10", _FunctionType.ufunc, "dimensionless", None),
    ("expm1", _FunctionType.ufunc, "dimensionless", None),
    ("log1p", _FunctionType.ufunc, "dimensionless", None),
    ("sin", _FunctionType.ufunc, "radian", "dimensionless"),
    ("cos", _FunctionType.ufunc, "radian", "dimensionless"),
    ("tan", _FunctionType.ufunc, "radian", "dimensionless"),
    ("arcsin", _FunctionType.ufunc, "dimensionless", "radian"),
    ("arccos", _FunctionType.ufunc, "dimensionless", "radian"),
    ("arctan", _FunctionType.ufunc, "dimensionless", "radian"),
    ("sinh", _FunctionType.ufunc, "radian", "dimensionless"),
    ("cosh", _FunctionType.ufunc, "radian", "dimensionless"),
    ("tanh", _FunctionType.ufunc, "radian", "dimensionless"),
    ("arcsinh", _FunctionType.ufunc, "dimensionless", "radian"),
    ("arccosh", _FunctionType.ufunc, "dimensionless", "radian"),
    ("arctanh", _FunctionType.ufunc, "dimensionless", "radian"),
    ("radians", _FunctionType.ufunc, "degree", "radian"),
    ("degrees", _FunctionType.ufunc, "radian", "degree"),
    ("deg2rad", _FunctionType.ufunc, "degree", "radian"),
    ("rad2deg", _FunctionType.ufunc, "radian", "degree"),
):
    _unary_requires_unit(func_name, func_type, in_units, out_units)

# Unary functions that output boolean arrays.
for func_name, func_type in (
    ("isfinite", _FunctionType.ufunc),
    ("isinf", _FunctionType.ufunc),
    ("isnan", _FunctionType.ufunc),
    ("signbit", _FunctionType.ufunc),
):
    _unary_boolean_output(func_name, func_type)

# Functions for which the output units match the units of the first argument.
for func_name, func_type, requires_compatible_units in (
    ("add", _FunctionType.ufunc, True),
    ("subtract", _FunctionType.ufunc, True),
    ("remainder", _FunctionType.ufunc, False),
    ("mod", _FunctionType.ufunc, False),
    ("fmod", _FunctionType.ufunc, False),
    ("hypot", _FunctionType.ufunc, True),
    ("copysign", _FunctionType.ufunc, True),
    ("nextafter", _FunctionType.ufunc, True),
):
    _binary_unchanged_units(func_name, func_type, requires_compatible_units)

# Functions requiring both arguments to be dimensionless.
for func_name, func_type, in_unit, out_unit in (
    ("logaddexp", _FunctionType.ufunc, "dimensionless", None),
    ("logaddexp2", _FunctionType.ufunc, "dimensionless", None),
    ("power", _FunctionType.ufunc, "dimensionless", None),
    ("float_power", _FunctionType.ufunc, "dimensionless", None),
    ("arctan2", _FunctionType.ufunc, "dimensionless", None),
):
    _binary_requires_units(func_name, func_type, in_unit, out_unit)

# Binary functions that output boolean arrays.
for func_name, func_type in (
    ("greater", _FunctionType.ufunc),
    ("greater_equal", _FunctionType.ufunc),
    ("less", _FunctionType.ufunc),
    ("less_equal", _FunctionType.ufunc),
    ("not_equal", _FunctionType.ufunc),
    ("equal", _FunctionType.ufunc),
):
    _binary_boolean_output(func_name, func_type)


# ==================================================
# universal functions (ufunc)
#
# Functions that need specialized implementations
# (e.g. non-standard unit algebra).
# ==================================================
@_implements("multiply", _FunctionType.ufunc)
def _multiply(
    x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
) -> smoot.Quantity:
    return x1.__class__(
        value=np.multiply(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u * x2.u,
        registry=x1._Quantity__registry,
    )


@_implements("matmul", _FunctionType.ufunc)
def _matmul(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x1.__class__(
        value=np.matmul(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u * x2.u,
        registry=x1._Quantity__registry,
    )


@_implements("divide", _FunctionType.ufunc)
def _divide(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x1.__class__(
        value=np.divide(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u / x2.u,
        registry=x1._Quantity__registry,
    )


@_implements("true_divide", _FunctionType.ufunc)
def _true_divide(
    x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
) -> smoot.Quantity:
    return x1.__class__(
        value=np.true_divide(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u / x2.u,
        registry=x1._Quantity__registry,
    )


@_implements("floor_divide", _FunctionType.ufunc)
def _floor_divide(
    x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
) -> smoot.Quantity:
    return x1.__class__(
        value=np.true_divide(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u / x2.u,
        registry=x1._Quantity__registry,
    )


@_implements("sqrt", _FunctionType.ufunc)
def _sqrt(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.sqrt(x.m, *args, **kwargs),
        units=x.units**0.5,
        registry=x._Quantity__registry,
    )


@_implements("square", _FunctionType.ufunc)
def _square(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.square(x.m, *args, **kwargs),
        units=x.units**2,
        registry=x._Quantity__registry,
    )


@_implements("cbrt", _FunctionType.ufunc)
def _cbrt(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.cbrt(x.m, *args, **kwargs),
        units=x.units ** (1 / 3),
        registry=x._Quantity__registry,
    )


@_implements("reciprocal", _FunctionType.ufunc)
def _reciprocal(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.reciprocal(x.m, *args, **kwargs),
        units=1 / x.units,
        registry=x._Quantity__registry,
    )


@_implements("maximum", _FunctionType.ufunc)
def _maximum(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x1.__class__(
        value=np.maximum(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.units,
        registry=x1._Quantity__registry,
    )


@_implements("minimum", _FunctionType.ufunc)
def _minimum(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x1.__class__(
        value=np.minimum(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.units,
        registry=x1._Quantity__registry,
    )


# ==================================================
# higher order array functions
# ==================================================
# TODO(jwh)
