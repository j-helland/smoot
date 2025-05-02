"""Bespoke implementations of numpy functions using numpy's __array_ufunc__ and __array_function__ mechanisms."""

from __future__ import annotations
import copy
from enum import Enum, auto
from typing import Any, Callable, Final, Iterable, Iterator
import warnings

import numpy as np
from numpy.typing import NDArray, ArrayLike

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
def _upcast(
    x: smoot.Quantity | Any, y: smoot.Quantity | Any
) -> tuple[smoot.Quantity, smoot.Quantity]:
    """Assumes that x and y cannot both be non-Quantity types."""
    x_is_quantity = isinstance(x, smoot.Quantity)
    y_is_quantity = isinstance(y, smoot.Quantity)

    if x_is_quantity and y_is_quantity:
        return (x, y)
    if x_is_quantity:
        return (x, x.__class__(y))
    return (y.__class__(x), y)


def _unary_unchanged_units(func_name: str, func_type: _FunctionType) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
        return x.__class__(
            value=func(x.magnitude, *args, **kwargs),
            units=x.units,
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
        )


def _unary_boolean_output(func_name: str, func_type: _FunctionType) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(x: smoot.Quantity, *args, **kwargs) -> NDArray[np.bool]:
        return func(x.magnitude, *args, **kwargs)


def _unary_internal_impl(
    func_name: str, func_type: _FunctionType, internal_name: str
) -> None:
    @_implements(func_name, func_type)
    def impl(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
        new = object.__new__(x.__class__)
        new._Quantity__inner = getattr(x._Quantity__inner, internal_name)()
        return new


def _join_arrays_impl(
    func_name: str,
    func_type: _FunctionType,
) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(
        arrays: Iterable[smoot.Quantity | ArrayLike], *args, **kwargs
    ) -> smoot.Quantity:
        magnitudes: list[ArrayLike] = []
        unit: smoot.Unit | None = None

        for item in arrays:
            if isinstance(item, smoot.Quantity):
                if unit is None:
                    unit = item.u
                elif not item.u.is_compatible_with(unit):
                    msg = f"Expected consistent units for concatenate. Got '{unit}' and '{item.u}'."
                    raise ValueError(msg)
        if unit is None:
            msg = "Concatenate found no units in arrays"
            raise ValueError(msg)

        for item in arrays:
            if isinstance(item, smoot.Quantity):
                magnitudes.append(item.m_as(unit))
            else:
                magnitudes.append(item)

        result_magnitude = func(tuple(magnitudes), *args, **kwargs)
        return unit._Unit__registry.Quantity(result_magnitude, unit)


def _binary_unchanged_units(
    func_name: str,
    func_type: _FunctionType,
    requires_compatible_units: bool,
) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(
        x1: smoot.Quantity | Any,
        x2: smoot.Quantity | Any,
        *args,
        **kwargs,
    ) -> smoot.Quantity:
        x1, x2 = _upcast(x1, x2)
        m2 = x2.m_as(x1) if requires_compatible_units else x2.magnitude

        return x1.__class__(
            value=func(x1.magnitude, m2, *args, **kwargs),
            units=x1.units,
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
        x1, x2 = _upcast(x1, x2)
        return x1.__class__(
            value=func(x1.m_as(in_unit), x2.m_as(in_unit), *args, **kwargs),
            units=out_unit,
        )


def _binary_boolean_output(func_name: str, func_type: _FunctionType) -> None:
    func = getattr(np, func_name)

    @_implements(func_name, func_type)
    def impl(
        x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
    ) -> NDArray[np.bool]:
        x1, x2 = _upcast(x1, x2)
        return func(x1.magnitude, x2.m_as(x1), *args, **kwargs)


# Functions for which the output units match the input units.
for func_name, func_type in (
    # ufunc
    ("negative", _FunctionType.ufunc),
    ("positive", _FunctionType.ufunc),
    ("rint", _FunctionType.ufunc),
    ("sign", _FunctionType.ufunc),
    ("floor", _FunctionType.ufunc),
    ("ceil", _FunctionType.ufunc),
    ("trunc", _FunctionType.ufunc),
    # array dispatch function
    ("sum", _FunctionType.function),
    ("nansum", _FunctionType.function),
    ("amax", _FunctionType.function),
    ("amin", _FunctionType.function),
    ("max", _FunctionType.function),
    ("nanmax", _FunctionType.function),
    ("min", _FunctionType.function),
    ("nanmin", _FunctionType.function),
    ("around", _FunctionType.function),
    ("average", _FunctionType.function),
    ("mean", _FunctionType.function),
    ("nanmean", _FunctionType.function),
    ("median", _FunctionType.function),
    ("nanmedian", _FunctionType.function),
    ("std", _FunctionType.function),
    ("var", _FunctionType.function),
    ("nanvar", _FunctionType.function),
    ("nanstd", _FunctionType.function),
    ("broadcast_to", _FunctionType.function),
    ("cumsum", _FunctionType.function),
    ("nancumsum", _FunctionType.function),
    ("delete", _FunctionType.function),
    ("diagonal", _FunctionType.function),
    ("diff", _FunctionType.function),
    ("expand_dims", _FunctionType.function),
    ("ediff1d", _FunctionType.function),
    ("fix", _FunctionType.function),
    ("flip", _FunctionType.function),
    ("gradient", _FunctionType.function),
    ("moveaxis", _FunctionType.function),
    ("nan_to_num", _FunctionType.function),
    ("percentile", _FunctionType.function),
    ("nanpercentile", _FunctionType.function),
    ("quantile", _FunctionType.function),
    ("nanquantile", _FunctionType.function),
    ("ptp", _FunctionType.function),
    ("swapaxes", _FunctionType.function),
    ("sort", _FunctionType.function),
    ("tile", _FunctionType.function),
    ("trim_zeros", _FunctionType.function),
    ("ravel", _FunctionType.function),
    ("squeeze", _FunctionType.function),
    ("round", _FunctionType.function),
    ("rot90", _FunctionType.function),
    ("rollaxis", _FunctionType.function),
    ("roll", _FunctionType.function),
    ("resize", _FunctionType.function),
    ("repeat", _FunctionType.function),
    ("take", _FunctionType.function),
    ("trace", _FunctionType.function),
):
    _unary_unchanged_units(func_name, func_type)

# Functions that require dimensionless quantities as inputs.
for func_name, func_type, in_units, out_units in (
    # ufunc
    ("exp2", _FunctionType.ufunc, "dimensionless", None),
    ("expm1", _FunctionType.ufunc, "dimensionless", None),
    ("log1p", _FunctionType.ufunc, "dimensionless", None),
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
    # array dispatch function
    ("cumprod", _FunctionType.function, "dimensionless", None),
    ("nancumprod", _FunctionType.function, "dimensionless", None),
):
    _unary_requires_unit(func_name, func_type, in_units, out_units)

# Unary functions that output boolean arrays.
for func_name, func_type in (
    # ufunc
    ("isfinite", _FunctionType.ufunc),
    ("isinf", _FunctionType.ufunc),
    ("isnan", _FunctionType.ufunc),
    ("signbit", _FunctionType.ufunc),
    # array dispatch function
    ("all", _FunctionType.function),
    ("any", _FunctionType.function),
):
    _unary_boolean_output(func_name, func_type)

for func_name, func_type, internal_name in (
    ("sqrt", _FunctionType.ufunc, "sqrt"),
    ("absolute", _FunctionType.ufunc, "__abs__"),
    ("fabs", _FunctionType.ufunc, "__abs__"),
    ("sin", _FunctionType.ufunc, "sin"),
    ("cos", _FunctionType.ufunc, "cos"),
    ("tan", _FunctionType.ufunc, "tan"),
    ("arcsin", _FunctionType.ufunc, "arcsin"),
    ("arccos", _FunctionType.ufunc, "arccos"),
    ("arctan", _FunctionType.ufunc, "arctan"),
    ("log", _FunctionType.ufunc, "log"),
    ("log10", _FunctionType.ufunc, "log10"),
    ("log2", _FunctionType.ufunc, "log2"),
    ("exp", _FunctionType.ufunc, "exp"),
):
    _unary_internal_impl(func_name, func_type, internal_name)

for func_name, func_type in (
    ("concatenate", _FunctionType.function),
    ("hstack", _FunctionType.function),
    ("vstack", _FunctionType.function),
    ("column_stack", _FunctionType.function),
    ("dstack", _FunctionType.function),
    ("stack", _FunctionType.function),
):
    _join_arrays_impl(func_name, func_type)

# Functions for which the output units match the units of the first argument.
for func_name, func_type, requires_compatible_units in (
    # ufunc
    ("add", _FunctionType.ufunc, True),
    ("subtract", _FunctionType.ufunc, True),
    ("remainder", _FunctionType.ufunc, False),
    ("mod", _FunctionType.ufunc, False),
    ("fmod", _FunctionType.ufunc, False),
    ("hypot", _FunctionType.ufunc, True),
    ("copysign", _FunctionType.ufunc, True),
    ("nextafter", _FunctionType.ufunc, True),
    # array dispatch function
    ("append", _FunctionType.function, True),
    ("insert", _FunctionType.function, True),
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
    # ufunc
    ("greater", _FunctionType.ufunc),
    ("greater_equal", _FunctionType.ufunc),
    ("less", _FunctionType.ufunc),
    ("less_equal", _FunctionType.ufunc),
    ("not_equal", _FunctionType.ufunc),
    ("equal", _FunctionType.ufunc),
    # array dispatch function
    ("allclose", _FunctionType.function),
    ("array_equal", _FunctionType.function),
    ("isclose", _FunctionType.function),
    ("isin", _FunctionType.function),
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
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.multiply(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u * x2.u,
    )


@_implements("matmul", _FunctionType.ufunc)
def _matmul(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.matmul(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u * x2.u,
    )


@_implements("divide", _FunctionType.ufunc)
def _divide(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.divide(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u / x2.u,
    )


@_implements("true_divide", _FunctionType.ufunc)
def _true_divide(
    x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
) -> smoot.Quantity:
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.true_divide(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u / x2.u,
    )


@_implements("floor_divide", _FunctionType.ufunc)
def _floor_divide(
    x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs
) -> smoot.Quantity:
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.true_divide(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.u / x2.u,
    )


@_implements("square", _FunctionType.ufunc)
def _square(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    new = object.__new__(x.__class__)
    new._Quantity__inner = x._Quantity__inner**2
    return new


@_implements("reciprocal", _FunctionType.ufunc)
def _reciprocal(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.reciprocal(x.m, *args, **kwargs),
        units=1 / x.units,
    )


@_implements("maximum", _FunctionType.ufunc)
def _maximum(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.maximum(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.units,
    )


@_implements("minimum", _FunctionType.ufunc)
def _minimum(x1: smoot.Quantity, x2: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    x1, x2 = _upcast(x1, x2)
    return x1.__class__(
        value=np.minimum(x1.m, x2.m_as(x1), *args, **kwargs),
        units=x1.units,
    )


# ==================================================
# higher order array functions
# ==================================================


def _get_first_unit(arr: Iterable[Any]) -> smoot.Unit:
    for val in arr:
        if isinstance(val, smoot.Quantity):
            return val.units

    msg = "No quantities found in array"
    raise ValueError(msg)


def _upcast_many(arr: Iterable[Any], units: smoot.Unit) -> Iterator[smoot.Quantity]:
    return (
        a.to(units)
        if isinstance(a, smoot.Quantity)
        else units._Unit__registry.Quantity(a)
        for a in arr
    )


@_implements("argmax", _FunctionType.function)
def _argmax(a: smoot.Quantity, *args, **kwargs) -> np.intp:
    return np.argmax(a.magnitude, *args, **kwargs)


@_implements("argmin", _FunctionType.function)
def _argmin(a: smoot.Quantity, *args, **kwargs) -> np.intp:
    return np.argmin(a.magnitude, *args, **kwargs)


@_implements("argsort", _FunctionType.function)
def _argsort(a: smoot.Quantity, *args, **kwargs) -> NDArray[np.intp]:
    return np.argsort(a.magnitude, *args, **kwargs)


@_implements("argwhere", _FunctionType.function)
def _argwhere(a: smoot.Quantity) -> NDArray[np.intp]:
    return np.argwhere(a.magnitude)


@_implements("array2string", _FunctionType.function)
def _array2string(a: smoot.Quantity, *args, **kwargs) -> str:
    return np.array2string(a.magnitude, *args, **kwargs)


@_implements("array_equiv", _FunctionType.function)
def _array_equiv(a1: smoot.Quantity | Any, a2: smoot.Quantity | Any) -> bool:
    a1, a2 = _upcast(a1, a2)
    return np.array_equiv(a1.m, a2.m_as(a1))


@_implements("array_repr", _FunctionType.function)
def _array_repr(arr: smoot.Quantity, *args, **kwargs) -> str:
    return np.array_repr(arr.magnitude, *args, **kwargs)


@_implements("array_str", _FunctionType.function)
def _array_str(a: smoot.Quantity, *args, **kwargs) -> str:
    return np.array_str(a.magnitude, *args, **kwargs)


@_implements("atleast_1d", _FunctionType.function)
def _atleast_1d(
    *arys: smoot.Quantity | Any,
) -> smoot.Quantity | tuple[smoot.Quantity | NDArray[Any], ...]:
    res = tuple(
        val.__class__(
            value=np.atleast_1d(val.magnitude),
            units=val.units,
        )
        if isinstance(val, smoot.Quantity)
        else np.atleast_1d(val)
        for val in arys
    )
    if len(res) == 1:
        return res[0]
    return res


@_implements("atleast_2d", _FunctionType.function)
def _atleast_2d(
    *arys: smoot.Quantity | Any,
) -> smoot.Quantity | tuple[smoot.Quantity | NDArray[Any], ...]:
    res = tuple(
        val.__class__(
            value=np.atleast_2d(val.magnitude),
            units=val.units,
        )
        if isinstance(val, smoot.Quantity)
        else np.atleast_2d(val)
        for val in arys
    )
    if len(res) == 1:
        return res[0]
    return res


@_implements("atleast_3d", _FunctionType.function)
def _atleast_3d(
    *arys: smoot.Quantity | Any,
) -> smoot.Quantity | tuple[smoot.Quantity | NDArray[Any], ...]:
    res = tuple(
        val.__class__(
            value=np.atleast_3d(val.magnitude),
            units=val.units,
        )
        if isinstance(val, smoot.Quantity)
        else np.atleast_3d(val)
        for val in arys
    )
    if len(res) == 1:
        return res[0]
    return res


@_implements("block", _FunctionType.function)
def _block(arrays: Iterable[smoot.Quantity | Any]) -> smoot.Quantity:
    units = _get_first_unit(arrays)
    return units._Unit__registry.Quantity(
        np.block(
            [
                arr.m_as(units) if isinstance(arr, smoot.Quantity) else arr
                for arr in arrays
            ]
        ),
        units,
    )


@_implements("broadcast_arrays", _FunctionType.function)
def _broadcast_arrays(
    *args: smoot.Quantity | Any, **kwargs
) -> tuple[smoot.Quantity | NDArray[Any], ...]:
    arrs = np.broadcast_arrays(
        *(arr.magnitude if isinstance(arr, smoot.Quantity) else arr for arr in args)
    )
    return tuple(
        v2.__class__(v1, v2.units) if isinstance(v2, smoot.Quantity) else v1
        for v1, v2 in zip(arrs, args)
    )


@_implements("clip", _FunctionType.function)
def _clip(
    a: smoot.Quantity,
    a_min: smoot.Quantity | float,
    a_max: smoot.Quantity | float,
    *args,
    **kwargs,
) -> smoot.Quantity:
    a_min, a_max = _upcast_many((a_min, a_max), a.units)
    return a.__class__(
        value=np.clip(a.m, a_min.m_as(a), a_max.m_as(a), *args, **kwargs),
        units=a.units,
    )


@_implements("compress", _FunctionType.function)
def _compress(
    condition: ArrayLike, a: smoot.Quantity, *args, **kwargs
) -> smoot.Quantity:
    result_magnitude = np.compress(condition, a.m, *args, **kwargs)
    return a.__class__(result_magnitude, a.u)


@_implements("copy", _FunctionType.function)
def _copy(a: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return copy.copy(a)


@_implements("copyto", _FunctionType.function)
def _copyto(
    dst: smoot.Quantity | Any, src: smoot.Quantity | Any, *args, **kwargs
) -> None:
    if isinstance(dst, smoot.Quantity):
        # TODO(jwh): Can we eliminate the extra array copy going into Rust?
        dst._Quantity__inner = dst.__class__(src, dst.units)._Quantity__inner
    else:
        warnings.warn(
            "The unit of the quantity is stripped when copying to non-quantity",
            stacklevel=2,
        )
        np.copyto(dst, src.magnitude, *args, **kwargs)


@_implements("correlate", _FunctionType.function)
def _correlate(
    a: smoot.Quantity | ArrayLike,
    v: smoot.Quantity | ArrayLike,
    *args,
    **kwargs,
) -> smoot.Quantity:
    a, v = _upcast(a, v)
    magnitude = np.correlate(a.m, v.m, *args, **kwargs)
    units = a.u * v.u
    return a.__class__(magnitude, units)


@_implements("count_nonzero", _FunctionType.function)
def _count_nonzero(a, *args, **kwargs) -> int:
    return a._Quantity__inner.count_nonzero()


@_implements("cross", _FunctionType.function)
def _cross(a, b, *args, **kwargs) -> smoot.Quantity:
    a, b = _upcast(a, b)
    magnitude = np.cross(a.m, b.m, *args, **kwargs)
    units = a.u * b.u
    return a.__class__(magnitude, units)


@_implements("dot", _FunctionType.function)
def _dot(
    x: smoot.Quantity | Any, y: smoot.Quantity | Any, *args, **kwargs
) -> smoot.Quantity:
    x, y = _upcast(x, y)
    magnitude = np.dot(x.m, y.m, *args, **kwargs)
    units = x.u * y.u
    return x.__class__(magnitude, units)


@_implements("einsum", _FunctionType.function)
def _einsum(
    subscripts: str, *operands: smoot.Quantity | Any, **kwargs
) -> smoot.Quantity:
    units = _get_first_unit(operands)
    result_units = units.parse("dimensionless")
    arrs = []
    for operand in _upcast_many(operands, units):
        result_units *= operand.units
        arrs.append(operand.magnitude)

    return units._Unit__registry.Quantity(
        value=np.einsum(subscripts, *arrs, **kwargs), units=units
    )


@_implements("full_like", _FunctionType.function)
def _full_like(a: smoot.Quantity, fill_value, *args, **kwargs) -> smoot.Quantity:
    magnitude = np.full_like(a.m, fill_value, *args, **kwargs)
    return a.__class__(magnitude, a.u)


@_implements("insert", _FunctionType.function)
def _insert(arr, obj, values, *args, **kwargs) -> smoot.Quantity:
    unit: smoot.Unit | None = None

    # Check if the main array is a Quantity
    if isinstance(arr, smoot.Quantity):
        unit = arr.u

    # Check if values is a Quantity
    if isinstance(values, smoot.Quantity):
        if unit is None:
            unit = values.u
        elif not values.u.is_compatible_with(unit):
            msg = f"Unit mismatch between arr ({unit}) and values ({values.u})"
            raise ValueError(msg)

    # Check if values contains any Quantity elements (for non-Quantity values)
    if unit is None and not isinstance(values, smoot.Quantity):
        # Flatten and check each element in values for Quantity instances
        def check_elements(item):
            nonlocal unit
            if isinstance(item, smoot.Quantity):
                if unit is None:
                    unit = item.u
                elif not item.u.is_compatible_with(unit):
                    msg = f"Unit mismatch in values: {item.u} vs {unit}"
                    raise ValueError(msg)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    check_elements(sub_item)

        try:
            check_elements(values)
        except TypeError:
            # values is not iterable, skip
            pass

    if unit is None:
        msg = "No units found in arr or values"
        raise ValueError(msg)

    # Convert arr to magnitude
    if isinstance(arr, smoot.Quantity):
        arr_magnitude = arr.m_as(unit)
    else:
        arr_magnitude = arr

    # Convert values to magnitude
    def convert_values(item):
        if isinstance(item, smoot.Quantity):
            return item.m_as(unit)
        elif isinstance(item, (list, tuple)):
            return [convert_values(sub_item) for sub_item in item]
        else:
            return item

    if isinstance(values, smoot.Quantity):
        values_magnitude = values.m_as(unit)
    else:
        values_magnitude = convert_values(values)

    # Perform the insert operation
    result_magnitude = np.insert(arr_magnitude, obj, values_magnitude, *args, **kwargs)

    # Return as a Quantity with the determined unit
    return unit._Unit__registry.Quantity(result_magnitude, unit)


@_implements("interp", _FunctionType.function)
def _interp(x, xp, fp, *args, **kwargs) -> smoot.Quantity:
    if isinstance(x, smoot.Quantity):
        units = x.u
    elif isinstance(xp, smoot.Quantity):
        units = xp.u
    else:
        units = fp.u

    x, xp, fp = _upcast_many((x, xp, fp), units)
    magnitude = np.interp(x.m, xp.m, fp.m, *args, **kwargs)
    return x.__class__(magnitude, units)


@_implements("intersect1d", _FunctionType.function)
def _intersect1d(ar1, ar2, *args, **kwargs) -> smoot.Quantity:
    ar1, ar2 = _upcast(ar1, ar2)
    magnitude = np.intersect1d(ar1.m, ar2.m_as(ar1), *args, **kwargs)
    return ar1.__class__(magnitude, ar1.u)


@_implements("iscomplex", _FunctionType.function)
def _iscomplex(_) -> bool:
    # TODO(jwh): support for complex numbers
    return False


@_implements("isreal", _FunctionType.function)
def _isreal(_) -> bool:
    # TODO(jwh): support for complex numbers
    return True


@_implements("linspace", _FunctionType.function)
def _linspace(start, stop, *args, **kwargs) -> smoot.Quantity:
    start, stop = _upcast(start, stop)
    magnitude = np.linspace(start.m, stop.m_as(start), *args, **kwargs)
    return start.__class__(magnitude, start.u)


@_implements("meshgrid", _FunctionType.function)
def _meshgrid(*xi, **kwargs) -> tuple[smoot.Quantity, ...]:
    for x in xi:
        if isinstance(x, smoot.Quantity):
            units = x.u
            break
    else:
        msg = "Meshgrid found no quantity in its inputs"
        raise TypeError(msg)

    xi = _upcast_many(xi, units)
    magnitudes = np.meshgrid(*(x.m for x in xi), **kwargs)
    return tuple(units._Unit__registry.Quantity(m, units) for m in magnitudes)


@_implements("nanargmax", _FunctionType.function)
def _nanargmax(a: smoot.Quantity, *args, **kwargs) -> np.intp:
    return np.nanargmax(a.magnitude, *args, **kwargs)


@_implements("nanargmin", _FunctionType.function)
def _nanargmin(a: smoot.Quantity, *args, **kwargs) -> np.intp:
    return np.nanargmin(a.magnitude, *args, **kwargs)


@_implements("nonzero", _FunctionType.function)
def _nonzero(a: smoot.Quantity) -> tuple[NDArray[np.intp], ...]:
    return np.nonzero(a.m)


@_implements("ones_like", _FunctionType.function)
def _ones_like(a, *args, **kwargs) -> smoot.Quantity:
    magnitude = np.ones_like(a.m, *args, **kwargs)
    return a.__class__(magnitude, a.u)


@_implements("pad", _FunctionType.function)
def _pad(
    array: smoot.Quantity,
    pad_width: ArrayLike,
    *args,
    **kwargs,
) -> smoot.Quantity:
    units = array.u

    def _convert(arg):
        if isinstance(arg, Iterable):
            return tuple(map(_convert, arg))
        elif not isinstance(arg, smoot.Quantity):
            if arg == 0 or np.isnan(arg):
                return array.__class__(arg, units)
            return array.__class__(arg, "dimensionless")
        return arg.m_as(array)

    if vals := kwargs.get("constant_values"):
        kwargs["constant_values"] = _convert(vals)
    if vals := kwargs.get("end_values"):
        kwargs["end_values"] = _convert(vals)

    magnitude = np.pad(array.m, pad_width, *args, **kwargs)
    return array.__class__(magnitude, units)


@_implements("transpose", _FunctionType.function)
def _transpose(a: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    magnitude = np.transpose(a.m, *args, **kwargs)
    return a.__class__(magnitude, a.u)


@_implements("prod", _FunctionType.function)
def _prod(
    a: smoot.Quantity, axis: int | tuple[int, ...] | None = None, *args, **kwargs
) -> smoot.Quantity:
    magnitude = np.prod(a.m, axis, *args, **kwargs)
    if axis is None:
        new_dim = a.size
    elif isinstance(axis, Iterable):
        new_dim = 1
        shape = a.shape
        for ax in axis:
            new_dim *= shape[ax]
    else:
        new_dim = a.shape[axis]
    return a.__class__(magnitude, a.u**new_dim)


@_implements("nanprod", _FunctionType.function)
def _nanprod(
    a: smoot.Quantity, axis: int | tuple[int, ...] | None = None, *args, **kwargs
) -> smoot.Quantity:
    magnitude = np.nanprod(a.m, axis, *args, **kwargs)
    if axis is None:
        new_dim = a.size
    elif isinstance(axis, Iterable):
        new_dim = 1
        shape = a.shape
        for ax in axis:
            new_dim *= shape[ax]
    else:
        new_dim = a.shape[axis]
    return a.__class__(magnitude, a.u**new_dim)


@_implements("reshape", _FunctionType.function)
def _reshape(
    a: smoot.Quantity, /, shape: tuple[int, ...], *args, **kwargs
) -> smoot.Quantity:
    a._Quantity__inner.reshape(shape)
    return a


@_implements("searchsorted", _FunctionType.function)
def _searchsorted(
    a: smoot.Quantity, v: smoot.Quantity | int | float, *args, **kwargs
) -> np.intp | NDArray[np.intp]:
    a, v = _upcast(a, v)
    return np.searchsorted(a.m, v.m_as(a), *args, **kwargs)


@_implements("shape", _FunctionType.function)
def _shape(a: smoot.Quantity) -> tuple[int, ...]:
    return a._Quantity__inner.shape


@_implements("size", _FunctionType.function)
def _size(a: smoot.Quantity) -> int:
    return a._Quantity__inner.size


@_implements("ndim", _FunctionType.function)
def _ndim(a: smoot.Quantity) -> int:
    return a._Quantity__inner.ndim


@_implements("trapz", _FunctionType.function)
@_implements("trapezoid", _FunctionType.function)
def _trapezoid(y: smoot.Quantity, x=None, dx=1.0, **kwargs) -> smoot.Quantity:
    units = y.u
    if x is not None:
        if isinstance(x, smoot.Quantity):
            units *= x.u
            xm = x.m
        else:
            xm = x
        magnitude = np.trapezoid(y.m, xm, **kwargs)
    else:
        if isinstance(dx, smoot.Quantity):
            units *= dx.u
            dxm = dx.m
        else:
            dxm = dx
        magnitude = np.trapezoid(y.m, dx=dxm, **kwargs)

    return y.__class__(magnitude, units)


@_implements("unwrap", _FunctionType.function)
def _unwrap(p: smoot.Quantity, discont=None, axis=-1, **kwargs) -> smoot.Quantity:
    discont = np.pi if discont is None else discont
    magnitude = np.unwrap(p.m_as("radian"), discont, axis, **kwargs)
    return p.__class__(magnitude, "radian").to(p)


@_implements("where", _FunctionType.function)
def _where(
    condition, x: smoot.Quantity | ArrayLike, y: smoot.Quantity | ArrayLike
) -> smoot.Quantity:
    x, y = _upcast(x, y)
    magnitude = np.where(condition, x.m, y.m_as(x))
    return x.__class__(magnitude, x.u)


@_implements("zeros_like", _FunctionType.function)
def _zeros_like(a: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    magnitude = np.zeros_like(a.m, *args, **kwargs)
    return a.__class__(magnitude, a.u)
