"""Bespoke implementations of numpy functions using numpy's __array_ufunc__ and __array_function__ mechanisms."""

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable, Final, Iterable, Iterator, SupportsIndex
import warnings

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
def _upcast(
    x: smoot.Quantity | Any, y: smoot.Quantity
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
    ("absolute", _FunctionType.ufunc),
    ("fabs", _FunctionType.ufunc),
    ("rint", _FunctionType.ufunc),
    ("sign", _FunctionType.ufunc),
    ("floor", _FunctionType.ufunc),
    ("ceil", _FunctionType.ufunc),
    ("trunc", _FunctionType.ufunc),
    # array dispatch function
    ("sum", _FunctionType.function),
    ("amax", _FunctionType.function),
    ("amin", _FunctionType.function),
    ("around", _FunctionType.function),
    ("average", _FunctionType.function),
    ("broadcast_to", _FunctionType.function),
):
    _unary_unchanged_units(func_name, func_type)

# Functions that require dimensionless quantities as inputs.
for func_name, func_type, in_units, out_units in (
    # ufunc
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
    # array dispatch function
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


@_implements("sqrt", _FunctionType.ufunc)
def _sqrt(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.sqrt(x.m, *args, **kwargs),
        units=x.units**0.5,
    )


@_implements("square", _FunctionType.ufunc)
def _square(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.square(x.m, *args, **kwargs),
        units=x.units**2,
    )


@_implements("cbrt", _FunctionType.ufunc)
def _cbrt(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    return x.__class__(
        value=np.cbrt(x.m, *args, **kwargs),
        units=x.units ** (1 / 3),
    )


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
        a.to(units) if isinstance(a, smoot.Quantity)
        else units.__registry.Quantity(a)
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


@_implements("astype", _FunctionType.function)
def _astype(x, dtype, /, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("astype")


@_implements("atleast_1d", _FunctionType.function)
def _atleast_1d(
    *arys: smoot.Quantity | Any,
) -> smoot.Quantity | tuple[smoot.Quantity | NDArray[Any], ...]:
    res = tuple(
        val.__class__(
            value=np.atleast_1d(val.magnitude),
            units=val.units,
        ) if isinstance(val, smoot.Quantity)
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
        ) if isinstance(val, smoot.Quantity)
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
        ) if isinstance(val, smoot.Quantity)
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
        np.block([
            arr.m_as(units) if isinstance(arr, smoot.Quantity)
            else arr
            for arr in arrays
        ]),
        units
    )


@_implements("broadcast_arrays", _FunctionType.function)
def _broadcast_arrays(*args: smoot.Quantity | Any, **kwargs) -> tuple[smoot.Quantity | NDArray[Any], ...]:
    arrs = np.broadcast_arrays(*(
        arr.magnitude if isinstance(arr, smoot.Quantity)
        else arr
        for arr in args
    ))
    return tuple(
        v2.__class__(v1, v2.units) if isinstance(v2, smoot.Quantity)
        else v1
        for v1, v2 in zip(arrs, args)
    )


@_implements("clip", _FunctionType.function)
def _clip(a: smoot.Quantity, a_min: smoot.Quantity | float, a_max: smoot.Quantity | float, *args, **kwargs) -> smoot.Quantity:
    a_min, a_max = _upcast_many((a_min, a_max), a.units)
    return a.__class__(
        value=np.clip(a.m, a_min.m_as(a), a_max.m_as(a), *args, **kwargs),
        units=a.units,
    )


@_implements("column_stack", _FunctionType.function)
def _column_stack(tup) -> smoot.Quantity:
    raise NotImplementedError("column_stack")
@_implements("common_type", _FunctionType.function)
def _common_type(*arrays) -> smoot.Quantity:
    raise NotImplementedError("common_type")
@_implements("compress", _FunctionType.function)
def _compress(condition, a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("compress")
@_implements("concatenate", _FunctionType.function)
def _concatenate(arrays: tuple[smoot.Quantity | Any, ...], *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("concatenate")
@_implements("convolve", _FunctionType.function)
def _convolve(a, v, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("convolve")
@_implements("copy", _FunctionType.function)
def _copy(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("copy")


@_implements("copyto", _FunctionType.function)
def _copyto(dst: smoot.Quantity | Any, src: smoot.Quantity | Any, *args, **kwargs) -> None:
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
def _correlate(a, v, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("correlate")
@_implements("count_nonzero", _FunctionType.function)
def _count_nonzero(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("count_nonzero")
@_implements("cov", _FunctionType.function)
def _cov(m, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("cov")
@_implements("cross", _FunctionType.function)
def _cross(a, b, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("cross")
@_implements("cumprod", _FunctionType.function)
def _cumprod(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("cumprod")
@_implements("cumsum", _FunctionType.function)
def _cumsum(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("cumsum")
@_implements("cumulative_prod", _FunctionType.function)
def _cumulative_prod(x, /, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("cumulative_prod")
@_implements("cumulative_sum", _FunctionType.function)
def _cumulative_sum(x, /, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("cumulative_sum")
@_implements("delete", _FunctionType.function)
def _delete(arr, obj, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("delete")
@_implements("diag", _FunctionType.function)
def _diag(v, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("diag")
@_implements("diag_indices_from", _FunctionType.function)
def _diag_indices_from(arr) -> smoot.Quantity:
    raise NotImplementedError("diag_indices_from")
@_implements("diagflat", _FunctionType.function)
def _diagflat(v, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("diagflat")
@_implements("diagonal", _FunctionType.function)
def _diagonal(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("diagonal")
@_implements("diff", _FunctionType.function)
def _diff(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("diff")
@_implements("digitize", _FunctionType.function)
def _digitize(x, bins, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("digitize")
@_implements("dot", _FunctionType.function)
def _dot(x: smoot.Quantity | Any, y: smoot.Quantity | Any, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("dot")
@_implements("dsplit", _FunctionType.function)
def _dsplit(ary, indices_or_sections) -> smoot.Quantity:
    raise NotImplementedError("dsplit")
@_implements("dstack", _FunctionType.function)
def _dstack(tup) -> smoot.Quantity:
    raise NotImplementedError("dstack")
@_implements("ediff1d", _FunctionType.function)
def _ediff1d(ary, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("ediff1d")


@_implements("einsum", _FunctionType.function)
def _einsum(subscripts: str, *operands: smoot.Quantity | Any, **kwargs) -> smoot.Quantity:
    units = _get_first_unit(operands)
    result_units = units.parse("dimensionless")
    arrs = []
    for operand in _upcast_many(operands, units):
        result_units *= operand.units
        arrs.append(operand.magnitude)

    return units._Unit__registry.Quantity(
        value=np.einsum(subscripts, *arrs, **kwargs),
        units=units
    )


@_implements("einsum_path", _FunctionType.function)
def _einsum_path(*operands, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("einsum_path")
@_implements("empty_like", _FunctionType.function)
def __empty_like(x: smoot.Quantity, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("empty_like")
@_implements("expand_dims", _FunctionType.function)
def _expand_dims(a, axis) -> smoot.Quantity:
    raise NotImplementedError("expand_dims")
@_implements("extract", _FunctionType.function)
def _extract(condition, arr) -> smoot.Quantity:
    raise NotImplementedError("extract")
@_implements("fill_diagonal", _FunctionType.function)
def _fill_diagonal(a, val, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("fill_diagonal")
@_implements("fix", _FunctionType.function)
def _fix(x, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("fix")
@_implements("flatnonzero", _FunctionType.function)
def _flatnonzero(a) -> smoot.Quantity:
    raise NotImplementedError("flatnonzero")
@_implements("flip", _FunctionType.function)
def _flip(m, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("flip")
@_implements("fliplr", _FunctionType.function)
def _fliplr(m) -> smoot.Quantity:
    raise NotImplementedError("fliplr")
@_implements("flipud", _FunctionType.function)
def _flipud(m) -> smoot.Quantity:
    raise NotImplementedError("flipud")
@_implements("full_like", _FunctionType.function)
def _full_like(a, fill_value, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("full_like")
@_implements("geomspace", _FunctionType.function)
def _geomspace(start, stop, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("geomspace")
@_implements("gradient", _FunctionType.function)
def _gradient(f, *varargs, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("gradient")
@_implements("histogram", _FunctionType.function)
def _histogram(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("histogram")
@_implements("histogram2d", _FunctionType.function)
def _histogram2d(x, y, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("histogram2d")
@_implements("histogram_bin_edges", _FunctionType.function)
def _histogram_bin_edges(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("histogram_bin_edges")
@_implements("histogramdd", _FunctionType.function)
def _histogramdd(sample, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("histogramdd")
@_implements("hsplit", _FunctionType.function)
def _hsplit(ary, indices_or_sections) -> smoot.Quantity:
    raise NotImplementedError("hsplit")
@_implements("hstack", _FunctionType.function)
def _hstack(tup, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("hstack")
@_implements("i0", _FunctionType.function)
def _i0(x) -> smoot.Quantity:
    raise NotImplementedError("i0")
@_implements("imag", _FunctionType.function)
def _imag(val) -> smoot.Quantity:
    raise NotImplementedError("imag")
@_implements("in1d", _FunctionType.function)
def _in1d(ar1, ar2, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("in1d")
@_implements("insert", _FunctionType.function)
def _insert(arr, obj, values, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("insert")
@_implements("interp", _FunctionType.function)
def _interp(x, xp, fp, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("interp")
@_implements("intersect1d", _FunctionType.function)
def _intersect1d(ar1, ar2, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("intersect1d")
@_implements("isclose", _FunctionType.function)
def _isclose(a, b, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("isclose")
@_implements("iscomplex", _FunctionType.function)
def _iscomplex(x) -> smoot.Quantity:
    raise NotImplementedError("iscomplex")
@_implements("iscomplexobj", _FunctionType.function)
def _iscomplexobj(x) -> smoot.Quantity:
    raise NotImplementedError("iscomplexobj")
@_implements("isin", _FunctionType.function)
def _isin(element, test_elements, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("isin")
@_implements("isneginf", _FunctionType.function)
def _isneginf(x, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("isneginf")
@_implements("isposinf", _FunctionType.function)
def _isposinf(x, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("isposinf")
@_implements("isreal", _FunctionType.function)
def _isreal(x) -> smoot.Quantity:
    raise NotImplementedError("isreal")
@_implements("isrealobj", _FunctionType.function)
def _isrealobj(x) -> smoot.Quantity:
    raise NotImplementedError("isrealobj")
@_implements("ix_", _FunctionType.function)
def _ix_(*args) -> smoot.Quantity:
    raise NotImplementedError("ix_")
@_implements("kron", _FunctionType.function)
def _kron(a, b) -> smoot.Quantity:
    raise NotImplementedError("kron")
@_implements("linspace", _FunctionType.function)
def _linspace(start, stop, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("linspace")
@_implements("logspace", _FunctionType.function)
def _logspace(start, stop, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("logspace")
@_implements("matrix_transpose", _FunctionType.function)
def _matrix_transpose(x, /) -> smoot.Quantity:
    raise NotImplementedError("matrix_transpose")
@_implements("max", _FunctionType.function)
def __max(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("max")
@_implements("mean", _FunctionType.function)
def _mean(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("mean")
@_implements("median", _FunctionType.function)
def _median(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("median")
@_implements("meshgrid", _FunctionType.function)
def _meshgrid(*xi, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("meshgrid")
@_implements("min", _FunctionType.function)
def _min(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("min")
@_implements("moveaxis", _FunctionType.function)
def _moveaxis(a, source, destination) -> smoot.Quantity:
    raise NotImplementedError("moveaxis")
@_implements("nan_to_num", _FunctionType.function)
def _nan_to_num(x, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nan_to_num")
@_implements("nanargmax", _FunctionType.function)
def _nanargmax(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanargmax")
@_implements("nanargmin", _FunctionType.function)
def _nanargmin(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanargmin")
@_implements("nancumprod", _FunctionType.function)
def _nancumprod(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nancumprod")
@_implements("nancumsum", _FunctionType.function)
def _nancumsum(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nancumsum")
@_implements("nanmax", _FunctionType.function)
def _nanmax(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanmax")
@_implements("nanmean", _FunctionType.function)
def _nanmean(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanmean")
@_implements("nanmedian", _FunctionType.function)
def _nanmedian(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanmedian")
@_implements("nanmin", _FunctionType.function)
def _nanmin(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanmin")
@_implements("nanpercentile", _FunctionType.function)
def _nanpercentile(a, q, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanpercentile")
@_implements("nanprod", _FunctionType.function)
def _nanprod(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanprod")
@_implements("nanquantile", _FunctionType.function)
def _nanquantile(a, q, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanquantile")
@_implements("nanstd", _FunctionType.function)
def _nanstd(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanstd")
@_implements("nansum", _FunctionType.function)
def _nansum(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nansum")
@_implements("nanvar", _FunctionType.function)
def _nanvar(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("nanvar")
@_implements("ndim", _FunctionType.function)
def _ndim(a) -> smoot.Quantity:
    raise NotImplementedError("ndim")
@_implements("nonzero", _FunctionType.function)
def _nonzero(a) -> smoot.Quantity:
    raise NotImplementedError("nonzero")
@_implements("ones_like", _FunctionType.function)
def _ones_like(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("ones_like")
@_implements("outer", _FunctionType.function)
def _outer(a, b, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("outer")
@_implements("pad", _FunctionType.function)
def _pad(array, pad_width, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("pad")
@_implements("partition", _FunctionType.function)
def _partition(a, kth, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("partition")
@_implements("percentile", _FunctionType.function)
def _percentile(a, q, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("percentile")
@_implements("transpose", _FunctionType.function)
def _transpose(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("transpose")
@_implements("piecewise", _FunctionType.function)
def _piecewise(x, condlist, funclist, *args, **kw) -> smoot.Quantity:
    raise NotImplementedError("piecewise")
@_implements("place", _FunctionType.function)
def _place(arr, mask, vals) -> smoot.Quantity:
    raise NotImplementedError("place")
@_implements("poly", _FunctionType.function)
def _poly(seq_of_zeros) -> smoot.Quantity:
    raise NotImplementedError("poly")
@_implements("polyadd", _FunctionType.function)
def _polyadd(a1, a2) -> smoot.Quantity:
    raise NotImplementedError("polyadd")
@_implements("polyder", _FunctionType.function)
def _polyder(p, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("polyder")
@_implements("polydiv", _FunctionType.function)
def _polydiv(u, v) -> smoot.Quantity:
    raise NotImplementedError("polydiv")
@_implements("polyfit", _FunctionType.function)
def _polyfit(x, y, deg, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("polyfit")
@_implements("polyint", _FunctionType.function)
def _polyint(p, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("polyint")
@_implements("polymul", _FunctionType.function)
def _polymul(a1, a2) -> smoot.Quantity:
    raise NotImplementedError("polymul")
@_implements("polysub", _FunctionType.function)
def _polysub(a1, a2) -> smoot.Quantity:
    raise NotImplementedError("polysub")
@_implements("polyval", _FunctionType.function)
def _polyval(p, x) -> smoot.Quantity:
    raise NotImplementedError("polyval")
@_implements("prod", _FunctionType.function)
def _prod(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("prod")
@_implements("ptp", _FunctionType.function)
def _ptp(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("ptp")
@_implements("put", _FunctionType.function)
def _put(a, ind, v, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("put")
@_implements("put_along_axis", _FunctionType.function)
def _put_along_axis(arr, indices, values, axis) -> smoot.Quantity:
    raise NotImplementedError("put_along_axis")
@_implements("quantile", _FunctionType.function)
def _quantile(a, q, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("quantile")
@_implements("ravel", _FunctionType.function)
def _ravel(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("ravel")
@_implements("real", _FunctionType.function)
def _real(val) -> smoot.Quantity:
    raise NotImplementedError("real")
@_implements("real_if_close", _FunctionType.function)
def _real_if_close(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("real_if_close")
@_implements("repeat", _FunctionType.function)
def _repeat(a, repeats, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("repeat")
@_implements("reshape", _FunctionType.function)
def _reshape(a, /, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("reshape")
@_implements("resize", _FunctionType.function)
def _resize(a, new_shape) -> smoot.Quantity:
    raise NotImplementedError("resize")
@_implements("roll", _FunctionType.function)
def _roll(a, shift, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("roll")
@_implements("rollaxis", _FunctionType.function)
def _rollaxis(a, axis, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("rollaxis")
@_implements("roots", _FunctionType.function)
def _roots(p) -> smoot.Quantity:
    raise NotImplementedError("roots")
@_implements("rot90", _FunctionType.function)
def _rot90(m, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("rot90")
@_implements("round", _FunctionType.function)
def _round(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("round")
@_implements("save", _FunctionType.function)
def _save(file, arr, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("save")
@_implements("savetxt", _FunctionType.function)
def _savetxt(fname, X, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("savetxt")
@_implements("savez", _FunctionType.function)
def _savez(file, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("savez")
@_implements("savez_compressed", _FunctionType.function)
def _savez_compressed(file, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("savez_compressed")
@_implements("searchsorted", _FunctionType.function)
def _searchsorted(a, v, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("searchsorted")
@_implements("select", _FunctionType.function)
def _select(condlist, choicelist, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("select")
@_implements("setdiff1d", _FunctionType.function)
def _setdiff1d(ar1, ar2, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("setdiff1d")
@_implements("setxor1d", _FunctionType.function)
def _setxor1d(ar1, ar2, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("setxor1d")
@_implements("shape", _FunctionType.function)
def _shape(a) -> smoot.Quantity:
    raise NotImplementedError("shape")
@_implements("sinc", _FunctionType.function)
def _sinc(x) -> smoot.Quantity:
    raise NotImplementedError("sinc")
@_implements("size", _FunctionType.function)
def _size(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("size")
@_implements("sort", _FunctionType.function)
def _sort(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("sort")
@_implements("sort_complex", _FunctionType.function)
def _sort_complex(a) -> smoot.Quantity:
    raise NotImplementedError("sort_complex")
@_implements("split", _FunctionType.function)
def _split(ary, indices_or_sections, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("split")
@_implements("squeeze", _FunctionType.function)
def _squeeze(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("squeeze")
@_implements("stack", _FunctionType.function)
def _stack(arrays, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("stack")
@_implements("std", _FunctionType.function)
def _std(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("std")
@_implements("swapaxes", _FunctionType.function)
def _swapaxes(a, axis1, axis2) -> smoot.Quantity:
    raise NotImplementedError("swapaxes")
@_implements("take", _FunctionType.function)
def _take(a, indices, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("take")
@_implements("take_along_axis", _FunctionType.function)
def _take_along_axis(arr, indices, axis) -> smoot.Quantity:
    raise NotImplementedError("take_along_axis")
@_implements("tensordot", _FunctionType.function)
def _tensordot(a, b, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("tensordot")
@_implements("tile", _FunctionType.function)
def _tile(A, reps) -> smoot.Quantity:
    raise NotImplementedError("tile")
@_implements("trace", _FunctionType.function)
def _trace(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("trace")
@_implements("trapezoid", _FunctionType.function)
def _trapezoid(y, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("trapezoid")
@_implements("tril", _FunctionType.function)
def _tril(m, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("tril")
@_implements("tril_indices_from", _FunctionType.function)
def _tril_indices_from(arr, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("tril_indices_from")
@_implements("trim_zeros", _FunctionType.function)
def _trim_zeros(filt, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("trim_zeros")
@_implements("triu", _FunctionType.function)
def _triu(m, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("triu")
@_implements("triu_indices_from", _FunctionType.function)
def _triu_indices_from(arr, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("triu_indices_from")
@_implements("union1d", _FunctionType.function)
def _union1d(ar1, ar2) -> smoot.Quantity:
    raise NotImplementedError("union1d")
@_implements("unique", _FunctionType.function)
def _unique(ar, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("unique")
@_implements("unique_all", _FunctionType.function)
def _unique_all(x) -> smoot.Quantity:
    raise NotImplementedError("unique_all")
@_implements("unique_counts", _FunctionType.function)
def _unique_counts(x) -> smoot.Quantity:
    raise NotImplementedError("unique_counts")
@_implements("unique_inverse", _FunctionType.function)
def _unique_inverse(x) -> smoot.Quantity:
    raise NotImplementedError("unique_inverse")
@_implements("unique_values", _FunctionType.function)
def _unique_values(x) -> smoot.Quantity:
    raise NotImplementedError("unique_values")
@_implements("unstack", _FunctionType.function)
def _unstack(x, /, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("unstack")
@_implements("unwrap", _FunctionType.function)
def _unwrap(p, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("unwrap")
@_implements("vander", _FunctionType.function)
def _vander(x, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("vander")
@_implements("var", _FunctionType.function)
def _var(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("var")
@_implements("vsplit", _FunctionType.function)
def _vsplit(ary, indices_or_sections) -> smoot.Quantity:
    raise NotImplementedError("vsplit")
@_implements("vstack", _FunctionType.function)
def _vstack(tup, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("vstack")
@_implements("where", _FunctionType.function)
def _where(condition, x: smoot.Quantity, y: smoot.Quantity) -> smoot.Quantity:
    raise NotImplementedError("where")
@_implements("zeros_like", _FunctionType.function)
def _zeros_like(a, *args, **kwargs) -> smoot.Quantity:
    raise NotImplementedError("zeros_like")
