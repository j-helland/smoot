from __future__ import annotations

import inspect
import math
import operator
import pickle
from typing import Any, Callable, Iterable
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest

from smoot import UnitRegistry


units = UnitRegistry()
Q = units.Quantity


def test_all_base_unit_strings() -> None:
    """Make sure smoot can parse all of its unit strings."""
    # Do this in a loop because there are too many unit combinations to parameterize.
    for unit in dir(units):
        Q(1, unit)


@pytest.mark.parametrize(
    argnames=("quantity", "expected"),
    argvalues=(
        (Q(1), None),
        (Q("1 radian"), None),
        (Q("1 steradian"), None),
        (Q("1 kiloradian"), None),
        (Q("1 meter"), {"[length]": 1.0}),
        (Q([1, 2, 3], "meter"), {"[length]": 1.0}),
        (Q("1 newton"), {"[length]": 1.0, "[mass]": 1.0, "[time]": -2.0}),
    ),
)
def test_dimensionality(
    quantity: Q,
    expected: dict[str, float] | None,
) -> None:
    assert quantity.dimensionality == expected


def test_len() -> None:
    assert len(Q([1, 2, 3])) == 3
    with pytest.raises(TypeError):
        len(Q(1))


@pytest.mark.parametrize(
    argnames=("quantity", "expected"),
    argvalues=(
        (Q("1 newton"), "1 newton"),
        (Q("1 km"), "1 kilometer"),
        (Q("meter") / Q("second"), "1 meter / second"),
        (Q("meter ** 2"), "1 meter ** 2"),
        (Q("1 / meter"), "1 / meter"),
        (Q("1.0 meter"), "1 meter"),
        (Q("1.1 meter"), "1.1 meter"),
    ),
)
def test_str(
    quantity: Q,
    expected: str,
) -> None:
    assert str(quantity) == expected


@pytest.mark.parametrize(
    argnames=("value", "unit", "expected"),
    argvalues=(
        (Q("1 km"), "meter", Q("1000 meter")),
        (Q("1000 meter"), "km", Q("1 km")),
        (Q(2), "radian", Q("2 radian")),
        (Q("2 radian"), "dimensionless", Q("2")),
    ),
)
def test_conversion(
    value: Q,
    unit: str,
    expected: Q,
) -> None:
    """Unit conversions produce the expected value."""
    assert value.to(unit) == expected

    # inplace
    value.ito(unit)
    assert value == expected


@pytest.mark.parametrize(
    argnames=("value", "expected"),
    argvalues=(
        (Q("1 km"), Q("1000 m")),
        (Q("1 km / hour"), Q(1000 / 60 / 60, "m / s")),
        (Q("1 km * hour"), Q(1000 * 60 * 60, "m * s")),
        (Q("1 joule / newton"), Q("1 meter")),
        (Q("1 Hz"), Q("1 / second")),
    ),
)
def test_to_root_units(value: Q, expected: Q) -> None:
    assert value.to_root_units() == expected

    # in-place
    value.ito_root_units()
    assert value == expected


def test_is_compatible_with() -> None:
    assert Q("1 meter").is_compatible_with("km")
    assert Q("1 meter").is_compatible_with(units.km)
    assert Q("1 meter").is_compatible_with(Q("1 km"))
    assert not Q("1 meter").is_compatible_with(units.gram)


def test_eq() -> None:
    """`==` operator works."""
    assert Q(1) == Q(1)
    assert Q(1) == 1
    assert 1 == Q(1)

    assert Q(2) != Q(1)
    assert Q("1 meter") != 1
    assert Q(math.nan) != Q(math.nan)


# fmt: off
@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        #### +
        (Q(1), operator.add, Q(1), Q(2)),
        (Q(1), operator.add, 1, Q(2)),
        (1, operator.add, Q(1), Q(2)),
        (Q(1, "meter"), operator.add, Q(1, "meter"), Q(2, "meter")),
        (Q(1, "meter"), operator.add, Q(1, "km"), Q(1001, "meter")),
        (Q(1, "km"), operator.add, Q(1, "meter"), Q(1001, "meter")),
        #### -
        (Q(2), operator.sub, Q(1), Q(1)),
        (Q(2), operator.sub, 1, Q(1)),
        (2, operator.sub, Q(1), Q(1)),
        (Q(2, "meter"), operator.sub, Q(1, "meter"), Q(1, "meter")),
        (Q(2, "meter"), operator.sub, Q(1, "km"), Q(-998, "meter")),
        # integer precision
        (Q(2, "km"), operator.sub, Q(1, "meter"), Q(1.999, "km")),
        #### *
        (Q(2), operator.mul, Q(2), Q(4)),
        (Q(2), operator.mul, 2, Q(4)),
        (2, operator.mul, Q(2), Q(4)),
        (Q(2, "meter"), operator.mul, Q(2, "meter"), Q(4, "meter ** 2")),
        (Q(2, "meter"), operator.mul, Q(2, "km"), Q(4, "km * meter")),
        (Q(2, "km"), operator.mul, Q(2, "meter"), Q(4, "km * meter")),
        (Q(2, "meter"), operator.mul, 2, Q(4, "meter")),
        #### **
        (Q(2), operator.pow, Q(3), Q(8)),
        (2, operator.pow, Q(3), Q(8)),
        (Q(2), operator.pow, 3, Q(8)),
        (Q(2, "meter"), operator.pow, Q(3, "meter"), ValueError),
        (Q(2, "meter"), operator.pow, Q(3), Q(8, "meter ** 3")),
        (Q(2, "meter"), operator.pow, 3, Q(8, "meter ** 3")),
        #### /
        (Q(4), operator.truediv, Q(2), Q(2)),
        (Q(4), operator.truediv, 2, Q(2)),
        (4, operator.truediv, Q(2), Q(2)),
        (Q(4, "meter"), operator.truediv, Q(2, "meter"), Q(2)),
        (Q(4, "meter"), operator.truediv, 2, Q(2, "meter")),
        (4, operator.truediv, Q(2, "meter"), Q(2, "1 / meter")),
        #### //
        (Q(4), operator.floordiv, Q(2), Q(2)),
        (Q(4), operator.floordiv, 2, Q(2)),
        (4, operator.floordiv, Q(2), Q(2)),
        (Q(4, "meter"), operator.floordiv, Q(2, "meter"), Q(2)),
        (Q(4, "meter"), operator.floordiv, 2, Q(2, "meter")),
        (4, operator.floordiv, Q(2, "meter"), Q(2, "1 / meter")),
        #### %
        (Q(4), operator.mod, Q(2), Q(0)),
        (4, operator.mod, Q(2), Q(0)),
        (Q(4), operator.mod, 2, Q(0)),
        (Q(4, "meter"), operator.mod, Q(2, "meter"), Q(0, "meter")),
        (Q(4, "meter"), operator.mod, Q(2, "km"), Q(4, "meter")),
        (Q(1, "km"), operator.mod, Q(1, "meter"), Q(1 % 1e-3, "km")),
    ),
)
# fmt: on
def test_binary_operators(
    x: Q | int,
    op: Callable[[Q | int, Q | int], Q],
    y: Q | int,
    expected: Q | type[Exception],
) -> None:
    """Binary operators applied to quantities produce the expected values."""
    if inspect.isclass(expected) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = op(x, y)
    else:
        assert op(x, y) == expected


@pytest.mark.parametrize(
    argnames=("x", "op", "expected"),
    argvalues=(
        (Q(1), operator.neg, Q(-1)),
        (Q(-1), operator.neg, Q(1)),
        (Q(-1), operator.abs, Q(1)),
        (Q(1.5), math.floor, Q(1.0)),
        (Q(1.5), math.ceil, Q(2.0)),
        (Q(1.5), round, Q(2.0)),
        (Q(2), math.sqrt, math.sqrt(2)),
        (Q(math.nan), math.isnan, True),
        (Q(1.0), int, 1),
    ),
)
def test_unary_operators(
    x: Q,
    op: Callable[[Q], Q],
    expected: Any,
) -> None:
    """Unary operators applied to quantities produce the expected values."""
    assert op(x) == expected


@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        (Q(1), operator.iadd, Q(1), Q(2)),
        (Q(1), operator.iadd, 1, Q(2)),
        (Q(1), operator.isub, Q(1), Q(0)),
        (Q(1), operator.isub, 1, Q(0)),
        (Q(2), operator.imul, Q(2), Q(4)),
        (Q(2), operator.imul, 2, Q(4)),
        (Q(2), operator.ipow, Q(3), Q(8)),
        (Q(2), operator.ipow, 3, Q(8)),
        (Q(4), operator.itruediv, Q(2), Q(2)),
        (Q(4), operator.itruediv, 2, Q(2)),
        (Q(4), operator.imod, 2, Q(0)),
    ),
)
def test_binary_inplace_operators(
    x: Q,
    op: Callable[[Q, Q | int], Q],
    y: Q | int,
    expected: Q,
) -> None:
    """In-place binary operators produce expected values."""
    op(x, y)
    assert x == expected


# fmt: off
@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        (Q([1, 2, 3]), operator.eq, Q([1, 0, 3]), np.array([True, False, True])),
        ([1, 2, 3], operator.eq, Q([1, 0, 3]), np.array([True, False, True])),
        (Q([1, 2, 3]), operator.eq, [1, 0, 3], np.array([True, False, True])),
        # compatible units
        (Q([1000, 2000, 3000], "meter"), operator.eq, Q([1, 2, 3], "km"), np.array([True, True, True])),
        # incompatible units
        (Q([1, 2, 3], "meter"), operator.eq, Q([1, 2, 3], "gram"), np.array([False, False, False])),
        (Q([1, 2, 3]), operator.add, Q([3, 2, 1]), Q([4, 4, 4])),
        ([1, 2, 3], operator.add, Q([3, 2, 1]), Q([4, 4, 4])),
        (Q([1, 2, 3]), operator.add, [3, 2, 1], Q([4, 4, 4])),
        # scalar addition
        (Q([1, 2, 3]), operator.add, 1, Q([2, 3, 4])),
        (1, operator.add, Q([1, 2, 3]), Q([2, 3, 4])),
        # compatible units
        (Q([1000, 2000, 3000], "meter"), operator.add, Q([3, 2, 1], "km"), Q([4000, 4000, 4000], "meter")),
        (Q([1, 2, 3], "km"), operator.add, Q([3000, 2000, 1000], "meter"), Q([4, 4, 4], "km")),
        (Q([1, 2, 3]), operator.sub, Q([3, 2, 1]), Q([-2, 0, 2])),
        ([1, 2, 3], operator.sub, Q([3, 2, 1]), Q([-2, 0, 2])),
        (Q([1, 2, 3]), operator.sub, [3, 2, 1], Q([-2, 0, 2])),
        (Q([1, 2, 3]), operator.mul, Q([3, 2, 1]), Q([3, 4, 3])),
        ([1, 2, 3], operator.mul, Q([3, 2, 1]), Q([3, 4, 3])),
        (Q([1, 2, 3]), operator.mul, [3, 2, 1], Q([3, 4, 3])),
        (Q([[1], [2], [3]]), operator.matmul, Q([[3, 2, 1]]), Q([[3, 2, 1], [6, 4, 2], [9, 6, 3]])),
        ([[1], [2], [3]], operator.matmul, Q([[3, 2, 1]]), Q([[3, 2, 1], [6, 4, 2], [9, 6, 3]])),
        (Q([[1], [2], [3]]), operator.matmul, [[3, 2, 1]], Q([[3, 2, 1], [6, 4, 2], [9, 6, 3]])),
    ),
)
# fmt: on
def test_binary_array_operators(
    x: Q | NDArray[np.int64],
    op: Callable[[Q | NDArray[np.int64], Q | NDArray[np.int64]], Q],
    y: Q | NDArray[np.int64],
    expected: Q,
) -> None:
    """Binary array operators produce expected values."""
    assert (op(x, y) == expected).all()


def test_matmul_produces_scalar() -> None:
    """vector/vector matmul should produce a scalar quantity"""
    assert (Q([1, 2, 3]) @ Q([3, 2, 1])) == Q(10)


# fmt: off
@pytest.mark.parametrize(
    argnames=("value", "ufunc", "expected"),
    argvalues=(
        #### ufuncs
        (Q([1, 2, 3], "meter"), np.negative, Q([-1, -2, -3], "meter")),
        (Q([1, 2, 3], "meter"), np.positive, Q([1, 2, 3], "meter")),
        (Q([-1, -2, -3], "meter"), np.absolute, Q([1, 2, 3], "meter")),
        (Q([-1, -2, -3], "meter"), np.fabs, Q([1, 2, 3], "meter")),
        (Q([1.1, 1.5], "meter"), np.rint, Q([1, 2], "meter")),
        (Q([-2, 2], "meter"), np.sign, Q([-1, 1], "meter")),
        (Q([1, 2, 3]), np.exp, Q(np.exp([1, 2, 3]))),
        (Q([1, 2, 3]), np.exp2, Q(np.exp2([1, 2, 3]))),
        (Q([1, 2, 3]), np.log, Q(np.log([1, 2, 3]))),
        (Q([1, 2, 3]), np.log2, Q(np.log2([1, 2, 3]))),
        (Q([1, 2, 3]), np.log10, Q(np.log10([1, 2, 3]))),
        (Q([1, 2, 3]), np.expm1, Q(np.expm1([1, 2, 3]))),
        (Q([1, 2, 3]), np.log1p, Q(np.log1p([1, 2, 3]))),
        (Q([1, 2, 3], "meter"), np.square, Q(np.square([1, 2, 3]), "meter ** 2")),
        (Q([1, 2, 3], "meter ** 2"), np.sqrt, Q(np.sqrt([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.reciprocal, Q([1, 1 / 2, 1 / 3], 1 / units.meter)),
        (Q([1, 2, 3], "radian"), np.sin, Q(np.sin([1, 2, 3]))),
        (Q([1, 2, 3], "radian"), np.cos, Q(np.cos([1, 2, 3]))),
        (Q([1, 2, 3], "radian"), np.tan, Q(np.tan([1, 2, 3]))),
        (Q([0.1, 0.2, 0.3], "radian"), np.arcsin, Q(np.arcsin([0.1, 0.2, 0.3]))),
        (Q([0.1, 0.2, 0.3], "radian"), np.arccos, Q(np.arccos([0.1, 0.2, 0.3]))),
        (Q([0.1, 0.2, 0.3], "radian"), np.arctan, Q(np.arctan([0.1, 0.2, 0.3]))),
        (Q([1, 2, 3], "radian"), np.sinh, Q(np.sinh([1, 2, 3]))),
        (Q([1, 2, 3], "radian"), np.cosh, Q(np.cosh([1, 2, 3]))),
        (Q([1, 2, 3], "radian"), np.tanh, Q(np.tanh([1, 2, 3]))),
        (Q([0.1, 0.2, 0.3], "radian"), np.arcsinh, Q(np.arcsinh([0.1, 0.2, 0.3]))),
        (Q([1, 2, 3], "radian"), np.arccosh, Q(np.arccosh([1, 2, 3]))),
        (Q([0.1, 0.2, 0.3], "radian"), np.arctanh, Q(np.arctanh([0.1, 0.2, 0.3]))),
        (Q([1, 2, 3], "radian"), np.degrees, Q(np.degrees([1, 2, 3]), "degree")),
        (Q([1, 2, 3], "degree"), np.radians, Q(np.radians([1, 2, 3]), "radian")),
        (Q([1, 2, 3], "degree"), np.deg2rad, Q(np.deg2rad([1, 2, 3]), "radian")),
        (Q([1, 2, 3], "radian"), np.rad2deg, Q(np.rad2deg([1, 2, 3]), "degree")),
        (Q([1, 2, 3], "meter"), np.isfinite, np.array([True, True, True])),
        (Q([1, 2, 3], "meter"), np.isinf, np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.isnan, np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.floor, Q(np.floor([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.ceil, Q(np.ceil([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.trunc, Q(np.trunc([1, 2, 3]), "meter")),
        #### higher order array functions
        (Q([1, 2, 3], "meter"), np.all, True),
        (Q([0, 1, 0], "meter"), np.any, True),
        (Q([1, 2, 3], "meter"), np.amax, Q(3, "meter")),
        (Q([1, 2, 3], "meter"), np.amin, Q(1, "meter")),
        (Q([1, 2, 3], "meter"), np.max, Q(3, "meter")),
        (Q([1, 2, 3], "meter"), np.min, Q(1, "meter")),
        (Q([1, 2, 3], "meter"), np.nanmax, Q(3, "meter")),
        (Q([1, 2, 3], "meter"), np.nanmin, Q(1, "meter")),
        (Q([1, 2, 3], "meter"), np.argmax, 2),
        (Q([1, 2, 3], "meter"), np.argmin, 0),
        (Q([1, 2, 3], "meter"), np.nanargmax, 2),
        (Q([1, 2, 3], "meter"), np.nanargmin, 0),
        (Q([3, 2, 1], "meter"), np.argsort, np.array([2, 1, 0])),
        (Q([0, 1, 2], "meter"), np.argwhere, np.array([[1], [2]])),
        (Q([0.1, 1.1, 2.5], "meter"), np.around, Q(np.around([0.1, 1.1, 2.5]), "meter")),
        (Q([1, 2, 3], "meter"), np.array2string, "[1. 2. 3.]"),
        (Q([1, 2, 3], "meter"), np.array_repr, "array([1., 2., 3.])"),
        (Q([1, 2, 3], "meter"), np.array_str, "[1. 2. 3.]"),
        (Q(1, "meter"), np.atleast_1d, Q([1], "meter")),
        (Q(1, "meter"), np.atleast_2d, Q([[1]], "meter")),
        (Q(1, "meter"), np.atleast_3d, Q([[[1]]], "meter")),
        (Q([1, 2, 3], "meter"), np.average, Q(np.average([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.mean, Q(np.mean([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.nanmean, Q(np.mean([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.median, Q(np.median([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.nanmedian, Q(np.median([1, 2, 3]), "meter")),
        ([Q([1, 2]), np.array([3, 4])], np.block, Q([1, 2, 3, 4])),
        (Q([1, 2, 3], "meter"), np.sum, Q(6, "meter")),
        (Q([1, 2, 3], "meter"), np.nansum, Q(6, "meter")),
        (Q([0, 0, 1, 2, 3], "meter"), np.count_nonzero, 3),
        (Q([1, 2, 3]), np.cumprod, Q([1, 2, 6])),
        (Q([1, 2, 3]), np.nancumprod, Q([1, 2, 6])),
        (Q([1, 2, 3]), np.nancumprod, Q([1, 2, 6])),
        (Q([1, 2, 3], "meter"), np.cumsum, Q([1, 3, 6], "meter")),
        (Q([1, 2, 3], "meter"), np.nancumsum, Q([1, 3, 6], "meter")),
        (Q([[1, 0], [0, 1]], "meter"), np.diagonal, Q([1, 1], "meter")),
        (Q([1, 2, 3], "meter"), np.diff, Q([1, 1], "meter")),
        (Q([1, 2, 3], "meter"), np.ediff1d, Q([1, 1], "meter")),
        (Q([1, 2, 3], "meter"), np.fix, Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), np.gradient, Q([1, 1, 1], "meter")),
        (Q([1, 2, math.nan], "meter"), np.nan_to_num, Q([1, 2, 0], "meter")),
        (Q([[1], [2], [3]], "meter"), np.transpose, Q([[1, 2, 3]], "meter")),
        (Q([1, 2, 3], "meter"), np.ptp, Q(2, "meter")),
        (Q([1, 2, 3], "meter"), np.std, Q(np.std([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.var, Q(np.var([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.nanvar, Q(np.var([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.nanstd, Q(np.nanstd([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), np.shape, (3,)),
        (Q([1, 2, 3], "meter"), np.size, 3),
        (Q([1, 2, 3], "meter"), np.ndim, 1),
        (Q([3, 2, 1], "meter"), np.sort, Q([1, 2, 3], "meter")),
        (Q([0, 0, 1, 0, 2, 0, 3, 0, 0], "meter"), np.trim_zeros, Q([1, 0, 2, 0, 3], "meter")),
        (Q([1, 2, 3], "radian"), np.unwrap, Q(np.unwrap([1, 2, 3]), "radian")),
        (Q([1, 2, 3], "radian"), np.zeros_like, Q([0, 0, 0], "radian")),
        (Q([1, 2, 3], "radian"), np.ones_like, Q([1, 1, 1], "radian")),
        (Q([1, 2, 3], "radian"), np.iscomplex, False),
        (Q([1, 2, 3], "radian"), np.isreal, True),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.ravel, Q([1, 2, 3, 4, 5, 6], "meter")),
        (Q(np.array([1]), "meter"), np.squeeze, Q(np.array(1), "meter")),
        (Q([1, 2, 3], "meter"), np.round, Q([1, 2, 3], "meter")),
        (Q([[1, 2, 3]], "meter"), np.rot90, Q([[3], [2], [1]], "meter")),
        (Q([1, 2, 3], units.meter), np.copy, Q([1, 2, 3], units.meter)),
        ([[1], Q([2], units.meter), Q([3], units.km)], np.concatenate, Q([1, 2, 3000], units.meter)),
        ([[1], Q([2], units.meter), Q([3], units.km)], np.vstack, Q([[1], [2], [3000]], units.meter)),
        ([[1], Q([2], units.meter), Q([3], units.km)], np.hstack, Q([1, 2, 3000], units.meter)),
        ([[1], Q([2], units.meter), Q([3], units.km)], np.stack, Q([[1], [2], [3000]], units.meter)),
        ([[1], Q([2], units.meter), Q([3], units.km)], np.dstack, Q([[[1, 2, 3000]]], units.meter)),
        ([[1], Q([2], units.meter), Q([3], units.km)], np.column_stack, Q([[1, 2, 3000]], units.meter)),
    ),
)
# fmt: on
def test_unary_array_funcs(value: Q, ufunc: np.ufunc, expected: Any) -> None:
    """Arbitrary numpy ufuncs can be invoked with expected results."""
    actual = ufunc(value)
    eq = actual == expected
    is_eq = eq.all() if isinstance(eq, np.ndarray) else eq
    assert is_eq, f"{actual} != {expected}"


# fmt: off
@pytest.mark.parametrize(
    argnames=("value1", "func", "value2", "expected"),
    argvalues=(
        #### ufuncs
        (Q([1, 2, 3], "meter"), np.add, Q([3, 2, 1], "meter"), Q([4, 4, 4], "meter")),
        ([1, 2, 3], np.add, Q([3, 2, 1]), Q([4, 4, 4])),
        (Q([1, 2, 3]), np.add, [3, 2, 1], Q([4, 4, 4])),
        (Q([1, 2, 3], "meter"), np.subtract, Q([3, 2, 1], "meter"), Q([-2, 0, 2], "meter")),
        ([1, 2, 3], np.subtract, Q([3, 2, 1]), Q([-2, 0, 2])),
        (Q([1, 2, 3]), np.subtract, [3, 2, 1], Q([-2, 0, 2])),
        (Q([1, 2, 3], "meter"), np.multiply, Q([3, 2, 1], "meter"), Q([3, 4, 3], "meter ** 2")),
        ([1, 2, 3], np.multiply, Q([3, 2, 1]), Q([3, 4, 3])),
        (Q([1, 2, 3]), np.multiply, [3, 2, 1], Q([3, 4, 3])),
        (Q([1, 2, 3], "meter"), np.matmul, Q([3, 2, 1], "meter"), Q(10, "meter ** 2")),
        ([1, 2, 3], np.matmul, Q([3, 2, 1]), Q(10)),
        (Q([1, 2, 3]), np.matmul, [3, 2, 1], Q(10)),
        (Q([1, 2, 3], "meter"), np.divide, Q([3, 2, 1], "meter"), Q([1 / 3, 1, 3])),
        ([1, 2, 3], np.divide, Q([3, 2, 1]), Q([1 / 3, 1, 3])),
        (Q([1, 2, 3]), np.divide, [3, 2, 1], Q([1 / 3, 1, 3])),
        (Q([1, 2, 3], "meter"), np.true_divide, Q([3, 2, 1], "meter"), Q([1 / 3, 1, 3])),
        ([1, 2, 3], np.true_divide, Q([3, 2, 1]), Q([1 / 3, 1, 3])),
        (Q([1, 2, 3]), np.true_divide, [3, 2, 1], Q([1 / 3, 1, 3])),
        (Q([1, 2, 3], "meter"), np.floor_divide, Q([3, 2, 1], "meter"), Q([1 / 3, 1, 3])),
        ([1, 2, 3], np.floor_divide, Q([3, 2, 1]), Q([1 / 3, 1, 3])),
        (Q([1, 2, 3]), np.floor_divide, [3, 2, 1], Q([1 / 3, 1, 3])),
        (Q([1, 2, 3]), np.logaddexp, Q([3, 2, 1]), Q(np.logaddexp([1, 2, 3], [3, 2, 1]))),
        (Q([1, 2, 3]), np.logaddexp2, Q([3, 2, 1]), Q(np.logaddexp2([1, 2, 3], [3, 2, 1]))),
        (Q([1, 2, 3]), np.power, Q([3, 2, 1]), Q(np.power([1, 2, 3], [3, 2, 1]))),
        (Q([1, 2, 3]), np.float_power, Q([3, 2, 1]), Q(np.float_power([1, 2, 3], [3, 2, 1]))),
        (Q([1, 2, 3], "meter"), np.remainder, Q([3, 2, 1], "meter"), Q([1, 0, 0], "meter")),
        (Q([1, 2, 3], "meter"), np.mod, Q([3, 2, 1], "meter"), Q([1, 0, 0], "meter")),
        (Q([1, 2, 3], "meter"), np.fmod, Q([3, 2, 1], "meter"), Q([1, 0, 0], "meter")),
        (Q([0.1, 0.2, 0.3], "radian"), np.arctan2, Q([0.3, 0.2, 0.1], "radian"), Q(np.arctan2([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]))),
        (Q([1, 2, 3], "meter"), np.hypot, Q([3, 2, 1], "meter"), Q(np.hypot([1, 2, 3], [3, 2, 1]), "meter")),
        (Q([1, 2, 3], "meter"), np.greater, Q([3, 2, 1], "km"), np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.greater_equal, Q([3, 2, 1], "km"), np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.less, Q([3, 2, 1], "km"), np.array([True, True, True])),
        (Q([1, 2, 3], "meter"), np.less_equal, Q([3, 2, 1], "km"), np.array([True, True, True])),
        (Q([1, 2, 3], "meter"), np.not_equal, Q([3, 2, 1], "km"), np.array([True, True, True])),
        (Q([1, 2, 3], "meter"), np.equal, Q([3, 2, 1], "km"), np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.maximum, Q([3, 2, 1], "km"), Q([3e3, 2e3, 1e3], "meter")),
        ([1, 2, 3], np.maximum, Q([3, 2, 1]), Q([3, 2, 3])),
        (Q([1, 2, 3]), np.maximum, [3, 2, 1], Q([3, 2, 3])),
        (Q([1, 2, 3], "meter"), np.minimum, Q([3, 2, 1], "km"), Q([1, 2, 3], "meter")),
        ([1, 2, 3], np.minimum, Q([3, 2, 1]), Q([1, 2, 1])),
        (Q([1, 2, 3]), np.minimum, [3, 2, 1], Q([1, 2, 1])),
        (Q([1, 2, 3], "meter"), np.copysign, Q([-3, -2, -1], "km"), Q([-1, -2, -3], "meter")),
        (Q([1, 2, 3], "meter"), np.nextafter, Q([3, 2, 1], "km"), Q([1, 2, 3], "meter")),
        #### higher order array functions
        (Q([1, 2, 3], "m"), np.allclose, Q([1, 2, 3], "km"), False),
        ([1, 2, 3], np.append, [Q(4)], Q([1, 2, 3, 4])),
        (Q([1, 2, 3]), np.append, Q(4), Q([1, 2, 3, 4])),
        (Q([1, 2, 3]), np.append, [Q(4)], Q([1, 2, 3, 4])),
        (Q([[1, 2, 3]]), np.append, [Q(4), Q(5), Q(6)], Q([1, 2, 3, 4, 5, 6])),
        (Q([1, 2, 3], "meter"), np.append, Q(4, "km"), Q([1, 2, 3, 4000], "meter")),
        (Q([1, 2, 3], "meter"), np.array_equal, Q([1, 2, 3], "km"), np.array([False, False, False])),
        (Q([[1], [2], [3]], "meter"), np.array_equiv, Q([1, 2, 3], "km"), np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.correlate, Q([3, 2, 1], "gram"), Q([10], "meter * gram")),
        (Q([1, 2, 3], "meter"), np.cross, Q([3, 2, 1], "gram"), Q([-4, 8, -4], "meter * gram")),
        (Q([1, 2, 3], "meter"), np.dot, Q([3, 2, 1], "gram"), Q(10, "meter * gram")),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.delete, [1, 0], Q([3, 4, 5, 6], "meter")),
        (Q([1, 2, 3], "meter"), np.expand_dims, 0, Q([[1, 2, 3]], "meter")),
        (Q([1, 2, 3], "meter"), np.flip, 0, Q([3, 2, 1], "meter")),
        (Q([1, 2, 3], "meter"), np.full_like, 1, Q([1, 1, 1], "meter")),
        (Q([1, 2, 3], "meter"), np.isclose, Q([3, 2, 1], "meter"), np.array([False, True, False])),
        (Q([1, 2, 3], "meter"), np.isclose, Q([3, 2, 1], "km"), np.array([False, False, False])),
        (Q([1, 2, 3], "meter"), np.intersect1d, Q([3e-3, 3, 2, 1], "km"), Q([3], "meter")),
        (Q(1, "meter"), np.isin, Q([1e-3], "km"), True),
        (Q(1, "meter"), np.linspace, Q(3e-3, "km"), Q(np.linspace(1, 3), "meter")),
        (Q([1, 2, 3], "meter"), np.percentile, 0.5, Q(1.01, "meter")),
        (Q([1, 2, 3], "meter"), np.quantile, 0.5, Q(2.0, "meter")),
        (Q([1, 2, 3], "meter"), np.nanpercentile, 0.5, Q(1.01, "meter")),
        (Q([1, 2, 3], "meter"), np.nanquantile, 0.5, Q(2.0, "meter")),
        (Q([1, 2, 3], "meter"), np.tile, 2, Q([1, 2, 3, 1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), np.rollaxis, 0, Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), np.roll, 0, Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), np.resize, (1,3), Q([[1, 2, 3]], "meter")),
        (Q([1, 2, 3], "meter"), np.reshape, (1,3), Q([[1, 2, 3]], "meter")),
        (Q([1, 2, 3], "meter"), np.pad, 1, Q([0, 1, 2, 3, 0], "meter")),
        (Q([1, 2, 3], "meter"), np.searchsorted, Q(1, "km"), 3),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.prod, None, Q(720, "meter ** 6")),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.prod, 0, Q([4, 10, 18], "meter ** 2")),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.prod, 1, Q([6, 120], "meter ** 3")),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.prod, (0,1), Q(720, "meter ** 6")),
        (Q([[1, 2, 3], [4, 5, 6]], "meter"), np.nanprod, None, Q(720, "meter ** 6")),
    ),
)
# fmt: on
def test_binary_array_funcs(
    value1: Q, value2: Q, func: Callable, expected: Q | np.ndarray
) -> None:
    actual = func(value1, value2)
    eq = actual == expected
    is_eq = eq.all() if isinstance(eq, np.ndarray) else eq
    assert is_eq, f"{actual} != {expected}"


def test_broadcast_arrays() -> None:
    actual = np.broadcast_arrays(Q([1, 2], "meter"), [[3], [4]])
    expected = (
        Q([[1, 2], [1, 2]], "meter"),
        np.array([[3, 3], [4, 4]])
    )
    for a, e in zip(actual, expected):
        assert (a == e).all()


def test_broadcast_to() -> None:
    actual = np.broadcast_to(Q([1, 2], "meter"), (2,2))
    expected = Q([[1, 2], [1, 2]], "meter")
    assert (actual == expected).all()


def test_clip() -> None:
    actual = np.clip(Q([1, 2, 3], "km"), Q(0, "m"), Q(1000, "m"))
    expected = Q([1, 1, 1], "km")
    assert (actual == expected).all()


def test_einsum() -> None:
    actual = np.einsum("i,i", Q([1, 2, 3], "meter"), Q([1, 2, 3], "km"))
    expected = Q(14000, "meter")
    assert actual == expected


def test_copyto() -> None:
    q1 = Q([1, 2, 3], "meter")
    q2 = Q([1, 2, 3], "km")
    np.copyto(q1, q2)
    assert (q1 == Q([1000, 2000, 3000], "meter")).all()

    arr = np.array([0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        np.copyto(arr, q2)
    assert (arr == np.array([1.0, 2.0, 3.0])).all()


def test_interp() -> None:
    xp = Q([1, 2, 3], "meter")
    fp = Q([3, 2, 0], "meter")
    assert np.interp(Q(2.5e-3, "km"), xp, fp) == Q(1.0, "meter")


def test_meshgrid() -> None:
    x = np.linspace(Q(1, "meter"), Q(3, "meter"))
    y = np.linspace(Q(1, "km"), Q(3, "km"))
    xx, yy = np.meshgrid(x, y)
    xx_e, yy_e = np.meshgrid(np.linspace(1.0, 3.0), 1000 * np.linspace(1.0, 3.0))
    assert xx.u == yy.u == units.meter
    assert np.allclose(xx.m, xx_e)
    assert np.allclose(yy.m, yy_e)


def test_moveaxis() -> None:
    q = Q([[1], [2]])
    qq = np.moveaxis(q, 0, 1)
    assert (qq == Q([[1, 2]])).all()


def test_swapaxes() -> None:
    q = Q([[1], [2]])
    qq = np.swapaxes(q, 0, 1)
    assert (qq == Q([[1, 2]])).all()


def test_trapezoid() -> None:
    # x flow
    y = Q([1, 2, 3], "meter")
    x = Q([3, 2, 1], "gram")
    res = np.trapezoid(y, x)
    expected = np.trapezoid([1, 2, 3], [3, 2, 1])
    assert res == Q(expected, "meter * gram")

    # dx flow
    expected = np.trapezoid([1, 2, 3], dx=0.1)
    assert np.trapezoid(y, dx=Q(0.1, "gram")) == Q(expected, "meter * gram")


def test_nonzero() -> None:
    x = Q([1, 2, 3], "meter")
    actual = np.nonzero(x)
    expected = np.nonzero(np.array([1, 2, 3]))
    assert all((a == e).all() for a, e in zip(actual, expected))
    assert all((a == e).all() for a, e in zip(x.nonzero(), expected))


def test_insert() -> None:
    x = Q([1, 2, 3], "meter")
    actual = np.insert(x, 0, Q(1, "km"))
    expected = Q([1000, 1, 2, 3], "meter")
    assert (actual == expected).all()


def test_pickle_roundtrip() -> None:
    """Quantities survive a roundtrip through pickle."""
    q = Q("1 meter")
    q_: Q = pickle.loads(pickle.dumps(q))
    assert q == q_

    # methods requiring a unit registry still work
    assert q.to_root_units() == q_.to_root_units()


def test_quantity_from_unit() -> None:
    """Quantities built from unit expressions work like regular quantities."""
    q = 1 * units.km
    assert q == Q("1 km")
    assert q.to(units.meter) == Q("1000 meter")

    # dimensionless
    q = 1 * units.radian
    assert q == Q("1 radian")
    assert q.to("dimensionless") == Q(1)

    # array
    q = [1, 2] * units.meter
    assert (q == Q([1, 2], "meter")).all()
    assert (q.to("km") == Q([1e-3, 2e-3], "km")).all()


@pytest.mark.parametrize(
    argnames=("quantity", "op", "unit", "expected"),
    argvalues=(
        (Q(1, "gram"), operator.mul, units.c ** 2, Q(1, "gram * c ** 2")),
        (Q(1, "meter"), operator.truediv, units.meter, Q(1)),
        (Q(1, "meter"), operator.sub, units.meter, Q(0, "meter")),
        (Q(1, "meter"), operator.add, units.meter, Q(2, "meter")),
    ),
)
def test_quantity_x_unit_operations(
    quantity: Q,
    op: Callable,
    unit: units.Unit,
    expected: Q,
) -> None:
    """Units and quantities must interop."""
    assert op(quantity, unit) == expected
    assert op(unit, quantity) == expected


def test_array_quantity_is_iterable() -> None:
    # Iteration returns scalar quantities
    q = Q([1, 2, 3], units.meter)
    assert list(iter(q)) == [Q(x, units.meter) for x in range(1, 4)]

    # Iteration returns array quantities
    q = Q([[1], [2], [3]], units.meter)
    assert list(iter(q)) == [Q([x], units.meter) for x in range(1, 4)]


def test_non_array_quantity_is_not_iterable() -> None:
    with pytest.raises(TypeError):
        iter(Q(1))


@pytest.mark.parametrize(
    argnames=("value", "func", "args", "expected"),
    argvalues=(
        (Q([1, 2, 3], "meter"), Q.argmax, (), 2),
        (Q([1, 2, 3], "meter"), Q.argmin, (), 0),
        (Q([1, 2, 3], "meter"), Q.argsort, (), [0, 1, 2]),
        (Q([1, 2, 3], "meter"), Q.clip, (Q(0, "meter"), Q(1, "meter")), Q([1, 1, 1], "meter")),
        (Q([[1, 2], [3, 4], [5, 6]], "meter"), Q.compress, ([0, 1],), Q([2], "meter")),
        (Q([1, 2, 3], "meter"), Q.cumsum, (), Q([1, 3, 6], "meter")),
        (Q([[1, 0], [0, 1]], "meter"), Q.diagonal, (), Q([1, 1], "meter")),
        (Q([1, 2, 3], "meter"), Q.dot, (Q([3, 2, 1], "meter"),), Q(10, "meter ** 2")),
        (Q([1, 2, 3], "meter"), Q.max, (), Q(3, "meter")),
        (Q([1, 2, 3], "meter"), Q.min, (), Q(1, "meter")),
        (Q([1, 2, 3], "meter"), Q.mean, (), Q(2, "meter")),
        (Q([1, 2, 3], "meter"), Q.prod, (), Q(6, "meter ** 3")),
        (Q([1, 2, 3], "meter"), Q.ravel, (), Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), Q.repeat, (2,), Q([1, 1, 2, 2, 3, 3], "meter")),
        (Q([1, 2, 3], "meter"), Q.reshape, ((1,3),), Q([[1, 2, 3]], "meter")),
        (Q([1, 2, 3], "meter"), Q.round, (), Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), Q.searchsorted, (Q(1, "meter"),), 0),
        (Q([1, 2, 3], "meter"), Q.sort, (), Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), Q.squeeze, (), Q([1, 2, 3], "meter")),
        (Q([1, 2, 3], "meter"), Q.std, (), Q(np.std([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), Q.sum, (), Q(6, "meter")),
        (Q([1, 2, 3], "meter"), Q.take, ([0],), Q([1], "meter")),
        (Q([[1, 0], [0, 1]], "meter"), Q.trace, (), Q(2, "meter")),
        (Q([1, 2, 3], "meter"), Q.var, (), Q(np.var([1, 2, 3]), "meter")),
        (Q([1, 2, 3], "meter"), Q.transpose, (), Q([1, 2, 3], "meter")),
    ),
)
def test_numpy_api(value: Q, func: Callable, args: tuple[Any, ...], expected: Any) -> None:
    actual = func(value, *args) 
    eq = actual == expected
    if isinstance(eq, Iterable):
        assert eq.all(), f"{actual} != {expected}"
    else:
        assert eq, f"{actual} != {expected}"
