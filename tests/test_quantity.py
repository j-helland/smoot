from __future__ import annotations

import inspect
import math
import operator
import pickle
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import pytest
import hypothesis
import hypothesis.strategies as st

from smoot import UnitRegistry


units = UnitRegistry()
Q = units.Quantity


@hypothesis.given(st.floats())
def test_quantity_floats(value: float) -> None:
    """All floats can be wrapped in a quantity."""
    if math.isnan(value):
        assert math.isnan(Q(value))
    else:
        assert Q(value).magnitude == value


# 2**53 + 1 is the first integer that can't be represented by a 64-bit float.
@hypothesis.given(st.integers(min_value=-(2**53), max_value=2**53))
def test_quantity_integers(value: int) -> None:
    assert Q(value).magnitude == value


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
        (Q("1 km ** 0.5 / m"), Q(math.sqrt(1000.0), "1 / m ** 0.5")),
        (Q("1 Hz"), Q("1 / second")),
    ),
)
def test_to_root_units(value: Q, expected: Q) -> None:
    assert value.to_root_units() == expected

    # in-place
    value.ito_root_units()
    assert value == expected


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
        (Q([1, 2, 3]), operator.pow, Q([3, 2, 1]), Q([1, 4, 3])),
        ([1, 2, 3], operator.pow, Q([3, 2, 1]), Q([1, 4, 3])),
        (Q([1, 2, 3]), operator.pow, [3, 2, 1], Q([1, 4, 3])),
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


@pytest.mark.parametrize(
    argnames=("x", "op", "expected"),
    argvalues=(
        (Q([-1, -2, -3]), abs, Q([1, 2, 3])),
        (Q([-1, -2, -3]), operator.neg, Q([1, 2, 3])),
        # ufuncs
        (Q([1, 2, 3]), np.sqrt, Q(np.sqrt(np.array([1, 2, 3])))),
    ),
)
def test_unary_array_operators(
    x: Q | NDArray[np.int64],
    op: Callable[[Q | NDArray[np.int64]], Q],
    expected: Q,
) -> None:
    """Unary array operators produce expected values."""
    assert (op(x) == expected).all()


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
        (Q([1, 2, 3], "meter"), np.sqrt, Q(np.sqrt([1, 2, 3]), "meter ** 0.5")),
        (Q([1, 2, 3], "meter"), np.square, Q(np.square([1, 2, 3]), "meter ** 2.0")),
        (Q([1, 2, 3], "meter"), np.cbrt, Q(np.cbrt([1, 2, 3]), units.meter ** (1 / 3))),
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
        (Q([1, 2, 3], "meter"), np.sum, Q(6, "meter")),
        # (Q([1, 2, 3], "meter"), np.cumsum, Q([1, 3, 6], "meter")),
        # (Q([1, 2, 3], "meter"), np.cumprod, Q([1, 2, 6], "meter ** 3")),
    ),
)
# fmt: on
def test_unary_array_funcs(value: Q, ufunc: np.ufunc, expected: Q | np.ndarray) -> None:
    """Arbitrary numpy ufuncs can be invoked with expected results."""
    actual = ufunc(value)
    eq = actual == expected
    is_eq = eq.all() if isinstance(eq, np.ndarray) else eq
    assert is_eq, f"{actual} != {expected}"


# fmt: off
@pytest.mark.parametrize(
    argnames=("value1", "func", "value2", "expected"),
    argvalues=(
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
