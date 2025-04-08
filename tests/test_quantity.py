import math
import operator
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from smoot import Quantity as Q


@pytest.mark.parametrize(
    argnames=("value", "unit", "expected"),
    argvalues=(
        (Q("1 km"), "meter", Q("1000 meter")),
        (Q("1000 meter"), "km", Q("1 km")),
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


def test_eq() -> None:
    """`==` operator works."""
    assert Q(1) == Q(1)
    assert Q(1) == 1
    assert 1 == Q(1)

    assert Q(2) != Q(1)
    assert Q("1 meter") != 1
    assert Q(math.nan) != Q(math.nan)


@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        (Q(1), operator.add, Q(1), Q(2)),
        (Q(1), operator.add, 1, Q(2)),
        (1, operator.add, Q(1), Q(2)),
        (Q(2), operator.sub, Q(1), Q(1)),
        (Q(2), operator.sub, 1, Q(1)),
        (2, operator.sub, Q(1), Q(1)),
        (Q(2), operator.mul, Q(2), Q(4)),
        (Q(2), operator.mul, 2, Q(4)),
        (2, operator.mul, Q(2), Q(4)),
        (Q(2), operator.pow, Q(3), Q(8)),
        (2, operator.pow, Q(3), Q(8)),
        (Q(2), operator.pow, 3, Q(8)),
        (Q(4), operator.truediv, Q(2), Q(2)),
        (Q(4), operator.truediv, 2, Q(2)),
        (4, operator.truediv, Q(2), Q(2)),
        (Q(4), operator.floordiv, Q(2), Q(2)),
        (Q(4), operator.floordiv, 2, Q(2)),
        (4, operator.floordiv, Q(2), Q(2)),
        (Q(4), operator.mod, Q(2), Q(0)),
        (4, operator.mod, Q(2), Q(0)),
        (Q(4), operator.mod, 2, Q(0)),
    ),
)
def test_binary_operators(
    x: Q | int,
    op: Callable[[Q | int, Q | int], Q],
    y: Q | int,
    expected: Q,
) -> None:
    """Binary operators applied to quantities produce the expected values."""
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


@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        (Q([1, 2, 3]), operator.eq, Q([1, 0, 3]), np.array([True, False, True])),
        ([1, 2, 3], operator.eq, Q([1, 0, 3]), np.array([True, False, True])),
        (Q([1, 2, 3]), operator.eq, [1, 0, 3], np.array([True, False, True])),
        # compatible units
        (
            Q([1000, 2000, 3000], "meter"),
            operator.eq,
            Q([1, 2, 3], "km"),
            np.array([True, True, True]),
        ),
        # incompatible units
        (
            Q([1, 2, 3], "meter"),
            operator.eq,
            Q([1, 2, 3], "gram"),
            np.array([False, False, False]),
        ),
        (Q([1, 2, 3]), operator.add, Q([3, 2, 1]), Q([4, 4, 4])),
        ([1, 2, 3], operator.add, Q([3, 2, 1]), Q([4, 4, 4])),
        (Q([1, 2, 3]), operator.add, [3, 2, 1], Q([4, 4, 4])),
        # scalar addition
        (Q([1, 2, 3]), operator.add, 1, Q([2, 3, 4])),
        (1, operator.add, Q([1, 2, 3]), Q([2, 3, 4])),
        # compatible units
        (
            Q([1000, 2000, 3000], "meter"),
            operator.add,
            Q([3, 2, 1], "km"),
            Q([4000, 4000, 4000], "meter"),
        ),
        (
            Q([1, 2, 3], "km"),
            operator.add,
            Q([3000, 2000, 1000], "meter"),
            Q([4, 4, 4], "km"),
        ),
        (Q([1, 2, 3]), operator.sub, Q([3, 2, 1]), Q([-2, 0, 2])),
        ([1, 2, 3], operator.sub, Q([3, 2, 1]), Q([-2, 0, 2])),
        (Q([1, 2, 3]), operator.sub, [3, 2, 1], Q([-2, 0, 2])),
        (Q([1, 2, 3]), operator.mul, Q([3, 2, 1]), Q([3, 4, 3])),
        ([1, 2, 3], operator.mul, Q([3, 2, 1]), Q([3, 4, 3])),
        (Q([1, 2, 3]), operator.mul, [3, 2, 1], Q([3, 4, 3])),
        (Q([1, 2, 3]), operator.pow, Q([3, 2, 1]), Q([1, 4, 3])),
        ([1, 2, 3], operator.pow, Q([3, 2, 1]), Q([1, 4, 3])),
        (Q([1, 2, 3]), operator.pow, [3, 2, 1], Q([1, 4, 3])),
        (
            Q([[1], [2], [3]]),
            operator.matmul,
            Q([[3, 2, 1]]),
            Q([[3, 2, 1], [6, 4, 2], [9, 6, 3]]),
        ),
        (
            [[1], [2], [3]],
            operator.matmul,
            Q([[3, 2, 1]]),
            Q([[3, 2, 1], [6, 4, 2], [9, 6, 3]]),
        ),
        (
            Q([[1], [2], [3]]),
            operator.matmul,
            [[3, 2, 1]],
            Q([[3, 2, 1], [6, 4, 2], [9, 6, 3]]),
        ),
    ),
)
def test_binary_array_operators(
    x: Q | NDArray[np.int64],
    op: Callable[[Q | NDArray[np.int64], Q | NDArray[np.int64]], Q],
    y: Q | NDArray[np.int64],
    expected: Q,
) -> None:
    """Binary array operators produce expected values."""
    print(op(x, y))
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


def test_ufunc() -> None:
    """Arbitrary numpy ufuncs can be invoked with expected results."""
    expected = np.sqrt(np.array([1, 2, 3]))
    actual = np.sqrt(Q([1, 2, 3])).magnitude
    assert np.allclose(expected, actual)
