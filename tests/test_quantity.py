import math
import operator
from typing import Any, Callable

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
    ),
)
def test_binary_operators(
    x: Q | int,
    op: Callable[[Q | int, Q | int], Q[int, int]],
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
    ),
)
def test_inplace_operators(
    x: Q,
    op: Callable[[Q, Q | int], Q],
    y: Q | int,
    expected: Q,
) -> None:
    op(x, y)
    assert x == expected
