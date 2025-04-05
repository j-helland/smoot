import operator
from typing import Callable

import pytest

from smoot import Quantity as Q


@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        (Q(1), operator.add, Q(1), Q(2)),
        (Q(1), operator.add, 1, Q(2)),
        (Q(1), operator.sub, Q(1), Q(0)),
        (Q(2), operator.mul, Q(2), Q(4)),
        # (Q(2), operator.pow, Q(2), Q(4)),
    ),
)
def test_basic_operators(
    x: Q,
    op: Callable[[Q, Q], Q],
    y: Q,
    expected: Q,
) -> None:
    assert op(x, y) == expected


@pytest.mark.parametrize(
    argnames=("x", "op", "y", "expected"),
    argvalues=(
        (Q(1), operator.iadd, Q(1), Q(2)),
        (Q(1), operator.isub, Q(1), Q(0)),
        (Q(2), operator.imul, Q(2), Q(4)),
    ),
)
def test_inplace_operators(
    x: Q,
    op: Callable[[Q, Q], Q],
    y: Q,
    expected: Q,
) -> None:
    op(x, y)
    assert x == expected
