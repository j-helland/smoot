from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

class F64Unit:
    def parse(self, s: str) -> F64Unit: ...

class F64Quantity:
    def __init__(self, value: float, units: F64Unit) -> None: ...
    @staticmethod
    def parse(expression: str) -> F64Quantity: ...
    @property
    def m(self) -> float: ...
    @property
    def magnitude(self) -> float: ...
    @property
    def u(self) -> F64Unit: ...
    @property
    def units(self) -> F64Unit: ...
    def to(self, units: str | F64Unit) -> F64Quantity: ...
    def ito(self, units: str | F64Unit) -> None: ...
    def m_as(self, units: str | F64Unit) -> float: ...

class I64Quantity:
    def __init__(self, value: int, units: F64Unit) -> None: ...
    @staticmethod
    def parse(expression: str) -> I64Quantity: ...
    @property
    def m(self) -> int: ...
    @property
    def magnitude(self) -> int: ...
    @property
    def u(self) -> F64Unit: ...
    @property
    def units(self) -> F64Unit: ...
    def to(self, units: str | F64Unit) -> I64Quantity: ...
    def ito(self, units: str | F64Unit) -> None: ...
    def m_as(self, units: str | F64Unit) -> int: ...

class ArrayF64Quantity:
    def __init__(self, value: NDArray[np.float64], units: F64Unit) -> None: ...
    @staticmethod
    def parse(expression: str) -> ArrayF64Quantity: ...
    @property
    def m(self) -> NDArray[np.float64]: ...
    @property
    def magnitude(self) -> NDArray[np.float64]: ...
    @property
    def u(self) -> F64Unit: ...
    @property
    def units(self) -> F64Unit: ...
    def to(self, units: str | F64Unit) -> ArrayF64Quantity: ...
    def ito(self, units: str | F64Unit) -> None: ...
    def m_as(self, units: str | F64Unit) -> NDArray[np.float64]: ...

class ArrayI64Quantity:
    def __init__(self, value: NDArray[np.int64], units: F64Unit) -> None: ...
    @staticmethod
    def parse(expression: str) -> ArrayI64Quantity: ...
    @property
    def m(self) -> NDArray[np.int64]: ...
    @property
    def magnitude(self) -> NDArray[np.int64]: ...
    @property
    def u(self) -> F64Unit: ...
    @property
    def units(self) -> F64Unit: ...
    def to(self, units: str | F64Unit) -> ArrayI64Quantity: ...
    def ito(self, units: str | F64Unit) -> None: ...
    def m_as(self, units: str | F64Unit) -> NDArray[np.int64]: ...
