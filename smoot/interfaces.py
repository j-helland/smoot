from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import smoot
from smoot.numpy_functions import NP_HANDLED_FUNCTIONS, NP_HANDLED_UFUNCS
from smoot.smoot import ArrayF64Quantity, F64Quantity

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray, ArrayLike


class SupportsPickle:
    @staticmethod
    def _load(inner: F64Quantity | ArrayF64Quantity) -> smoot.Quantity:
        # Use the currently set application registry.
        # WARNING: This may not be the same as the registry the pickled quantity was created with.
        new = object.__new__(smoot.ApplicationRegistry.get().Quantity)
        new.__inner = inner
        return new

    def __reduce__(self):
        return (smoot.Quantity._load, (self._Quantity__inner,))


class SupportsNumpy:
    """Numpy API for quantities."""

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        _method: str,
        *inputs: smoot.Quantity,
        **kwargs: Any,
    ) -> smoot.Quantity | type(NotImplemented):
        if (func := NP_HANDLED_UFUNCS.get(ufunc.__name__)) is None:
            return NotImplemented
        return func(*inputs, **kwargs)

    def __array_function__(
        self,
        func: Callable,
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> smoot.Quantity | type(NotImplemented):
        func_name = ".".join(func.__module__.split(".")[1:] + [func.__name__])
        if (func := NP_HANDLED_FUNCTIONS.get(func_name)) is None:
            return NotImplemented
        return func(*args, **kwargs)

    # TODO(jwh): documentation
    # TODO(jwh): replace *args, **kwargs with real arguments
    def argmax(self, *args, **kwargs) -> np.intp | NDArray[np.intp]:
        return np.argmax(self, *args, **kwargs)

    def argmin(self, *args, **kwargs) -> np.intp | NDArray[np.intp]:
        return np.argmin(self, *args, **kwargs)

    def argsort(self, *args, **kwargs) -> NDArray[np.intp]:
        return np.argsort(self, *args, **kwargs)

    def clip(self, *args, **kwargs) -> smoot.Quantity:
        return np.clip(self, *args, **kwargs)

    def compress(self, condition: ArrayLike[Any], *args, **kwargs) -> smoot.Quantity:
        return np.compress(condition, self, *args, **kwargs)

    def cumsum(self, *args, **kwargs) -> smoot.Quantity:
        return np.cumsum(self, *args, **kwargs)

    def diagonal(self, *args, **kwargs) -> smoot.Quantity:
        return np.diagonal(self, *args, **kwargs)

    def dot(self, *args, **kwargs) -> smoot.Quantity:
        return np.dot(self, *args, **kwargs)

    def max(self, *args, **kwargs) -> smoot.Quantity:
        return np.max(self, *args, **kwargs)

    def mean(self, *args, **kwargs) -> smoot.Quantity:
        return np.mean(self, *args, **kwargs)

    def min(self, *args, **kwargs) -> smoot.Quantity:
        return np.min(self, *args, **kwargs)

    def nonzero(self, *args, **kwargs) -> tuple[NDArray[np.intp], ...]:
        return np.nonzero(self, *args, **kwargs)

    def prod(self, *args, **kwargs) -> smoot.Quantity:
        return np.prod(self, *args, **kwargs)

    def ravel(self, *args, **kwargs) -> smoot.Quantity:
        return np.ravel(self, *args, **kwargs)

    def repeat(self, *args, **kwargs) -> smoot.Quantity:
        return np.repeat(self, *args, **kwargs)

    def reshape(self, shape: tuple[int, ...], *args, **kwargs) -> smoot.Quantity:
        _ = self._Quantity__inner.reshape(shape)
        return self

    def round(self, *args, **kwargs) -> smoot.Quantity:
        return np.round(self, *args, **kwargs)

    def searchsorted(self, *args, **kwargs) -> np.intp | NDArray[np.intp]:
        return np.searchsorted(self, *args, **kwargs)

    def sort(self, *args, **kwargs) -> smoot.Quantity:
        return np.sort(self, *args, **kwargs)

    def squeeze(self, *args, **kwargs) -> smoot.Quantity:
        return np.squeeze(self, *args, **kwargs)

    def std(self, *args, **kwargs) -> smoot.Quantity:
        return np.std(self, *args, **kwargs)

    def sum(self, *args, **kwargs) -> smoot.Quantity:
        return np.sum(self, *args, **kwargs)

    def take(self, *args, **kwargs) -> smoot.Quantity:
        return np.take(self, *args, **kwargs)

    def trace(self, *args, **kwargs) -> smoot.Quantity:
        return np.trace(self, *args, **kwargs)

    def var(self, ddof: float = 0.0, *args, **kwargs) -> smoot.Quantity:
        return np.var(self, *args, **kwargs)

    def transpose(self) -> smoot.Quantity:
        new = object.__new__(self.__class__)
        new._Quantity__inner = self._Quantity__inner.transpose()
        return new

    @property
    def T(self) -> smoot.Quantity:
        new = object.__new__(self.__class__)
        new._Quantity__inner = self._Quantity__inner.transpose()
        return new

    @property
    def shape(self) -> tuple[int, ...]:
        return self._Quantity__inner.shape

    @property
    def size(self) -> int:
        return self._Quantity__inner.size

    @property
    def ndim(self) -> int:
        return self._Quantity__inner.ndim
