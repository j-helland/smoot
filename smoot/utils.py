import warnings

from typing import Any

from numpy.typing import NDArray


def warn_for_large_arrays(arr: NDArray[Any]) -> None:
    if arr.size > 1000:
        warnings.warn(
            f"Created a large array with {arr.size} elements. "
            "Large arrays might incur noticeable overhead with Smoot due to copying",
            UserWarning,
            stacklevel=2,
        )
