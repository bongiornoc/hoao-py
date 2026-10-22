"""Shared array and input types for the public API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd

FloatArray: TypeAlias = NDArray[np.float64]
ReturnInput: TypeAlias = Union[ArrayLike, "pd.DataFrame"]
WeightInput: TypeAlias = Union[ArrayLike, "pd.Series[float]"]
Device: TypeAlias = Literal["cpu", "cuda"]
