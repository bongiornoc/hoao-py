"""Input validation and explicit pandas alignment; no pandas dependency."""
from __future__ import annotations

import sys
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, TypeGuard, cast

import numpy as np

from ._typing import FloatArray, ReturnInput, WeightInput

if TYPE_CHECKING:
    import pandas as pd


def is_frame(value: object) -> TypeGuard[pd.DataFrame]:
    module = sys.modules.get("pandas")
    return module is not None and isinstance(value, module.DataFrame)


def is_series(value: object) -> TypeGuard[pd.Series[float]]:
    module = sys.modules.get("pandas")
    return module is not None and isinstance(value, module.Series)


def labels(values: Sequence[Hashable], size: int) -> tuple[Hashable, ...]:
    result = tuple(values)
    try:
        unique = len(set(result)) == size
    except TypeError as exc:
        raise ValueError("Asset labels must be hashable.") from exc
    if len(result) != size or not unique:
        raise ValueError("Asset labels must be unique and match the asset dimension.")
    pandas = sys.modules.get("pandas")
    for label in result:
        missing = label is None
        if pandas is not None:
            marker = pandas.isna(label)
            missing = missing or (isinstance(marker, (bool, np.bool_)) and bool(marker))
        elif isinstance(label, (float, np.floating)):
            missing = missing or bool(np.isnan(label))
        elif isinstance(label, np.datetime64):
            missing = missing or bool(np.isnat(label))
        if missing:
            raise ValueError("Asset labels must not be missing.")
    return result


def observations(
    returns: ReturnInput, minimum: int = 3,
) -> tuple[FloatArray, tuple[Hashable, ...] | None, tuple[Hashable, ...] | None]:
    assets = None
    dates = None
    if is_frame(returns):
        assets = labels(cast(Sequence[Hashable], returns.columns.tolist()), len(returns.columns))
        if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
            raise ValueError("Observation index must be unique and increasing.")
        dates = tuple(cast(Sequence[Hashable], returns.index.tolist()))
        raw = returns.to_numpy()
    else:
        raw = np.asarray(returns)
    if raw.dtype.kind not in "iuf":
        raise TypeError("Returns must contain real numeric values.")
    x = np.array(raw, dtype=np.float64, order="C", copy=True)
    if x.ndim != 2 or x.shape[0] < minimum or x.shape[1] < 1:
        raise ValueError(f"Expected (observations, assets) with at least {minimum} rows and one asset.")
    if not np.isfinite(x).all():
        raise ValueError("Fitting returns must be finite; NaN and infinity are not imputed.")
    return x, assets, dates


def weights_array(
    weights: WeightInput, size: int, assets: tuple[Hashable, ...] | None,
) -> FloatArray:
    if is_series(weights):
        if assets is None:
            raise ValueError("Labeled weights require a model fitted with asset labels.")
        if not weights.index.is_unique or set(weights.index) != set(assets):
            raise ValueError("Weight labels must match the model's unique asset labels exactly.")
        raw = weights.reindex(list(assets)).to_numpy()
    else:
        raw = np.asarray(weights)
    if raw.dtype.kind not in "iuf":
        raise TypeError("Weights must be real numeric values.")
    w = np.asarray(raw, dtype=np.float64)
    if w.shape != (size,) or not np.isfinite(w).all():
        raise ValueError(f"Weights must be finite with shape ({size},).")
    return w


def readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
