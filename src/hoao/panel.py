"""Return histories and asset universes for historical calibration.

A ReturnsPanel records which assets were available on each observation
date. Eligibility combines that information with the trailing return
history, so asset selection does not depend on subsequent returns.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from datetime import date as Date, datetime, timezone
import re
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from ._typing import FloatArray, ReturnInput
from ._validation import is_frame, labels as validate_labels

if TYPE_CHECKING:
    import pandas as pd

DateLike: TypeAlias = str | Date | datetime | np.datetime64
Cutoff: TypeAlias = DateLike | int
DateInput: TypeAlias = Union[Sequence[DateLike], NDArray[np.datetime64], "pd.DatetimeIndex"]
Availability: TypeAlias = Union[
    NDArray[np.bool_], "pd.DataFrame", Mapping[Cutoff, Sequence[Hashable]],
    Sequence[Sequence[Hashable]],
]


def _date(value: Any) -> np.datetime64:
    """Convert a date-like scalar to an unambiguous UTC timestamp."""
    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        raise ValueError("A dated panel needs a date-like cutoff, not an integer")
    try:
        if hasattr(value, "to_datetime64"):
            if value.tzinfo is not None:
                value = value.tz_convert("UTC").tz_localize(None)
            stamp = value.to_datetime64()
        elif isinstance(value, np.datetime64):
            stamp = value
        elif isinstance(value, str):
            # Perform timezone conversion before constructing any NumPy ns
            # timestamp: a valid UTC instant can have an out-of-range local
            # clock reading. Python integers also preserve the submicrosecond
            # fraction discarded by datetime.fromisoformat.
            parsed = datetime.fromisoformat(value)
            utc = (parsed.astimezone(timezone.utc).replace(tzinfo=None)
                   if parsed.tzinfo is not None else parsed)
            elapsed = utc - datetime(1970, 1, 1)
            nanoseconds = ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000
                           + elapsed.microseconds) * 1000
            fraction = re.search(r"[T ]\d{2}(?::?\d{2})?(?::?\d{2})?[.,](\d+)", value)
            if fraction:
                digits = fraction.group(1)
                if any(digit != "0" for digit in digits[9:]):
                    raise ValueError("Date cannot be represented at nanosecond precision")
                nanoseconds += int(digits.ljust(9, "0")[6:9])
            if not -2**63 < nanoseconds < 2**63:
                raise ValueError("Date lies outside the datetime64[ns] range")
            return np.datetime64(nanoseconds, "ns")
        else:
            if isinstance(value, datetime) and value.tzinfo is not None:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            if not isinstance(value, (datetime, Date)):
                raise ValueError("Expected an ISO date, datetime, or numpy.datetime64")
            stamp = np.datetime64(value)
        if np.isnat(stamp):
            raise ValueError("Missing date")
        unit, multiple = np.datetime_data(stamp.dtype)
        if unit in ("Y", "M"):
            year = int(stamp.astype("datetime64[Y]").astype(np.int64)) + 1970
            if year < 1677 or year > 2262:
                raise ValueError("Date lies outside the datetime64[ns] range")
            stamp = stamp.astype("datetime64[D]")
            unit, multiple = "D", 1
        count = int(stamp.astype(np.int64)) * multiple
        factors = {"W": 604800_000_000_000, "D": 86400_000_000_000,
                   "h": 3600_000_000_000, "m": 60_000_000_000,
                   "s": 1_000_000_000, "ms": 1_000_000, "us": 1000, "ns": 1}
        if unit in factors:
            nanoseconds = count * factors[unit]
        else:
            divisor = {"ps": 1000, "fs": 1_000_000, "as": 1_000_000_000}[unit]
            nanoseconds, remainder = divmod(count, divisor)
            if remainder:
                raise ValueError("Date cannot be represented at nanosecond precision")
        # Python integers avoid overflow in NumPy's casts near the negative
        # ns boundary, including its ns->us flooring implementation.
        if not -2**63 < nanoseconds < 2**63:
            raise ValueError("Date lies outside the datetime64[ns] range")
        return np.datetime64(nanoseconds, "ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid date: {value!r}") from exc


def _positive_int(value: Any, name: str, *, zero: bool = False) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < (0 if zero else 1)):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")
    return int(value)


class ReturnsPanel:
    """A return history with an asset universe specified for each date.

    Construct a panel once and pass it to :func:`calibrate` or
    :meth:`Calibration.update`. The panel stores a copy of the data and
    caches eligibility calculations for each training-window length.

    Parameters
    ----------
    returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
        Real returns with observations in chronological row order and
        assets in columns. NaN denotes a missing observation; infinity
        is rejected. Both axes must be nonempty. Values are copied to
        float64 without annualization or filling missing observations.
    dates : sequence of date-like, optional
        Unique, strictly increasing dates for array input, one per row.
        A DataFrame supplies its own index, so this argument must be
        omitted for DataFrame input. Without dates, row positions are
        used; a positional DataFrame must have index ``0, 1, ..., T-1``.
        Timezone-aware timestamps are converted to UTC, and dates must
        be representable as ``datetime64[ns]``.
    assets : sequence of hashable, optional
        Unique, non-missing column labels for array input. Defaults to
        integer column positions. Omit for a DataFrame, whose column
        labels are preserved.
    available : ndarray of bool, pandas.DataFrame, mapping, or sequence, optional
        Asset availability known on each row's signal date. Supply a
        boolean array of the same shape as `returns`, a boolean DataFrame
        with exactly matching row and column labels in the same order,
        a mapping from dates or row positions to lists of asset labels,
        or one such list per row. Mapping keys must identify panel rows;
        omitted rows have an empty universe. Duplicate or unknown asset
        labels are errors. If omitted, all assets are candidates.

    Attributes
    ----------
    values : ndarray of float64, shape (n_observations, n_assets)
        Read-only return values, including any NaNs in the input.
    dates : ndarray of datetime64[ns], shape (n_observations,), or None
        Read-only timestamps, or None for a positional panel.
    assets : tuple of hashable
        Asset labels in return-column order.
    available : ndarray of bool, shape (n_observations, n_assets)
        Read-only availability mask before checking training histories.
        All entries are True when no availability was supplied.

    Raises
    ------
    ValueError
        If returns, dates or labels are invalid, or availability cannot
        be aligned exactly with the panel.

    See Also
    --------
    eligible : Inspect which assets have a usable trailing history.
    calibrate : Sample historical training and target windows.

    Notes
    -----
    On a signal date, an asset is eligible only if it is marked available
    and has a complete, finite, nonconstant training history ending on
    that date. A missing observation in the subsequent target window
    does not exclude it or cause a replacement asset to be selected.

    Returns alone can establish whether the training history is usable;
    they cannot reconstruct liquidity or capitalization screens. Encode
    such criteria in `available`, using information known at each date.

    Examples
    --------
    Automatic eligibility excludes an asset while a missing observation
    remains in its trailing window:

    >>> import numpy as np
    >>> from hoao import ReturnsPanel
    >>> returns = np.array([[0.01, 0.02, 0.03],
    ...                     [0.02, 0.01, np.nan],
    ...                     [-0.01, 0.02, 0.01],
    ...                     [0.03, -0.01, 0.02],
    ...                     [0.01, 0.03, 0.04]])
    >>> panel = ReturnsPanel(returns, assets=["A", "B", "C"])
    >>> panel.eligible(train_size=3)[2].tolist()
    [True, True, False]
    >>> panel.eligible(train_size=3)[4].tolist()
    [True, True, True]

    An explicit universe restricts these candidates further. In this
    positional panel, the mapping keys are row numbers:

    >>> available = {2: ["A", "C"], 4: ["B", "C"]}
    >>> panel = ReturnsPanel(returns, assets=["A", "B", "C"],
    ...                      available=available)
    >>> panel.eligible(train_size=3)[2].tolist()
    [True, False, False]
    >>> panel.eligible(train_size=3)[4].tolist()
    [False, True, True]
    """

    def __init__(self, returns: ReturnInput, *, dates: DateInput | None = None,
                 assets: Sequence[Hashable] | None = None,
                 available: Availability | None = None) -> None:
        frame = is_frame(returns)
        if frame:
            if dates is not None or assets is not None:
                raise ValueError("DataFrame input supplies its own dates and assets")
            if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
                raise ValueError("Return index must be unique and increasing")
            if returns.index.dtype.kind not in "iu":
                dates = list(returns.index)
            elif not np.array_equal(np.asarray(returns.index), np.arange(len(returns))):
                raise ValueError("A positional DataFrame must use row labels 0, 1, ..., T-1")
            assets = tuple(returns.columns)
        try:
            raw = np.asarray(returns)
            if raw.dtype.kind not in "iuf":
                raise ValueError("Return values must be real numeric values or NaN")
            values = np.array(raw, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("returns must contain numeric values or NaN") from exc
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError("returns must have shape (observations, assets), with nonempty axes")
        if np.isinf(values).any():
            raise ValueError("returns must not contain infinity")
        self._values = values
        self._values.flags.writeable = False
        self._assets = validate_labels(tuple(range(values.shape[1])) if assets is None else assets,
                                       values.shape[1])
        self._positions = {asset: i for i, asset in enumerate(self._assets)}
        self._dates: NDArray[np.datetime64] | None = None
        if dates is not None:
            self._dates = np.array([_date(value) for value in dates], dtype="datetime64[ns]")
            if (len(self._dates) != len(values)
                    or np.any(self._dates[1:] <= self._dates[:-1])):
                raise ValueError("Dates must be unique, increasing, and match the number of rows")
            self._dates.flags.writeable = False
        self._available = self._availability(available, returns if frame else None)
        self._available.flags.writeable = False
        self._eligibility_cache: dict[int, NDArray[np.bool_]] = {}

    @property
    def values(self) -> FloatArray:
        """Read-only float64 returns, shaped (n_observations, n_assets)."""
        return self._values.view()

    @property
    def dates(self) -> NDArray[np.datetime64] | None:
        """Read-only datetime64[ns] timestamps, or None for positional data."""
        return None if self._dates is None else self._dates.view()

    @property
    def assets(self) -> tuple[Hashable, ...]:
        """Asset labels in return-column order."""
        return self._assets

    @property
    def available(self) -> NDArray[np.bool_]:
        """Read-only availability flags before checking training histories."""
        return self._available.view()

    def __len__(self) -> int:
        return len(self._values)

    def _availability(self, supplied: Any, frame: Any) -> NDArray[np.bool_]:
        if supplied is None:
            return np.ones(self._values.shape, dtype=bool)
        if is_frame(supplied):
            if frame is not None:
                aligned = supplied.index.equals(frame.index) and supplied.columns.equals(frame.columns)
            else:
                actual_index = (np.array([_date(value) for value in supplied.index], dtype="datetime64[ns]")
                                if self._dates is not None else np.asarray(supplied.index))
                expected_index = self._dates if self._dates is not None else np.arange(len(self))
                aligned = np.array_equal(actual_index, expected_index) and tuple(supplied.columns) == self.assets
            if not aligned:
                raise ValueError("available must have exactly the same row and column labels as returns")
            if supplied.isna().any().any() or not all(d.kind == "b" for d in supplied.dtypes):
                raise ValueError("available table must contain boolean flags without missing values")
            return supplied.to_numpy(dtype=bool, copy=True)
        if not isinstance(supplied, Mapping):
            try:
                array = np.asarray(supplied)
            except ValueError:
                array = np.asarray(supplied, dtype=object)
            if array.dtype == bool:
                if array.shape != self._values.shape:
                    raise ValueError("Boolean available mask must have the same shape as returns")
                return array.copy()
        mask = np.zeros(self._values.shape, dtype=bool)
        if isinstance(supplied, Mapping):
            rows = []
            seen: set[int] = set()
            for label, labels in supplied.items():
                row = self._row(label)
                if row in seen:
                    raise ValueError("Multiple availability keys refer to the same date")
                seen.add(row)
                rows.append((row, labels))
        else:
            if len(supplied) != len(self):
                raise ValueError("Daily available lists must have one entry per return row")
            rows = enumerate(supplied)
        for row, labels in rows:
            if isinstance(labels, (str, bytes)):
                raise ValueError("Each available entry must be a list of asset labels")
            try:
                labels = list(labels)
                if len(set(labels)) != len(labels):
                    raise ValueError(f"Duplicate available asset at row {row}")
                columns = [self._positions[label] for label in labels]
            except (TypeError, KeyError) as exc:
                raise ValueError(f"Unknown or invalid available asset at row {row}: {exc}") from exc
            mask[row, columns] = True
        return mask

    def _row(self, value: Any) -> int:
        if self._dates is None:
            row = _positive_int(value, "availability row", zero=True)
            if row >= len(self):
                raise ValueError("Availability row lies outside the panel")
            return row
        stamp = _date(value)
        row = int(np.searchsorted(self._dates, stamp))
        if row == len(self) or self._dates[row] != stamp:
            raise ValueError(f"Availability date {value!r} is absent from returns")
        return row

    def _cutoff(self, value: Any) -> int | np.datetime64:
        return (_positive_int(value, "cutoff", zero=True)
                if self._dates is None else _date(value))

    def _location(self, value: Any) -> int:
        cutoff = self._cutoff(value)
        return int(cutoff) if self._dates is None else int(np.searchsorted(self._dates, cutoff))

    def eligible(self, *, train_size: int) -> NDArray[np.bool_]:
        """Identify available assets with complete, varying training histories.

        Parameters
        ----------
        train_size : int
            Positive number of consecutive observations in the training
            window, including the signal row itself.

        Returns
        -------
        ndarray of bool, shape (n_observations, n_assets)
            True where an asset is available on that row, all returns in its
            trailing window are finite, and the window contains at least two
            distinct values. Rows without enough history are False. A
            one-observation window cannot have positive variance, so
            ``train_size=1`` returns an all-False mask. The returned array is
            an independent copy.

        Raises
        ------
        ValueError
            If `train_size` is not a positive integer.

        Notes
        -----
        Only the training window and availability on the signal row enter
        this calculation. Future return values and future availability are
        not inspected. Calibration separately checks whether enough target
        dates exist before its cutoff.

        Finite-observation counts and adjacent-value changes are accumulated
        once per window length. This avoids repeatedly computing rolling
        variances, including cancellation errors for nearly constant data.

        Examples
        --------
        The second asset has a complete history but is constant:

        >>> from hoao import ReturnsPanel
        >>> panel = ReturnsPanel([[0.01, 0.02], [0.03, 0.02], [0.02, 0.02]])
        >>> panel.eligible(train_size=3).tolist()
        [[False, False], [False, False], [True, False]]
        """
        train_size = _positive_int(train_size, "train_size")
        if train_size not in self._eligibility_cache:
            result = np.zeros(self._values.shape, dtype=bool)
            if train_size > 1 and train_size <= len(self):
                finite = np.isfinite(self._values)
                counts = np.vstack([np.zeros((1, finite.shape[1]), dtype=np.int64),
                                    np.cumsum(finite, axis=0, dtype=np.int64)])
                complete = counts[train_size:] - counts[:-train_size] == train_size
                changes = np.zeros(finite.shape, dtype=np.int64)
                changes[1:] = self._values[1:] != self._values[:-1]
                changed = np.cumsum(changes, axis=0)
                different = changed[train_size - 1:] - changed[:len(self) - train_size + 1] > 0
                result[train_size - 1:] = complete & different & self._available[train_size - 1:]
            result.flags.writeable = False
            self._eligibility_cache[train_size] = result
        return self._eligibility_cache[train_size].copy()

    def _signals(self, *, n_assets: int, train_size: int, test_size: int,
                 gap: int, before: Any, start: Any = None) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        upper = min(len(self), self._location(before))
        lower = train_size - 1 if start is None else max(train_size - 1, self._location(start))
        signals = np.arange(lower, upper - gap - test_size, dtype=np.int64)
        eligible = self.eligible(train_size=train_size)
        signals = signals[eligible[signals].sum(axis=1) >= n_assets]
        if len(signals) == 0:
            raise ValueError("No signal dates have enough eligible training histories and complete target dates before the cutoff")
        return signals, eligible
