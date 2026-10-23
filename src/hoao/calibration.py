"""Historical calibration of AO correlation and HOAO cokurtosis.

Calibration averages coefficients measured in later return windows
along bases estimated from earlier windows. The resulting coefficients
can be applied to a current window, updated with more historical pairs,
and saved together with their sampling and chronology information.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._numerics import (BASIS_CONVENTION, correlation_basis, fourth_basis,
                        normalize_correlation, pair_core, pair_cores_batch, standardize)
from ._typing import Device, FloatArray, ReturnInput
from ._validation import observations
from .models import CorrelationEstimate, CokurtosisEstimate, MomentModel
from .panel import Cutoff, ReturnsPanel, _date, _positive_int

_SCHEMA = 1
_MEMBERS = {"metadata", "sum2", "sum4", "filled_locations"}
_METADATA_FIELDS = {"schema", "basis", "n_assets", "train_size", "test_size",
                    "gap", "n_pairs", "seed", "dated", "before", "latest_target",
                    "missing", "filled_count", "location_bounds"}
MissingPolicy = Literal["zero", "raise"]


def _seed(random_state: int | None) -> int:
    if random_state is None:
        return secrets.randbits(128)
    return _positive_int(random_state, "random_state", zero=True)


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    result = array.copy()
    result.flags.writeable = False
    return result


class Calibration:
    """Historical AO and HOAO coefficients, averaged by spectral rank.

    Create a calibration with :func:`calibrate`, or restore one with
    :meth:`load`. Its fitting methods combine the historical coefficients
    with bases and marginal volatilities estimated from a current return
    window. The calibration is specific to its asset count and window
    lengths, but can be applied to different asset identities.

    Attributes
    ----------
    n_assets : int
        Number of assets sampled in each pair and required when fitting.
    train_size, test_size : int
        Observations in each training and subsequent target window.
        Fitting a current window requires exactly `train_size` rows.
    gap : int
        Observations skipped between training and target windows.
    n_pairs : int
        Total number of accumulated pairs. Each contributes equal weight
        to both cores, including after updates and save/load cycles.
    correlation_core : ndarray of float64, shape (n_assets,)
        Read-only mean target variances along training correlation
        eigenvectors, indexed by decreasing training eigenvalue rank.
    cokurtosis_core : ndarray of float64, shape (n_assets, n_assets)
        Read-only mean products of squared target coordinates in the
        training fourth-order basis. Both indices follow decreasing
        singular-value rank in that basis.
    before : int or numpy.datetime64
        Exclusive target cutoff of the latest calibration or update.
        An application date must be at least this cutoff.
    latest_target : int or numpy.datetime64
        Latest target observation actually sampled. It is strictly
        earlier than `before` and any permitted application date.
    missing : {'zero', 'raise'}
        Policy for missing target returns, fixed at initial calibration.
    filled_count : int
        Number of sampled target NaNs replaced with zero. An observation
        sampled in several pairs contributes once per occurrence.
    filled_locations : ndarray of int64, shape (filled_count, 3)
        Read-only records of ``(pair_id, panel_row, panel_column)``, in
        pair order. All positions are zero-based and refer to the panel
        used for that pair. Retain the corresponding panel axes if
        later updates use different rows or columns.

    See Also
    --------
    calibrate : Construct a calibration from a return history.
    ReturnsPanel : Store history and availability known on each date.

    Notes
    -----
    Coefficients are dimensionless: training and target windows are
    standardized independently before projection. Current marginal
    volatilities are restored when fitting. Historical asset identities
    and volatility levels are not carried into the fitted model.

    Persistence retains coefficient sums, the pair count, random seed,
    dimensions, cutoffs, missing-return records and basis convention in
    an NPZ file with JSON metadata. For the same panel, sampling bounds
    and numerical backend, splitting a run into updates preserves the
    sampling sequence and accumulation order. Changing CPU worker count
    does not change those results. Changing a bound changes the sampling
    distribution for subsequent pairs.

    Examples
    --------
    This positional history uses rows before 80 for calibration. The
    current fitting window ends at row 99, after all calibration targets:

    >>> import numpy as np
    >>> from hoao import ReturnsPanel, calibrate
    >>> history = 0.01 * np.random.default_rng(2).normal(size=(100, 5))
    >>> calibration = calibrate(ReturnsPanel(history), n_assets=3,
    ...                         train_size=20, test_size=10, n_pairs=4,
    ...                         before=80, random_state=7)
    >>> model = calibration.fit(history[-20:, :3], as_of=99)
    >>> model.covariance.shape, model.cokurtosis.core.shape
    ((3, 3), (3, 3))
    """

    n_assets: int
    train_size: int
    test_size: int
    gap: int
    n_pairs: int
    before: int | np.datetime64
    latest_target: int | np.datetime64 | None
    missing: MissingPolicy

    def __init__(self, *, n_assets: int, train_size: int, test_size: int,
                 gap: int, seed: int, dated: bool, before: int | np.datetime64,
                 missing: MissingPolicy) -> None:
        self.n_assets = n_assets
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.n_pairs = 0
        self.before = before
        self.latest_target: int | np.datetime64 | None = None
        self.missing = missing
        self._seed = seed
        self._dated = dated
        self._sum2 = np.zeros(n_assets, dtype=np.float64)
        self._sum4 = np.zeros((n_assets, n_assets), dtype=np.float64)
        self._filled = np.empty((0, 3), dtype=np.int64)
        self._location_bounds: list[tuple[int, int, int, int]] = []

    @property
    def correlation_core(self) -> FloatArray:
        """Read-only AO coefficients indexed by decreasing training eigenvalue rank."""
        if self.n_pairs < 1:
            raise ValueError("Calibration is empty")
        return _readonly(self._sum2 / self.n_pairs)

    @property
    def cokurtosis_core(self) -> FloatArray:
        """Read-only HOAO even core, shaped (n_assets, n_assets)."""
        if self.n_pairs < 1:
            raise ValueError("Calibration is empty")
        return _readonly(self._sum4 / self.n_pairs)

    @property
    def filled_locations(self) -> NDArray[np.int64]:
        """Read-only zero-fill records: (pair_id, panel_row, panel_column)."""
        return _readonly(self._filled)

    @property
    def filled_count(self) -> int:
        """Number of target NaNs replaced with zero across all sampled pairs."""
        return len(self._filled)

    def _fitting_data(self, returns: ReturnInput, as_of: Any) -> tuple[Any, ...]:
        x, assets, dates = observations(returns, minimum=3)
        if x.shape != (self.train_size, self.n_assets):
            raise ValueError(f"This calibration requires exactly {self.train_size} observations and {self.n_assets} assets; received {x.shape}")
        last = None
        if dates is not None:
            if self._dated:
                last = _date(dates[-1])
            elif isinstance(dates[-1], (int, np.integer)):
                last = int(dates[-1])
            else:
                raise ValueError("Dated fitting observations require a dated calibration panel")
        if as_of is None:
            if last is None:
                raise ValueError("as_of is required for fitting arrays without dates")
            as_of = last
        signal = _date(as_of) if self._dated else _positive_int(as_of, "as_of", zero=True)
        if last is not None and signal < last:
            raise ValueError("as_of cannot precede the last fitting observation")
        if signal < self.before or self.latest_target is None or self.latest_target >= signal:
            raise ValueError("Calibration target history must end strictly before as_of, and as_of must be at least the calibration cutoff")
        z, mean, scales = standardize(x)
        return x, z, mean, scales, assets

    def fit(self, returns: ReturnInput, *, as_of: Cutoff | None = None,
            device: Device = "cpu") -> MomentModel:
        """Apply historical AO and HOAO coefficients to a current return window.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (train_size, n_assets)
            Current finite returns, with exactly the dimensions used in
            calibration and positive variance in every column. Asset
            identities may differ from the historical panel. DataFrame
            columns must be unique, non-missing asset labels, and its index
            must be unique and increasing.
        as_of : date-like or int, optional
            Application date, or absolute row position for a positional
            calibration. Required for arrays. For a DataFrame, it defaults
            to the final index entry, which must use the calibration's date
            convention. It must be at least the calibration cutoff and no
            earlier than the last fitting observation; all calibration
            targets must be strictly earlier.
        device : {'cpu', 'cuda'}, default='cpu'
            Backend for the current bases. Experimental CUDA support requires
            a compatible CuPy installation and a working NVIDIA driver.
            Results use NumPy float64 arrays with either backend.

        Returns
        -------
        MomentModel
            Current sample mean, AO correlation and covariance, HOAO
            cokurtosis, and raw equal-weight sample variance. Marginal scales
            use ``ddof=0``; outputs retain the input units and frequency.

        Raises
        ------
        TypeError
            If returns do not have a real numeric dtype.
        ValueError
            If returns, dimensions, labels, dates or `device` are invalid.
        ImportError
            If CUDA is requested but CuPy is not installed.
        RuntimeError
            If the requested CUDA device cannot be initialized.

        See Also
        --------
        fit_correlation : Apply only the second-order coefficients.
        fit_cokurtosis : Apply only the fourth-order coefficients.

        Notes
        -----
        The current window supplies separate bases for correlation and
        cokurtosis. Historical coefficients are assigned by rank in each
        basis. Correlation is rescaled congruently to unit diagonal before
        current marginal volatilities are restored. The fourth-order core
        retains its own diagonal; it is not normalized to one.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import ReturnsPanel, calibrate
        >>> history = 0.01 * np.random.default_rng(2).normal(size=(100, 5))
        >>> calibration = calibrate(ReturnsPanel(history), n_assets=3,
        ...                         train_size=20, test_size=10, n_pairs=4,
        ...                         before=80, random_state=7)
        >>> current = history[-20:, :3]
        >>> model = calibration.fit(current, as_of=99)
        >>> np.allclose(np.diag(model.covariance), current.var(axis=0))
        True
        """
        x, z, mean, scales, assets = self._fitting_data(returns, as_of)
        _, u2 = correlation_basis(z, device=device)
        corr = normalize_correlation((u2 * self.correlation_core) @ u2.T)
        c = CorrelationEstimate(corr, scales, assets=assets)
        k = CokurtosisEstimate(scales=scales, basis=fourth_basis(z, device=device),
                              core=self.cokurtosis_core, assets=assets)
        centered_equal_weight = (x - mean).mean(axis=1)
        reference_variance = float(np.var(centered_equal_weight, ddof=0))
        return MomentModel(mean, c, k, reference_variance=reference_variance, assets=assets)

    def fit_correlation(self, returns: ReturnInput, *, as_of: Cutoff | None = None,
                        device: Device = "cpu") -> CorrelationEstimate:
        """Apply AO coefficients to estimate current correlation and covariance.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (train_size, n_assets)
            Finite current returns with positive variance in every column.
            Dimensions and DataFrame labels must meet the requirements of
            :meth:`fit`.
        as_of : date-like or int, optional
            Application date or absolute row position. Required for arrays;
            otherwise inferred from the final DataFrame index entry. The
            chronology checks are the same as for :meth:`fit`.
        device : {'cpu', 'cuda'}, default='cpu'
            Backend for the current correlation basis. CUDA is experimental
            and requires CuPy and a working NVIDIA driver.

        Returns
        -------
        CorrelationEstimate
            Unit-diagonal correlation, current marginal standard deviations
            and covariance in squared return units. No fourth-order basis
            or tensor is computed.

        Raises
        ------
        TypeError
            If returns do not have a real numeric dtype.
        ValueError
            If fitting data, dimensions, labels, chronology or device are
            invalid.
        ImportError
            If CUDA is requested but CuPy is not installed.
        RuntimeError
            If the requested CUDA device cannot be initialized.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import ReturnsPanel, calibrate
        >>> history = np.random.default_rng(2).normal(size=(100, 5))
        >>> calibration = calibrate(ReturnsPanel(history), n_assets=3,
        ...                         train_size=20, test_size=10, n_pairs=4,
        ...                         before=80, random_state=7)
        >>> estimate = calibration.fit_correlation(history[-20:, :3], as_of=99)
        >>> np.allclose(np.diag(estimate.correlation), 1.0)
        True
        """
        _, z, _, scales, assets = self._fitting_data(returns, as_of)
        _, u = correlation_basis(z, device=device)
        return CorrelationEstimate(normalize_correlation((u * self.correlation_core) @ u.T),
                                   scales, assets=assets)

    def fit_cokurtosis(self, returns: ReturnInput, *, as_of: Cutoff | None = None,
                      device: Device = "cpu") -> CokurtosisEstimate:
        """Apply the HOAO even core in the current fourth-order basis.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (train_size, n_assets)
            Finite current returns with positive variance in every column.
            Dimensions and DataFrame labels must meet the requirements of
            :meth:`fit`.
        as_of : date-like or int, optional
            Application date or absolute row position. Required for arrays;
            otherwise inferred from the final DataFrame index entry. The
            chronology checks are the same as for :meth:`fit`.
        device : {'cpu', 'cuda'}, default='cpu'
            Backend for the current fourth-order basis. CUDA is experimental
            and requires CuPy and a working NVIDIA driver.

        Returns
        -------
        CokurtosisEstimate
            Current basis, historical even core and current marginal
            volatilities. Portfolio fourth moments use the original return
            units. Correlation estimation is skipped.

        Raises
        ------
        TypeError
            If returns do not have a real numeric dtype.
        ValueError
            If fitting data, dimensions, labels, chronology or device are
            invalid.
        ImportError
            If CUDA is requested but CuPy is not installed.
        RuntimeError
            If the requested CUDA device cannot be initialized.

        Examples
        --------
        The fitted estimate keeps the calibrated core and supplies a new
        basis for the current window:

        >>> import numpy as np
        >>> from hoao import ReturnsPanel, calibrate
        >>> history = np.random.default_rng(2).normal(size=(100, 5))
        >>> calibration = calibrate(ReturnsPanel(history), n_assets=3,
        ...                         train_size=20, test_size=10, n_pairs=4,
        ...                         before=80, random_state=7)
        >>> estimate = calibration.fit_cokurtosis(history[-20:, :3], as_of=99)
        >>> np.array_equal(estimate.core, calibration.cokurtosis_core)
        True
        >>> estimate.basis.shape
        (3, 3)
        """
        _, z, _, scales, assets = self._fitting_data(returns, as_of)
        return CokurtosisEstimate(scales=scales, basis=fourth_basis(z, device=device),
                                  core=self.cokurtosis_core, assets=assets)

    def update(self, panel: ReturnsPanel, *, n_pairs: int, before: Cutoff,
               start: Cutoff | None = None, n_jobs: int = 1,
               device: Device = "cpu") -> Calibration:
        """Add historical pairs to the existing calibration.

        Parameters
        ----------
        panel : ReturnsPanel
            History and availability for the new draws. It must use the same
            dated or positional convention as the original panel, but may
            contain different assets. The saved asset count, window lengths,
            gap and missing-return policy continue to apply.
        n_pairs : int
            Positive number of additional pairs. Sampling continues from the
            saved seed and next pair ID; earlier pairs retain their weight.
        before : date-like or int
            Exclusive cutoff for all new target observations. It must be no
            earlier than the previous cutoff. Use a date for a dated panel
            and an absolute row position for a positional panel.
        start : date-like or int, optional
            Inclusive lower bound for new signal dates. If omitted, sample
            from all eligible dates before the cutoff, including old dates.
            Set a later bound to restrict the new draws to a recent period;
            their training windows may begin before that bound.
        n_jobs : int, default=1
            Positive number of CPU workers. Results are accumulated in pair
            order, with BLAS limited to one thread during calibration. CUDA
            requires ``n_jobs=1``.
        device : {'cpu', 'cuda'}, default='cpu'
            Numerical backend. CUDA is experimental and requires CuPy and a
            working NVIDIA driver; it processes at most four pairs per batch.

        Returns
        -------
        Calibration
            This object, with new sums, count, cutoff and missing-return
            records. Changes are committed only after all new pairs succeed;
            a failed update leaves the existing calibration unchanged.

        Raises
        ------
        TypeError
            If `panel` is not a ReturnsPanel.
        ValueError
            If settings or date conventions are incompatible, no eligible
            pairs exist, or a target NaN is encountered with ``missing='raise'``.
        ImportError
            If CUDA is requested but CuPy is not installed.
        RuntimeError
            If the requested CUDA device cannot be initialized.

        See Also
        --------
        calibrate : Sampling rules and independent window standardization.
        save : Preserve the updated sums and sampling state.

        Examples
        --------
        Splitting eight draws into two calls gives the same cores as one
        eight-pair calibration when the panel, bounds and seed are unchanged:

        >>> import numpy as np
        >>> from hoao import ReturnsPanel, calibrate
        >>> panel = ReturnsPanel(np.random.default_rng(2).normal(size=(100, 5)))
        >>> settings = dict(n_assets=3, train_size=20, test_size=10,
        ...                 before=80, random_state=7)
        >>> calibration = calibrate(panel, n_pairs=4, **settings)
        >>> _ = calibration.update(panel, n_pairs=4, before=80, n_jobs=2)
        >>> uninterrupted = calibrate(panel, n_pairs=8, **settings)
        >>> calibration.n_pairs
        8
        >>> np.array_equal(calibration.cokurtosis_core, uninterrupted.cokurtosis_core)
        True
        """
        if not isinstance(panel, ReturnsPanel):
            raise TypeError("panel must be a ReturnsPanel")
        n_pairs = _positive_int(n_pairs, "n_pairs")
        n_jobs = _positive_int(n_jobs, "n_jobs")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda" and n_jobs != 1:
            raise ValueError("CUDA calibration requires n_jobs=1 to bound device memory")
        if self._dated != (panel.dates is not None):
            raise ValueError("Panel and calibration must use the same dated or positional convention")
        cutoff = panel._cutoff(before)
        if cutoff < self.before:
            raise ValueError("An update cutoff cannot precede the previous cutoff")
        signals, eligible = panel._signals(n_assets=self.n_assets,
            train_size=self.train_size, test_size=self.test_size, gap=self.gap,
            before=cutoff, start=start)
        values = panel.values
        dates = panel.dates

        def select(identifier: int) -> tuple[Any, ...]:
            rng = np.random.default_rng(np.random.SeedSequence([self._seed, identifier]))
            signal = int(rng.choice(signals))
            columns = np.sort(rng.choice(np.flatnonzero(eligible[signal]), self.n_assets, replace=False))
            train = values[signal - self.train_size + 1:signal + 1, columns]
            first = signal + self.gap + 1
            target = values[first:first + self.test_size, columns].copy()
            missing = np.argwhere(np.isnan(target))
            if len(missing) and self.missing == "raise":
                row, col = missing[0]
                label = first + int(row) if dates is None else dates[first + row]
                raise ValueError(f"Missing target return for asset {panel.assets[columns[col]]!r} at {label}; selected assets were not replaced")
            target[np.isnan(target)] = 0.0
            records = np.empty((len(missing), 3), dtype=np.int64)
            if len(missing):
                records[:, 0] = identifier
                records[:, 1] = first + missing[:, 0]
                records[:, 2] = columns[missing[:, 1]]
            last_row = first + self.test_size - 1
            latest = last_row if dates is None else dates[last_row]
            return train, target, records, latest

        def one(identifier: int) -> tuple[Any, ...]:
            train, target, records, latest = select(identifier)
            a, b = pair_core(train, target, device=device, order="both")
            return a, b, records, latest

        sum2, sum4 = self._sum2.copy(), self._sum4.copy()
        records = [self._filled]
        latest = self.latest_target

        def accumulate(result: tuple[Any, ...]) -> None:
            nonlocal latest
            a, b, filled, last = result
            sum2[:] += a
            sum4[:] += b
            if len(filled):
                records.append(filled)
            latest = last if latest is None else max(latest, last)

        context = nullcontext()
        if device == "cpu":
            from threadpoolctl import threadpool_limits
            context = threadpool_limits(limits=1, user_api="blas")
        with context:
            if device == "cuda":
                for begin in range(self.n_pairs, self.n_pairs + n_pairs, 4):
                    end = min(begin + 4, self.n_pairs + n_pairs)
                    selected = [select(identifier) for identifier in range(begin, end)]
                    a, b = pair_cores_batch(np.stack([item[0] for item in selected]),
                                           np.stack([item[1] for item in selected]),
                                           device="cuda", order="both")
                    if a is None or b is None:
                        raise RuntimeError("Both calibration cores were requested")
                    for index, item in enumerate(selected):
                        accumulate((a[index], b[index], item[2], item[3]))
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    for begin in range(self.n_pairs, self.n_pairs + n_pairs, 2 * n_jobs):
                        end = min(begin + 2 * n_jobs, self.n_pairs + n_pairs)
                        futures = [executor.submit(one, i) for i in range(begin, end)]
                        for future in futures:
                            accumulate(future.result())
        if latest is None or latest >= cutoff:
            raise RuntimeError("Calibration target reaches its exclusive cutoff")
        if not np.isfinite(sum2).all() or not np.isfinite(sum4).all():
            raise ValueError("Calibration overflowed; check the input return scales")
        self._sum2, self._sum4 = sum2, sum4
        self._filled = np.concatenate(records, axis=0)
        self._location_bounds.append((self.n_pairs, self.n_pairs + n_pairs,
                                      len(panel), len(panel.assets)))
        self.n_pairs += n_pairs
        self.before, self.latest_target = cutoff, latest
        return self

    def save(self, path: str | Path) -> None:
        """Save the calibration coefficients and the state needed to resume it.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination filename, conventionally ending in ``.npz``. The
            exact name is used; no extension is appended. An existing file
            at this path is replaced only after the new file is complete.

        Raises
        ------
        ValueError
            If the calibration contains no pairs.
        OSError
            If the destination cannot be written or replaced.

        See Also
        --------
        load : Restore a saved calibration and validate its contents.

        Notes
        -----
        The NPZ archive contains numeric arrays and versioned JSON metadata,
        with no pickle. It preserves sums rather than only averages, so a
        later update gives old and new pairs equal weight. The return panel
        itself is not saved. Writes use a temporary file in the destination
        directory and atomic replacement; a failed write preserves any
        previously saved calibration at that path.

        Examples
        --------
        >>> import numpy as np
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from hoao import Calibration, ReturnsPanel, calibrate
        >>> panel = ReturnsPanel(np.random.default_rng(2).normal(size=(100, 5)))
        >>> calibration = calibrate(panel, n_assets=3, train_size=20,
        ...                         test_size=10, n_pairs=4, before=80,
        ...                         random_state=7)
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "calibration.npz"
        ...     calibration.save(path)
        ...     restored = Calibration.load(path)
        >>> restored.n_pairs
        4
        >>> np.array_equal(restored.cokurtosis_core, calibration.cokurtosis_core)
        True
        """
        if self.n_pairs < 1 or self.latest_target is None:
            raise ValueError("Cannot save an empty calibration")
        metadata = dict(schema=_SCHEMA, basis=BASIS_CONVENTION,
            n_assets=self.n_assets, train_size=self.train_size,
            test_size=self.test_size, gap=self.gap, n_pairs=self.n_pairs,
            seed=self._seed, dated=self._dated,
            before=str(self.before) if self._dated else int(self.before),
            latest_target=str(self.latest_target) if self._dated else int(self.latest_target),
            missing=self.missing, filled_count=self.filled_count,
            location_bounds=self._location_bounds)
        destination = Path(path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                                    sum2=self._sum2, sum4=self._sum4,
                                    filled_locations=self._filled)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        """Restore and validate a calibration saved with :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            NPZ file produced by a compatible schema and basis convention.

        Returns
        -------
        Calibration
            Restored coefficient sums, pair count, sampling seed, dimensions,
            cutoffs and missing-return records. It can be used immediately
            for fitting or extended by :meth:`update` with a return panel.

        Raises
        ------
        ValueError
            If the file's schema or basis convention is incompatible, or its
            metadata, array values, dimensions, counts or chronology fail
            validation. Object arrays and pickle are not accepted.
        OSError
            If the file cannot be opened or read.

        Examples
        --------
        A restored calibration continues the original pair sequence:

        >>> import numpy as np
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from hoao import Calibration, ReturnsPanel, calibrate
        >>> panel = ReturnsPanel(np.random.default_rng(2).normal(size=(100, 5)))
        >>> calibration = calibrate(panel, n_assets=3, train_size=20,
        ...                         test_size=10, n_pairs=4, before=80,
        ...                         random_state=7)
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "calibration.npz"
        ...     calibration.save(path)
        ...     restored = Calibration.load(path)
        >>> _ = restored.update(panel, n_pairs=2, before=80)
        >>> restored.n_pairs, calibration.n_pairs
        (6, 4)
        """
        try:
            with np.load(path, allow_pickle=False) as state:
                if set(state.files) != _MEMBERS:
                    raise ValueError("Unexpected calibration checkpoint members")
                raw = state["metadata"]
                if raw.shape != () or raw.dtype.kind not in "US":
                    raise ValueError("Checkpoint metadata must be a JSON string")
                metadata = json.loads(str(raw.item()))
                if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
                    raise ValueError("Unexpected calibration metadata fields")
                if type(metadata["schema"]) is not int or metadata["schema"] != _SCHEMA:
                    raise ValueError("Unsupported calibration schema version")
                if metadata["basis"] != BASIS_CONVENTION:
                    raise ValueError("Incompatible calibration basis convention")
                n = _positive_int(metadata["n_assets"], "n_assets")
                train = _positive_int(metadata["train_size"], "train_size")
                test = _positive_int(metadata["test_size"], "test_size")
                gap = _positive_int(metadata["gap"], "gap", zero=True)
                count = _positive_int(metadata["n_pairs"], "n_pairs")
                seed = _positive_int(metadata["seed"], "seed", zero=True)
                filled_count = _positive_int(metadata["filled_count"], "filled_count", zero=True)
                if min(train, test) < 3:
                    raise ValueError("Calibration windows must each contain at least three observations")
                if type(metadata["dated"]) is not bool or metadata["missing"] not in ("zero", "raise"):
                    raise ValueError("Invalid calibration date convention or missing policy")
                dated = metadata["dated"]
                before = _date(metadata["before"]) if dated else _positive_int(metadata["before"], "before", zero=True)
                latest = _date(metadata["latest_target"]) if dated else _positive_int(metadata["latest_target"], "latest_target", zero=True)
                if latest >= before:
                    raise ValueError("Calibration targets must end strictly before its cutoff")
                a, b, filled = state["sum2"].copy(), state["sum4"].copy(), state["filled_locations"].copy()
                if a.dtype != np.float64 or b.dtype != np.float64 or a.shape != (n,) or b.shape != (n, n):
                    raise ValueError("Invalid calibration array dtype or shape")
                if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
                    raise ValueError("Calibration sums must be finite and nonnegative")
                if not np.allclose(b, b.T, rtol=1e-12, atol=1e-14):
                    raise ValueError("Fourth-order calibration sum must be symmetric")
                tolerance = 1e-10 * max(1.0, float(np.linalg.norm(b, ord=2)))
                if np.linalg.eigvalsh(b).min() < -tolerance:
                    raise ValueError("Fourth-order calibration sum must be positive semidefinite")
                if filled.dtype != np.int64 or filled.shape != (filled_count, 3) or np.any(filled < 0):
                    raise ValueError("Invalid missing-return location records")
                if filled_count:
                    if (metadata["missing"] != "zero" or np.any(filled[:, 0] >= count)
                            or np.any(np.diff(filled[:, 0]) < 0)):
                        raise ValueError("Missing-return records conflict with calibration count or policy")
                    if len(np.unique(filled, axis=0)) != filled_count:
                        raise ValueError("Duplicate missing-return records")
                bounds = metadata["location_bounds"]
                if not isinstance(bounds, list) or not bounds:
                    raise ValueError("Missing panel location bounds")
                next_pair = 0
                checked_bounds: list[tuple[int, int, int, int]] = []
                for item in bounds:
                    if not isinstance(item, list) or len(item) != 4:
                        raise ValueError("Malformed panel location bounds")
                    first = _positive_int(item[0], "first pair", zero=True)
                    stop = _positive_int(item[1], "stop pair")
                    rows = _positive_int(item[2], "panel rows")
                    cols = _positive_int(item[3], "panel columns")
                    if first != next_pair or stop <= first or rows < train + gap + test or cols < n:
                        raise ValueError("Panel location bounds conflict with calibration dimensions or count")
                    selected = filled[(filled[:, 0] >= first) & (filled[:, 0] < stop)]
                    if len(selected) and (np.any(selected[:, 1] >= rows)
                            or np.any(selected[:, 1] < train + gap)
                            or np.any(selected[:, 2] >= cols)):
                        raise ValueError("Filled observation lies outside its source panel")
                    checked_bounds.append((first, stop, rows, cols))
                    next_pair = stop
                if next_pair != count:
                    raise ValueError("Panel location bounds disagree with calibration count")
                if not dated and (latest < train + gap + test - 1
                                   or (filled_count and np.any(filled[:, 1] > latest))):
                    raise ValueError("Target locations conflict with latest_target")
                result = cls(n_assets=n, train_size=train, test_size=test, gap=gap,
                             seed=seed, dated=dated, before=before, missing=metadata["missing"])
                result._sum2, result._sum4, result._filled = a, b, filled
                result._location_bounds = checked_bounds
                result.n_pairs, result.latest_target = count, latest
                return result
        except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
            raise ValueError("Malformed calibration checkpoint") from exc


def calibrate(panel: ReturnsPanel, *, n_assets: int, train_size: int,
              test_size: int, n_pairs: int, before: Cutoff, start: Cutoff | None = None,
              gap: int = 1, missing: MissingPolicy = "zero",
              random_state: int | None = None, n_jobs: int = 1,
              device: Device = "cpu") -> Calibration:
    """Estimate AO correlation and HOAO cokurtosis coefficients from history.

    Each draw selects a signal date and an asset subset, estimates bases
    from the trailing training window, and measures coefficients in the
    subsequent target window. The returned calibration averages these
    coefficients by rank for application to later return windows.

    Parameters
    ----------
    panel : ReturnsPanel
        Return history and asset availability known on each date. A
        candidate asset must have a complete, finite, nonconstant
        training history ending on the selected signal date.
    n_assets : int
        Positive number of assets selected uniformly without replacement
        on each date. The selected assets remain fixed throughout the
        target window. Calibration depends on this count, not on the
        particular asset identities.
    train_size, test_size : int
        Observations in the training and target windows, each at least
        three. Training ends on the signal date. Each window is centered
        and standardized independently with ``ddof=0``. A constant
        target column is retained as zero after standardization.
    n_pairs : int
        Positive number of historical pairs. Each contributes equally
        to both coefficient averages.
    before : date-like or int
        Exclusive upper bound for every target observation. Use a date
        for a dated panel and an absolute row position otherwise. Later
        application dates must be at least this cutoff.
    start : date-like or int, optional
        Inclusive lower bound for signal dates. Training windows may
        begin earlier. If omitted, all dates with enough training history
        and target dates before the cutoff are considered.
    gap : int, default=1
        Nonnegative number of observations skipped between windows. With
        a signal on row ``t``, the first target row is ``t + gap + 1``.
        The default skips the execution day of the manuscript's protocol;
        zero makes the two windows adjacent.
    missing : {'zero', 'raise'}, default='zero'
        Policy for NaNs in the selected target returns. ``'zero'`` uses
        the manuscript's zero-return convention and records each filled
        location. ``'raise'`` stops at a missing target return without
        choosing replacement assets or dates. Missing training returns
        always exclude the asset, and the panel rejects infinities.
    random_state : int or None, default=None
        Nonnegative sampling seed. None generates a fresh seed. Each pair
        uses a stream determined by the seed and its sequential pair ID;
        both the seed and pair count are retained for continuation.
    n_jobs : int, default=1
        Positive number of CPU workers. At most twice this many pairs
        are in flight, BLAS uses one thread during calibration, and
        results are accumulated in pair order. CUDA requires one worker.
    device : {'cpu', 'cuda'}, default='cpu'
        Numerical backend. Experimental CUDA support requires CuPy and a
        working NVIDIA driver, and processes up to four pairs per batch.
        Public results are NumPy float64 arrays with either backend.

    Returns
    -------
    Calibration
        Averaged second- and fourth-order coefficients with the sums,
        count, dimensions, cutoffs and missing-return records needed for
        fitting, inspection, persistence and subsequent updates.

    Raises
    ------
    TypeError
        If `panel` is not a ReturnsPanel.
    ValueError
        If parameters are invalid, no admissible dates have enough
        eligible assets, or a selected target contains NaN with
        ``missing='raise'``.
    ImportError
        If CUDA is requested but CuPy is not installed.
    RuntimeError
        If the requested CUDA device cannot be initialized.

    See Also
    --------
    ReturnsPanel : Availability formats and trailing-history eligibility.
    Calibration.fit : Apply the coefficients to a current return window.
    CV : Estimate coefficients from random splits of one current window.

    Notes
    -----
    Dates are drawn uniformly with replacement from those meeting the
    training-history and calendar requirements. Assets are then drawn
    uniformly within that date's eligible universe. Target return values
    affect neither selection step. The same windows supply both orders,
    using separate correlation and fourth-order bases.

    Zero filling is an explicit modeling convention for missing target
    data; a NaN alone does not establish that a stock exited the universe.
    Use the stored count and locations to inspect how often it applies.

    Examples
    --------
    With a dated DataFrame, the application date is inferred from the
    last row of the current window:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hoao import ReturnsPanel, calibrate
    >>> history = pd.DataFrame(
    ...     0.01 * np.random.default_rng(10).normal(size=(120, 4)),
    ...     index=pd.bdate_range("2020-01-01", periods=120),
    ...     columns=["A", "B", "C", "D"],
    ... )
    >>> calibration = calibrate(ReturnsPanel(history), n_assets=3,
    ...                         train_size=20, test_size=10, n_pairs=5,
    ...                         before=history.index[80], random_state=4)
    >>> model = calibration.fit(history.iloc[-20:, :3])
    >>> model.assets
    ('A', 'B', 'C')
    >>> calibration.n_pairs, calibration.filled_count
    (5, 0)
    """
    if not isinstance(panel, ReturnsPanel):
        raise TypeError("panel must be a ReturnsPanel")
    n_assets = _positive_int(n_assets, "n_assets")
    train_size = _positive_int(train_size, "train_size")
    test_size = _positive_int(test_size, "test_size")
    gap = _positive_int(gap, "gap", zero=True)
    if min(train_size, test_size) < 3:
        raise ValueError("train_size and test_size must each be at least three")
    if n_assets > len(panel.assets):
        raise ValueError("n_assets exceeds the panel's asset count")
    if missing not in ("zero", "raise"):
        raise ValueError("missing must be 'zero' or 'raise'")
    result = Calibration(n_assets=n_assets, train_size=train_size, test_size=test_size,
                         gap=gap, seed=_seed(random_state), dated=panel.dates is not None,
                         before=panel._cutoff(before), missing=missing)
    return result.update(panel, n_pairs=n_pairs, before=before, start=start,
                         n_jobs=n_jobs, device=device)
