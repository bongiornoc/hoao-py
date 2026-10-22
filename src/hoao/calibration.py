"""Historical AO/HOAO calibration with reproducible sampling and safe state."""
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
    """Reusable historical AO and HOAO cores, averaged by spectral rank.

    Create instances with :func:`calibrate` or :meth:`load`. Calibration is
    specific to its asset count and training-window length, but not to asset
    identities. Each historical pair contributes equally, including after
    updates and save/load cycles. All arithmetic uses float64.

    Attributes
    ----------
    n_assets, train_size, test_size, gap : int
        Sampling dimensions. ``gap`` counts sessions omitted between the
        training and target windows; the default of one skips execution day.
    n_pairs : int
        Number of accumulated historical pairs, also the next sampling ID.
    correlation_core : numpy.ndarray of shape (n_assets,)
        Mean target variances along the training correlation eigenvectors.
    cokurtosis_core : numpy.ndarray of shape (n_assets, n_assets)
        Mean cross-moments of squared target coordinates in the training
        fourth-moment basis. This is an even core, not a dense fourth tensor.
    before : int or numpy.datetime64
        Exclusive cutoff supplied at the latest calibration/update. Applying
        the core at an earlier signal date is rejected conservatively.
    latest_target : int or numpy.datetime64
        Latest target observation actually used; always strictly before
        ``before`` and the signal date used for fitting.
    filled_count : int
        Count of NaN target observations explicitly replaced by zero.
    filled_locations : numpy.ndarray of int64, shape (filled_count, 3)
        Columns are sampling pair ID, absolute panel row, and panel column.
        Rows preserve pair order. Positions refer to the panel supplied for
        that pair; they do not claim that a missing return was an economic
        exit. Updating with a different panel requires retaining that panel's
        axes if these locations will be audited later.

    Notes
    -----
    A saved core contains arrays and JSON metadata, never executable pickle.
    The random seed and next pair ID are saved. For the same panel, cutoff,
    and optional start bound, splitting a calibration into several updates
    produces exactly the same ordered sums as one uninterrupted run, even
    when the CPU worker count changes. Changing the sampling bounds changes
    the sampling distribution for the new pair IDs.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import ReturnsPanel, calibrate
    >>> returns = np.random.default_rng(2).normal(size=(120, 3)) * 0.01
    >>> panel = ReturnsPanel(returns)
    >>> core = calibrate(panel, n_assets=3, train_size=20, test_size=10,
    ...                  n_pairs=4, before=90, random_state=7)
    >>> fitted = core.fit(returns[80:100], as_of=99)
    >>> fitted.correlation.shape
    (3, 3)
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
        """Read-only mean AO eigenvalues, ordered by decreasing sample rank."""
        if self.n_pairs < 1:
            raise ValueError("Calibration is empty")
        return _readonly(self._sum2 / self.n_pairs)

    @property
    def cokurtosis_core(self) -> FloatArray:
        """Read-only mean HOAO even-core matrix, without an n**4 allocation."""
        if self.n_pairs < 1:
            raise ValueError("Calibration is empty")
        return _readonly(self._sum4 / self.n_pairs)

    @property
    def filled_locations(self) -> NDArray[np.int64]:
        """Read-only (pair ID, panel row, panel column) zero-fill records."""
        return _readonly(self._filled)

    @property
    def filled_count(self) -> int:
        """Number of target NaNs replaced by the explicit zero-return policy."""
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
        """Fit both moments using the historical cores and current scales.

        Parameters
        ----------
        returns : pandas.DataFrame or array_like, shape (train_size, n_assets)
            Current finite return observations in chronological order. Their
            asset identities may differ from the historical calibration's.
            Each column must have positive variance. Sample moments use
            ``ddof=0`` and are returned in the supplied return units.
        as_of : date-like or int, optional
            Signal date (or absolute row for a positional calibration).
            Inferred from the final index entry for a dated DataFrame;
            required for arrays. It must not precede any fitting observation
            or the calibration cutoff. All historical targets must be earlier.
        device : {"cpu", "cuda"}, default="cpu"
            Compute backend for the bases. CUDA requires the optional GPU
            dependency; returned public arrays are NumPy float64 arrays.

        Returns
        -------
        MomentModel
            Sample mean, AO correlation/covariance, HOAO even cokurtosis, and
            the raw equal-weight sample variance used for optimization.

        Examples
        --------
        >>> # core was calibrated using only observations before row 90.
        >>> model = core.fit(returns[80:100], as_of=99)  # doctest: +SKIP
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
        """Fit only AO correlation and covariance, skipping fourth-order work.

        Parameters
        ----------
        returns : pandas.DataFrame or array_like, shape (train_size, n_assets)
            Current finite returns with positive variance in every column.
        as_of : date-like or int, optional
            Signal date, inferred from a dated DataFrame's final row. Required
            for arrays. The same causal checks as :meth:`fit` apply.
        device : {"cpu", "cuda"}, default="cpu"
            Backend for the current correlation basis.

        Returns
        -------
        CorrelationEstimate
            Unit-diagonal correlation and covariance in current return units.

        Examples
        --------
        >>> c = core.fit_correlation(returns[80:100], as_of=99)  # doctest: +SKIP
        """
        _, z, _, scales, assets = self._fitting_data(returns, as_of)
        _, u = correlation_basis(z, device=device)
        return CorrelationEstimate(normalize_correlation((u * self.correlation_core) @ u.T),
                                   scales, assets=assets)

    def fit_cokurtosis(self, returns: ReturnInput, *, as_of: Cutoff | None = None,
                      device: Device = "cpu") -> CokurtosisEstimate:
        """Fit only HOAO even cokurtosis, skipping correlation estimation.

        Parameters
        ----------
        returns : pandas.DataFrame or array_like, shape (train_size, n_assets)
            Current finite returns with positive variance in every column.
        as_of : date-like or int, optional
            Signal date, inferred from a dated DataFrame's final row. Required
            for arrays. The same causal checks as :meth:`fit` apply.
        device : {"cpu", "cuda"}, default="cpu"
            Backend for the current fourth-moment basis.

        Returns
        -------
        CokurtosisEstimate
            Compact fourth central moments in current return units. The
            standardized dependence tensor is never rescaled to unit diagonal.

        Examples
        --------
        >>> k = core.fit_cokurtosis(returns[80:100], as_of=99)  # doctest: +SKIP
        """
        _, z, _, scales, assets = self._fitting_data(returns, as_of)
        return CokurtosisEstimate(scales=scales, basis=fourth_basis(z, device=device),
                                  core=self.cokurtosis_core, assets=assets)

    def update(self, panel: ReturnsPanel, *, n_pairs: int, before: Cutoff,
               start: Cutoff | None = None, n_jobs: int = 1,
               device: Device = "cpu") -> Calibration:
        """Add historical pairs without resetting accumulated sums or counts.

        Parameters
        ----------
        panel : ReturnsPanel
            Return history and point-in-time universe. Its date convention
            must match the original calibration. Asset identities may differ.
        n_pairs : int
            Number of additional pairs, sampled with the saved seed and the
            next consecutive pair IDs.
        before : date-like or int
            Exclusive target cutoff, no earlier than the previous cutoff.
        start : date-like or int, optional
            Inclusive lower bound for signal dates. None uses all available
            history; specify the previous update date to sample only recent
            signal dates. Training history may precede this bound.
        n_jobs : int, default=1
            CPU workers. Work in flight is bounded by twice this number, BLAS
            threads are capped during concurrent work, and results are added
            in pair order. CUDA requires one worker.
        device : {"cpu", "cuda"}, default="cpu"
            Backend for each pair. GPU work uses batches of at most four pairs
            to bound memory while sharing transfers and linear algebra.

        Returns
        -------
        Calibration
            This object, updated only after every requested pair succeeds.
            A failure leaves the existing calibration unchanged.

        Examples
        --------
        >>> core.update(panel, n_pairs=20, before=120)  # doctest: +SKIP
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
        """Save accumulated state as NPZ arrays and versioned JSON metadata.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. The exact supplied name is used, preferably
            ending in ``.npz``. A temporary sibling file is flushed and synced
            before atomic replacement; failed writes preserve an existing file.

        Examples
        --------
        >>> core.save("historical-core.npz")  # doctest: +SKIP
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
        """Load and validate a non-pickle calibration checkpoint.

        Parameters
        ----------
        path : str or pathlib.Path
            Checkpoint produced by :meth:`save` using this schema and basis
            convention. Incompatible versions and malformed state are rejected.

        Returns
        -------
        Calibration
            Historical sums, exact count, sampling seed, and causal cutoffs.

        Raises
        ------
        ValueError
            If metadata, shapes, counts, finite values, or cutoff chronology
            are invalid. Object arrays and pickle are never accepted.

        Examples
        --------
        >>> from hoao import Calibration
        >>> core = Calibration.load("historical-core.npz")  # doctest: +SKIP
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
    """Calibrate AO correlation and HOAO cokurtosis from a return panel.

    Parameters
    ----------
    panel : ReturnsPanel
        Return history and optional daily available stocks. Only complete,
        nonconstant training histories can enter a sampled asset subset.
    n_assets : int
        Assets sampled uniformly without replacement on each signal date.
        Calibration is specific to this count, not to asset identities.
    train_size, test_size : int
        Observations in each training and subsequent target window, each at
        least three. Training ends on the sampled signal date. Each window is
        centered and standardized using its own population standard deviations.
        A constant target component is retained as zero after standardization.
    n_pairs : int
        Number of historical pairs. Each pair has equal weight in both cores.
    before : date-like or int
        Exclusive upper bound for *all target observations*. Use a date for a
        dated panel and an absolute row position otherwise. This cutoff is
        checked when the calibrated core is subsequently applied.
    start : date-like or int, optional
        Inclusive lower bound for signal dates. Training history may precede
        it. None uses all signal dates with enough available history.
    gap : int, default=1
        Sessions skipped between training and target windows. With one, a
        signal after session t's close is executed on t+1 and targets start
        on t+2. Zero makes the windows adjacent.
    missing : {"zero", "raise"}, default="zero"
        Policy for target NaNs **after** selecting stocks. "zero" reproduces
        the paper's explicit zero-return marking convention and records every
        filled location. "raise" stops without substituting stocks or dates.
        Missing training returns always exclude that asset from eligibility;
        infinity is always an error. Zero filling does not establish an exit.
    random_state : int or None, default=None
        Nonnegative sampling seed. Each pair has an independent stream based
        on its stable ID. The seed is saved for exact continuation.
    n_jobs : int, default=1
        Number of bounded CPU workers; results are accumulated in pair order.
        CUDA requires one worker. Concurrent CPU work limits BLAS to one thread
        per worker to avoid oversubscription.
    device : {"cpu", "cuda"}, default="cpu"
        Numerical backend. CUDA needs the optional CuPy dependency and a GPU;
        pairs run in batches of at most four to bound memory. Public outputs
        remain NumPy.

    Returns
    -------
    Calibration
        Historical compact cores, dimensions, count, cutoff, and fill records.

    Notes
    -----
    Signal dates are uniform among dates meeting training-history and calendar
    constraints. Assets are uniform within that date's eligible universe.
    Target values cannot change either selection. No annualization is applied.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import ReturnsPanel, calibrate
    >>> data = np.random.default_rng(10).normal(size=(100, 4)) * 0.01
    >>> core = calibrate(ReturnsPanel(data), n_assets=3, train_size=20,
    ...                  test_size=10, n_pairs=5, before=80, random_state=4)
    >>> core.cokurtosis_core.shape
    (3, 3)
    >>> core.n_pairs
    5
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
