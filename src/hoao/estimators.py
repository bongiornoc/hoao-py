"""Sample and cross-validated estimators of correlation and cokurtosis.

Each estimator accepts a return matrix with one row per observation and one
column per asset. ``fit`` returns a :class:`~hoao.MomentModel` holding both
moments; ``fit_correlation`` and ``fit_cokurtosis`` estimate only the
second- or fourth-order component.
"""

from __future__ import annotations

from collections.abc import Hashable
from copy import deepcopy

import numpy as np

from ._backend import array_namespace, as_host, validate_device
from ._numerics import (
    MomentOrder,
    correlation_basis,
    fourth_basis,
    normalize_correlation,
    pair_core,
    standardize,
)
from ._typing import Device, FloatArray, ReturnInput
from ._validation import observations
from .models import CorrelationEstimate, CokurtosisEstimate, MomentModel


def _reference_variance(x: FloatArray, mean: FloatArray) -> float:
    portfolio = (x - mean).mean(axis=1)
    return float(np.var(portfolio))


class Sample:
    """Sample estimator of correlation and cokurtosis.

    Means, variances and correlations are the sample moments of the fitting
    window. The fourth central moment is either the full sample moment or,
    with ``even=True``, its even-core projection in a basis estimated from
    the same window (the ``MLE_even`` estimator of the accompanying
    manuscript).

    Parameters
    ----------
    even : bool, default=False
        If False, keep the sample fourth central moment exactly; it is
        stored as the centered observations. If True, keep only its even
        core, as defined in the Notes. The correlation and covariance do not
        depend on this setting.
    device : {'cpu', 'cuda'}, default='cpu'
        Keyword-only. Array backend for matrix products and
        eigendecompositions. ``'cuda'`` is experimental and requires a
        separately installed CuPy build that matches the local CUDA runtime;
        fitting raises ImportError if CuPy is missing and RuntimeError if no
        CUDA device can be initialized. Results are NumPy float64 arrays in
        host memory with either backend. For small windows the CPU is
        usually faster.

    Raises
    ------
    TypeError
        If `even` is not a Python or NumPy boolean.
    ValueError
        If `device` is neither ``'cpu'`` nor ``'cuda'``.

    See Also
    --------
    CV : Cross-validated estimator with the same interface.
    calibrate : Estimator with coefficients calibrated on historical windows.

    Notes
    -----
    For a window of ``T`` observations of ``n`` assets, every central moment
    divides by ``T`` (``ddof=0``); the covariance therefore equals
    ``numpy.cov(returns, rowvar=False, ddof=0)``. The fourth central moment
    is not converted to excess kurtosis.

    For ``even=True``, the returns are standardized,
    ``Z = (X - mean) / std``, and rotated into the basis ``U4`` of the
    higher-order singular value decomposition (HOSVD) of their fourth-moment
    tensor: the left singular vectors of the tensor reshaped to an
    ``(n, n**3)`` matrix, ordered by decreasing singular value. With
    ``Y = Z @ U4``, the even core is the ``(n, n)`` matrix
    ``B[i, j] = mean(Y[:, i]**2 * Y[:, j]**2)``. It contains the rotated
    tensor entries in which each index occurs an even number of times,
    ``(i, i, i, i)`` and the permutations of ``(i, i, j, j)``. These entries
    are unchanged when a basis vector is negated; all other rotated entries
    are set to zero.

    No dense ``n**4`` tensor is formed. The sample representation stores
    ``T * n`` values and the even representation ``O(n**2)``. ``U4`` is
    obtained from an ``(n, n)`` Gram matrix in
    ``O(T**2 * n + T * n**2 + n**3)`` operations. If ``Z`` is rank
    deficient, as when ``n >= T``, the leading basis vectors span its row
    space and the remaining ones complete it deterministically. Numerically
    repeated singular values are also resolved by a deterministic
    convention. The even core does not depend on the signs of the basis
    vectors, but it can depend on the basis chosen within a repeated
    singular subspace.

    Unlike scikit-learn estimators, ``fit`` returns a new
    :class:`~hoao.MomentModel`, not the estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import Sample
    >>> rng = np.random.default_rng(7)
    >>> returns = 0.01 * rng.standard_t(df=5, size=(250, 4))
    >>> model = Sample().fit(returns)
    >>> np.allclose(model.covariance, np.cov(returns, rowvar=False, ddof=0))
    True

    The even estimate has the same covariance, but stores a ``(4, 4)``
    basis and core in place of the 250 centered observations:

    >>> even_model = Sample(even=True).fit(returns)
    >>> np.allclose(even_model.covariance, model.covariance)
    True
    >>> model.cokurtosis.centered.shape, even_model.cokurtosis.core.shape
    ((250, 4), (4, 4))
    """

    even: bool
    device: Device

    def __init__(self, even: bool = False, *, device: Device = "cpu") -> None:
        if not isinstance(even, (bool, np.bool_)):
            raise TypeError("even must be a boolean.")
        validate_device(device)
        self.even = bool(even)
        self.device: Device = device

    def _correlation(
        self, z: FloatArray, scales: FloatArray, assets: tuple[Hashable, ...] | None
    ) -> CorrelationEstimate:
        namespace = array_namespace(self.device)
        data = namespace.asarray(z)
        c = as_host(data.T @ data / len(z), namespace)
        return CorrelationEstimate(normalize_correlation(c), scales, assets=assets)

    def _cokurtosis(
        self,
        x: FloatArray,
        z: FloatArray,
        mean: FloatArray,
        scales: FloatArray,
        assets: tuple[Hashable, ...] | None,
    ) -> CokurtosisEstimate:
        if not self.even:
            return CokurtosisEstimate(scales=scales, centered=z * scales, assets=assets)
        basis = fourth_basis(z, self.device)
        namespace = array_namespace(self.device)
        squared = (namespace.asarray(z) @ namespace.asarray(basis)) ** 2
        core = as_host(squared.T @ squared / len(z), namespace)
        return CokurtosisEstimate(scales=scales, basis=basis, core=core, assets=assets)

    def fit(self, returns: ReturnInput) -> MomentModel:
        """Estimate the mean, covariance and fourth central moment.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
            Real-valued returns, one row per observation and one column per
            asset. At least three rows are required; all values must be
            finite, and every column must vary. Missing values are rejected,
            not imputed. For a DataFrame, the columns become the asset
            labels and must be unique, hashable and non-missing, and the
            index must be unique and increasing.

        Returns
        -------
        MomentModel
            The sample mean, covariance and correlation, the fourth central
            moment selected by `even`, and the variance of the equal-weight
            portfolio over the window, stored as ``reference_variance`` and
            used to scale the optimization objective. Moments are in the
            units and sampling frequency of `returns`; nothing is annualized.

        Raises
        ------
        TypeError
            If `returns` does not have an integer or floating-point dtype.
        ValueError
            If `returns` does not meet the requirements above.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> rng = np.random.default_rng(0)
        >>> returns = 0.01 * rng.standard_t(df=5, size=(500, 3))
        >>> model = Sample().fit(returns)

        Portfolio moments equal those of the realized portfolio returns:

        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> realized = returns @ weights
        >>> moments = model.portfolio(weights)
        >>> bool(np.isclose(moments.variance, np.var(realized)))
        True
        >>> centered = realized - realized.mean()
        >>> bool(np.isclose(moments.kurtosis, np.mean(centered**4) / np.var(realized)**2))
        True

        The reference variance is that of the equal-weight portfolio:

        >>> bool(np.isclose(model.reference_variance, np.var(returns.mean(axis=1))))
        True
        """
        x, assets, _ = observations(returns)
        z, mean, scales = standardize(x)
        return MomentModel(
            mean,
            self._correlation(z, scales, assets),
            self._cokurtosis(x, z, mean, scales, assets),
            reference_variance=_reference_variance(x, mean),
            assets=assets,
        )

    def fit_correlation(self, returns: ReturnInput) -> CorrelationEstimate:
        """Estimate the sample correlation and covariance only.

        No fourth-order quantity is computed, so `even` has no effect.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
            Returns with the same requirements as in `fit`.

        Returns
        -------
        CorrelationEstimate
            Sample correlation with unit diagonal and the column standard
            deviations (``ddof=0``) that scale it to the covariance. It
            equals the correlation component of `fit` for the same returns.

        Raises
        ------
        TypeError
            If `returns` does not have an integer or floating-point dtype.
        ValueError
            If `returns` does not meet the requirements of `fit`.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> rng = np.random.default_rng(3)
        >>> common = rng.normal(size=(250, 1))
        >>> returns = 0.01 * (common + rng.normal(size=(250, 3)))
        >>> estimate = Sample().fit_correlation(returns)
        >>> np.allclose(estimate.correlation, np.corrcoef(returns, rowvar=False))
        True
        >>> np.allclose(estimate.scales, returns.std(axis=0))
        True
        """
        x, assets, _ = observations(returns)
        z, _, scales = standardize(x)
        return self._correlation(z, scales, assets)

    def fit_cokurtosis(self, returns: ReturnInput) -> CokurtosisEstimate:
        """Estimate the fourth central moment only.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
            Returns with the same requirements as in `fit`.

        Returns
        -------
        CokurtosisEstimate
            Fourth central moment, in the units of `returns` to the fourth
            power. With ``even=False`` it stores the centered observations;
            with ``even=True`` it stores the basis ``U4`` and the even core
            defined in the class Notes. It equals the fourth-order component
            of `fit` for the same returns.

        Raises
        ------
        TypeError
            If `returns` does not have an integer or floating-point dtype.
        ValueError
            If `returns` does not meet the requirements of `fit`.

        Notes
        -----
        Portfolio kurtosis also requires a variance estimate; use `fit` to
        obtain both moments.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(4).standard_t(df=5, size=(500, 3))
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> fourth = Sample().fit_cokurtosis(returns)
        >>> centered = returns @ weights - np.mean(returns @ weights)
        >>> bool(np.isclose(fourth.fourth_moment(weights), np.mean(centered**4)))
        True

        For weights proportional to ``U4[:, k] / scales``, only the retained
        core entry ``B[k, k]`` contributes, so the even estimate equals the
        sample moment:

        >>> even = Sample(even=True).fit_cokurtosis(returns)
        >>> direction = even.basis[:, 0] / even.scales
        >>> bool(np.isclose(even.fourth_moment(direction), fourth.fourth_moment(direction)))
        True
        """
        x, assets, _ = observations(returns)
        z, mean, scales = standardize(x)
        return self._cokurtosis(x, z, mean, scales, assets)


class CV:
    """Cross-validated estimator of correlation and cokurtosis.

    Correlation eigenvalues and the even core of the fourth central moment
    are estimated out of sample. Each of `n_splits` random partitions of the
    window fits eigenvectors, or a fourth-order basis, on a training subset
    and measures the corresponding coefficients on the complementary test
    subset. The coefficients are averaged over partitions and combined with
    the eigenvectors, or basis, of the complete window. In-sample
    eigenvalues are distorted by overfitting, increasingly so as the number
    of assets approaches the number of observations; coefficients measured
    on held-out observations do not share this overfitting.

    Parameters
    ----------
    n_splits : int, default=10
        Number of random partitions, at least one. Each partition places
        every observation in exactly one subset. Partitions are drawn
        independently, so an observation can be used for training in one
        partition and for testing in another.
    train_fraction : float, default=0.65
        Fraction of the observations in each training subset, strictly
        between 0 and 1. The training subset holds
        ``int(train_fraction * n_observations)`` observations and the test
        subset the rest; each must hold at least three when fitting.
    random_state : int, numpy.random.Generator or None, default=None
        Seed or generator for the partitions. A generator is copied, so its
        state is not advanced. ``None`` draws fresh entropy once, when the
        estimator is created. Every fitting call replays the same sequence
        of partitions, so `fit_correlation` and `fit_cokurtosis` return the
        components of `fit` for the same returns.
    device : {'cpu', 'cuda'}, default='cpu'
        Array backend for matrix products and eigendecompositions, with the
        same requirements and errors as in :class:`Sample`. Results are
        NumPy float64 arrays in host memory with either backend.

    Raises
    ------
    ValueError
        If `n_splits` is not a positive integer, `train_fraction` is not a
        finite number strictly between 0 and 1, or `device` is neither
        ``'cpu'`` nor ``'cuda'``.
    TypeError
        If `random_state` is not an integer, a NumPy Generator or None.

    See Also
    --------
    Sample : In-sample estimator; defines the basis and the even core.
    calibrate : Estimator with coefficients calibrated on historical windows.

    Notes
    -----
    The complete window ``X`` of ``T`` observations is standardized once,
    ``Z = (X - mean) / std`` with ``ddof=0``. Within each partition, the
    training and test subsets of ``Z`` are recentered by their own means but
    keep the complete-window scales.

    * Second order: the eigenvectors ``U2`` of the training covariance are
      paired with the test variances ``mean((Z_test @ U2)**2, axis=0)``.
      The averaged variances replace the eigenvalues of ``Z.T @ Z / T``;
      the resulting matrix ``C`` is rescaled to unit diagonal,
      ``C[i, j] / sqrt(C[i, i] * C[j, j])``, and the complete-window
      standard deviations convert it to a covariance.
    * Fourth order: the training subset gives the basis ``U4`` and the test
      subset the even core, both defined in the Notes of :class:`Sample`.
      The averaged core is paired with the ``U4`` of the complete window.

    The mean and the marginal variances are the sample values of the
    complete window; only the dependence structure is cross-validated. The
    two moments are filtered separately and need not belong to a single
    distribution, so the Pearson kurtosis of a portfolio can fall below one,
    the lower bound for any distribution. Such values are reported
    unchanged.

    Numerically repeated eigenvalues are resolved by a deterministic
    convention. The even core does not depend on the signs of the basis
    vectors, but it can depend on the basis chosen within a repeated
    eigenspace. Windows with more assets than observations are supported,
    and no dense ``n**4`` tensor is formed.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import CV
    >>> rng = np.random.default_rng(10)
    >>> returns = 0.01 * rng.standard_t(df=5, size=(120, 6))
    >>> estimator = CV(n_splits=20, random_state=42)
    >>> model = estimator.fit(returns)

    The correlations are cross-validated; the variances are not:

    >>> np.allclose(np.diag(model.covariance), returns.var(axis=0))
    True

    The order-specific methods replay the same partitions:

    >>> correlation = estimator.fit_correlation(returns)
    >>> np.allclose(correlation.correlation, model.correlation)
    True
    """

    n_splits: int
    train_fraction: float
    random_state: int | np.random.Generator | None
    device: Device

    def __init__(
        self,
        n_splits: int = 10,
        train_fraction: float = 0.65,
        random_state: int | np.random.Generator | None = None,
        device: Device = "cpu",
    ) -> None:
        if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 1:
            raise ValueError("n_splits must be a positive integer.")
        if (
            not isinstance(train_fraction, (int, float, np.integer, np.floating))
            or isinstance(train_fraction, (bool, np.bool_))
            or not np.isfinite(float(train_fraction))
            or not 0.0 < float(train_fraction) < 1.0
        ):
            raise ValueError("train_fraction must be strictly between zero and one.")
        if isinstance(random_state, (bool, np.bool_)) or (
            random_state is not None
            and not isinstance(random_state, (int, np.integer, np.random.Generator))
        ):
            raise TypeError("random_state must be an integer, numpy Generator, or None.")
        validate_device(device)
        self.n_splits = int(n_splits)
        self.train_fraction = float(train_fraction)
        self.random_state = random_state
        self.device: Device = device
        self._generator = deepcopy(np.random.default_rng(random_state))

    def _fit_components(
        self, x: FloatArray, assets: tuple[Hashable, ...] | None, order: MomentOrder
    ) -> tuple[FloatArray, CorrelationEstimate | None, CokurtosisEstimate | None]:
        z, mean, scales = standardize(x)
        cut = int(self.train_fraction * len(z))
        if min(cut, len(z) - cut) < 3:
            raise ValueError("CV training and test partitions each require at least three observations.")
        use_second = order in ("both", "correlation")
        use_fourth = order in ("both", "cokurtosis")
        n = z.shape[1]
        sum_second = np.zeros(n, dtype=np.float64) if use_second else None
        sum_fourth = np.zeros((n, n), dtype=np.float64) if use_fourth else None
        generator = deepcopy(self._generator)
        for _ in range(self.n_splits):
            permutation = generator.permutation(len(z))
            second, fourth = pair_core(
                z[permutation[:cut]],
                z[permutation[cut:]],
                standardized=True,
                device=self.device,
                order=order,
            )
            if sum_second is not None and second is not None:
                sum_second += second
            if sum_fourth is not None and fourth is not None:
                sum_fourth += fourth
        correlation = None
        cokurtosis = None
        if sum_second is not None:
            _, basis2 = correlation_basis(z, self.device)
            c = (basis2 * (sum_second / self.n_splits)) @ basis2.T
            correlation = CorrelationEstimate(normalize_correlation(c), scales, assets=assets)
        if sum_fourth is not None:
            cokurtosis = CokurtosisEstimate(
                scales=scales,
                basis=fourth_basis(z, self.device),
                core=sum_fourth / self.n_splits,
                assets=assets,
            )
        return mean, correlation, cokurtosis

    def fit(self, returns: ReturnInput) -> MomentModel:
        """Estimate the mean, cross-validated covariance and cokurtosis.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
            Real-valued returns, one row per observation and one column per
            asset. All values must be finite and every column must vary;
            missing values are rejected, not imputed. Both
            ``int(train_fraction * n_observations)`` and the remaining
            number of observations must be at least three. For a DataFrame,
            the columns become the asset labels and must be unique, hashable
            and non-missing, and the index must be unique and increasing.

        Returns
        -------
        MomentModel
            The sample mean, the cross-validated correlation and covariance,
            the cross-validated fourth central moment, and the variance of
            the equal-weight portfolio over the window, stored as
            ``reference_variance`` and used to scale the optimization
            objective. Moments are in the units and sampling frequency of
            `returns`; nothing is annualized. The estimator is not modified.

        Raises
        ------
        TypeError
            If `returns` does not have an integer or floating-point dtype.
        ValueError
            If `returns` does not meet the requirements above.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from hoao import CV
        >>> rng = np.random.default_rng(1)
        >>> returns = pd.DataFrame(
        ...     0.01 * rng.standard_t(df=5, size=(150, 3)),
        ...     index=pd.date_range("2024-01-01", periods=150, freq="B"),
        ...     columns=["bonds", "equities", "gold"],
        ... )
        >>> model = CV(random_state=42).fit(returns)
        >>> model.assets
        ('bonds', 'equities', 'gold')
        >>> reference = returns.mean(axis=1).var(ddof=0)
        >>> bool(np.isclose(model.reference_variance, reference))
        True
        """
        x, assets, _ = observations(returns)
        mean, correlation, cokurtosis = self._fit_components(x, assets, "both")
        assert correlation is not None and cokurtosis is not None
        return MomentModel(
            mean,
            correlation,
            cokurtosis,
            reference_variance=_reference_variance(x, mean),
            assets=assets,
        )

    def fit_correlation(self, returns: ReturnInput) -> CorrelationEstimate:
        """Estimate the cross-validated correlation and covariance only.

        No fourth-order quantity is computed. The result equals the
        correlation component of `fit` for the same returns.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
            Returns with the same requirements as in `fit`.

        Returns
        -------
        CorrelationEstimate
            Cross-validated correlation with unit diagonal and the sample
            standard deviations (``ddof=0``) of the window, which scale it
            to the covariance.

        Raises
        ------
        TypeError
            If `returns` does not have an integer or floating-point dtype.
        ValueError
            If `returns` does not meet the requirements of `fit`.

        Examples
        --------
        With 60 assets and 40 observations the sample correlation is
        singular. Each eigenvalue is replaced by an averaged held-out
        variance; these are positive here, so the estimate is positive
        definite:

        >>> import numpy as np
        >>> from hoao import CV
        >>> returns = 0.01 * np.random.default_rng(8).standard_normal((40, 60))
        >>> estimate = CV(random_state=7).fit_correlation(returns)
        >>> bool(np.linalg.eigvalsh(estimate.correlation).min() > 0)
        True
        """
        x, assets, _ = observations(returns)
        _, estimate, _ = self._fit_components(x, assets, "correlation")
        assert estimate is not None
        return estimate

    def fit_cokurtosis(self, returns: ReturnInput) -> CokurtosisEstimate:
        """Estimate the cross-validated fourth central moment only.

        No correlation is computed. The result equals the fourth-order
        component of `fit` for the same returns.

        Parameters
        ----------
        returns : array_like or pandas.DataFrame, shape (n_observations, n_assets)
            Returns with the same requirements as in `fit`.

        Returns
        -------
        CokurtosisEstimate
            The fourth-order basis ``U4`` of the complete window, the even
            core averaged over partitions, and the sample standard
            deviations (``ddof=0``) of the window. Portfolio fourth moments
            are evaluated from these ``(n_assets, n_assets)`` and
            ``(n_assets,)`` arrays without forming a dense tensor.

        Raises
        ------
        TypeError
            If `returns` does not have an integer or floating-point dtype.
        ValueError
            If `returns` does not meet the requirements of `fit`.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import CV
        >>> returns = 0.01 * np.random.default_rng(9).standard_t(df=5, size=(150, 5))
        >>> estimator = CV(random_state=7)
        >>> fourth = estimator.fit_cokurtosis(returns)
        >>> fourth.basis.shape, fourth.core.shape
        ((5, 5), (5, 5))
        >>> np.allclose(fourth.core, estimator.fit(returns).cokurtosis.core)
        True
        """
        x, assets, _ = observations(returns)
        _, _, estimate = self._fit_components(x, assets, "cokurtosis")
        assert estimate is not None
        return estimate
