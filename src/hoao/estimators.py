"""Sample and cross-validated dependence estimators for return matrices."""

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
    """Estimate sample moments, optionally projecting the fourth-order core.

    Parameters
    ----------
    even : bool, default=False
        If ``False``, retain the empirical fourth moment exactly, using
        centered observations as its compact representation. If ``True``,
        use the manuscript's ``MLE_even`` estimator: rotate standardized
        observations into their fourth-order HOSVD basis and retain only
        core entries with even index multiplicities. Correlation is the
        ordinary sample correlation in both cases.
    device : {'cpu', 'cuda'}, keyword-only, default='cpu'
        Device for basis calculations. ``'cuda'`` requires a separately
        installed, compatible CuPy package and a working NVIDIA driver.
        CUDA support is experimental; all public results are NumPy float64
        arrays on the CPU. Small datasets are usually faster on the CPU.

    Notes
    -----
    Returns are rows by assets. Every central moment uses the maximum
    likelihood convention, dividing by the number of observations rather
    than ``n_observations - 1``. The fourth moment is not excess kurtosis.

    ``fit`` returns a new :class:`~hoao.MomentModel`, not the estimator
    itself. No dense four-dimensional tensor is constructed. The unprojected
    sample representation stores ``O(n_observations * n_assets)`` values;
    the even estimate stores ``O(n_assets**2)`` values.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import Sample
    >>> returns = np.random.default_rng(7).normal(0, 0.01, size=(100, 4))
    >>> sample_model = Sample().fit(returns)
    >>> even_model = Sample(even=True).fit(returns)
    >>> correlation = Sample().fit_correlation(returns)
    >>> fourth_moment = Sample(even=True).fit_cokurtosis(returns)
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
        """Fit a joint sample covariance and fourth-moment model.

        Parameters
        ----------
        returns : numpy.ndarray or pandas.DataFrame, shape (n_observations, n_assets)
            Finite real asset returns, with at least three observations.
            Rows are observation dates and columns are assets. For a
            DataFrame, column labels are retained and its index must be
            unique and increasing. Every asset must have positive variance.
            Missing observations are rejected rather than imputed.

        Returns
        -------
        model : MomentModel
            Sample means, correlation and covariance, a compact fourth
            moment estimate, and the sample variance of the equal-weight
            portfolio used to normalize portfolio optimization.

        Raises
        ------
        ValueError
            If shapes, labels, observations, or asset variances are invalid.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(0).normal(size=(80, 3)) * 0.01
        >>> model = Sample().fit(returns)
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
        """Fit only the sample correlation and covariance.

        Parameters
        ----------
        returns : numpy.ndarray or pandas.DataFrame, shape (n_observations, n_assets)
            At least three finite observations per asset, in columns.
            DataFrame labels are preserved. Each asset must have positive
            variance; missing values and duplicate labels are rejected.

        Returns
        -------
        estimate : CorrelationEstimate
            Unit-diagonal sample correlation and population sample scales,
            from which covariance and portfolio variance are available.
            This method does not compute a fourth-order basis.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(3).normal(size=(60, 2))
        >>> estimate = Sample().fit_correlation(returns)
        """
        x, assets, _ = observations(returns)
        z, _, scales = standardize(x)
        return self._correlation(z, scales, assets)

    def fit_cokurtosis(self, returns: ReturnInput) -> CokurtosisEstimate:
        """Fit only the sample or even-projected fourth central moment.

        Parameters
        ----------
        returns : numpy.ndarray or pandas.DataFrame, shape (n_observations, n_assets)
            At least three finite observations per asset, in columns.
            DataFrame labels are preserved. Each asset must have positive
            variance; missing values and duplicate labels are rejected.

        Returns
        -------
        estimate : CokurtosisEstimate
            A compact fourth-moment estimate in raw-return units. With
            ``even=False``, it retains centered sample observations. With
            ``even=True``, it retains an orthogonal basis and an even-core
            matrix. This method does not fit a covariance estimator.

        Notes
        -----
        Portfolio kurtosis also requires a variance estimate. Use ``fit``
        when both second and fourth moments are needed together.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(4).normal(size=(90, 3))
        >>> fourth = Sample(even=True).fit_cokurtosis(returns)
        """
        x, assets, _ = observations(returns)
        z, mean, scales = standardize(x)
        return self._cokurtosis(x, z, mean, scales, assets)


class CV:
    """Cross-validate correlation and the sign-invariant cokurtosis core.

    Parameters
    ----------
    n_splits : int, default=10
        Number of repeated random, disjoint training/test partitions.
        Every partition uses all observations once; observations may occur
        in different roles across different partitions.
    train_fraction : float, default=0.65
        Fraction of observations assigned to training, rounded down to an
        integer. Both halves must contain at least three observations.
    random_state : int or numpy.random.Generator or None, default=None
        Source of the random partitions. An integer makes separate CV
        instances reproducible. A Generator is copied without advancing
        the caller's state. A fresh state is created for ``None``. Repeated
        calls on one instance reuse the same partitions for a given input
        length, including calls to the order-specific fitting methods.
    device : {'cpu', 'cuda'}, default='cpu'
        Computation device. ``'cuda'`` requires compatible CuPy and NVIDIA
        installations and is experimental. Outputs remain NumPy float64
        arrays. CPU execution is the default for small fitting windows.

    Notes
    -----
    The complete fitting window is centered and divided by its population
    sample volatilities. Each partition is then recentered independently
    without changing those scales. Training observations determine the
    basis; test observations determine the even core. Averaged test cores
    are reconstructed in the basis of the complete fitting window.

    Correlation and fourth moments use separate bases. Filtered correlation
    is normalized to unit diagonal before restoring asset volatilities.
    This follows the manuscript's main estimators; it does not enforce joint
    moment compatibility, so estimated portfolio kurtosis can be below one.
    Cokurtosis means the fourth central moment tensor, not excess kurtosis.

    Repeated eigenvalues use a deterministic projector-based basis convention.
    More assets than observations are supported without dense fourth tensors.
    The estimator is invariant to basis signs, but a different convention
    within a tied eigenspace can produce a different even-core estimate.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import CV
    >>> returns = np.random.default_rng(10).normal(0, 0.01, size=(100, 5))
    >>> estimator = CV(random_state=42)
    >>> model = estimator.fit(returns)
    >>> correlation = estimator.fit_correlation(returns)
    >>> fourth = estimator.fit_cokurtosis(returns)
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
        """Fit cross-validated correlation and cokurtosis together.

        Parameters
        ----------
        returns : numpy.ndarray or pandas.DataFrame, shape (n_observations, n_assets)
            Finite asset returns, one observation per row and asset per
            column. The configured split must leave at least three rows
            in each partition. DataFrame columns become asset labels;
            its index must be unique and increasing. Constant assets and
            missing observations are rejected.

        Returns
        -------
        model : MomentModel
            Sample means, filtered correlation and covariance, compact
            filtered fourth moment, and the sample equal-weight reference
            variance used in portfolio optimization. The model is a new
            object; the estimator retains its original configuration.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import CV
        >>> returns = np.random.default_rng(1).normal(size=(100, 8)) * 0.01
        >>> model = CV(n_splits=10, random_state=42).fit(returns)
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
        """Fit cross-validated correlation without fourth-moment calculations.

        Parameters
        ----------
        returns : numpy.ndarray or pandas.DataFrame, shape (n_observations, n_assets)
            Finite returns in columns, with positive asset variances and
            enough rows for three observations in each split partition.
            DataFrame labels are retained; missing values are rejected.

        Returns
        -------
        estimate : CorrelationEstimate
            Filtered unit-diagonal correlation and current sample asset
            volatilities. Its result matches the correlation component of
            ``fit`` for the same input and this estimator's random state.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import CV
        >>> returns = np.random.default_rng(8).normal(size=(100, 5))
        >>> correlation = CV(random_state=7).fit_correlation(returns)
        """
        x, assets, _ = observations(returns)
        _, estimate, _ = self._fit_components(x, assets, "correlation")
        assert estimate is not None
        return estimate

    def fit_cokurtosis(self, returns: ReturnInput) -> CokurtosisEstimate:
        """Fit cross-validated cokurtosis without fitting a correlation model.

        Parameters
        ----------
        returns : numpy.ndarray or pandas.DataFrame, shape (n_observations, n_assets)
            Finite returns in columns, with positive asset variances and
            enough rows for three observations in each split partition.
            DataFrame labels are retained; missing values are rejected.

        Returns
        -------
        estimate : CokurtosisEstimate
            The full-window orthogonal fourth-order basis, averaged even
            core, and current sample volatilities. Portfolio fourth moments
            are evaluated without constructing a dense tensor. The result
            matches the fourth-order component of ``fit`` for the same input.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import CV
        >>> returns = np.random.default_rng(9).normal(size=(100, 5))
        >>> fourth = CV(random_state=7).fit_cokurtosis(returns)
        """
        x, assets, _ = observations(returns)
        _, _, estimate = self._fit_components(x, assets, "cokurtosis")
        assert estimate is not None
        return estimate
