"""Fitted correlation and cokurtosis estimates and their portfolio moments.

Estimators return a MomentModel when both orders are requested, or a
CorrelationEstimate or CokurtosisEstimate when only one is needed.
These objects retain the asset order and marginal scales of the fitting
window. Portfolio calculations use the original return units.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._typing import FloatArray, WeightInput
from ._validation import labels, readonly, weights_array

if TYPE_CHECKING:
    import pandas as pd
    from .optimization import OptimizationResult


@dataclass(frozen=True)
class PortfolioMoments:
    """Estimated mean, variance and fourth central moment of a portfolio.

    Returned by :meth:`MomentModel.portfolio` and included in every
    :class:`OptimizationResult`. The record is immutable. Its values refer
    to one observation period at the frequency of the fitting data; no
    annualization is applied.

    Attributes
    ----------
    mean : float
        Weighted sample mean of the fitting returns, in return units.
    variance : float
        Estimated second central moment, in squared return units.
    volatility : float
        Square root of `variance`, in return units.
    fourth_moment : float
        Estimated fourth central moment, in return units to the fourth
        power. This is an unstandardized moment, not a kurtosis.
    kurtosis : float
        Pearson kurtosis, ``fourth_moment / variance**2``. Three is not
        subtracted. The value is NaN when variance is zero.

    Notes
    -----
    A distribution with positive variance has Pearson kurtosis at least
    one. Separately filtered covariance and cokurtosis need not be moments
    of the same distribution, so their ratio can fall below one. Such a
    result is retained, as discussed in the accompanying manuscript.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import Sample
    >>> returns = np.array([[-0.01], [0.0], [0.01]])
    >>> moments = Sample().fit(returns).portfolio([1.0])
    >>> bool(np.isclose(moments.variance, np.var(returns)))
    True
    >>> round(moments.kurtosis, 6)
    1.5
    """

    mean: float
    variance: float
    volatility: float
    fourth_moment: float
    kurtosis: float


class CorrelationEstimate:
    """Correlation and covariance fitted to a window of asset returns.

    Returned by an estimator's ``fit_correlation`` method. The correlation
    describes dependence; the marginal standard deviations convert it to a
    covariance in the units of the current fitting window.

    Parameters
    ----------
    correlation : ndarray of float64, shape (n_assets, n_assets)
        Finite, symmetric, positive semidefinite matrix with unit diagonal.
    scales : ndarray of float64, shape (n_assets,)
        Finite, strictly positive marginal standard deviations. Estimators
        compute these from the current returns with ``ddof=0``.
    assets : sequence of hashable, optional
        Unique, non-missing labels in matrix order. If omitted, arrays are
        interpreted by position.

    Attributes
    ----------
    correlation : ndarray of float64, shape (n_assets, n_assets)
        Read-only correlation matrix.
    covariance : ndarray of float64, shape (n_assets, n_assets)
        Read-only covariance matrix,
        ``correlation * np.outer(scales, scales)``.
    scales : ndarray of float64, shape (n_assets,)
        Read-only marginal standard deviations, in return units.
    assets : tuple of hashable or None
        Asset labels in array order, or None for unlabeled input.

    Raises
    ------
    ValueError
        If dimensions disagree, values are non-finite, scales are not
        positive, the correlation is invalid, or labels are invalid.

    See Also
    --------
    Sample.fit_correlation : Estimate sample correlation and covariance.
    CV.fit_correlation : Estimate correlation by cross-validation.
    Calibration.fit_correlation : Apply historical AO coefficients.

    Examples
    --------
    The covariance diagonal retains the current sample variances even
    when the correlation has been filtered:

    >>> import numpy as np
    >>> from hoao import CV
    >>> returns = 0.01 * np.random.default_rng(4).standard_t(5, size=(100, 3))
    >>> estimate = CV(random_state=7).fit_correlation(returns)
    >>> np.allclose(np.diag(estimate.covariance), returns.var(axis=0))
    True
    """

    correlation: FloatArray
    covariance: FloatArray
    scales: FloatArray
    assets: tuple[Hashable, ...] | None

    def __init__(
        self, correlation: FloatArray, scales: FloatArray, *,
        assets: Sequence[Hashable] | None = None,
    ) -> None:
        c, s = np.asarray(correlation, dtype=np.float64), np.asarray(scales, dtype=np.float64)
        if s.ndim != 1 or not s.size or c.shape != (s.size, s.size):
            raise ValueError("Correlation and scale dimensions do not agree.")
        if not np.isfinite(c).all() or not np.isfinite(s).all() or np.any(s <= 0):
            raise ValueError("Correlation must be finite and scales strictly positive.")
        if not np.allclose(c, c.T, atol=1e-12, rtol=1e-12) or not np.allclose(np.diag(c), 1):
            raise ValueError("Correlation must be symmetric with diagonal one.")
        eigenvalues = np.linalg.eigvalsh(c)
        if eigenvalues[0] < -1e-10 * max(1.0, float(eigenvalues[-1])):
            raise ValueError("Correlation must be positive semidefinite.")
        self.correlation = readonly(c)
        self.scales = readonly(s)
        self.covariance = readonly(c * np.outer(s, s))
        self.assets = labels(assets, len(s)) if assets is not None else None

    def to_pandas(self, kind: Literal["correlation", "covariance"] = "correlation") -> pd.DataFrame:
        """Export the correlation or covariance as a labeled DataFrame.

        Parameters
        ----------
        kind : {'correlation', 'covariance'}, default='correlation'
            Matrix to export. Correlation is dimensionless; covariance is in
            squared return units at the original observation frequency.

        Returns
        -------
        pandas.DataFrame, shape (n_assets, n_assets)
            An independent copy with asset labels on both axes. For an
            unlabeled estimate, both axes use integer positions.

        Raises
        ------
        ImportError
            If pandas is not installed.
        ValueError
            If `kind` is neither ``'correlation'`` nor ``'covariance'``.

        Examples
        --------
        >>> import pandas as pd
        >>> from hoao import Sample
        >>> returns = pd.DataFrame({"A": [0.01, -0.02, 0.03],
        ...                         "B": [-0.01, 0.01, 0.02]})
        >>> estimate = Sample().fit_correlation(returns)
        >>> covariance = estimate.to_pandas("covariance")
        >>> covariance.index.tolist(), covariance.columns.tolist()
        (['A', 'B'], ['A', 'B'])
        """
        import pandas as pd
        if kind not in ("correlation", "covariance"):
            raise ValueError("kind must be 'correlation' or 'covariance'.")
        matrix = self.correlation if kind == "correlation" else self.covariance
        return pd.DataFrame(matrix.copy(), index=self.assets, columns=self.assets)


class CokurtosisEstimate:
    """Fourth central co-moments stored for direct portfolio evaluation.

    Obtain an estimate from ``Sample``, ``CV`` or ``Calibration``. The
    unfiltered sample estimate retains centered returns. The even sample,
    CV and HOAO estimates retain a fourth-order basis and an even-core
    matrix. Neither representation requires a dense fourth-order tensor.

    Parameters
    ----------
    scales : ndarray of float64, shape (n_assets,)
        Finite, strictly positive sample standard deviations of the
        fitting returns, computed with ``ddof=0``.
    basis : ndarray of float64, shape (n_assets, n_assets), optional
        Orthogonal basis for the fourth moment of standardized returns.
        Columns are basis vectors, ordered by decreasing singular value
        of the sample fourth tensor's mode-one unfolding.
    core : ndarray of float64, shape (n_assets, n_assets), optional
        Symmetric, nonnegative even-core matrix in `basis` coordinates.
        Its entries estimate ``mean(Y[:, i]**2 * Y[:, j]**2)`` for
        centered, rotated, standardized observations ``Y``.
    centered : ndarray of float64, shape (n_observations, n_assets), optional
        Finite, column-centered observations in original return units.
        Supply these for an exact sample estimate, or supply both `basis`
        and `core` for an even estimate; the two representations are
        mutually exclusive.
    assets : sequence of hashable, optional
        Unique, non-missing asset labels in column order.

    Attributes
    ----------
    scales : ndarray of float64, shape (n_assets,)
        Read-only marginal standard deviations.
    basis, core : ndarray of float64, shape (n_assets, n_assets), or None
        Read-only basis and even core. Both are None for the unfiltered
        sample estimate.
    centered : ndarray of float64, shape (n_observations, n_assets), or None
        Read-only centered returns for the sample estimate; None for an
        even estimate.
    assets : tuple of hashable or None
        Labels used to align pandas weight Series, or None for unlabeled
        estimates.

    Raises
    ------
    ValueError
        If the representation is incomplete, its dimensions or values
        are invalid, the basis is not orthogonal, the core is not
        symmetric and nonnegative, or asset labels are invalid.

    See Also
    --------
    Sample : Exact sample moments or the even-core sample projection.
    CV : Estimate an even core from random splits of a return window.
    calibrate : Estimate an even core from historical window pairs.

    Notes
    -----
    For an even estimate, let ``v = basis.T @ (scales * weights)`` and
    ``q = v**2``. The portfolio fourth moment is
    ``3 * q @ core @ q - 2 * np.diag(core) @ v**4``. The coefficient
    three accounts for the six permutations of each paired-index term;
    the subtraction counts each four-equal-index term only once.
    Evaluation, gradients and Hessian-vector products cost
    ``O(n_assets**2)``. For the sample representation they cost
    ``O(n_observations * n_assets)``.

    Use the estimators to construct cores from observations. Symmetry and
    nonnegative entries alone do not establish that an arbitrary supplied
    core is a valid moment estimate.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import Sample
    >>> returns = 0.01 * np.random.default_rng(2).standard_t(5, size=(100, 3))
    >>> estimate = Sample(even=True).fit_cokurtosis(returns)
    >>> estimate.basis.shape, estimate.core.shape
    ((3, 3), (3, 3))

    Doubling every position multiplies the fourth moment by sixteen:

    >>> weights = np.array([0.5, 0.3, 0.2])
    >>> ratio = estimate.fourth_moment(2 * weights) / estimate.fourth_moment(weights)
    >>> bool(np.isclose(ratio, 16.0))
    True
    """

    scales: FloatArray
    basis: FloatArray | None
    core: FloatArray | None
    centered: FloatArray | None
    assets: tuple[Hashable, ...] | None
    _factor: FloatArray | None

    def __init__(
        self, *, scales: FloatArray, basis: FloatArray | None = None,
        core: FloatArray | None = None, centered: FloatArray | None = None,
        assets: Sequence[Hashable] | None = None,
    ) -> None:
        s = np.asarray(scales, dtype=np.float64)
        if s.ndim != 1 or not s.size or not np.isfinite(s).all() or np.any(s <= 0):
            raise ValueError("Scales must be a positive finite vector.")
        n = len(s)
        self.scales = readonly(s)
        self.assets = labels(assets, n) if assets is not None else None
        if centered is not None:
            x = np.asarray(centered, dtype=np.float64)
            if basis is not None or core is not None:
                raise ValueError("Supply centered observations OR a basis and core.")
            if x.ndim != 2 or x.shape[1] != n or len(x) < 1 or not np.isfinite(x).all():
                raise ValueError("Centered observations must be finite with shape (time, assets).")
            if not np.allclose(x.mean(axis=0), 0, atol=1e-12 * max(1.0, float(np.max(abs(x))))):
                raise ValueError("Sample observations must be centered.")
            self.centered = readonly(x)
            self.basis = self.core = self._factor = None
        else:
            if basis is None or core is None:
                raise ValueError("Both basis and core are required for an even estimate.")
            u, b = np.asarray(basis, dtype=np.float64), np.asarray(core, dtype=np.float64)
            if u.shape != (n, n) or b.shape != (n, n) or not np.isfinite(u).all() or not np.isfinite(b).all():
                raise ValueError("Basis and core must be finite square arrays.")
            if not np.allclose(u.T @ u, np.eye(n), atol=1e-10, rtol=1e-10):
                raise ValueError("The fourth-order basis must be orthogonal.")
            if np.any(b < 0) or not np.allclose(b, b.T, atol=1e-12, rtol=1e-12):
                raise ValueError("Even core must be nonnegative and symmetric.")
            self.basis, self.core = readonly(u), readonly(b)
            self._factor = readonly(s[:, None] * u)
            self.centered = None

    def _value_gradient(self, w: FloatArray) -> tuple[float, FloatArray]:
        if self.centered is not None:
            r = self.centered @ w
            return float(np.mean(r**4)), 4 * (self.centered.T @ (r**3)) / len(r)
        assert self._factor is not None and self.core is not None
        v = self._factor.T @ w
        squared, diagonal = v * v, np.diag(self.core)
        product = self.core @ squared
        value = 3 * squared @ product - 2 * diagonal @ (squared * squared)
        gradient = self._factor @ (12 * v * product - 8 * diagonal * v**3)
        return float(value), gradient

    def fourth_moment(self, weights: WeightInput) -> float:
        """Evaluate the fourth central moment of a weighted portfolio.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Finite real positions. Short positions and non-unit budgets are
            allowed. Arrays follow the fitting-data column order. A Series
            must contain each asset label exactly once and is reordered to
            match the estimate; missing or additional labels are errors.

        Returns
        -------
        float
            Fourth central moment in return units to the fourth power, at the
            frequency of the fitting data. This is not divided by variance.

        Raises
        ------
        TypeError
            If weights do not have a real numeric dtype.
        ValueError
            If weights have the wrong shape, contain non-finite
            values, or have labels incompatible with the estimate.

        Examples
        --------
        For the sample estimator, evaluation agrees with the fourth power of
        centered portfolio returns:

        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(3).normal(size=(100, 3))
        >>> estimate = Sample().fit_cokurtosis(returns)
        >>> weights = np.array([0.6, 0.5, -0.1])
        >>> realized = returns @ weights
        >>> expected = np.mean((realized - realized.mean())**4)
        >>> bool(np.isclose(estimate.fourth_moment(weights), expected))
        True
        """
        w = weights_array(weights, len(self.scales), self.assets)
        return self._value_gradient(w)[0]

    def gradient(self, weights: WeightInput) -> FloatArray:
        """Differentiate the fourth central moment with respect to weights.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Finite real weights at which to evaluate the derivative. A Series
            is aligned by the estimate's asset labels, with the same checks
            as :meth:`fourth_moment`.

        Returns
        -------
        ndarray of float64, shape (n_assets,)
            Analytic derivative in model asset order. Each entry measures the
            change in the fourth moment per unit change in that asset's
            weight. The derivative is unconstrained: it does not impose a
            unit budget.

        Raises
        ------
        TypeError
            If weights do not have a real numeric dtype.
        ValueError
            If weights or their labels are invalid.

        See Also
        --------
        hessian_vector_product : Directional derivative of the gradient.

        Examples
        --------
        The fourth moment is homogeneous of degree four, so its gradient
        satisfies ``weights @ gradient == 4 * fourth_moment``:

        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(3).normal(size=(100, 3))
        >>> estimate = Sample(even=True).fit_cokurtosis(returns)
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> derivative = weights @ estimate.gradient(weights)
        >>> bool(np.isclose(derivative, 4 * estimate.fourth_moment(weights)))
        True
        """
        return self._value_gradient(weights_array(weights, len(self.scales), self.assets))[1]

    def hessian_vector_product(self, weights: WeightInput, direction: WeightInput) -> FloatArray:
        """Apply the fourth-moment Hessian to a direction in weight space.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Finite real weights at which to evaluate the Hessian.
        direction : array_like or pandas.Series, shape (n_assets,)
            Finite real direction. Each Series is aligned independently by
            asset label, using the same rules as :meth:`fourth_moment`.

        Returns
        -------
        ndarray of float64, shape (n_assets,)
            Hessian times `direction`, in model asset order. The calculation
            forms neither the full Hessian nor a fourth-order tensor.

        Raises
        ------
        TypeError
            If either vector does not have a real numeric dtype.
        ValueError
            If either vector or its labels are invalid.

        Examples
        --------
        Compare the analytic product with a directional finite difference of
        the gradient:

        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(3).normal(size=(100, 3))
        >>> estimate = Sample(even=True).fit_cokurtosis(returns)
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> direction = np.array([1.0, -1.0, 0.0])
        >>> step = 1e-6
        >>> difference = (estimate.gradient(weights + step * direction)
        ...               - estimate.gradient(weights - step * direction)) / (2 * step)
        >>> np.allclose(estimate.hessian_vector_product(weights, direction), difference)
        True
        """
        w = weights_array(weights, len(self.scales), self.assets)
        direction_array = weights_array(direction, len(self.scales), self.assets)
        if self.centered is not None:
            r, d = self.centered @ w, self.centered @ direction_array
            return 12 * self.centered.T @ (r * r * d) / len(r)
        assert self._factor is not None and self.core is not None
        v, h = self._factor.T @ w, self._factor.T @ direction_array
        answer = 24 * v * (self.core @ (v * h))
        answer += (12 * (self.core @ (v * v)) - 24 * np.diag(self.core) * v * v) * h
        return self._factor @ answer

    def to_dense(self, *, standardized: bool = False, max_bytes: int = 512 * 1024**2) -> FloatArray:
        """Reconstruct the fourth-order tensor within a specified memory budget.

        Parameters
        ----------
        standardized : bool, default=False
            If False, return co-moments in the original return units. If
            True, divide each axis by its marginal standard deviation. A
            standardized fourth tensor does not generally have unit diagonal.
        max_bytes : int, default=536870912
            Positive upper bound for the estimated output and working storage
            in bytes; the default is 512 MiB. The check precedes allocation.

        Returns
        -------
        ndarray of float64, shape (n_assets, n_assets, n_assets, n_assets)
            Symmetric fourth central co-moments. All axes follow model asset
            order. In a C-order unfolding, the last index varies fastest.

        Raises
        ------
        MemoryError
            If estimated dense working memory exceeds `max_bytes`.
        ValueError
            If `max_bytes` is not a positive integer.

        Notes
        -----
        The tensor alone occupies ``8 * n_assets**4`` bytes. The budget also
        allows for reconstruction work arrays. Fitting, calibration and
        portfolio calculations use the compact representation directly; dense
        export is useful for small examples and comparisons with other code.

        Examples
        --------
        Contracting a small dense tensor reproduces direct evaluation:

        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(3).normal(size=(100, 3))
        >>> estimate = Sample(even=True).fit_cokurtosis(returns)
        >>> tensor = estimate.to_dense(max_bytes=1024**2)
        >>> weights = np.array([0.5, 0.3, 0.2])
        >>> dense_value = np.einsum("ijkl,i,j,k,l", tensor, weights, weights, weights, weights)
        >>> bool(np.isclose(dense_value, estimate.fourth_moment(weights)))
        True
        """
        n = len(self.scales)
        temporary = 8 * (4 * n**4 + (len(self.centered) * n * n if self.centered is not None else 0))
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, np.integer)) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        if temporary > max_bytes:
            raise MemoryError(f"Dense export needs approximately {temporary:,} bytes; cap is {max_bytes:,}.")
        if self.centered is not None:
            x = self.centered / self.scales if standardized else self.centered
            pairs = (x[:, :, None] * x[:, None, :]).reshape(len(x), n * n)
            return (pairs.T @ pairs / len(x)).reshape((n,) * 4)
        assert self.core is not None and self.basis is not None and self._factor is not None
        tensor = np.zeros((n,) * 4, dtype=np.float64)
        for i in range(n):
            tensor[i, i, i, i] = self.core[i, i]
            for j in range(i):
                for index in ((i,i,j,j), (i,j,i,j), (i,j,j,i), (j,i,i,j), (j,i,j,i), (j,j,i,i)):
                    tensor[index] = self.core[i, j]
        factor = self.basis if standardized else self._factor
        for axis in range(4):
            tensor = np.moveaxis(np.tensordot(factor, tensor, axes=(1, axis)), 0, axis)
        return tensor


class MomentModel:
    """Fitted moments for evaluating and optimizing asset portfolios.

    Returned by ``Sample.fit``, ``CV.fit`` and ``Calibration.fit``. The
    model combines the current sample mean and marginal volatilities with
    the estimator's second- and fourth-order dependence estimates. It can
    evaluate any finite weight vector or solve the variance–fourth-moment
    allocation problem described in the manuscript.

    Parameters
    ----------
    mean : ndarray of float64, shape (n_assets,)
        Finite sample means of the fitting returns.
    correlation_estimate : CorrelationEstimate
        Correlation and current marginal standard deviations.
    cokurtosis_estimate : CokurtosisEstimate
        Fourth central co-moments with exactly the same marginal scales
        and asset labels as `correlation_estimate`.
    reference_variance : float
        Raw sample variance of the equal-weight portfolio in the fitting
        window, computed with ``ddof=0``. Must be finite and nonnegative;
        optimization additionally requires it to be strictly positive.
    assets : sequence of hashable, optional
        Unique, non-missing asset labels. If supplied, they must agree
        with both component estimates. Otherwise the component labels
        are used.

    Attributes
    ----------
    mean : ndarray of float64, shape (n_assets,)
        Read-only sample means in original return units.
    correlation, covariance : ndarray of float64, shape (n_assets, n_assets)
        Read-only dependence and covariance matrices. The covariance
        diagonal equals the current sample marginal variances.
    correlation_estimate : CorrelationEstimate
        Second-order component, including its scales and pandas export.
    cokurtosis : CokurtosisEstimate
        Fourth-order component, with portfolio evaluation and derivatives.
    reference_variance : float
        Equal-weight sample variance used to normalize optimization.
    assets : tuple of hashable or None
        Asset labels in output array order, or None for unlabeled input.

    Raises
    ------
    ValueError
        If the mean or reference variance is invalid, or the component
        dimensions, scales or labels disagree.

    See Also
    --------
    portfolio : Evaluate moments for given weights.
    optimize : Find a unit-budget allocation for a chosen penalty.
    frontier : Solve for several penalties with successive warm starts.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import CV
    >>> returns = 0.01 * np.random.default_rng(1).standard_t(5, size=(250, 4))
    >>> model = CV(random_state=2).fit(returns)
    >>> weights = np.full(4, 0.25)
    >>> moments = model.portfolio(weights)
    >>> bool(np.isclose(moments.variance, weights @ model.covariance @ weights))
    True
    >>> allocation = model.optimize(gamma=1.0)
    >>> bool(np.isclose(allocation.weights.sum(), 1.0))
    True
    """

    mean: FloatArray
    correlation: FloatArray
    covariance: FloatArray
    cokurtosis: CokurtosisEstimate
    correlation_estimate: CorrelationEstimate
    reference_variance: float
    assets: tuple[Hashable, ...] | None

    def __init__(
        self, mean: FloatArray, correlation_estimate: CorrelationEstimate,
        cokurtosis_estimate: CokurtosisEstimate, *, reference_variance: float,
        assets: Sequence[Hashable] | None = None,
    ) -> None:
        mu = np.asarray(mean, dtype=np.float64)
        n = len(correlation_estimate.scales)
        if mu.shape != (n,) or not np.isfinite(mu).all():
            raise ValueError("Mean must be a finite vector matching the estimates.")
        if not np.array_equal(correlation_estimate.scales, cokurtosis_estimate.scales):
            raise ValueError("Second and fourth estimates must use identical marginal scales.")
        asset_labels = labels(assets, n) if assets is not None else correlation_estimate.assets
        if asset_labels != correlation_estimate.assets or asset_labels != cokurtosis_estimate.assets:
            raise ValueError("Component asset labels must agree.")
        if not np.isfinite(reference_variance) or reference_variance < 0:
            raise ValueError("reference_variance must be finite and nonnegative.")
        self.mean = readonly(mu)
        self.correlation_estimate = correlation_estimate
        self.correlation, self.covariance = correlation_estimate.correlation, correlation_estimate.covariance
        self.cokurtosis, self.reference_variance, self.assets = cokurtosis_estimate, float(reference_variance), asset_labels

    def portfolio(self, weights: WeightInput) -> PortfolioMoments:
        """Evaluate the estimated moments of a portfolio with given weights.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Finite real weights. Negative positions are allowed, and weights
            need not sum to one. Arrays follow the fitting-data column order.
            A Series is reordered by asset label and must contain exactly
            the model's labels, without duplicates or missing labels.

        Returns
        -------
        PortfolioMoments
            Mean, variance, volatility, fourth central moment and Pearson
            kurtosis in the original units and observation frequency.

        Raises
        ------
        TypeError
            If weights do not have a real numeric dtype.
        ValueError
            If weights have an invalid shape, values or labels. A weight
            Series also requires a model with asset labels.
        FloatingPointError
            If the computed variance is negative beyond floating-point
            roundoff. A negative value within roundoff is set to zero.

        Notes
        -----
        The mean is the weighted sample mean. Variance and fourth moment use
        the fitted dependence estimates. Their ratio is Pearson kurtosis,
        without subtracting three; it is NaN for zero variance. Separately
        filtered moments can give kurtosis below one, which is reported
        unchanged rather than treated as a numerical error.

        Examples
        --------
        A labeled weight vector can be supplied in a different order from
        the return columns:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from hoao import Sample
        >>> returns = pd.DataFrame({"A": [0.01, -0.02, 0.03],
        ...                         "B": [-0.01, 0.01, 0.02]})
        >>> model = Sample().fit(returns)
        >>> weights = pd.Series({"B": 0.4, "A": 0.6})
        >>> moments = model.portfolio(weights)
        >>> realized = returns @ weights
        >>> bool(np.isclose(moments.variance, realized.var(ddof=0)))
        True
        """
        w = weights_array(weights, len(self.mean), self.assets)
        variance = float(w @ self.covariance @ w)
        if variance < 0:
            tolerance = 64 * np.finfo(float).eps * float(np.linalg.norm(self.covariance, ord=2)) * float(w @ w)
            if variance < -tolerance:
                raise FloatingPointError("Negative portfolio variance beyond roundoff.")
            variance = 0.0
        fourth = self.cokurtosis._value_gradient(w)[0]
        return PortfolioMoments(float(self.mean @ w), variance, float(np.sqrt(variance)), fourth,
                                fourth / variance**2 if variance > 0 else float("nan"))

    def optimize(self, gamma: float = 0.0, *, tol: float = 1e-8, max_iter: int = 1000) -> OptimizationResult:
        """Minimize variance and fourth moment subject to a unit budget.

        Parameters
        ----------
        gamma : float, default=0.0
            Finite, nonnegative penalty on the normalized fourth central
            moment. Zero gives the global minimum-variance portfolio. Larger
            values place more weight on the fourth moment relative to
            variance; this is not a penalty on kurtosis itself.
        tol : float, default=1e-8
            Positive, finite target for the infinity norm of the objective
            gradient in coordinates that satisfy the budget constraint.
        max_iter : int, default=1000
            Maximum BFGS iterations. A subsequent Hessian-vector refinement,
            when needed, can use up to ``min(max_iter, 100)`` more iterations.

        Returns
        -------
        OptimizationResult
            Unit-budget weights, fitted portfolio moments, objective value
            and convergence diagnostics. The weights follow model asset order.

        Raises
        ------
        OptimizationError
            If the candidate fails the budget or gradient-residual checks.
            The exception's ``result`` contains the candidate and diagnostics.
        ValueError
            If `gamma`, `tol` or `max_iter` is invalid, or the equal-weight
            reference variance is not strictly positive.

        See Also
        --------
        frontier : Reuse each solution as the starting point for the next penalty.
        OptimizationResult : Meaning of the returned diagnostics.

        Notes
        -----
        The objective is ``variance / s2 + gamma * fourth_moment / s2**2``,
        where ``s2 = reference_variance`` is the unfiltered sample variance
        of equal-weight returns. Using this common scale makes the penalty
        independent of whether returns are supplied as fractions or percents.

        Weights sum to one and short sales are allowed. No expected-return
        forecast enters the objective. For ``gamma=0``, a direct constrained
        covariance solve is used; singular systems use a minimum-norm
        least-squares solution. For positive penalties, SciPy uses analytic
        gradients and Hessian-vector products on the CPU.

        Acceptance requires a budget error at most ``1e-10`` and a reduced
        gradient residual at most ``max(1e-6, 10 * tol)``. These independent
        checks determine success, including when the solver reports that
        floating-point precision prevented further improvement.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = 0.01 * np.random.default_rng(2).standard_t(5, size=(250, 4))
        >>> model = Sample().fit(returns)
        >>> allocation = model.optimize(gamma=1.0)
        >>> bool(np.isclose(allocation.weights.sum(), 1.0))
        True
        >>> allocation.success
        True
        >>> scale = model.reference_variance
        >>> expected = (allocation.moments.variance / scale
        ...             + allocation.moments.fourth_moment / scale**2)
        >>> bool(np.isclose(allocation.objective, expected))
        True
        """
        from .optimization import optimize
        return optimize(self, gamma, tol=tol, max_iter=max_iter)

    def frontier(self, gammas: Sequence[float], *, tol: float = 1e-8, max_iter: int = 1000) -> list[OptimizationResult]:
        """Solve the variance–fourth-moment allocation for a sequence of penalties.

        Parameters
        ----------
        gammas : sequence of float
            Finite, nonnegative penalties, evaluated in the supplied order.
            Each solution initializes the next optimization. An increasing
            sequence is convenient for tracing the tradeoff from minimum
            variance toward smaller fourth moments; no sorting is applied.
        tol : float, default=1e-8
            Positive, finite target for the reduced objective gradient in
            each solve. See :meth:`optimize` for the acceptance threshold.
        max_iter : int, default=1000
            Maximum BFGS iterations per solve, followed by at most
            ``min(max_iter, 100)`` refinement iterations when needed.

        Returns
        -------
        list of OptimizationResult
            Allocations in the order of `gammas`. An empty sequence returns
            an empty list. All portfolio moments are fitted estimates from
            this model, not measurements on future returns.

        Raises
        ------
        OptimizationError
            If a solve fails its feasibility or stationarity checks. The
            exception contains that solve's diagnostic result.
        ValueError
            If a penalty or solver setting is invalid, or the reference
            variance is not strictly positive when a solve is attempted.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> returns = np.random.default_rng(2).standard_t(5, size=(250, 4))
        >>> model = Sample().fit(returns)
        >>> allocations = model.frontier([0.0, 1.0, 10.0])
        >>> [item.gamma for item in allocations]
        [0.0, 1.0, 10.0]
        >>> weights = np.stack([item.weights for item in allocations])
        >>> np.allclose(weights.sum(axis=1), 1.0)
        True
        """
        from .optimization import optimize
        result: list[OptimizationResult] = []
        initial: FloatArray | None = None
        for gamma in gammas:
            item = optimize(self, gamma, tol=tol, max_iter=max_iter, initial=initial)
            result.append(item)
            initial = item.weights
        return result
