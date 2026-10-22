"""Compact fitted estimates and portfolio-level results."""
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
    """Central moments of a portfolio in the input return units.

    Attributes
    ----------
    mean : float
        Weighted sample mean of the fitting returns.
    variance : float
        Estimated second central moment, in squared return units.
    volatility : float
        Square root of variance, without annualization.
    fourth_moment : float
        Estimated fourth central moment, in fourth-power return units.
    kurtosis : float
        Pearson kurtosis: fourth moment divided by variance squared.
        NaN when variance is zero. Separately filtered moments may give
        values below one; the value is not clipped.

    Examples
    --------
    >>> from hoao import Sample
    >>> risk = Sample().fit([[0.01], [-0.02], [0.03]]).portfolio([1.0])
    >>> risk.variance > 0
    True
    """

    mean: float
    variance: float
    volatility: float
    fourth_moment: float
    kurtosis: float


class CorrelationEstimate:
    """A correlation estimate with current marginal return volatilities.

    Normally returned by an estimator's ``fit_correlation`` method.

    Parameters
    ----------
    correlation : ndarray of float64, shape (n_assets, n_assets)
        Symmetric positive semidefinite matrix with diagonal one.
    scales : ndarray of float64, shape (n_assets,)
        Positive sample standard deviations, computed with ``ddof=0``.
    assets : sequence of hashable, optional
        Unique labels in matrix-column order.

    Attributes
    ----------
    correlation, covariance : ndarray of float64
        Read-only square arrays. Covariance is scaled to return units.
    scales : ndarray of float64
        Read-only marginal volatilities.
    assets : tuple of hashable or None
        Asset labels, when supplied by the fitting data.

    Examples
    --------
    >>> from hoao import Sample
    >>> estimate = Sample().fit_correlation([[1., 2.], [2., 1.], [4., 3.]])
    >>> estimate.correlation.shape
    (2, 2)
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
        """Return a labeled copy of the correlation or covariance matrix.

        Parameters
        ----------
        kind : {'correlation', 'covariance'}, default 'correlation'
            Matrix to export. Covariance retains the input return units.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_assets, n_assets)``. Uses integer labels if the
            estimate was fitted from an unlabeled array.

        Raises
        ------
        ImportError
            If pandas is not installed; install ``hoao[pandas]``.
        ValueError
            If ``kind`` does not name a supported matrix.

        Examples
        --------
        >>> from hoao import Sample
        >>> estimate = Sample().fit_correlation([[1.], [2.], [3.]])
        >>> estimate.to_pandas().iloc[0, 0]
        np.float64(1.0)
        """
        import pandas as pd
        if kind not in ("correlation", "covariance"):
            raise ValueError("kind must be 'correlation' or 'covariance'.")
        matrix = self.correlation if kind == "correlation" else self.covariance
        return pd.DataFrame(matrix.copy(), index=self.assets, columns=self.assets)


class CokurtosisEstimate:
    """A fourth central moment represented without a dense fourth tensor.

    Use ``Sample``, ``CV`` or ``Calibration`` to construct this estimate.
    The sample estimator retains centered observations. Filtered estimators
    retain an orthogonal basis and the even-core matrix of squared rotated
    returns. No odd-order moment is estimated by this object.

    Parameters
    ----------
    scales : ndarray of float64, shape (n_assets,)
        Positive current sample marginal volatilities.
    basis : ndarray of float64, shape (n_assets, n_assets), optional
        Orthogonal fourth-order basis in standardized coordinates.
    core : ndarray of float64, shape (n_assets, n_assets), optional
        Even-core matrix, produced by the library's estimation procedures.
        Arbitrary user-created cores are not a supported estimation method.
    centered : ndarray of float64, shape (n_observations, n_assets), optional
        Raw centered return observations for the sample estimator. Supply
        this instead of ``basis`` and ``core``.
    assets : sequence of hashable, optional
        Unique asset labels in column order.

    Attributes
    ----------
    scales, basis, core, centered : ndarray of float64 or None
        Read-only numerical representation. Unused components are None.
    assets : tuple of hashable or None
        Labels used to align pandas weight Series.

    Examples
    --------
    >>> from hoao import Sample
    >>> fourth = Sample(even=True).fit_cokurtosis([[1.], [2.], [3.]])
    >>> round(fourth.fourth_moment([1.]), 6)
    0.666667
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
        """Evaluate the portfolio fourth central moment in return units.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Finite positions; short positions and non-unit budgets are
            allowed. A Series is aligned by exact asset labels.

        Returns
        -------
        float
            Fourth central moment, without annualization. A filtered
            estimate costs O(n_assets**2); the sample estimate costs
            O(n_observations * n_assets).

        Examples
        --------
        >>> from hoao import Sample
        >>> k = Sample().fit_cokurtosis([[-1.], [0.], [1.]])
        >>> round(k.fourth_moment([1.]), 6)
        0.666667
        """
        w = weights_array(weights, len(self.scales), self.assets)
        return self._value_gradient(w)[0]

    def gradient(self, weights: WeightInput) -> FloatArray:
        """Differentiate the fourth central moment with respect to weights.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Evaluation weights. A Series must match all asset labels.

        Returns
        -------
        ndarray of float64, shape (n_assets,)
            Analytic gradient in model asset order, in return units.

        Examples
        --------
        >>> from hoao import Sample
        >>> k = Sample().fit_cokurtosis([[-1.], [0.], [1.]])
        >>> round(float(k.gradient([1.])[0]), 6)
        2.666667
        """
        return self._value_gradient(weights_array(weights, len(self.scales), self.assets))[1]

    def hessian_vector_product(self, weights: WeightInput, direction: WeightInput) -> FloatArray:
        """Apply the fourth-moment Hessian to a direction without forming it.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Weights at which to evaluate the derivative.
        direction : array_like or pandas.Series, shape (n_assets,)
            Direction vector, aligned using the same label rules as weights.

        Returns
        -------
        ndarray of float64, shape (n_assets,)
            Hessian times direction in model asset order. The compact
            even-core path requires O(n_assets**2) operations.

        Examples
        --------
        >>> from hoao import Sample
        >>> k = Sample().fit_cokurtosis([[-1.], [0.], [1.]])
        >>> import numpy as np
        >>> np.allclose(k.hessian_vector_product([1.], [1.]), [8.0])
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
        """Explicitly materialize the fourth-order tensor under a memory cap.

        Parameters
        ----------
        standardized : bool, default False
            If True, divide each tensor axis by its marginal volatility.
            Otherwise return the tensor in raw return units.
        max_bytes : int, default 536870912
            Conservative cap for dense output plus work arrays. The method
            raises before allocation if the estimate exceeds this cap.

        Returns
        -------
        ndarray of float64, shape (n_assets,) * 4
            Central cokurtosis. Axes follow the input asset order. In a
            C-order unfolding, the last index varies fastest.

        Raises
        ------
        MemoryError
            If estimated dense working memory exceeds ``max_bytes``.

        Notes
        -----
        Fitting, calibration and portfolio methods never call this method.
        Dense tensors require O(n_assets**4) memory; use portfolio methods
        for routine risk calculations.

        Examples
        --------
        >>> from hoao import Sample
        >>> k = Sample().fit_cokurtosis([[-1.], [0.], [1.]])
        >>> k.to_dense().shape
        (1, 1, 1, 1)
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
    """Joint second- and fourth-moment estimates from one fitting window.

    Obtain a model from ``Sample.fit``, ``CV.fit`` or ``Calibration.fit``.
    The mean is always the raw sample mean; filtering affects dependence.

    Parameters
    ----------
    mean : ndarray of float64, shape (n_assets,)
        Sample mean of the current fitting window.
    correlation_estimate : CorrelationEstimate
        Correlation and current marginal volatilities.
    cokurtosis_estimate : CokurtosisEstimate
        Fourth moment fitted to the same asset order and scale.
    reference_variance : float
        Raw sample variance of equal-weight returns in the fitting window.
        This common scale is used for every optimization penalty.
    assets : sequence of hashable, optional
        Unique input labels. Must agree with the component estimates.

    Attributes
    ----------
    mean : ndarray of float64, shape (n_assets,)
        Read-only sample return means.
    correlation, covariance : ndarray of float64
        Read-only square estimated matrices.
    cokurtosis : CokurtosisEstimate
        Compact fourth central moment, retaining raw return scales.
    reference_variance : float
        Raw equal-weight sample variance, without annualization.
    assets : tuple of hashable or None
        Asset labels in output array order.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import CV
    >>> returns = np.random.default_rng(1).normal(size=(100, 4)) * 0.01
    >>> model = CV(random_state=2).fit(returns)
    >>> model.portfolio(np.full(4, 0.25)).volatility > 0
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
        """Compute portfolio mean, variance, volatility and fourth moments.

        Parameters
        ----------
        weights : array_like or pandas.Series, shape (n_assets,)
            Finite weights, including short positions. They need not sum
            to one. Series are aligned by label, never silently by position.

        Returns
        -------
        PortfolioMoments
            All moments use the original return frequency and units.
            Kurtosis is Pearson kurtosis, or NaN for zero variance.

        Notes
        -----
        Separately filtered covariance and cokurtosis need not be moments
        of a single distribution. Kurtosis below one is reported unchanged.

        Examples
        --------
        >>> from hoao import Sample
        >>> risk = Sample().fit([[-1.], [0.], [1.]]).portfolio([1.])
        >>> round(risk.kurtosis, 6)
        1.5
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
        """Minimize normalized variance plus a fourth-moment penalty.

        Parameters
        ----------
        gamma : float, default 0.0
            Nonnegative fourth-moment penalty. Zero gives global minimum
            variance. The objective is variance / s2 + gamma * m4 / s2**2,
            where s2 is ``reference_variance``.
        tol : float, default 1e-8
            Target infinity norm of the reduced objective gradient.
        max_iter : int, default 1000
            Maximum quasi-Newton iterations before final Newton polishing.

        Returns
        -------
        OptimizationResult
            Unit-budget weights, estimated moments and convergence metrics.

        Raises
        ------
        OptimizationError
            If feasibility or stationarity checks fail. The exception's
            ``result`` attribute contains the diagnostic candidate.
        ValueError
            If the penalty is negative or the reference variance is zero.

        Notes
        -----
        Short sales are allowed. No expected-return target, leverage cap,
        transaction cost or compatibility correction is imposed.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> model = Sample().fit(np.random.default_rng(2).normal(size=(60, 3)))
        >>> bool(abs(model.optimize().weights.sum() - 1) < 1e-12)
        True
        """
        from .optimization import optimize
        return optimize(self, gamma, tol=tol, max_iter=max_iter)

    def frontier(self, gammas: Sequence[float], *, tol: float = 1e-8, max_iter: int = 1000) -> list[OptimizationResult]:
        """Optimize a sequence of penalties with successive warm starts.

        Parameters
        ----------
        gammas : sequence of float
            Nonnegative penalties, evaluated in the supplied order.
        tol : float, default 1e-8
            Target reduced-gradient infinity norm for each portfolio.
        max_iter : int, default 1000
            Maximum iterations per optimization.

        Returns
        -------
        list of OptimizationResult
            Results in input order. These are fitted allocations; an
            out-of-sample frontier requires separate future observations.

        Examples
        --------
        >>> import numpy as np
        >>> from hoao import Sample
        >>> model = Sample().fit(np.random.default_rng(2).normal(size=(60, 3)))
        >>> len(model.frontier([0., 1., 10.]))
        3
        """
        from .optimization import optimize
        result: list[OptimizationResult] = []
        initial: FloatArray | None = None
        for gamma in gammas:
            item = optimize(self, gamma, tol=tol, max_iter=max_iter, initial=initial)
            result.append(item)
            initial = item.weights
        return result
