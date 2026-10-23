"""Variance–fourth-moment allocation under a unit-budget constraint.

Use MomentModel.optimize for one penalty or MomentModel.frontier for
several. Both return weights together with portfolio moments and an
independent check of feasibility and stationarity.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from ._typing import FloatArray
from ._validation import readonly
from .models import MomentModel, PortfolioMoments

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class OptimizationResult:
    """Portfolio weights, fitted moments and convergence diagnostics.

    Returned by :meth:`MomentModel.optimize` and :meth:`MomentModel.frontier`.
    Failed solves attach the same record to :class:`OptimizationError`,
    allowing the candidate and its diagnostics to be inspected.

    Attributes
    ----------
    weights : ndarray of float64, shape (n_assets,)
        Read-only allocation in model asset order. Successful allocations
        sum to one within ``1e-10``; individual weights may be negative.
    moments : PortfolioMoments
        Estimated moments of this allocation in the original return
        units and frequency, without annualization.
    objective : float
        ``variance / s2 + gamma * fourth_moment / s2**2``, where ``s2``
        is the model's raw equal-weight sample variance.
    iterations : int
        Total iterations across the initial solve and any refinement.
        Zero for the direct minimum-variance solve or a single asset.
    gradient_residual : float
        Infinity norm of the objective gradient in orthonormal
        coordinates tangent to the budget constraint. A stationary
        allocation has zero residual. The acceptance threshold is
        ``max(1e-6, 10 * tol)`` for the requested solver tolerance.
    success : bool
        Whether weights and objective are finite and the budget and
        gradient-residual checks passed. This is checked independently
        of the underlying SciPy termination flag.
    gamma : float
        Fourth-moment penalty used in this solve.
    message : str
        Solver termination message, including any refinement step.
    assets : tuple of hashable or None
        Asset labels in weight order, or None for unlabeled data.

    Notes
    -----
    Convergence describes the optimization of the fitted model. The
    reported moments are estimates from that model, not subsequent
    realized portfolio performance.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import Sample
    >>> returns = 0.01 * np.random.default_rng(4).normal(size=(100, 3))
    >>> model = Sample().fit(returns)
    >>> allocation = model.optimize(gamma=1.0)
    >>> allocation.success, allocation.gradient_residual <= 1e-6
    (True, True)
    >>> bool(np.isclose(allocation.weights.sum(), 1.0))
    True
    >>> allocation.moments == model.portfolio(allocation.weights)
    True
    """

    weights: FloatArray
    moments: PortfolioMoments
    objective: float
    iterations: int
    gradient_residual: float
    success: bool
    gamma: float
    message: str
    assets: tuple[Hashable, ...] | None

    def weights_to_pandas(self) -> pd.Series[float]:
        """Return the allocation as a labeled pandas Series.

        Returns
        -------
        pandas.Series of float64, shape (n_assets,)
            An independent copy named ``'weight'``. The index contains the
            model's asset labels in fitting-data order, or integer positions
            if the model was fitted from an unlabeled array.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from hoao import Sample
        >>> returns = pd.DataFrame(
        ...     np.random.default_rng(4).normal(size=(100, 3)),
        ...     columns=["A", "B", "C"],
        ... )
        >>> allocation = Sample().fit(returns).optimize()
        >>> weights = allocation.weights_to_pandas()
        >>> weights.index.tolist(), weights.name
        (['A', 'B', 'C'], 'weight')
        """
        import pandas as pd
        return pd.Series(self.weights.copy(), index=self.assets, name="weight")


class OptimizationError(RuntimeError):
    """An allocation failed the feasibility or stationarity checks.

    Raised by :meth:`MomentModel.optimize` or :meth:`MomentModel.frontier`
    when the numerical candidate does not meet the acceptance thresholds.
    The candidate is retained for diagnosis.

    Parameters
    ----------
    result : OptimizationResult
        Result of the unsuccessful solve, with ``success=False``.

    Attributes
    ----------
    result : OptimizationResult
        Candidate weights, fitted moments, objective, iteration count and
        gradient residual. Inspect ``result.message`` for the solver's
        explanation of termination.

    Examples
    --------
    Catch the exception when a larger workflow needs to record a failure
    and continue. Here the ordinary solve succeeds; the same code exposes
    the candidate if a later dataset causes convergence to fail:

    >>> import numpy as np
    >>> from hoao import OptimizationError, Sample
    >>> returns = np.random.default_rng(4).normal(size=(100, 3))
    >>> model = Sample().fit(returns)
    >>> try:
    ...     result = model.optimize(gamma=1.0)
    ... except OptimizationError as error:
    ...     result = error.result
    >>> result.success
    True
    """

    result: OptimizationResult

    def __init__(self, result: OptimizationResult) -> None:
        self.result = result
        super().__init__(f"Portfolio optimization failed (gradient residual {result.gradient_residual:.3g}): {result.message}")


def optimize(
    model: MomentModel, gamma: float = 0.0, *, tol: float = 1e-8,
    max_iter: int = 1000, initial: FloatArray | None = None,
) -> OptimizationResult:
    """Internal solver used by the documented MomentModel interface."""
    if not np.isfinite(gamma) or gamma < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("gamma must be finite and nonnegative; tol must be positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    scale = model.reference_variance
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Optimization needs a strictly positive equal-weight reference variance.")
    n = len(model.mean)
    base = np.ones(n) / n
    # Householder maps the first coordinate to the budget direction. The
    # remaining coordinates parameterize all and only sum-to-one portfolios.
    householder = np.ones(n) / np.sqrt(n)
    householder[0] += 1
    householder /= np.linalg.norm(householder)

    def reflect(vector: FloatArray) -> FloatArray:
        return vector - 2 * householder * (householder @ vector)

    def project(vector: FloatArray) -> FloatArray:
        return reflect(vector)[1:]

    def lift(vector: FloatArray) -> FloatArray:
        embedded = np.empty(n)
        embedded[0], embedded[1:] = 0, vector
        return reflect(embedded)

    def full(w: FloatArray) -> tuple[float, FloatArray]:
        sw = model.covariance @ w
        value, gradient = float(w @ sw) / scale, 2 * sw / scale
        if gamma:
            fourth, derivative = model.cokurtosis._value_gradient(w)
            value += gamma * fourth / scale**2
            gradient += gamma * derivative / scale**2
        return value, gradient

    def reduced(z: FloatArray) -> tuple[float, FloatArray]:
        value, gradient = full(base + lift(z))
        return value, project(gradient)

    def hessian_product(z: FloatArray, p: FloatArray) -> FloatArray:
        direction = lift(p)
        product = 2 * (model.covariance @ direction) / scale
        if gamma:
            product += gamma * model.cokurtosis.hessian_vector_product(base + lift(z), direction) / scale**2
        return project(product)

    iterations, message = 0, "Single feasible allocation"
    if n == 1:
        weights, residual = base, 0.0
    else:
        z = project(initial - base) if initial is not None else np.zeros(n - 1)
        if gamma == 0:
            covariance = model.covariance / scale
            product = covariance @ householder
            rotated = covariance - 2 * np.outer(householder, product) - 2 * np.outer(product, householder)
            rotated += 4 * float(householder @ product) * np.outer(householder, householder)
            matrix, rhs = rotated[1:, 1:], -project(covariance @ base)
            # Least squares selects the minimum Euclidean norm coordinates
            # if a genuine numerical nullspace makes the optimum nonunique.
            if np.linalg.cond(matrix) < 1 / np.sqrt(np.finfo(float).eps):
                z = np.asarray(np.linalg.solve(matrix, rhs), dtype=np.float64)
                message = "Direct budget-constrained covariance solve"
            else:
                z = np.asarray(np.linalg.lstsq(matrix, rhs, rcond=None)[0], dtype=np.float64)
                message = "Minimum-norm solve for singular covariance"
        else:
            solution = minimize(reduced, z, method="BFGS", jac=True,
                                options={"gtol": tol, "maxiter": max_iter})
            z = np.asarray(solution.x, dtype=np.float64)
            iterations, message = int(solution.nit), str(solution.message)
            if np.isfinite(z).all() and np.max(np.abs(reduced(z)[1])) > tol:
                # Hessian-vector polishing avoids constructing a fourth tensor
                # or a full dense Hessian; it is useful after precision loss.
                polish = minimize(reduced, z, method="trust-ncg", jac=True,
                                  hessp=hessian_product,
                                  options={"gtol": tol, "maxiter": min(max_iter, 100)})
                candidate = np.asarray(polish.x, dtype=np.float64)
                if np.isfinite(candidate).all() and np.linalg.norm(reduced(candidate)[1]) < np.linalg.norm(reduced(z)[1]):
                    z = candidate
                iterations += int(polish.nit)
                message += "; Hessian-vector polishing: " + str(polish.message)
        weights = base + lift(z)
        residual = float(np.max(np.abs(reduced(z)[1])))
    objective = full(weights)[0]
    success = bool(np.isfinite(weights).all() and np.isfinite(objective)
                   and abs(float(weights.sum()) - 1) <= 1e-10
                   and residual <= max(1e-6, 10 * tol))
    if np.isfinite(weights).all():
        moments = model.portfolio(weights)
    else:
        moments = PortfolioMoments(*(float("nan"),) * 5)
    result = OptimizationResult(readonly(weights), moments, objective, iterations, residual,
                                success, float(gamma), message, model.assets)
    if not success:
        raise OptimizationError(result)
    return result
