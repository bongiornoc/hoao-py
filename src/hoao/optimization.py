"""Budget-only variance/fourth-moment allocation with analytic derivatives."""
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
    """Portfolio allocation and numerical optimality diagnostics.

    Attributes
    ----------
    weights : ndarray of float64, shape (n_assets,)
        Read-only unit-budget allocation in fitting-data asset order.
    moments : PortfolioMoments
        Estimated moments of this allocation, without annualization.
    objective : float
        Variance / reference_variance + gamma * fourth / reference_variance**2.
    iterations : int
        Total optimizer iterations. Zero for a direct minimum-variance solve.
    gradient_residual : float
        Infinity norm of the objective gradient in orthonormal budget
        coordinates. Successful results have residual <= max(1e-6, 10*tol).
    success : bool
        Whether independent feasibility and stationarity checks passed.
    gamma : float
        Fourth-moment penalty used for this result.
    message : str
        Solver termination information.
    assets : tuple of hashable or None
        Asset labels for conversion to pandas.

    Examples
    --------
    >>> import numpy as np
    >>> from hoao import Sample
    >>> model = Sample().fit(np.random.default_rng(4).normal(size=(100, 3)))
    >>> allocation = model.optimize(gamma=1.)
    >>> allocation.success
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
        """Return a labeled copy of the allocation weights.

        Returns
        -------
        pandas.Series
            Finite weights in model asset order, named ``weight``. Integer
            positions are used if the model has no asset labels.

        Raises
        ------
        ImportError
            If pandas is unavailable; install ``hoao[pandas]``.

        Examples
        --------
        >>> from hoao import Sample
        >>> result = Sample().fit([[-1.], [0.], [1.]]).optimize()
        >>> result.weights_to_pandas().tolist()
        [1.0]
        """
        import pandas as pd
        return pd.Series(self.weights.copy(), index=self.assets, name="weight")


class OptimizationError(RuntimeError):
    """An allocation did not pass feasibility and optimality checks.

    Parameters
    ----------
    result : OptimizationResult
        Diagnostic candidate; its ``success`` field is False.

    Attributes
    ----------
    result : OptimizationResult
        Candidate weights, moments and stationarity residual. The caller
        can inspect these without treating an unsuccessful solve as valid.

    Examples
    --------
    Catch this exception around ``model.optimize`` and inspect
    ``error.result.gradient_residual`` before changing numerical settings.
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
