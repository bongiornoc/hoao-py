"""HOAO: nonlinear shrinkage of covariance and cokurtosis for large portfolios.

Sample and CV estimate moments from a current return window. For
historical estimation, pass a ReturnsPanel to calibrate, then apply the
resulting Calibration to current returns. Each estimator returns a
MomentModel for portfolio evaluation and variance–fourth-moment optimization.

Inputs have one row per observation and one column per asset. NumPy arrays
and pandas DataFrames are accepted; numerical outputs are NumPy float64
arrays with explicit methods for labeled pandas conversion. Moments use
population normalization (ddof=0) and the original return units and
frequency. No annualization is applied.
"""
from .calibration import Calibration as Calibration, calibrate as calibrate
from .estimators import CV as CV, Sample as Sample
from .models import (
    CokurtosisEstimate as CokurtosisEstimate,
    CorrelationEstimate as CorrelationEstimate,
    MomentModel as MomentModel,
    PortfolioMoments as PortfolioMoments,
)
from .optimization import OptimizationError as OptimizationError, OptimizationResult as OptimizationResult
from .panel import ReturnsPanel as ReturnsPanel

__version__ = "0.1.0"
__all__ = [
    "CV", "Sample", "ReturnsPanel", "calibrate", "Calibration",
    "CorrelationEstimate", "CokurtosisEstimate", "MomentModel", "PortfolioMoments",
    "OptimizationResult", "OptimizationError", "__version__",
]
