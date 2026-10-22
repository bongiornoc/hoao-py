"""Higher-order moment estimation and portfolio allocation.

Fit sample, cross-validated or historically calibrated moments using
observation-by-asset arrays. Numerical results use float64 and the original
return units. See the individual objects for executable examples.
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
