API reference
=============

All public objects can be imported directly from ``hoao``. ``Sample`` and
``CV`` estimate moments from a current return window. Historical estimation
starts with ``ReturnsPanel`` and ``calibrate``, which produce a reusable
``Calibration``. Each estimator provides ``fit`` for a joint ``MomentModel``
and ``fit_correlation`` or ``fit_cokurtosis`` when only one order is needed.

The examples below use small synthetic datasets. For a complete workflow,
including installation and data preparation, see the :doc:`guide`.

Current-window estimators
-------------------------

.. autoclass:: hoao.Sample
   :members: fit, fit_correlation, fit_cokurtosis

.. autoclass:: hoao.CV
   :members: fit, fit_correlation, fit_cokurtosis

Historical calibration
----------------------

``ReturnsPanel`` defines the history and the universe available on each date.
``calibrate`` samples historical window pairs; ``Calibration`` stores their
averaged coefficients and applies them to a later return window.

.. autoclass:: hoao.ReturnsPanel
   :members: eligible

.. autofunction:: hoao.calibrate

.. autoclass:: hoao.Calibration
   :members: fit, fit_correlation, fit_cokurtosis, update, save, load

Fitted estimates and portfolio moments
--------------------------------------

Numerical results are NumPy arrays. The fitted objects retain asset labels
from pandas input and align labeled weight vectors explicitly. Conversion
methods return independent pandas copies.

.. autoclass:: hoao.MomentModel
   :members: portfolio, optimize, frontier

.. autoclass:: hoao.CorrelationEstimate
   :members: to_pandas

.. autoclass:: hoao.CokurtosisEstimate
   :members: fourth_moment, gradient, hessian_vector_product, to_dense

.. autoclass:: hoao.PortfolioMoments

Optimization results
--------------------

Every successful solve returns its fitted moments and convergence
diagnostics. When the checks fail, ``OptimizationError.result`` retains the
candidate and the same diagnostics for inspection.

.. autoclass:: hoao.OptimizationResult
   :members: weights_to_pandas

.. autoclass:: hoao.OptimizationError
