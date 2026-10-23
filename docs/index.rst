HOAO
====

Nonlinear shrinkage of covariance and cokurtosis for large portfolios.

HOAO estimates second- and fourth-order dependence from asset returns. It
implements the sample, cross-validated and historical estimators described in
*Nonlinear Shrinkage for Higher Moments Estimation in Large Portfolios* by
Paul Ruelloux, Christian Bongiorno and Damien Challet. Fitted models evaluate
portfolio moments and optimize the tradeoff between variance and the fourth
central moment.

Start with the :doc:`user guide <guide>` for complete workflows with NumPy
and pandas, including historical calibration and changing asset universes.
The :doc:`API reference <api>` documents individual objects and methods with
array shapes, return types and executable examples. The
:doc:`mathematical conventions <mathematics>` specify the estimators,
normalization and compact fourth-order representation.

Input arrays have observations in rows and assets in columns. All estimates
use central moments with population normalization (``ddof=0``), retaining the
units and frequency of the supplied returns. The CPU implementation uses
NumPy and SciPy; optional CuPy support is experimental. See
:doc:`performance` for measured CPU workloads and their conditions.

.. toctree::
   :maxdepth: 2

   guide
   mathematics
   performance
   api
   development
