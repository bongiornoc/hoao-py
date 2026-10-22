hoao
====

Estimate correlation and cokurtosis, calibrate historical oracle coefficients,
and evaluate or optimize portfolio moments with a compact numerical model.

All public numerical arrays use observations in rows and assets in columns.
NumPy is the default; pandas preserves meaningful asset labels, and optional
CuPy acceleration is explicit. The implementation uses central moments with
population normalization (``ddof=0``).

.. toctree::
   :maxdepth: 2

   guide
   mathematics
   performance
   api
   development
