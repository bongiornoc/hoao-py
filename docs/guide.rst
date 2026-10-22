User guide
==========

From returns to a fitted model
------------------------------

A return matrix has shape ``(n_observations, n_assets)``. Rows are ordered
observation sessions and columns identify assets. All asset returns must use
the same units and frequency. The examples use simple daily fractional
returns: ``0.01`` is a one-percent daily return. The package does not infer
whether your observations are daily, monthly, logarithmic, or percentage
values; those interpretations remain yours.

``Sample().fit(returns)`` retains exact sample moments.
``Sample(even=True).fit(returns)`` applies the uncalibrated even projection
to the fourth-order core. ``CV(random_state=42).fit(returns)`` calibrates the
core on random held-out subsets. Each returns a ``MomentModel``:

.. code-block:: python

   import numpy as np
   from hoao import CV

   rng = np.random.default_rng(7)
   x = 0.01 * rng.standard_t(df=8, size=(120, 6))
   model = CV(random_state=42).fit(x)
   moments = model.portfolio(np.full(6, 1 / 6))
   assert moments.volatility > 0

``fit_correlation`` computes only the second-order estimate;
``fit_cokurtosis`` computes only the fourth-order estimate. Choose these
when you do not need a combined model. Portfolio evaluation itself accepts
any finite real weights; it does not require them to sum to one. The
optimization methods impose that budget constraint.

Sample means, variances, and co-moments use population normalization
(``ddof=0``). Returned volatility is in the frequency of the observations.
Kurtosis is the dimensionless Pearson quantity, with Gaussian value three.
Subtract three yourself if you need excess kurtosis.

Labels and input validation
---------------------------

NumPy arrays are interpreted positionally. pandas DataFrames attach asset
names to columns and observation dates to their index. A weight Series is
aligned with the fitted model's asset names; a plain weight array follows
the fitted column order. Duplicate, missing, and unexpected asset labels
are rejected rather than silently dropped.

Fitting requires finite observations and nonconstant asset columns. Clean
or explicitly select your input window before fitting; no fitting-time
NaN imputation or price-to-return conversion is implicit. Constructing a
historical ``ReturnsPanel`` may retain missing returns because its role is
to choose valid historical training windows and apply an explicit policy to
subsequent targets.

Numerical outputs are ordinary arrays. Use
``CorrelationEstimate.to_pandas(kind="correlation")``,
``CorrelationEstimate.to_pandas(kind="covariance")``, or
``OptimizationResult.weights_to_pandas()`` for labeled presentation.

Historical calibration
----------------------

A dataset on disk can be read with pandas before constructing the panel.
For a wide CSV with one ``date`` column and one numeric return column per
asset:

.. code-block:: python

   import pandas as pd
   from hoao import ReturnsPanel

   frame = pd.read_csv("returns.csv", index_col="date", parse_dates=["date"])
   panel = ReturnsPanel(frame)

Rows must be chronological and dates unique. Missing return cells remain
missing until the calibration's explicit target policy is applied. For
Parquet, ``pd.read_parquet`` can supply the same table when an appropriate
pandas Parquet engine is installed separately.

``ReturnsPanel`` wraps one historical return table. Pass dates and asset
identifiers separately for arrays, or let a DataFrame supply them. If dates
are omitted, observation coordinates are integer row positions; ``before``
and ``as_of`` must then use those coordinates too.

The ``available`` input specifies the eligible assets on each signal date.
A boolean mask has the same shape as the return table. It can encode a
listing universe, an index's historical constituents, or your own screens.
Eligibility describes what was available on that date; it must not be built
using future observations. Without a supplied mask, historical finite-data
and positive-variance checks determine the usable universe. These checks
cannot reconstruct financial security metadata from returns alone.

For each pair, calibration selects a signal date and a random subset of
``n_assets`` valid assets. It takes ``train_size`` observations ending at
the signal date, skips ``gap`` observations, then takes ``test_size``
subsequent observations. All test observations must precede ``before``.
The default ``gap=1`` follows the paper's next-close execution timing.

.. code-block:: python

   from hoao import ReturnsPanel, calibrate

   history = 0.01 * rng.standard_t(df=8, size=(300, 8))
   panel = ReturnsPanel(history)
   calibration = calibrate(
       panel, n_assets=4, train_size=60, test_size=20,
       n_pairs=100, before=200, random_state=42,
   )
   model = calibration.fit(history[-60:, :4], as_of=299)

The test data contribute historical calibration targets, but never choose
the signal-date asset universe or the training basis. With the default
``missing="zero"``, missing (NaN) target entries become zero returns while the
original asset subset remains fixed. This follows the paper's cash/exit
convention and also covers other missing target entries. To reject missing
targets, use the strict policy documented on ``calibrate``. A zero return
does not reconstruct unknown liquidation proceeds.

The calibration is tied to its asset count, train/test lengths, preprocessing,
and timing convention. It is independent of the particular asset names: the
coefficients describe modes ordered by magnitude, so a calibration can be
applied to a new group with the same size. Supply an ``as_of`` value for
array inputs; dated DataFrames provide the final observation date. Never
claim a later ``as_of`` for observations used in an earlier decision.

``Calibration.save`` and ``Calibration.load`` persist coefficients and
metadata without Python pickle objects. ``Calibration.update`` incorporates
additional historical pairs into the existing running sums and count. Use
only observations and targets already available at the next cutoff. A
calibration checkpoint does not contain the original return dataset.

Allocation
----------

``model.optimize(gamma=0)`` returns a minimum-variance allocation.
``model.optimize(gamma=1)`` adds a normalized fourth-central-moment penalty.
``model.frontier([0, 0.1, 1, 10])`` evaluates a list of penalties and returns
their optimization results. Each result contains the weights, fitted moments,
objective value, iteration count, and convergence diagnostics.

The reference variance is measured from the original fitting window's
equally weighted portfolio. It is shared by estimators fitted to the same
observations. This is distinct from the fitted model's equal-weight variance,
which may change after filtering.

The optimizer allows short positions and imposes only the unit budget.
There are no return forecasts, long-only bounds, or leverage caps. The
objective uses the fourth central moment, not standardized kurtosis;
increasing its penalty does not ensure a monotonically decreasing kurtosis.
Inspect ``success`` and ``gradient_residual``. An in-sample frontier is not
an out-of-sample performance forecast.

Performance choices
-------------------

Keep filtered fourth moments in compact form. ``fourth_moment(weights)``,
``gradient(weights)``, and ``hessian_vector_product(weights, direction)``
operate on the compact representation. ``to_dense`` is an explicitly
requested export with a memory budget, intended for small examples and
independent verification.

CPU historical calibration can use ``n_jobs`` for independent pairs. The
default single job is straightforward and deterministic. Parallel execution
limits nested BLAS work to avoid multiplying worker count by BLAS thread
count. Numerical array layout, the number of observations, and the number
of assets all affect the best hardware choice.

An explicit ``device="cuda"`` requests optional CuPy kernels. GPU support
is experimental and is not locally benchmarked. Host preprocessing and
solver work may remain on the CPU. The package does not silently choose a
GPU based on availability. CUDA calibration requires ``n_jobs=1`` and uses
bounded batches of historical pairs internally.
