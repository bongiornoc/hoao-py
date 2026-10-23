User guide
==========

This guide explains how HOAO interprets return data, how its estimators
differ, how historical calibration handles time and missing data, and what
the optimizer computes. The :doc:`mathematical notes <mathematics>` define the
estimators formally, and the :doc:`API reference <api>` lists every parameter.

Return data and units
---------------------

Every estimator takes a return matrix of shape
``(n_observations, n_assets)``: one row per observation, in chronological
order, and one column per asset. All columns must share the same units and
frequency. HOAO uses the values exactly as given. It does not convert prices
to returns, percentages to fractions or log returns to simple returns, and it
never annualizes. The examples use simple daily returns written as fractions,
so ``0.01`` is a one-percent daily gain and every output is a daily quantity:
volatility in daily return units, variance in their squares and the fourth
moment in their fourth powers.

.. code-block:: python

   import numpy as np
   from hoao import CV

   rng = np.random.default_rng(7)
   x = 0.01 * rng.standard_t(df=8, size=(120, 6))

   model = CV(random_state=42).fit(x)
   moments = model.portfolio(np.full(6, 1 / 6))
   excess_kurtosis = moments.kurtosis - 3

Means, variances and co-moments are sample estimates, with sums divided by
the number of observations (``ddof=0``). Portfolio kurtosis is Pearson
kurtosis, the fourth central moment divided by the squared variance; it equals
3 for a Gaussian distribution and is NaN for a portfolio with zero variance.
The model's ``mean`` is always the raw sample mean, because the estimators
filter dependence, not expected returns.

A fitting window must contain at least three rows of finite values, and every
asset must have positive variance in it. Missing values are rejected rather
than imputed, so clean or select the window before fitting. ``CV`` also needs
at least three rows on each side of its splits: both
``floor(train_fraction * n_observations)`` and the remaining count must be at
least three. Windows with more assets than observations are supported.

``MomentModel.portfolio`` accepts any finite weights, including short
positions and weights that do not sum to one; only optimization imposes a
budget. It returns a ``PortfolioMoments`` record with ``mean``, ``variance``,
``volatility``, ``fourth_moment`` and ``kurtosis``.

Labels and alignment
--------------------

NumPy input is positional: asset ``i`` is column ``i``, and NumPy weights are
read in that order. A pandas DataFrame attaches its column labels to the
fitted model, and its index must be unique and increasing. Asset labels must
be unique, hashable and not missing.

A pandas Series of weights is aligned by label. It must contain each of the
model's assets exactly once and nothing else: a missing, unknown or duplicated
label is an error, as is a Series passed to a model fitted on an unlabeled
array. The same rules apply to the weights and directions accepted by
``CokurtosisEstimate.fourth_moment``, ``gradient`` and
``hessian_vector_product``.

.. code-block:: python

   import pandas as pd
   from hoao import Sample

   frame = pd.DataFrame(x, columns=["A", "B", "C", "D", "E", "F"])
   labeled = Sample().fit(frame)
   weights = pd.Series({"F": 0.5, "E": 0.5, "D": 0.0, "C": 0.0, "B": 0.0, "A": 0.0})
   labeled.portfolio(weights)  # matched by label, not by position

Model attributes are read-only NumPy float64 arrays, so the numerical API
works without pandas. ``CorrelationEstimate.to_pandas(kind="correlation")``,
``CorrelationEstimate.to_pandas(kind="covariance")`` and
``OptimizationResult.weights_to_pandas()`` return labeled copies; a model
fitted from an array is labeled by integer position.

Estimators
----------

Every estimator standardizes the fitting window, estimates dependence on that
scale and then restores the window's sample volatilities, so all of them share
the sample mean and the marginal variances. They differ in how they treat
correlation and cokurtosis.

``Sample()`` uses the sample correlation and the exact sample fourth central
moment, stored as the centered returns. A portfolio's fourth moment then costs
O(``n_observations * n_assets``) operations.

``Sample(even=True)``, the estimator called MLE_even in the manuscript, keeps
the sample correlation and projects the fourth moment. Standardized returns
are rotated into the HOSVD basis of their fourth-moment tensor: the left
singular vectors of its ``n_assets`` by ``n_assets**3`` unfolding, ordered by
decreasing singular value. Only the core entries in which every index appears
an even number of times are retained. With ``y`` the rotated returns, these
entries are determined by the matrix
``B[i, j] = mean(y[:, i]**2 * y[:, j]**2)``, so the estimate is stored and
evaluated with O(``n_assets**2``) numbers. Nothing is calibrated.

``CV(n_splits=10, train_fraction=0.65, random_state=None)`` cross-validates
both moments within the fitting window. The complete window is centered and
divided by its sample volatilities. Each of the ``n_splits`` random partitions
assigns ``floor(train_fraction * n_observations)`` rows to training and the
rest to testing, and recenters both parts without rescaling them. The training
rows determine the correlation eigenvectors and the fourth-order basis; the
test rows measure the variance along each eigenvector and the even core in
that basis. Averaged over partitions and matched by rank, these values replace
the eigenvalues and the even core of the full window, whose bases are kept.
The correlation is then renormalized to unit diagonal.

An integer ``random_state`` makes the partitions reproducible across
instances. A NumPy ``Generator`` is copied, so the caller's generator does not
advance, and ``None`` draws fresh randomness when the instance is created. An
instance reuses its partitions for inputs of the same length, so its ``fit``,
``fit_correlation`` and ``fit_cokurtosis`` results agree with one another.

Historical calibration, described below, uses the same basis and core
construction but standardizes every training and test window separately, with
its own mean and volatilities. This differs from the single standardization of
the complete window in ``CV``; both conventions are part of the methods'
definitions.

The filtered estimators treat the second and fourth moments separately, each
in its own basis, as the manuscript's main estimators do. Nothing forces the
pair to be the moments of one probability distribution, so a fitted portfolio
can show a Pearson kurtosis below one, a value no real distribution attains.
HOAO reports such values unchanged. Compatibility corrections, filtered
coskewness and tail-quantile estimators are not implemented.

When eigenvalues are tied or nearly tied, the even core depends on the basis
chosen inside the tied eigenspace. HOAO applies a deterministic
coordinate-projector convention, and a rank-deficient window, including one
with more assets than observations, uses the row space of its singular value
decomposition completed by a canonical basis of the null space. This keeps
fitting free of dense tensors, but in tied cases the results can differ from a
research implementation that lets a dense eigensolver choose the basis. Seeds
fix the random splits and samples; floating-point results can still differ
slightly across BLAS libraries and devices, most visibly near ties. The
:doc:`mathematical notes <mathematics>` give the details.

Historical calibration
----------------------

A historical calibration learns the corrections from many windows of a longer
history instead of the current window. Its input is a ``ReturnsPanel``, which
holds the return history and, optionally, the assets available on each date.

Building a panel
~~~~~~~~~~~~~~~~

A wide table with a ``date`` column and one return column per asset can be
read with pandas; ``pd.read_parquet`` works the same way when a Parquet engine
is installed.

.. code-block:: python

   import pandas as pd
   from hoao import ReturnsPanel

   frame = pd.read_csv("returns.csv", index_col="date", parse_dates=["date"])
   panel = ReturnsPanel(frame)

``ReturnsPanel(returns, *, dates=None, assets=None, available=None)`` copies
the returns to float64. NaN marks an unobserved return and is kept until a
calibration applies its missing-data rules; infinity is rejected. A DataFrame
supplies its dates from the index and its asset names from the columns, so
``dates`` and ``assets`` are only for arrays. Dates must be unique and
strictly increasing. Timezone-aware dates are converted to UTC, and all dates
must fit NumPy's nanosecond range, roughly the years 1677 to 2262.

Without dates, a panel is positional: rows are identified by their integer
position, and ``before``, ``start``, ``as_of`` and the keys of an availability
mapping must be row positions as well. A DataFrame with an integer index is
treated as positional and must be labeled ``0, 1, ..., n_observations - 1``.

Asset availability
~~~~~~~~~~~~~~~~~~

The ``available`` argument states which assets could have been selected on
each date: the listed stocks, the historical constituents of an index or the
assets passing your own liquidity screens, for example. It must be built only
from information known on that date. It can be a boolean array with the shape
of the returns, a boolean DataFrame with exactly the panel's dates and assets
as index and columns, a mapping from dates (or row positions) to lists of
asset labels, or a sequence with one list of labels per row. A date absent
from a mapping has an empty universe.

On a signal date, an asset is eligible if it is available on that date and its
training window, the ``train_size`` observations ending on that date, is
finite and contains at least two distinct values. Without ``available``, every
asset is a candidate and only the training-window checks apply. These checks
cannot reveal listing status, liquidity, security type or index membership,
so supply availability whenever such rules matter. Test windows never
influence eligibility. ``panel.eligible(train_size=...)`` returns the
resulting boolean mask, which shows how many assets each date offers.

Sampling pairs
~~~~~~~~~~~~~~

A calibration pair starts from a signal row ``s``. Its training window is rows
``s - train_size + 1`` through ``s``, the next ``gap`` rows are skipped, and
its test window is the ``test_size`` rows after them. Both windows must have
at least three rows. The default ``gap=1`` follows the paper's next-close
execution: a signal computed after the close of row ``s`` is traded at
``s + 1``, and its returns are measured from ``s + 2``. With ``gap=0`` the
windows are adjacent.

A signal row is valid if its training window lies inside the panel, it is not
earlier than ``start`` when one is given, its whole test window falls strictly
before ``before``, and at least ``n_assets`` assets are eligible on it. Each
pair draws a valid signal row uniformly at random and then ``n_assets`` of its
eligible assets uniformly without replacement. After each window is
standardized with its own mean and volatilities, the training window supplies
the correlation eigenvectors and the fourth-order basis, and the test window
supplies the variance along each eigenvector and the even core in that basis.
Every pair counts equally in the averages, which are available as
``correlation_core``, one value per eigenvalue rank, and ``cokurtosis_core``,
an ``n_assets`` by ``n_assets`` matrix.

.. code-block:: python

   from hoao import ReturnsPanel, calibrate

   history = 0.01 * rng.standard_t(df=8, size=(300, 8))
   panel = ReturnsPanel(history)  # positional: cutoffs are row numbers
   calibration = calibrate(
       panel, n_assets=4, train_size=60, test_size=20,
       n_pairs=100, before=200, random_state=42,
   )
   calibrated = calibration.fit(history[-60:, :4], as_of=299)

``random_state`` must be a nonnegative integer or ``None``, in which case a
fresh seed is drawn. Either way the seed is stored with the calibration. Pair
``k`` draws from a random stream derived from the seed and ``k``, so for a
given panel and set of bounds the result depends neither on ``n_jobs`` nor on
how the pairs are divided among updates.

Missing returns
~~~~~~~~~~~~~~~

Missing training returns are never filled; they make the asset ineligible on
that signal date. Test windows are not consulted during selection, so a
selected asset that stops trading keeps its place in the pair. With the
default ``missing="zero"``, its missing test returns become zero, the paper's
convention for a position that exits to cash. A test column that is missing
entirely is therefore constant and contributes zero after standardization.
The rule applies to every NaN in a test window, whatever its cause, and a zero
return does not reconstruct actual liquidation proceeds, so encode the panel
accordingly.

Each replacement is recorded: ``filled_count`` gives their number and
``filled_locations`` lists them as (pair ID, panel row, panel column) in pair
order, with positions referring to the panel used for that pair. With
``missing="raise"``, calibration stops at the first missing test return
without substituting assets or dates.

.. code-block:: python

   gappy = history.copy()
   gappy[150:, 7] = np.nan  # the last asset stops trading at row 150
   audited = calibrate(
       ReturnsPanel(gappy), n_assets=4, train_size=60, test_size=20,
       n_pairs=100, before=200, random_state=42,
   )
   print(audited.filled_count, audited.filled_locations[:3])

Applying a calibration
~~~~~~~~~~~~~~~~~~~~~~

``Calibration.fit(returns, *, as_of=None)`` takes exactly ``train_size`` rows
and ``n_assets`` columns of finite returns, each column with positive
variance. The assets need not be those seen during calibration. Coefficients
are indexed by the rank of each basis direction, so a calibration applies to
any group of the same size, whereas another size or window length requires a
new calibration. The fitted model combines the window's sample mean and
volatilities with the calibrated values placed in the window's own bases.
``fit_correlation`` and ``fit_cokurtosis`` compute a single moment.

``as_of`` is the decision date. It may not precede the last fitting
observation or the calibration's ``before`` cutoff, and every test observation
used by the calibration must be strictly earlier. A DataFrame with a date
index supplies ``as_of`` from its last row; for arrays it is required. Use
dates with a dated calibration and row positions with a positional one. The
check is only as reliable as the date supplied: never pass an ``as_of`` later
than the moment the decision is actually made.

Saving and updating
~~~~~~~~~~~~~~~~~~~

``Calibration.save`` writes a compressed NumPy archive containing the running
sums, the pair count, the seed, the cutoffs, the dimensions, the missing-data
policy and the fill records, as arrays and versioned JSON metadata. It uses no
pickle and stores no return data. The archive is written under a temporary
name and then moved into place, so a failed save leaves an existing file
intact. ``Calibration.load`` checks the format version, the basis convention,
the array shapes and values and the chronology, and rejects inconsistent
files.

``Calibration.update(panel, *, n_pairs, before, start=None)`` draws
``n_pairs`` further pairs, numbered after the existing ones, and adds them to
the running sums, so every pair keeps equal weight. The panel must follow the
original convention, dated or positional, although its rows and assets may
differ. ``before`` may stay the same or move forward, never back. ``start``
restricts the new signal dates, while their training windows may extend
before it; without ``start``, the entire valid history is sampled again. The
calibration changes only once every requested pair has succeeded.

.. code-block:: python

   from hoao import Calibration

   calibration.save("core.npz")
   restored = Calibration.load("core.npz")
   restored.update(panel, n_pairs=50, before=250, start=200)
   calibrated = restored.fit(history[-60:, :4], as_of=299)

Splitting a calibration into several updates over the same panel and bounds
reproduces a single run exactly. Changing ``before`` or ``start`` changes the
distribution from which the new pairs are drawn.

Allocation
----------

``MomentModel.optimize(gamma=0.0, *, tol=1e-8, max_iter=1000)`` solves

.. math::

   \min_w \; \frac{w^\top C w}{s^2} + \gamma \, \frac{m_4(w)}{s^4}
   \quad \text{subject to} \quad \sum_i w_i = 1,

where :math:`C` is the fitted covariance, :math:`m_4(w)` the fitted fourth
central moment of the portfolio and :math:`s^2` the model's
``reference_variance``, the sample variance (``ddof=0``) of the equally
weighted portfolio over the fitting window. The reference is computed from the
raw returns, so it is the same for every estimator fitted to the same window,
whereas the equal-weight variance implied by a filtered covariance generally
differs. Dividing by :math:`s^2` and :math:`s^4` makes the objective
dimensionless and gives ``gamma`` a common meaning across those estimators.

The budget is the only constraint. Short positions are allowed, and there are
no position bounds, leverage limits, transaction costs or expected-return
targets. The objective penalizes the fourth central moment, not standardized
kurtosis, so increasing ``gamma`` does not guarantee a lower kurtosis.

With ``gamma=0`` the problem is the global minimum-variance portfolio, solved
directly; if the covariance restricted to budget-preserving directions is
numerically singular, the minimum-norm solution is returned. A positive
``gamma`` is minimized by BFGS over the budget-constrained weights with
analytic gradients. If the gradient tolerance has not been reached, a
trust-region Newton-CG polish follows, using Hessian-vector products rather
than a dense Hessian. ``tol`` sets the target infinity norm of the reduced
gradient, and ``max_iter`` bounds the quasi-Newton iterations.

Before returning, ``optimize`` checks that the weights are finite and sum to
one within ``1e-10`` and that ``gradient_residual``, the infinity norm of the
gradient within the budget constraint, is at most ``max(1e-6, 10 * tol)``.
These checks establish feasibility and stationarity, not global optimality.
If they fail, ``OptimizationError`` is raised and its ``result`` attribute
holds the rejected candidate. For the model fitted at the start of this guide:

.. code-block:: python

   from hoao import OptimizationError

   try:
       allocation = model.optimize(gamma=1.0)
   except OptimizationError as error:
       print(error.result.gradient_residual, error.result.message)

``frontier(gammas)`` optimizes the penalties in the given order, starting each
solve from the previous weights, and returns the results in that order; it
raises ``OptimizationError`` if any solve fails. Each ``OptimizationResult``
reports ``weights``, ``moments``, ``objective``, ``iterations``,
``gradient_residual``, ``success``, ``gamma`` and the solver ``message``.
These describe the fitted model on its fitting window; evaluating an
allocation requires returns observed afterwards.

Performance and devices
-----------------------

Filtered fourth moments stay compact. For an estimate from
``Sample(even=True)``, ``CV`` or a calibration, ``fourth_moment(weights)``,
``gradient(weights)`` and ``hessian_vector_product(weights, direction)`` cost
O(``n_assets**2``) operations; for the exact sample estimate they cost
O(``n_observations * n_assets``). Fitting computes the fourth-order basis from
an ``n_assets`` by ``n_assets`` Gram matrix accumulated over blocks of
observations. Accumulating this matrix and diagonalizing it takes
O(``n_observations**2 * n_assets + n_observations * n_assets**2 + n_assets**3``)
operations, without constructing an ``n_assets**4`` array.

``to_dense(*, standardized=False, max_bytes=512 * 1024**2)`` materializes the
four-dimensional tensor for small problems and independent checks. Its axes
follow the asset order, and ``standardized=True`` divides each axis by the
asset's volatility. The output takes ``8 * n_assets**4`` bytes; the method
budgets four times that for output and work arrays, plus
``8 * n_observations * n_assets**2`` bytes for the exact sample estimate, and
raises ``MemoryError`` before allocating if the budget exceeds ``max_bytes``.

.. code-block:: python

   tensor = model.cokurtosis.to_dense(max_bytes=64 * 1024**2)  # shape (6, 6, 6, 6)

Calibration pairs are independent. ``n_jobs`` runs them on a pool of CPU
threads with at most twice that many pairs in flight, and CPU calibration
limits BLAS to one thread so that workers do not oversubscribe the cores.
Results are accumulated in pair order and are identical for any number of
workers. The fastest configuration depends on the numbers of assets and
observations and on the hardware; the :doc:`performance notes <performance>`
describe how the CPU benchmarks were run.

``device="cuda"`` requests the optional CuPy backend, installed with the
``hoao[cuda12]`` or ``hoao[cuda13]`` extra and requiring a working NVIDIA
driver. It is accepted by ``Sample``, ``CV``, ``calibrate``,
``Calibration.update`` and the ``Calibration`` fitting methods. GPU support is
experimental and has not been benchmarked. The device is never chosen
automatically, results are returned as NumPy float64 arrays, and validation,
standardization, pandas alignment, checkpoints and optimization remain on the
CPU. CUDA calibration requires ``n_jobs=1`` and processes pairs in batches of
at most four to bound device memory. Small problems can run faster on the CPU
because transfers and kernel launches have fixed costs.
