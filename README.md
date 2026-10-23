# HOAO: nonlinear shrinkage of covariance and cokurtosis for large portfolios

HOAO estimates the covariance and cokurtosis of asset returns and uses them to
evaluate and optimize portfolios when tail risk matters as well as variance.
Cokurtosis is the fourth-order counterpart of the covariance matrix, with one
central co-moment for every quadruple of assets. Combined with portfolio
weights, it gives the fourth central moment of the portfolio return and hence
its kurtosis. With n assets it has n⁴ entries, and its sample estimate becomes
very noisy as the number of assets grows relative to the number of
observations.

The package implements the main estimators of *Nonlinear Shrinkage for Higher
Moments Estimation in Large Portfolios* by Paul Ruelloux, Christian Bongiorno
and Damien Challet. Correlation and cokurtosis are each expressed in an
orthogonal basis estimated from training observations, and their coefficients
in that basis are measured on testing observations. These come from random
splits of the current window (cross-validation) or successive windows drawn
from a historical dataset (historical calibration). Fourth-order
estimates are kept in compact form, two n × n matrices for the filtered
estimators, so no n⁴ array is formed unless you request one.

## Installation

HOAO requires Python 3.11 or later and depends on NumPy, SciPy and
threadpoolctl. It is not yet available on PyPI; install the
[v0.1.0 release](https://github.com/bongiornoc/hoao-py/releases/tag/v0.1.0)
from its Git tag:

```bash
pip install "hoao @ git+https://github.com/bongiornoc/hoao-py.git@v0.1.0"
```

Optional extras add pandas support (DataFrame input, labeled weights and
labeled output, used in several examples below) or CuPy for experimental GPU
execution. Install at most one CUDA extra, matching the CUDA version of your
NVIDIA setup:

```bash
pip install "hoao[pandas] @ git+https://github.com/bongiornoc/hoao-py.git@v0.1.0"
pip install "hoao[cuda12] @ git+https://github.com/bongiornoc/hoao-py.git@v0.1.0"  # or hoao[cuda13]
```

The core package does not need pandas, CuPy or a CUDA runtime.

## From a return matrix to portfolio moments

HOAO works on a matrix of returns with one row per observation and one column
per asset. The example simulates 240 days of heavy-tailed returns for eight
assets, fits the cross-validated estimator `CV`, evaluates the equally weighted
portfolio and then optimizes the weights:

```python
import numpy as np
from hoao import CV

rng = np.random.default_rng(7)
returns = 0.01 * rng.standard_t(df=8, size=(240, 8))  # 240 days, 8 assets

model = CV(random_state=42).fit(returns)
weights = np.full(8, 1 / 8)
moments = model.portfolio(weights)

print(model.covariance.shape)   # (8, 8)
print(moments.volatility)       # daily standard deviation
print(moments.kurtosis)         # Pearson kurtosis, 3 for a Gaussian

result = model.optimize(gamma=1.0)
print(result.weights.round(3))  # fully invested: the weights sum to one
print(result.moments.kurtosis)
```

Returns are used exactly as given. The examples use simple daily returns
written as fractions, so `0.01` is a one-percent gain. HOAO does not convert
prices, percentages or log returns and never annualizes, so every output is in
the units and frequency of the input: `moments.volatility` is a daily standard
deviation and `moments.fourth_moment` is in fourth powers of daily returns.
Variances and higher central moments divide by the number of observations
(`ddof=0`). Kurtosis is Pearson kurtosis, the fourth central moment divided by
the squared variance; subtract 3 to obtain excess kurtosis. A fitting window
needs at least three rows of finite returns and positive variance for every
asset; missing values are rejected rather than imputed.

`fit` returns a `MomentModel`. Its `mean` is the plain sample mean, because
the estimators filter dependence rather than expected returns; `correlation`
and `covariance` are read-only NumPy arrays, and `cokurtosis` holds the compact
fourth-moment estimate. `portfolio(weights)` accepts any finite weights,
including short positions and weights that do not sum to one, and returns a
`PortfolioMoments` record with `mean`, `variance`, `volatility`,
`fourth_moment` and `kurtosis`. `optimize(gamma=1.0)` finds the fully invested
portfolio that minimizes variance plus a penalty on the fourth moment, as
explained under [Portfolio optimization](#portfolio-optimization).

## Choosing an estimator

The `fit` method of every estimator returns a `MomentModel` built on the
sample mean and sample volatilities of the fitted window. The estimators differ
in how they estimate correlation and cokurtosis.

`Sample()` uses the sample covariance and the exact sample fourth central
moment. It is the unfiltered reference, and its cokurtosis is very noisy when
there are many assets relative to observations.

`Sample(even=True)` keeps the sample covariance but simplifies the fourth
moment, without calibrating anything. Standardized returns are rotated into the
basis of the higher-order singular value decomposition (HOSVD) of their
fourth-moment tensor, a fourth-order analogue of principal components, and only
the *even* core entries are kept: those in which every index appears an even
number of times, such as (i, i, i, i) and (i, i, j, j). These entries form an
n × n matrix, which is what makes the compact representation possible.

`CV(n_splits=10, train_fraction=0.65, random_state=None)` also uses the even
core, and replaces both the correlation eigenvalues and the even-core values
with out-of-sample estimates. In each of `n_splits` random partitions of the
window, a fraction `train_fraction` of the rows (rounded down) estimates the
bases and the remaining rows measure the moments along them; the averages are
then placed in the bases of the full window. Each side of a partition needs at
least three rows. CV uses no data beyond the window being fitted, and an
integer `random_state` makes the partitions reproducible. The full window is
standardized before splitting; each part is then recentered without rescaling.

Historical calibration, `calibrate(...)` followed by `Calibration.fit(...)`,
learns the same kind of correction from many past windows of a longer history:
the average-oracle (AO) estimator for correlation and the higher-order average
oracle (HOAO) for cokurtosis. Each historical training and test window is
standardized independently. This method requires a historical panel and
chronology as described in [the next section](#calibrating-from-history).

```python
from hoao import Sample

sample = Sample().fit(returns)
sample_even = Sample(even=True).fit(returns)
cross_validated = CV(n_splits=10, train_fraction=0.65, random_state=42).fit(returns)

for fitted in (sample, sample_even, cross_validated):
    print(fitted.portfolio(weights).kurtosis)
```

Following the paper's main estimators, the filtered estimators treat the second
and fourth moments separately, each in its own basis, so the fitted pair is not
guaranteed to be the moments of any single probability distribution. Pearson
kurtosis is at least one for every real distribution, but a fitted portfolio
can show a smaller value; HOAO reports it unchanged rather than clipping it.
Supplementary compatibility corrections, filtered coskewness and tail-quantile
estimators are not part of the package.

When only one moment is needed, `fit_correlation` and `fit_cokurtosis` skip
the work for the other. With the same `random_state`, they reproduce the
corresponding parts of `fit`:

```python
correlation = CV(random_state=42).fit_correlation(returns)
cokurtosis = CV(random_state=42).fit_cokurtosis(returns)

covariance = correlation.covariance                # (8, 8)
fourth_moment = cokurtosis.fourth_moment(weights)  # portfolio fourth central moment
gradient = cokurtosis.gradient(weights)            # its derivative with respect to the weights
```

## Calibrating from history

A historical calibration learns its corrections from a long return panel. It
draws many *pairs* of windows. For each pair, a signal date is chosen at random
together with a random subset of `n_assets` assets eligible on that date; the
training window is the `train_size` observations ending on the signal date, and
the test window is the `test_size` observations that follow after skipping
`gap` observations. The training window determines the correlation and
fourth-order bases, and the test window measures the variances and even core
realized along them. Averaged over `n_pairs` pairs, these values form the
calibration, which `fit` then applies to a current window.

```python
import numpy as np
from hoao import ReturnsPanel, calibrate

rng = np.random.default_rng(11)
dates = np.arange("2020-01-01", "2021-05-15", dtype="datetime64[D]")
history = 0.01 * rng.standard_t(df=8, size=(len(dates), 8))
assets = [f"Asset_{i}" for i in range(8)]
panel = ReturnsPanel(history, dates=dates, assets=assets)

calibration = calibrate(
    panel,
    n_assets=4,
    train_size=60,
    test_size=20,
    n_pairs=100,
    before=dates[-100],
    random_state=42,
)

# Apply the calibration on the last date to the latest 60 rows of four assets.
current = history[-60:, :4]
model = calibration.fit(current, as_of=dates[-1])
print(model.portfolio(np.full(4, 0.25)).kurtosis)
```

The example uses consecutive calendar days for brevity; with real data, pass
the dates on which observations were actually recorded. Window lengths and
`gap` count rows, not calendar days. The default `gap=1` reproduces the paper's
next-close execution: a decision made after the close of observation t is
traded at t + 1, and the returns it earns are measured from t + 2.

Chronology is enforced. `before` is an exclusive cutoff for every test
observation, so nothing on or after it enters the calibration. When the
calibration is applied, `as_of` states the decision date: it may not precede
the last fitted row or the calibration's cutoff, and every test window used in
the calibration must end strictly before it. A DataFrame with a date index
supplies `as_of` from its last row; for a NumPy array, it must be passed
explicitly.

A calibration is tied to its `n_assets`, `train_size`, `test_size` and `gap`,
but not to particular assets. Its coefficients are indexed by the rank of each
basis direction, so a calibration learned on random four-asset subsets applies
to any four assets. `fit` accepts exactly `train_size` rows and `n_assets`
columns; another universe size or window length needs a new calibration.

### Point-in-time asset universes

On each signal date, assets are drawn only from those that are *available*,
meaning they belong to your investment universe on that date, and whose
training window is complete, finite and not constant. Availability must
reflect only what was known on that date: deriving it from today's index
constituents, for example, would introduce survivorship bias. Returns alone
reveal nothing about listing status, liquidity, security type or index
membership, so supply availability whenever such rules matter. For a dated
pandas panel, the simplest form is a boolean DataFrame with the same index and
columns as the returns:

```python
import pandas as pd

frame = pd.DataFrame(history, index=pd.to_datetime(dates), columns=assets)
available = pd.DataFrame(True, index=frame.index, columns=frame.columns)
available.loc[frame.index[:100], "Asset_7"] = False  # not investable for the first 100 days
panel = ReturnsPanel(frame, available=available)
```

Availability can also be a mapping from each date to the list of available
asset labels, or a sequence with one such list per row. A date missing from a
mapping has an empty universe. This panel is equivalent to the previous one:

```python
available_by_day = {
    date: assets[:7] if row < 100 else assets
    for row, date in enumerate(dates)
}
panel = ReturnsPanel(history, dates=dates, assets=assets, available=available_by_day)
```

Without `available`, every asset with a valid training window is a candidate.
Test windows never influence which assets are selected.

### Missing returns

A missing return (NaN) in a training window makes the asset ineligible on that
signal date; training data are never filled. Because test windows play no part
in selection, an asset that stops trading after the signal date keeps its place
in the pair. By default, `missing="zero"` sets its missing test returns to
zero, the paper's convention for a position that exits to cash, and records
each replacement in `calibration.filled_locations` as (pair ID, panel row,
panel column), with the total in `filled_count`. The rule applies to every
missing test return, whatever its cause, and a zero return does not
reconstruct actual liquidation proceeds, so prepare the panel accordingly.
With `missing="raise"`, calibration stops at the first missing test return
instead. Windows passed to `fit`, as to `Sample` and `CV`, may not contain
missing values.

### Saving and updating a calibration

`save` writes a compressed NumPy archive with JSON metadata and no pickled
objects. It holds the accumulated sums, the number of pairs, the random seed,
the cutoffs and the fill records, but not the return data.

```python
from hoao import Calibration

calibration.save("historical-core.npz")
restored = Calibration.load("historical-core.npz")
model = restored.fit(current, as_of=dates[-1])
```

`update` draws more pairs and adds them to the running sums, so every pair, old
or new, carries equal weight. The new `before` may equal or follow the previous
cutoff but never precede it. `start` restricts the new signal dates to recent
history, although their training windows may begin earlier; without `start`,
new pairs are drawn from the entire valid history.

```python
restored.update(panel, n_pairs=20, before=dates[-40], start=dates[-100])
model = restored.fit(current, as_of=dates[-1])
print(restored.n_pairs)  # 120
```

Each pair draws from a random stream derived from the saved seed and its
position in the sequence, so a series of updates over the same panel and
bounds reproduces one uninterrupted calibration exactly. The
[user guide](docs/guide.rst) and `Calibration.update` in the
[API reference](docs/api.rst) give the details.

## Labeled data with pandas

With the pandas extra installed, every estimator accepts a DataFrame whose
index is unique and increasing, typically dates, and whose columns are unique
asset names. Weights passed as a Series are aligned by name. A Series with a
missing, unknown or duplicated label raises an error instead of being matched
by position, as does a Series passed to a model fitted on an unlabeled array;
NumPy weights always follow the column order of the fitted data.

```python
frame = pd.DataFrame(
    returns,
    index=pd.bdate_range("2025-01-02", periods=len(returns)),
    columns=[f"Asset_{i}" for i in range(returns.shape[1])],
)
model = CV(random_state=42).fit(frame)
weights = pd.Series(1 / frame.shape[1], index=frame.columns)
print(model.portfolio(weights).volatility)

estimate = CV(random_state=42).fit_correlation(frame)
print(estimate.to_pandas(kind="correlation"))
print(model.optimize(gamma=0).weights_to_pandas())
```

Model attributes stay NumPy arrays, so the numerical API works without pandas.
`to_pandas(kind="correlation")` or `to_pandas(kind="covariance")` and
`weights_to_pandas()` return labeled copies on request.

## Portfolio optimization

`model.optimize(gamma)` solves

```text
minimize    w' C w / s2  +  gamma * m4(w) / s2**2
subject to  sum(w) = 1
```

where `C` is the fitted covariance, `m4(w)` the fitted fourth central moment of
the portfolio and `s2` the sample variance of the equally weighted portfolio
over the fitting window, stored as `model.reference_variance`. Dividing by
`s2` and `s2**2` makes both terms dimensionless. Because `s2` comes from the
raw returns rather than from a fitted model, it is the same for every estimator
fitted to those returns, and a given `gamma` expresses the same trade-off
across them. `gamma=0` gives the global minimum-variance portfolio, the fully
invested portfolio with the smallest fitted variance, by a direct linear solve.
A positive `gamma` trades variance for a smaller fourth moment and is solved
iteratively with analytic gradients.

The budget is the only constraint: short positions are allowed, and there are
no position bounds, leverage limits, transaction costs or expected-return
targets. A larger `gamma` weighs the fourth central moment more heavily but
does not guarantee a lower kurtosis, which also depends on the variance. All
of these quantities are in-sample estimates; judging how an allocation
performs requires returns observed after the fitting window.

`optimize` returns an `OptimizationResult` only after checks of the budget and
of stationarity, made independently of the solver, have passed;
`gradient_residual` reports the largest remaining gradient component within the
budget constraint. Otherwise it raises `OptimizationError`, whose `result`
attribute holds the rejected candidate for diagnosis. `frontier` solves a list
of penalties in the given order, starting each solve from the previous
solution. With the labeled model from the previous section:

```python
minimum_variance = model.optimize(gamma=0)
allocations = model.frontier([0.0, 0.1, 1.0, 10.0])
for allocation in allocations:
    print(allocation.gamma, allocation.moments.volatility, allocation.moments.kurtosis)
```

## Compact fourth moments and dense export

A filtered cokurtosis estimate, from `Sample(even=True)`, `CV` or a
calibration, stores an n × n basis and an n × n even core. Evaluating a
portfolio's fourth moment, its gradient or a Hessian-vector product therefore
costs O(n²) operations. The exact `Sample()` estimate keeps the centered
returns instead, so its cost also grows with the number of observations.
Fitting obtains the fourth-order basis from an n × n Gram matrix accumulated
over blocks of observations, again without forming the tensor.

When the full tensor is needed, for instance to compare results with another
implementation, `to_dense` builds it under an explicit memory cap:

```python
small = CV(random_state=42).fit_cokurtosis(returns[:, :4])
tensor = small.to_dense(max_bytes=8 * 1024**2)  # shape (4, 4, 4, 4)
```

The tensor alone occupies `8 * n**4` bytes. `to_dense` budgets four times that
for the output and its work arrays (more for an exact `Sample()` estimate) and
raises `MemoryError` before allocating anything if the total exceeds
`max_bytes`, which defaults to 512 MiB.

When eigenvalues are tied or nearly tied, the even core depends on which basis
is chosen inside the tied eigenspace. HOAO resolves this with a deterministic
coordinate-projector convention, and a rank-deficient window, including one
with more assets than observations, uses the row space of its singular value
decomposition completed by a canonical basis of the null space. This keeps
fitting free of dense tensors, but in tied cases the results can differ from a
research implementation that lets a dense eigensolver choose the basis. Seeds
fix the random splits and samples, yet floating-point results can still vary
slightly across BLAS libraries and devices, most visibly near ties. The
[mathematical notes](docs/mathematics.rst) give the details.

## Performance and GPU support

CPU timings measured on an Intel Core i7-11850H with one BLAS thread:

| Assets | Training rows | CV fit (seconds) | Calibration pairs per second |
|---:|---:|---:|---:|
| 50 | 100 | 0.009 | 914.5 |
| 50 | 500 | 0.016 | 404.6 |
| 100 | 100 | 0.033 | 278.9 |
| 100 | 500 | 0.036 | 202.3 |
| 200 | 100 | 0.103 | 97.6 |
| 200 | 500 | 0.106 | 80.0 |

Each case used synthetic returns, ten CV splits and 100 calibration pairs, with
Python 3.11.11, NumPy 2.4.6 and SciPy 1.17.1. Peak process memory, including
the interpreter and the data, was about 99–114 MiB. These are measurements on
one machine, not guarantees; the [benchmark protocol](docs/performance.rst) and
[raw results](benchmarks/cpu-results.json) document them.

Calibration pairs are independent, and `n_jobs` processes them in parallel CPU
threads while BLAS is limited to one thread, which avoids oversubscribing the
cores. Results are accumulated in pair order and do not depend on the number
of workers.

GPU execution through CuPy is experimental and must be requested with
`device="cuda"` on `Sample`, `CV`, `calibrate` or the `Calibration` methods;
HOAO never selects a device automatically. Not every stage moves to the GPU:
validation, pandas alignment, checkpoints and optimization run on the CPU, and
results are returned as NumPy float64 arrays. CUDA calibration requires
`n_jobs=1` and processes pairs in batches of at most four. GPU performance has
not been measured, and for small problems data transfers and kernel launches
can make the CPU faster.

## Documentation and development

The [user guide](docs/guide.rst) covers input conventions, calibration
chronology and the optimizer in more depth, the
[mathematical notes](docs/mathematics.rst) define the estimators, and the
[API reference](docs/api.rst) documents every public object. The scripts in
[`examples/`](examples/) run on synthetic data and do not require the paper's
licensed dataset.

To develop HOAO, install a clone of the repository in editable mode with the
`dev` extra and run the checks:

```bash
pip install -e ".[dev]"
python -m pytest
python -m pyright --pythonpath "$(command -v python)"
python docs/check_docstrings.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

Continuous integration builds and checks the release artifacts but does not
upload them to PyPI; the [release instructions](docs/development.rst) leave
publication to the maintainer.

## Citation and license

If you use HOAO in research, please cite the accompanying manuscript:

> Paul Ruelloux, Christian Bongiorno and Damien Challet, *Nonlinear Shrinkage
> for Higher Moments Estimation in Large Portfolios* (2026), manuscript
> prepared for *Annals of Operations Research*.

A DOI will be added when one is available. [`CITATION.cff`](CITATION.cff)
provides machine-readable citation metadata for both the software and the
paper.

The software is written by Christian Bongiorno and Paul Ruelloux and released
under the BSD 3-Clause license; see [`LICENSE`](LICENSE).
