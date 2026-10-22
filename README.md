# hoao

**Correlation and cokurtosis estimation, historical oracle calibration, and portfolio optimization.**

`hoao` implements the main methods in *Nonlinear Shrinkage for Higher Moments
Estimation in Large Portfolios*, by Paul Ruelloux, Christian Bongiorno, and
Damien Challet. The software is authored by Christian Bongiorno and Paul
Ruelloux and released under the BSD 3-Clause license.

Starting from a return matrix, you can fit a model, obtain its covariance,
evaluate portfolio moments, or optimize a variance–fourth-moment objective.
Historical calibration learns reusable coefficients from past windows and
daily asset universes. Fourth-order estimates are stored as a basis and a
small matrix, so ordinary estimation and portfolio evaluation do not require
an array with four asset axes.

## Installation

Python 3.11 or newer is required. Once a release has been uploaded to PyPI:

```bash
pip install hoao
```

For development or before the first PyPI release, install from this checkout:

```bash
pip install -e .
```

Optional dependencies are explicit:

```bash
pip install "hoao[pandas]"       # DataFrames, Series, and labeled outputs
pip install "hoao[cuda12]"       # CuPy for an appropriate CUDA 12 installation
# or: pip install "hoao[cuda13]"
```

The CPU package does not require pandas, CuPy, or a CUDA runtime. Install only
one CUDA extra, matching your NVIDIA environment. GPU support is experimental;
GPU speedups have not been measured on the development machine. A small fit
can be faster on CPU because moving arrays and launching kernels have a cost.

## A complete example

Returns have shape `(n_observations, n_assets)`. Supply simple daily returns
as fractions: `0.01` means one percent. No prices, annualization, or percent
conversion is performed implicitly.

```python
import numpy as np
from hoao import CV

rng = np.random.default_rng(7)
returns = 0.01 * rng.standard_t(df=8, size=(240, 8))

model = CV(random_state=42).fit(returns)
weights = np.full(8, 1 / 8)
moments = model.portfolio(weights)

print(model.correlation.shape)       # (8, 8)
print(model.covariance.shape)        # (8, 8)
print(moments.variance)              # variance per observation
print(moments.volatility)            # standard deviation per observation
print(moments.fourth_moment)         # fourth central moment in return units
print(moments.kurtosis)              # Pearson kurtosis, not excess kurtosis

result = model.optimize(gamma=1.0)
print(result.weights.sum())         # approximately 1
print(result.gradient_residual)     # convergence diagnostic
```

Each estimator offers separate methods when you need only one order:

```python
correlation = CV(random_state=42).fit_correlation(returns)
cokurtosis = CV(random_state=42).fit_cokurtosis(returns)

covariance = correlation.covariance
fourth_moment = cokurtosis.fourth_moment(weights)
gradient = cokurtosis.gradient(weights)
```

`fit_correlation` avoids computing the fourth-order basis. `fit` returns a
combined `MomentModel` and shares preprocessing between the two orders.

## Choosing an estimator

| Estimator | What it does | Historical dataset needed? |
|---|---|---|
| `Sample()` | Sample covariance and exact sample fourth central moment | No |
| `Sample(even=True)` | Sample covariance and uncalibrated even projection of cokurtosis | No |
| `CV()` | Calibrates coefficients using random splits of the supplied window | No |
| `calibrate(...).fit(...)` | Applies historical AO/HOAO coefficients to a current window | Yes |

```python
from hoao import Sample

sample = Sample().fit(returns)
sample_even = Sample(even=True).fit(returns)
```

CV defaults to ten random splits with a 65% training fraction. `random_state`
makes sampling reproducible. The full input window is standardized first;
each CV training and validation subsample is then re-centered. Historical
calibration instead standardizes each training and testing window separately.
This difference is part of the methods' definition.

## Calibrate from a dataset

`ReturnsPanel` contains observations, dates, asset identifiers, and optionally
an available-asset universe for each date. A calibration samples historical
signal dates and asset subsets, estimates training bases, and averages
the moments of subsequent observations in those bases.

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

# Apply the coefficients to four assets observed later.
current = history[-60:, :4]
model = calibration.fit(current, as_of=dates[-1])
print(model.portfolio(np.full(4, 0.25)).kurtosis)
```

These synthetic daily dates illustrate the interface. In a real dataset,
provide your actual ordered observation sessions. Window lengths count rows;
they do not count calendar days. `gap=1` skips one observation between the end
of training and the start of testing, matching next-close execution in the
paper. `before` is an exclusive cutoff: every calibration testing window
must end before it. When fitting from an ndarray, supply `as_of` explicitly;
for a dated DataFrame the final row supplies the decision date.

A calibration is specific to its number of assets and window lengths. It is
asset-agnostic: you can apply a four-asset calibration to a different group
of four assets. Recalibrate for a different number of assets or training
window length. Preserve the chronological meaning of `as_of`; a historical
calibration must never be applied retrospectively before its cutoff.

### Supply the available stocks for each day

For a dated pandas panel, an equally indexed boolean DataFrame is a simple
way to supply availability:

```python
import pandas as pd

frame = pd.DataFrame(history, index=pd.to_datetime(dates), columns=assets)
available = pd.DataFrame(True, index=frame.index, columns=frame.columns)
available.loc[frame.index[:100], "Asset_7"] = False
panel = ReturnsPanel(frame, available=available)
```

You can also pass a mapping from each date to a list of asset names, or one
list per return row. A missing date in a mapping means an empty universe:

```python
available_by_day = {
    date: assets[:7] if row < 100 else assets
    for row, date in enumerate(dates)
}
panel = ReturnsPanel(
    history, dates=dates, assets=assets, available=available_by_day
)
```

Availability describes what could have been selected at that date. Training
data completeness and positive variance are checked using the training
window. When `available` is omitted, these historical checks determine the
usable universe automatically. They do not infer listing status, liquidity,
security type, or membership in an index. Provide your own availability
when those rules matter.

Asset selection never examines future testing-window completeness. By
default, `missing="zero"` converts missing testing returns to zero and keeps
the selected assets fixed. This is the paper's cash/exit convention; it also
applies to other missing target entries, so prepare the input accordingly.
Use `missing="raise"` when a missing testing observation should stop
calibration. Missing fitting-window observations are not zero-filled.

### Save and reuse calibration

```python
from hoao import Calibration

calibration.save("historical-core.npz")
restored = Calibration.load("historical-core.npz")
model = restored.fit(current, as_of=dates[-1])
```

Checkpoints store numerical coefficients and calibration metadata, not the
original dataset. Incremental updates retain running sums and counts rather
than discarding previously calibrated pairs:

```python
restored.update(panel, n_pairs=20, before=dates[-40], start=dates[-100])
model = restored.fit(current, as_of=dates[-1])
```

`start` restricts newly sampled signal dates. Omitting it samples across the
entire valid historical pool again. The updated cutoff must not move backward.
See `Calibration.update` in the [API reference](docs/api.rst).

## pandas and asset labels

Use a DataFrame with increasing dates on its index and unique asset names on
its columns. A labeled weight Series is aligned by asset name. Unknown,
missing, or duplicated labels are errors rather than positional guesses.

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

Model arrays remain NumPy arrays by default. Conversion to pandas is explicit
so that the numerical API works without an optional dependency.

## Portfolio optimization

`model.optimize(gamma=0)` computes the global minimum-variance portfolio.
For positive `gamma`, it adds a penalty on the fourth **central moment**.
Both terms are normalized using the sample variance of the equal-weight
portfolio in the original fitting data, stored in the model. This provides
the same reference scale when comparing estimators fitted on the same data.

```python
minimum_variance = model.optimize(gamma=0)
allocations = model.frontier([0.0, 0.1, 1.0, 10.0])
for allocation in allocations:
    print(allocation.moments.volatility, allocation.moments.kurtosis)
```

The only portfolio constraint is that weights sum to one. Short sales are
allowed; there are no leverage limits or expected-return forecasts. Larger
`gamma` penalizes the fourth central moment, and does **not** guarantee a
decrease in standardized kurtosis. A frontier is a set of fitted-model
allocations; evaluating future performance requires your own unseen data.
Check `success` and `gradient_residual` before using an optimization result.

## Memory, speed, and GPU

Filtered cokurtosis is represented by two square matrices. Portfolio moments
and gradients cost quadratic time in the number of assets. The fourth-order
basis uses an implicit Gram calculation with blocked observation products;
the core calculation uses matrix multiplication of squared projected returns.
This avoids dense tensor construction on the ordinary path.

Dense reconstruction is explicit and protected by a byte budget:

```python
small = CV(random_state=42).fit_cokurtosis(returns[:, :4])
tensor = small.to_dense(max_bytes=8 * 1024**2)   # shape (4, 4, 4, 4)
```

The dense representation needs roughly `8 * n_assets**4` bytes before
temporary workspaces. Keep compact objects for portfolio calculations.
Near-tied basis eigenvalues require special care because the even projection
depends on the choice of basis within a tied eigenspace. A deterministic
coordinate-projector convention resolves ties. Rank-deficient windows use
their SVD row space and a canonical null-space completion, including when
there are more assets than observations. This keeps fitting tensor-free,
but can differ from a research implementation that lets a dense eigensolver
choose its own basis. See [the mathematical guide](docs/mathematics.rst).

Historical calibration exposes `n_jobs` to parallelize independent pairs on
CPU. Use `device="cuda"` on `CV` or `calibrate` to request the optional GPU
path explicitly. Array validation, pandas alignment, checkpoint metadata,
and optimization may still involve the CPU; GPU mode is not a promise that
every stage runs on the device. There is no automatic device selection.
CUDA calibration requires `n_jobs=1` and processes bounded batches internally.

Measured CPU timings on an Intel Core i7-11850H with one BLAS thread:

| Assets | Training rows | CV fit (seconds) | Calibration pairs/second |
|---:|---:|---:|---:|
| 50 | 100 | 0.009 | 914.5 |
| 50 | 500 | 0.016 | 404.6 |
| 100 | 100 | 0.033 | 278.9 |
| 100 | 500 | 0.036 | 202.3 |
| 200 | 100 | 0.103 | 97.6 |
| 200 | 500 | 0.106 | 80.0 |

These are measurements on finite synthetic returns, using ten CV splits and
100 calibration pairs per case, with Python 3.11.11, NumPy 2.4.6, and SciPy
1.17.1. Peak process memory was approximately 99–114 MiB, including the
interpreter and synthetic dataset. They are observations, not performance
guarantees; see the [benchmark protocol](docs/performance.rst) and
[raw results](benchmarks/cpu-results.json). GPU performance was not measured.

## Scientific boundaries

- Estimates use central moments with `ddof=0` and preserve the input return
  frequency. Kurtosis is Pearson kurtosis (Gaussian value 3).
- Covariance and cokurtosis use separate bases. Their combination is not
  guaranteed to be the moment pair of one probability distribution, so fitted
  kurtosis can be below one. Values are not silently clipped.
- The main estimators cover orders two and four. Supplementary compatibility
  corrections, filtered coskewness, and tail-quantile estimators are outside
  this package's scope.
- A seed fixes random splits and selections. Floating-point results can vary
  slightly across BLAS implementations and devices, especially near tied
  eigenvalues.

## Documentation, examples, and development

The [guide](docs/guide.rst), [mathematics](docs/mathematics.rst), and generated
[API reference](docs/api.rst) explain shapes, preprocessing, chronology, and
the optimization objective. Runnable scripts in [`examples/`](examples/)
use synthetic observations and do not require access to the paper's licensed
dataset.

```bash
pip install -e ".[dev]"
python -m pytest
python -m pyright --pythonpath "$(command -v python)"
python docs/check_docstrings.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

CI builds and checks release artifacts. It does not upload to PyPI; the
[release instructions](docs/development.rst) leave publication to the maintainer.

## Citation

Please cite the accompanying manuscript by **Paul Ruelloux, Christian
Bongiorno, and Damien Challet**, *Nonlinear Shrinkage for Higher Moments
Estimation in Large Portfolios* (2026), prepared for Annals of Operations
Research. A DOI will be added when available. Machine-readable software and
paper attribution is in [`CITATION.cff`](CITATION.cff).
