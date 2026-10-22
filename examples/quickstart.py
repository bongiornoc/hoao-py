"""Fit sample/CV moments and optimize a small synthetic portfolio.

Run from the repository after installation: ``python examples/quickstart.py``.
"""

import numpy as np

from hoao import CV, Sample

rng = np.random.default_rng(7)
returns = 0.01 * rng.standard_t(df=8, size=(120, 6))
weights = np.full(returns.shape[1], 1 / returns.shape[1])

sample = Sample().fit(returns)
sample_even = Sample(even=True).fit(returns)
model = CV(n_splits=4, random_state=42).fit(returns)
moments = model.portfolio(weights)
result = model.optimize(gamma=1.0)

if __name__ == "__main__":
    print("Equal-weight CV moments:", moments)
    print("Optimized weights:", result.weights)
    print("Convergence residual:", result.gradient_residual)
