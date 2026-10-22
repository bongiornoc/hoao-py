"""Use named assets and retrieve labeled correlation and portfolio weights.

Requires the ``pandas`` extra: ``pip install 'hoao[pandas]'``.
"""

import numpy as np
import pandas as pd

from hoao import CV

rng = np.random.default_rng(23)
frame = pd.DataFrame(
    0.01 * rng.standard_t(df=8, size=(120, 4)),
    index=pd.bdate_range("2025-01-02", periods=120),
    columns=["Asset_A", "Asset_B", "Asset_C", "Asset_D"],
)
estimator = CV(n_splits=4, random_state=42)
estimate = estimator.fit_correlation(frame)
model = estimator.fit(frame)
weights = pd.Series(0.25, index=frame.columns[::-1])
moments = model.portfolio(weights)
correlation = estimate.to_pandas(kind="correlation")
result = model.optimize(gamma=0)
labeled_weights = result.weights_to_pandas()

if __name__ == "__main__":
    print(correlation)
    print(labeled_weights)
    print("Equal-weight volatility:", moments.volatility)
