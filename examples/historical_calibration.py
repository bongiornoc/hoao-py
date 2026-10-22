"""Calibrate, checkpoint, and apply an asset-agnostic historical oracle.

The synthetic panel uses integer observation positions. Its cutoff and
``as_of`` values therefore use the same integer coordinate system.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hoao import Calibration, ReturnsPanel, calibrate

rng = np.random.default_rng(11)
returns = 0.01 * rng.standard_t(df=8, size=(300, 8))
available = np.ones_like(returns, dtype=bool)
available[:80, -1] = False
panel = ReturnsPanel(returns, available=available)

calibration = calibrate(
    panel,
    n_assets=4,
    train_size=60,
    test_size=20,
    n_pairs=12,
    before=200,
    random_state=42,
)
with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "core.npz"
    calibration.save(checkpoint)
    restored = Calibration.load(checkpoint)

restored.update(panel, n_pairs=4, before=240, start=200)
model = restored.fit(returns[-60:, :4], as_of=299)
moments = model.portfolio(np.full(4, 0.25))

if __name__ == "__main__":
    print("Historical oracle portfolio moments:", moments)
