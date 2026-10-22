"""The standard CPU API must not import optional dataframe/GPU packages."""
import os
from pathlib import Path
import subprocess
import sys


def test_cpu_works_when_optional_packages_are_unavailable(tmp_path):
    code = '''
import importlib.abc
import sys
class NoOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'pandas', 'cupy'}:
            raise ModuleNotFoundError('Optional dependency intentionally unavailable')
sys.meta_path.insert(0, NoOptional())
import numpy as np
import hoao
x = np.random.default_rng(4).normal(size=(120, 4))
hoao.Sample().fit(x).optimize()
hoao.CV(random_state=3).fit(x).portfolio([.25]*4)
panel = hoao.ReturnsPanel(x)
core = hoao.calibrate(panel, n_assets=3, train_size=20, test_size=10,
                      n_pairs=3, before=90, random_state=5)
core.fit(x[-20:, :3], as_of=119).optimize(1)
assert 'pandas' not in sys.modules and 'cupy' not in sys.modules
'''
    source = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=source, OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
