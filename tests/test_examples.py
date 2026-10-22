"""Run the user-facing examples as installed-package integration checks."""

from pathlib import Path
import re
import runpy

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("filename", ["quickstart.py", "historical_calibration.py"])
def test_numpy_examples(filename):
    namespace = runpy.run_path(str(EXAMPLES / filename))
    moments = namespace["moments"]
    assert np.isfinite(moments.variance) and moments.variance > 0
    assert np.isfinite(moments.fourth_moment) and moments.fourth_moment > 0
    if "result" in namespace:
        assert namespace["result"].success
        np.testing.assert_allclose(namespace["result"].weights.sum(), 1, atol=1e-10)


def test_pandas_example():
    pytest.importorskip("pandas")
    namespace = runpy.run_path(str(EXAMPLES / "pandas_labels.py"))
    assert namespace["correlation"].columns.equals(namespace["frame"].columns)
    assert namespace["labeled_weights"].index.equals(namespace["frame"].columns)
    np.testing.assert_allclose(namespace["labeled_weights"].sum(), 1, atol=1e-10)


def test_readme_python_examples(tmp_path, monkeypatch):
    """Keep the advertised end-to-end README workflow executable."""
    pytest.importorskip("pandas")
    readme = (EXAMPLES.parent / "README.md").read_text()
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "readme_examples"}
    for index, source in enumerate(re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)):
        exec(compile(source, f"README.md:example-{index + 1}", "exec"), namespace)
    assert namespace["tensor"].shape == (4, 4, 4, 4)
