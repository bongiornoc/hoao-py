Development and release
=======================

Local checks
------------

Use a fresh environment from the repository root:

.. code-block:: console

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e '.[dev]'
   python -m pytest
   python -m pyright --pythonpath "$(command -v python)"
   python docs/check_docstrings.py
   python -m ruff check src tests examples docs
   python -m sphinx -W --keep-going -b html docs docs/_build/html

The example tests execute complete workflows using deterministic synthetic
data. Numerical tests should compare compact calculations against independent
dense formulas at small dimensions, verify derivatives, check label alignment,
and reject temporal leakage. CPU workflows must work without pandas or CuPy.
An optional backend needs explicit tests and a clear skip when its runtime is
unavailable; an untested GPU must not be described as a measured speedup.

The release includes inline type annotations and ``py.typed`` for editors.
NumPy-style docstrings are the API documentation source. Every public
operation should explain shapes, units, normalization, exceptions, and a
small executable example. Add design explanations in the guide when a
choice changes the scientific estimator rather than only its implementation.

Build an artifact
-----------------

.. code-block:: console

   python -m build
   python -m twine check dist/*

Inspect both the wheel and source distribution before release. Confirm that
``hoao/py.typed`` and the license are in the wheel, and that no return datasets,
calibration checkpoints, credentials, or research output directories have
entered either artifact. Check the installed wheel in a fresh environment,
not only the editable source tree:

.. code-block:: console

   python -m venv /tmp/hoao-release-check
   /tmp/hoao-release-check/bin/python -m pip install dist/hoao-0.1.0-py3-none-any.whl
   /tmp/hoao-release-check/bin/python examples/quickstart.py

Run the public type-completeness check against an installed wheel. Editable
installations may not be discoverable by Pyright's ``--verifytypes`` resolver:

.. code-block:: console

   /tmp/hoao-release-check/bin/python -m pip install pyright
   . /tmp/hoao-release-check/bin/activate
   python -m pyright --verifytypes hoao --ignoreexternal

GitHub Actions tests supported Python versions, performs platform smoke
checks, validates types and documentation, and uploads build artifacts for
inspection. It contains no PyPI publication step and needs no PyPI token.

Maintainer publication
----------------------

PyPI publication is a separate maintainer action. Before uploading, verify
that the distribution name is available to the publisher, review the
repository visibility and attribution, and update the version in
``pyproject.toml``, ``src/hoao/__init__.py``, ``docs/conf.py``, and
``CITATION.cff`` together. A version already uploaded to PyPI cannot be
replaced with a different artifact.

After all checks pass, upload the reviewed artifacts using credentials
managed outside the repository:

.. code-block:: console

   python -m twine upload --repository testpypi dist/*
   # After reviewing the test release:
   python -m twine upload dist/*

The second command publishes publicly. The package does not execute either
command automatically. Repository tags and a GitHub release should refer
to the exact source used to build the uploaded artifacts.
