Development and release
=======================

Working on the package
----------------------

From a repository checkout, install the development extra in a virtual
environment. It includes the test, type-checking and documentation tools:

.. code-block:: console

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e '.[dev]'
   python -m pytest
   python -m pytest --doctest-modules src/hoao
   python -m pyright --pythonpath "$(command -v python)"
   python docs/check_docstrings.py
   python -m ruff check src tests examples docs
   python -m sphinx -W --keep-going -b html docs docs/_build/html

The suite compares compact moment calculations with independent dense
formulas at small dimensions, checks derivatives and label alignment, and
verifies calibration chronology and reproducibility. It also executes the
README and example scripts on synthetic data. The separate doctest command
runs examples embedded in public docstrings.

Keep CPU workflows usable without pandas or CuPy. Tests of optional
dependencies should state why they are skipped when the dependency or its
runtime is unavailable. GPU timing requires a working device and should
include transfers and synchronization.

Public docstrings supply the API reference and editor help. Describe array
shapes, units, normalization, exceptions and a small executable example in
NumPy format; keep inline type annotations consistent with that contract.
Explain estimator conventions in the user guide and mathematical notes,
where readers can follow their consequences across methods.

Building distributions
----------------------

.. code-block:: console

   python -m build
   python -m twine check dist/*

The wheel must include ``hoao/py.typed`` and the license. Inspect both
archives to ensure they contain only package code, documentation, tests and
synthetic examples, with no research datasets or calibration state. Verify
the built wheel in a fresh environment:

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

Publishing a release
--------------------

PyPI publication is performed by a maintainer. Confirm ownership of the
distribution name, then update the version in
``pyproject.toml``, ``src/hoao/__init__.py``, ``docs/conf.py``, and
``CITATION.cff`` together. A version already uploaded to PyPI cannot be
replaced with a different artifact.

After all checks pass, upload the reviewed artifacts using credentials
managed outside the repository:

.. code-block:: console

   python -m twine upload --repository testpypi dist/*
   # After reviewing the test release:
   python -m twine upload dist/*

The second command publishes the distribution on PyPI. Tag the exact source
commit used to build the artifacts and attach those same artifacts to the
corresponding GitHub release.
