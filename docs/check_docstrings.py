"""Validate essential NumPy-style contracts on public API operations.

Run ``python docs/check_docstrings.py`` after installing the development extra.
Checks focus on missing documentation, parameter names and descriptions,
return sections, and examples. Cosmetic recommendations remain advisory.
"""

from __future__ import annotations

import inspect
import sys

import hoao
from numpydoc.validate import validate


REQUIRED_CODES = {
    "GL08",  # Missing docstring.
    "PR01",  # A parameter is absent from the docstring.
    "PR02",  # Docstring contains an unknown parameter.
    "PR03",  # Parameters are documented out of order.
    "PR04",  # Parameter type is absent.
    "PR07",  # Parameter description is absent.
    "RT01",  # A function returning a value has no Returns section.
    "RT03",  # Return description is absent.
    "EX01",  # No Examples section.
}


def main() -> int:
    """Check published callables and report actionable documentation errors."""
    errors = []
    count = 0
    for name in hoao.__all__:
        obj = getattr(hoao, name)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        identifiers = [f"hoao.{name}"]
        if inspect.isclass(obj):
            identifiers.extend(
                f"hoao.{name}.{member}"
                for member, value in inspect.getmembers(obj)
                if not member.startswith("_")
                and (inspect.isfunction(value) or inspect.ismethod(value))
            )
        for identifier in identifiers:
            count += 1
            for code, message in validate(identifier)["errors"]:
                # These are factory-produced records. Their documented public
                # contract is the Attributes section, not constructor use.
                record_class = identifier in {
                    "hoao.Calibration", "hoao.PortfolioMoments", "hoao.OptimizationResult"
                }
                if code in REQUIRED_CODES and not (record_class and code == "PR01"):
                    errors.append(f"{identifier}: {code} {message}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Validated essential NumPy-style contracts for {count} public callables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
