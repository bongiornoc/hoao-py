"""Small, optional device boundary; importing hoao never imports CuPy."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ._typing import Device, FloatArray


def validate_device(device: str) -> None:
    """Reject misspelled devices without initializing a GPU runtime."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")


def array_namespace(device: Device) -> Any:
    """Return NumPy or lazily import and check the optional CUDA backend."""
    validate_device(device)
    if device == "cpu":
        return np
    try:
        cp = import_module("cupy")
    except ImportError as exc:
        raise ImportError(
            "device='cuda' requires a CuPy installation matching your CUDA "
            "runtime. Install CuPy separately, or use device='cpu'."
        ) from exc
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError(
            "The CUDA backend could not initialize. Check the NVIDIA driver "
            "and CuPy installation, or use device='cpu'."
        ) from exc
    if count < 1:
        raise RuntimeError("device='cuda' requires an available CUDA device.")
    return cp


def as_host(array: Any, namespace: Any) -> FloatArray:
    """Return a CPU float64 array, synchronizing a CUDA result if needed."""
    if namespace is np:
        return np.asarray(array, dtype=np.float64)
    return np.asarray(namespace.asnumpy(array), dtype=np.float64)
