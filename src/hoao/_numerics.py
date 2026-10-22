"""Exact even-core algorithms with bounded storage and no dense fourth tensor.

All observations have shape ``(n_observations, n_assets)``. The routines in
this private module assume that any asset labels have already been aligned.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._backend import array_namespace, as_host
from ._typing import Device, FloatArray

BASIS_CONVENTION = "descending-projector-canonical-v1"
MomentOrder = Literal["both", "correlation", "cokurtosis"]


def _matrix(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite (observations, assets) array.")
    return x


def standardize(
    x: FloatArray, allow_constant: bool = False
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Center columns and divide by their population standard deviations.

    Returns standardized observations, the sample mean, and the sample
    standard deviations (``ddof=0``). When explicitly allowed, a constant
    column has zero standardized observations and a zero scale.
    """
    x = _matrix(x)
    mean = x.mean(axis=0)
    centered = x - mean
    # Remove the small residual mean left by subtracting a large offset.
    # Keep the reported raw sample mean unchanged.
    centered -= centered.mean(axis=0)
    # A mean of identical nonbinary floating-point values can round away
    # from that value. Such a column must still have exactly zero variance.
    centered[:, np.ptp(x, axis=0) == 0.0] = 0.0
    scales = np.sqrt(np.mean(centered * centered, axis=0))
    if not np.isfinite(scales).all():
        raise ValueError("The return magnitudes overflowed the variance calculation.")
    if not allow_constant and np.any(scales == 0.0):
        raise ValueError("Returns contain a constant asset with zero variance.")
    z = np.divide(
        centered, scales, out=np.zeros_like(centered), where=scales > 0.0
    )
    return z, mean, scales


def normalize_correlation(c: FloatArray) -> FloatArray:
    """Restore a symmetric positive-diagonal matrix to unit diagonal."""
    c = _matrix(c)
    if c.shape[0] != c.shape[1]:
        raise ValueError("A correlation matrix must be square.")
    diagonal = np.diag(c)
    if np.any(diagonal <= 0.0):
        raise ValueError("Cannot normalize a correlation with nonpositive variance.")
    scale = np.sqrt(diagonal)
    result = (c / scale[:, None]) / scale[None, :]
    result = (result + result.T) * 0.5
    np.fill_diagonal(result, 1.0)
    return result


def _canonical_projector_basis(projector: FloatArray, rank: int) -> FloatArray:
    """Choose coordinates deterministically from an orthogonal projector.

    Pivot on the largest remaining projected coordinate norm, with a stable
    lowest-index choice for roundoff ties. Reorthogonalization avoids unstable
    normalization of almost-dependent projected coordinate vectors.
    """
    n = projector.shape[0]
    result = np.empty((n, rank), dtype=np.float64)
    residual = np.diag(projector).copy()
    tolerance = 100.0 * n * np.finfo(np.float64).eps
    for j in range(rank):
        maximum = float(np.max(residual))
        pivot = int(np.flatnonzero(residual >= maximum - tolerance)[0])
        vector = projector[:, pivot].copy()
        for _ in range(2):
            if j:
                previous = result[:, :j]
                vector -= previous @ (previous.T @ vector)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= tolerance:
            raise np.linalg.LinAlgError("Could not resolve the numerical eigenspace.")
        vector /= norm
        if vector[pivot] < 0.0:
            vector *= -1.0
        result[:, j] = vector
        residual -= vector * vector
    return result


def _canonical_eigenvectors(
    values: FloatArray, vectors: FloatArray
) -> FloatArray:
    """Canonicalize numerically unresolved eigenspaces in descending order."""
    n, count = vectors.shape
    if not count:
        return vectors.copy()
    tolerance = 100.0 * n * np.finfo(np.float64).eps * float(np.max(np.abs(values)))
    result = vectors.copy()
    start = 0
    while start < count:
        end = start + 1
        # Bound the width of a group, rather than merging a long chain of
        # individually close but collectively distinguishable eigenvalues.
        while end < count and values[start] - values[end] <= tolerance:
            end += 1
        if end - start > 1:
            subspace = vectors[:, start:end]
            result[:, start:end] = _canonical_projector_basis(
                subspace @ subspace.T, end - start
            )
        else:
            pivot = int(np.argmax(np.abs(result[:, start])))
            if result[pivot, start] < 0.0:
                result[:, start] *= -1.0
        start = end
    return result


def _eigh(matrix: FloatArray, device: Device) -> tuple[FloatArray, FloatArray]:
    namespace = array_namespace(device)
    values, vectors = namespace.linalg.eigh(namespace.asarray(matrix))
    return as_host(values, namespace)[::-1], as_host(vectors, namespace)[:, ::-1]


def correlation_basis(
    z: FloatArray, device: Device = "cpu"
) -> tuple[FloatArray, FloatArray]:
    """Return the second central moment and its descending orthogonal basis."""
    z = _matrix(z)
    namespace = array_namespace(device)
    data = namespace.asarray(z)
    c = as_host(data.T @ data / len(z), namespace)
    c = (c + c.T) * 0.5
    values, vectors = _eigh(c, device)
    return c, _canonical_eigenvectors(values, vectors)


def _fourth_gram_device(
    z: FloatArray, block_size: int, device: Device
) -> FloatArray:
    namespace = array_namespace(device)
    data = namespace.asarray(z)
    gram = namespace.zeros((z.shape[1], z.shape[1]), dtype=namespace.float64)
    for start in range(0, len(z), block_size):
        block = data[start : start + block_size]
        kernel = block @ data.T
        kernel *= kernel * kernel
        # This association avoids repeating an n-by-T @ T-by-n product for
        # every block, and limits temporary storage to block_size-by-T.
        gram += block.T @ (kernel @ data)
    gram /= float(len(z)) ** 2
    result = as_host(gram, namespace)
    if not np.isfinite(result).all():
        raise FloatingPointError("Fourth-moment Gram overflowed; check return magnitudes.")
    return (result + result.T) * 0.5


def fourth_gram(z: FloatArray, block_size: int = 256) -> FloatArray:
    """Compute the exact fourth-moment unfolding Gram in observation blocks.

    For already centered rows, evaluates
    ``Z.T @ ((Z @ Z.T)**3) @ Z / T**2`` without constructing either the
    fourth-order tensor or the full observation kernel. The cube is
    elementwise. Work is ``O(T**2*n + T*n**2)`` and temporary storage is
    ``O(block_size*T + n**2)`` in addition to the observations.
    """
    z = _matrix(z)
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    return _fourth_gram_device(z, int(block_size), "cpu")


def _rank_aware_fourth_basis(z: FloatArray, device: Device) -> FloatArray:
    """Separate the exact data row space from its unobserved directions."""
    n = z.shape[1]
    _, singular, right = np.linalg.svd(z, full_matrices=False)
    threshold = np.finfo(np.float64).eps * max(z.shape) * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    if not rank:
        return np.eye(n, dtype=np.float64)
    row_space = right[:rank].T
    reduced = z @ row_space
    gram = _fourth_gram_device(reduced, 256, device)
    values, vectors = _eigh(gram, device)
    observed = _canonical_eigenvectors(values, row_space @ vectors)
    if rank == n:
        return observed
    null_projector = np.eye(n) - row_space @ row_space.T
    null = _canonical_projector_basis(null_projector, n - rank)
    return np.column_stack((observed, null))


def fourth_basis(z: FloatArray, device: Device = "cpu") -> FloatArray:
    """Return the full fourth-order HOSVD basis without allocating ``n**4``.

    Eigenvectors are ordered by decreasing squared singular value. Within
    a numerically tied eigenspace, coordinate-projector canonicalization
    provides a deterministic convention. Rank-deficient data use the SVD
    row space followed by a canonical null-space completion. This remains
    valid when there are more assets than observations.

    A different choice within a tied eigenspace can change an even-projected
    estimate; calibration and fitting must use the same basis convention.
    """
    z = _matrix(z)
    # Validate optional CUDA support even for a zero-rank input.
    array_namespace(device)
    t, n = z.shape
    if n >= t:
        return _rank_aware_fourth_basis(z, device)
    gram = _fourth_gram_device(z, 256, device)
    values, vectors = _eigh(gram, device)
    threshold = 100.0 * n * np.finfo(np.float64).eps * max(float(values[0]), 0.0)
    if values[-1] <= threshold:
        return _rank_aware_fourth_basis(z, device)
    return _canonical_eigenvectors(values, vectors)


def pair_core(
    train: FloatArray,
    test: FloatArray,
    *,
    standardized: bool = False,
    device: Device = "cpu",
    order: MomentOrder = "both",
) -> tuple[FloatArray | None, FloatArray | None]:
    """Calibrate selected even cores using disjoint train and test samples.

    With ``standardized=False``, standardize each historical window using
    its own mean and population volatility. With ``standardized=True``,
    preserve the supplied scales and only recenter each window, as required
    by CV after standardizing the complete fitting window.

    Returns the oracle variance vector and the even fourth-core matrix;
    the unrequested component is ``None`` and is not computed. Constant
    historical test columns are retained as zero standardized components.
    """
    train, test = _matrix(train), _matrix(test)
    if train.shape[1] != test.shape[1]:
        raise ValueError("Training and test windows must have the same assets.")
    if order not in ("both", "correlation", "cokurtosis"):
        raise ValueError("order must be 'both', 'correlation', or 'cokurtosis'.")
    if standardized:
        zin = train - train.mean(axis=0)
        zout = test - test.mean(axis=0)
    else:
        zin, _, _ = standardize(train)
        zout, _, _ = standardize(test, allow_constant=True)
    namespace = array_namespace(device)
    target = namespace.asarray(zout)
    a = b = None
    if order in ("both", "correlation"):
        _, u2 = correlation_basis(zin, device)
        rotated = target @ namespace.asarray(u2)
        a = as_host(namespace.mean(rotated * rotated, axis=0), namespace)
    if order in ("both", "cokurtosis"):
        u4 = fourth_basis(zin, device)
        squared = (target @ namespace.asarray(u4)) ** 2
        b = as_host(squared.T @ squared / len(zout), namespace)
        b = (b + b.T) * 0.5
    return a, b


def _batch_fourth_gram(data: Any, namespace: Any, block_size: int = 256) -> Any:
    """Vectorized observation-block Grams; callers bound the batch dimension."""
    batch, t, n = data.shape
    gram = namespace.zeros((batch, n, n), dtype=namespace.float64)
    transposed = namespace.swapaxes(data, -1, -2)
    for start in range(0, t, block_size):
        block = data[:, start : start + block_size]
        kernel = block @ transposed
        kernel *= kernel * kernel
        gram += namespace.swapaxes(block, -1, -2) @ (kernel @ data)
    gram /= float(t) ** 2
    return (gram + namespace.swapaxes(gram, -1, -2)) * 0.5


def _batch_eigh(matrix: Any, namespace: Any) -> tuple[FloatArray, FloatArray]:
    values, vectors = namespace.linalg.eigh(matrix)
    values = as_host(values, namespace)[:, ::-1]
    vectors = as_host(vectors, namespace)[:, :, ::-1]
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise FloatingPointError("Batched moment decomposition returned nonfinite values.")
    return values, vectors


def _batch_rank_aware_fourth_basis(
    zin: FloatArray, data: Any, namespace: Any
) -> FloatArray:
    """Group windows by data rank and diagonalize their reduced Grams together."""
    batch, t, n = zin.shape
    result = np.empty((batch, n, n), dtype=np.float64)
    spaces: dict[int, list[tuple[int, FloatArray]]] = {}
    for index, window in enumerate(zin):
        _, singular, right = np.linalg.svd(window, full_matrices=False)
        threshold = np.finfo(np.float64).eps * max(t, n) * singular[0]
        rank = int(np.count_nonzero(singular > threshold))
        if rank == 0:
            result[index] = np.eye(n)
        else:
            spaces.setdefault(rank, []).append((index, right[:rank].T))
    for rank, members in spaces.items():
        indices = np.array([index for index, _ in members], dtype=np.int64)
        row_spaces = np.stack([space for _, space in members])
        reduced = data[namespace.asarray(indices)] @ namespace.asarray(row_spaces)
        values, vectors = _batch_eigh(_batch_fourth_gram(reduced, namespace), namespace)
        for local, (index, space) in enumerate(members):
            observed = _canonical_eigenvectors(values[local], space @ vectors[local])
            if rank == n:
                result[index] = observed
            else:
                null = _canonical_projector_basis(np.eye(n) - space @ space.T, n - rank)
                result[index] = np.column_stack((observed, null))
    return result


def _batch_fourth_basis(zin: FloatArray, data: Any, namespace: Any) -> FloatArray:
    """Batched counterpart of fourth_basis, with the same tie convention."""
    batch, t, n = zin.shape
    if n >= t:
        return _batch_rank_aware_fourth_basis(zin, data, namespace)
    values, vectors = _batch_eigh(_batch_fourth_gram(data, namespace), namespace)
    result = np.empty((batch, n, n), dtype=np.float64)
    threshold = 100.0 * n * np.finfo(np.float64).eps * np.maximum(values[:, 0], 0.0)
    deficient = values[:, -1] <= threshold
    for index in np.flatnonzero(~deficient):
        result[index] = _canonical_eigenvectors(values[index], vectors[index])
    if deficient.any():
        indices = np.flatnonzero(deficient)
        result[indices] = _batch_rank_aware_fourth_basis(
            zin[indices], data[namespace.asarray(indices)], namespace
        )
    return result


def pair_cores_batch(
    train: FloatArray,
    test: FloatArray,
    *,
    standardized: bool = False,
    device: Device = "cuda",
    order: MomentOrder = "both",
) -> tuple[FloatArray | None, FloatArray | None]:
    """Fit a bounded batch of independent historical or CV window pairs.

    Parameters are the batched counterparts of ``pair_core``: training
    observations have shape ``(batch, train_rows, assets)`` and targets have
    shape ``(batch, test_rows, assets)``. The caller bounds the batch size
    (calibration uses four pairs). CUDA computes batched Grams, symmetric
    eigendecompositions and projections, retaining data on the device
    between these operations. Deterministic eigenspace canonicalization
    and data-rank SVDs run on the CPU, using the same convention as the
    unbatched estimator. Results are host float64 arrays with shapes
    ``(batch, assets)`` and ``(batch, assets, assets)``.

    The CPU path delegates to the ordinary pair routine and is principally
    useful for backend parity tests. No path constructs a dense fourth tensor.
    """
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if (
        train.ndim != 3 or test.ndim != 3
        or not all(train.shape) or not all(test.shape)
        or train.shape[0] != test.shape[0] or train.shape[2] != test.shape[2]
        or not np.isfinite(train).all() or not np.isfinite(test).all()
    ):
        raise ValueError("Expected finite (batch, observations, assets) arrays with matching batch and assets.")
    if order not in ("both", "correlation", "cokurtosis"):
        raise ValueError("order must be 'both', 'correlation', or 'cokurtosis'.")
    namespace = array_namespace(device)
    if device == "cpu":
        pairs = [
            pair_core(a, b, standardized=standardized, device="cpu", order=order)
            for a, b in zip(train, test)
        ]
        second = np.stack([a for a, _ in pairs if a is not None]) if order != "cokurtosis" else None
        fourth = np.stack([b for _, b in pairs if b is not None]) if order != "correlation" else None
        return second, fourth
    if standardized:
        zin = train - train.mean(axis=1, keepdims=True)
        zout = test - test.mean(axis=1, keepdims=True)
    else:
        zin = np.stack([standardize(window)[0] for window in train])
        zout = np.stack([standardize(window, allow_constant=True)[0] for window in test])
    data, target = namespace.asarray(zin), namespace.asarray(zout)
    second = fourth = None
    if order in ("both", "correlation"):
        covariance = namespace.swapaxes(data, -1, -2) @ data / train.shape[1]
        covariance = (covariance + namespace.swapaxes(covariance, -1, -2)) * 0.5
        values, vectors = _batch_eigh(covariance, namespace)
        bases = np.stack([
            _canonical_eigenvectors(value, vector) for value, vector in zip(values, vectors)
        ])
        rotated = target @ namespace.asarray(bases)
        second = as_host(namespace.mean(rotated * rotated, axis=1), namespace)
    if order in ("both", "cokurtosis"):
        bases = _batch_fourth_basis(zin, data, namespace)
        squared = (target @ namespace.asarray(bases)) ** 2
        fourth = as_host(
            namespace.swapaxes(squared, -1, -2) @ squared / test.shape[1], namespace
        )
        fourth = (fourth + np.swapaxes(fourth, -1, -2)) * 0.5
    return second, fourth
