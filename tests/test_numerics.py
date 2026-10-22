"""Independent dense identities and rank-deficient basis checks."""

import numpy as np
import pytest

from hoao import _numerics as numerics
from hoao import _backend as backend
from hoao._backend import array_namespace


def dense_fourth(x):
    return np.einsum("ta,tb,tc,td->abcd", x, x, x, x) / len(x)


def dense_basis(x):
    tensor = dense_fourth(x)
    unfolding = tensor.reshape(x.shape[1], -1)
    _, vectors = np.linalg.eigh(unfolding @ unfolding.T)
    return vectors[:, ::-1]


def test_standardization_uses_population_variance_and_preserves_input():
    x = np.array([[1.0, 20.0], [3.0, 16.0], [-2.0, 9.0]])
    original = x.copy()
    z, mean, scales = numerics.standardize(x)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(mean, x.mean(0))
    np.testing.assert_allclose(scales, x.std(0, ddof=0))
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(z**2, axis=0), 1.0)
    constant = np.full((13, 2), 0.1)
    with pytest.raises(ValueError, match="constant"):
        numerics.standardize(constant)
    z, _, scales = numerics.standardize(constant, allow_constant=True)
    np.testing.assert_array_equal(z, np.zeros_like(z))
    np.testing.assert_array_equal(scales, np.zeros(2))


@pytest.mark.parametrize("block_size", [1, 7, 256])
def test_implicit_gram_matches_independent_dense_tensor(block_size):
    x = np.random.default_rng(56).normal(size=(31, 5))
    x -= x.mean(axis=0)
    tensor = dense_fourth(x)
    unfolding = tensor.reshape(5, -1)
    reference = unfolding @ unfolding.T
    np.testing.assert_allclose(
        numerics.fourth_gram(x, block_size), reference, rtol=2e-13, atol=2e-13
    )


@pytest.mark.parametrize("shape", [(8, 12), (100, 200), (32, 8)])
def test_rank_aware_basis_reconstructs_defining_gram(shape):
    x = np.random.default_rng(46).normal(size=shape)
    z, _, _ = numerics.standardize(x)
    basis = numerics.fourth_basis(z)
    h = numerics.fourth_gram(z)
    rotated = basis.T @ h @ basis
    diagonal = np.diag(rotated)
    scale = np.linalg.norm(h)
    np.testing.assert_allclose(basis.T @ basis, np.eye(shape[1]), atol=3e-12)
    assert np.linalg.norm(rotated - np.diag(diagonal)) / scale < 2e-12
    assert np.linalg.norm((basis * diagonal) @ basis.T - h) / scale < 2e-12
    assert np.min(-np.diff(diagonal)) >= -2e-12 * scale
    rank = np.linalg.matrix_rank(z)
    if rank < shape[1]:
        assert np.linalg.norm(z @ basis[:, rank:]) / np.linalg.norm(z) < 2e-12
    np.testing.assert_array_equal(basis, numerics.fourth_basis(z))


def test_duplicate_columns_and_zero_data_have_finite_complete_bases():
    x = np.random.default_rng(87).normal(size=(30, 4))
    x = np.column_stack((x, x[:, 0], x[:, 0] + x[:, 1]))
    z, _, _ = numerics.standardize(x)
    u = numerics.fourth_basis(z)
    np.testing.assert_allclose(u.T @ u, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(z @ u[:, 4:], 0.0, atol=2e-12)
    np.testing.assert_array_equal(numerics.fourth_basis(np.zeros((10, 3))), np.eye(3))


def test_canonical_tied_space_is_independent_of_supplied_orientation():
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(9, 4)))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    first = numerics._canonical_eigenvectors(np.ones(4), q)
    second = numerics._canonical_eigenvectors(np.ones(4), q @ rotation)
    np.testing.assert_allclose(first, second, atol=2e-13)
    z = np.vstack((np.eye(4), -np.eye(4)))
    _, u2 = numerics.correlation_basis(z)
    u4 = numerics.fourth_basis(z)
    np.testing.assert_allclose(u2, np.eye(4), atol=1e-13)
    np.testing.assert_allclose(u4, np.eye(4), atol=1e-13)


@pytest.mark.parametrize("standardized", [False, True])
def test_pair_core_matches_dense_projection(standardized):
    rng = np.random.default_rng(55)
    train = rng.normal(size=(41, 4)) * np.arange(1, 5) + np.arange(4)
    test = rng.normal(size=(27, 4)) * np.arange(2, 6) - np.arange(4)
    if standardized:
        zin, zout = train - train.mean(0), test - test.mean(0)
    else:
        zin = (train - train.mean(0)) / train.std(0)
        zout = (test - test.mean(0)) / test.std(0)
    c = zin.T @ zin / len(zin)
    _, u2 = np.linalg.eigh(c)
    u2 = u2[:, ::-1]
    u4 = dense_basis(zin)
    test_fourth = dense_fourth(zout)
    expected2 = np.diag(u2.T @ (zout.T @ zout / len(zout)) @ u2)
    expected4 = np.einsum("abcd,ai,bi,cj,dj->ij", test_fourth, u4, u4, u4, u4)
    a, b = numerics.pair_core(train, test, standardized=standardized)
    np.testing.assert_allclose(a, expected2, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(b, expected4, rtol=1e-11, atol=1e-11)


def test_unused_moment_order_is_not_computed(monkeypatch):
    x = np.random.default_rng(7).normal(size=(30, 4))

    def forbidden(*args, **kwargs):
        raise AssertionError("Unused basis was computed")

    with monkeypatch.context() as patch:
        patch.setattr(numerics, "fourth_basis", forbidden)
        a, b = numerics.pair_core(x[:18], x[18:], order="correlation")
        assert a.shape == (4,) and b is None
    with monkeypatch.context() as patch:
        patch.setattr(numerics, "correlation_basis", forbidden)
        a, b = numerics.pair_core(x[:18], x[18:], order="cokurtosis")
        assert a is None and b.shape == (4, 4)


def test_historical_constant_target_columns_are_retained_as_zero():
    train = np.random.default_rng(67).normal(size=(20, 3))
    target = np.full((12, 3), 0.2)
    a, b = numerics.pair_core(train, target)
    np.testing.assert_array_equal(a, np.zeros(3))
    np.testing.assert_array_equal(b, np.zeros((3, 3)))


def test_invalid_numerical_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="positive integer"):
        numerics.fourth_gram(np.ones((3, 2)), 0)
    with pytest.raises(ValueError, match="nonpositive"):
        numerics.normalize_correlation(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same assets"):
        numerics.pair_core(np.ones((3, 2)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="order"):
        numerics.pair_core(np.ones((3, 2)), np.ones((4, 2)), order="odd")


def test_cuda_import_is_optional_and_missing_backend_is_explicit(monkeypatch):
    def without_cupy(name):
        assert name == "cupy"
        raise ImportError("Simulated absent optional dependency")

    monkeypatch.setattr(backend, "import_module", without_cupy)
    assert array_namespace("cpu") is np
    with pytest.raises(ImportError, match="CuPy"):
        array_namespace("cuda")
    with pytest.raises(ValueError, match="device"):
        array_namespace("gpu")


def test_cuda_matches_cpu_when_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA driver unavailable")
    x = np.random.default_rng(763).normal(size=(40, 6))
    a_cpu, b_cpu = numerics.pair_core(x[:24], x[24:], device="cpu")
    a_gpu, b_gpu = numerics.pair_core(x[:24], x[24:], device="cuda")
    np.testing.assert_allclose(a_gpu, a_cpu, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(b_gpu, b_cpu, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("standardized", [False, True])
@pytest.mark.parametrize("shape", [(24, 5), (8, 12), (65, 200)])
def test_vectorized_batch_algebra_matches_scalar_pairs_without_gpu(
    monkeypatch, standardized, shape
):
    """Exercise the CUDA algorithm through NumPy; this is not a hardware test."""
    rng = np.random.default_rng(459)
    train = rng.normal(size=(3, *shape))
    test = rng.normal(size=(3, 11, shape[1]))
    expected_a, expected_b = numerics.pair_cores_batch(
        train, test, standardized=standardized, device="cpu"
    )
    monkeypatch.setattr(numerics, "array_namespace", lambda device: np)
    actual_a, actual_b = numerics.pair_cores_batch(
        train, test, standardized=standardized, device="cuda"
    )
    np.testing.assert_allclose(actual_a, expected_a, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(actual_b, expected_b, rtol=2e-10, atol=2e-11)


def test_batch_resolves_duplicate_columns_and_tied_eigenspaces(monkeypatch):
    rng = np.random.default_rng(63)
    train = rng.normal(size=(2, 25, 4))
    train = np.concatenate((train, train[:, :, :1]), axis=2)
    test = rng.normal(size=(2, 17, 5))
    reference = numerics.pair_cores_batch(train, test, device="cpu")
    monkeypatch.setattr(numerics, "array_namespace", lambda device: np)
    result = numerics.pair_cores_batch(train, test, device="cuda")
    for expected, actual in zip(reference, result):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    tied = np.stack([np.vstack((np.eye(5), -np.eye(5)))] * 2)
    a, b = numerics.pair_cores_batch(tied, test, standardized=True, device="cuda")
    for index in range(2):
        expected_a, expected_b = numerics.pair_core(tied[index], test[index], standardized=True)
        np.testing.assert_allclose(a[index], expected_a, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(b[index], expected_b, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("order", ["correlation", "cokurtosis"])
def test_batch_only_computes_requested_order(monkeypatch, order):
    rng = np.random.default_rng(53)
    train = rng.normal(size=(2, 12, 4))
    target = rng.normal(size=(2, 8, 4))
    monkeypatch.setattr(numerics, "array_namespace", lambda device: np)
    a, b = numerics.pair_cores_batch(train, target, order=order, device="cuda")
    if order == "correlation":
        assert a.shape == (2, 4) and b is None
    else:
        assert a is None and b.shape == (2, 4, 4)


def test_batch_validates_shapes_and_order():
    with pytest.raises(ValueError, match="matching batch and assets"):
        numerics.pair_cores_batch(np.ones((2, 5, 3)), np.ones((3, 5, 3)), device="cpu")
    with pytest.raises(ValueError, match="order"):
        numerics.pair_cores_batch(np.ones((2, 5, 3)), np.ones((2, 5, 3)), order="odd", device="cpu")
