"""Estimator behavior against independently expanded small tensor oracles."""

from copy import deepcopy
from itertools import permutations

import numpy as np
import pytest

from hoao import estimators as estimator_module
from hoao.estimators import CV, Sample


def dense_fourth(z):
    return np.einsum("ta,tb,tc,td->abcd", z, z, z, z) / len(z)


def basis4(z):
    unfolding = dense_fourth(z).reshape(z.shape[1], -1)
    _, u = np.linalg.eigh(unfolding @ unfolding.T)
    return u[:, ::-1]


def expand_even(b):
    n = len(b)
    out = np.zeros((n,) * 4)
    for i in range(n):
        out[i, i, i, i] = b[i, i]
        for j in range(i):
            for indices in set(permutations((i, i, j, j))):
                out[indices] = b[i, j]
    return out


def dense_cv_reference(x, seed, n_splits, fraction):
    scales = x.std(axis=0)
    z = (x - x.mean(0)) / scales
    rng = np.random.default_rng(seed)
    cut = int(fraction * len(z))
    n = x.shape[1]
    second = np.zeros(n)
    fourth = np.zeros((n, n))
    for _ in range(n_splits):
        order = rng.permutation(len(z))
        train, test = z[order[:cut]], z[order[cut:]]
        train, test = train - train.mean(0), test - test.mean(0)
        _, u2 = np.linalg.eigh(train.T @ train / len(train))
        u2 = u2[:, ::-1]
        u4 = basis4(train)
        second += np.diag(u2.T @ (test.T @ test / len(test)) @ u2) / n_splits
        fourth += np.einsum("abcd,ai,bi,cj,dj->ij", dense_fourth(test), u4, u4, u4, u4) / n_splits
    _, u2 = np.linalg.eigh(z.T @ z / len(z))
    u2 = u2[:, ::-1]
    c = (u2 * second) @ u2.T
    c /= np.sqrt(np.outer(np.diag(c), np.diag(c)))
    u4 = basis4(z)
    tensor = np.einsum("ia,jb,kc,ld,abcd->ijkl", u4, u4, u4, u4, expand_even(fourth), optimize=True)
    return c * np.outer(scales, scales), tensor, scales


@pytest.mark.parametrize("n", [1, 5])
def test_sample_recovers_direct_portfolio_moments(n):
    rng = np.random.default_rng(25)
    x = rng.normal(size=(57, n)) * np.arange(1, n + 1) + 0.1
    w = rng.normal(size=n)
    model = Sample().fit(x)
    raw = x @ w
    centered = raw - raw.mean()
    risk = model.portfolio(w)
    np.testing.assert_allclose(risk.mean, raw.mean(), rtol=1e-12)
    np.testing.assert_allclose(risk.variance, np.mean(centered**2), rtol=1e-12)
    np.testing.assert_allclose(risk.fourth_moment, np.mean(centered**4), rtol=1e-12)
    np.testing.assert_allclose(risk.kurtosis, np.mean(centered**4) / np.mean(centered**2) ** 2, rtol=1e-12)
    np.testing.assert_allclose(model.reference_variance, np.var(x.mean(axis=1)), rtol=1e-12)


def test_sample_even_matches_dense_even_projection():
    rng = np.random.default_rng(99)
    x = rng.normal(size=(44, 4)) * np.arange(1, 5)
    z = (x - x.mean(0)) / x.std(0)
    u = basis4(z)
    b = np.einsum("abcd,ai,bi,cj,dj->ij", dense_fourth(z), u, u, u, u)
    model = Sample(even=True).fit(x)
    w = rng.normal(size=4)
    v = u.T @ (x.std(0) * w)
    expected = np.einsum("ijkl,i,j,k,l", expand_even(b), v, v, v, v)
    np.testing.assert_allclose(model.cokurtosis.fourth_moment(w), expected, rtol=1e-12)
    np.testing.assert_allclose(model.covariance, Sample().fit(x).covariance, rtol=1e-13)


def test_large_offset_sample_matches_independent_twice_centered_moments():
    rng = np.random.default_rng(138)
    x = 1e10 + rng.normal(size=(20, 4)) * 0.01
    centered = x - x.mean(axis=0)
    centered -= centered.mean(axis=0)
    model = Sample().fit(x)
    np.testing.assert_allclose(model.cokurtosis.centered, centered, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(model.covariance, centered.T @ centered / len(x), rtol=1e-13)
    w = np.array([0.2, -0.3, 0.4, 0.7])
    portfolio = centered @ w
    moments = model.portfolio(w)
    np.testing.assert_allclose(moments.variance, np.mean(portfolio**2), rtol=1e-13)
    np.testing.assert_allclose(moments.fourth_moment, np.mean(portfolio**4), rtol=1e-13)
    np.testing.assert_array_equal(model.mean, x.mean(axis=0))
    np.testing.assert_allclose(
        model.reference_variance, np.var((x - x.mean(axis=0)).mean(axis=1)), rtol=1e-14
    )


def test_cv_matches_independent_dense_fold_and_tensor_reference():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(37, 4)) * np.arange(1, 5) + np.arange(4)
    expected_covariance, expected_tensor, scales = dense_cv_reference(x, 18, 4, 0.65)
    model = CV(n_splits=4, random_state=18).fit(x)
    np.testing.assert_allclose(model.covariance, expected_covariance, rtol=2e-11, atol=2e-11)
    w = rng.normal(size=4)
    scaled = scales * w
    expected = np.einsum("ijkl,i,j,k,l", expected_tensor, scaled, scaled, scaled, scaled)
    np.testing.assert_allclose(model.cokurtosis.fourth_moment(w), expected, rtol=2e-11)
    np.testing.assert_array_equal(np.diag(model.correlation), np.ones(4))


@pytest.mark.parametrize("estimator", [Sample(), Sample(even=True), CV(n_splits=3, random_state=4)])
def test_partial_fits_match_joint_fit(estimator):
    x = np.random.default_rng(56).normal(size=(35, 5))
    model = estimator.fit(x)
    second = estimator.fit_correlation(x)
    fourth = estimator.fit_cokurtosis(x)
    np.testing.assert_array_equal(model.correlation, second.correlation)
    w = np.arange(1, 6) / 15
    np.testing.assert_allclose(model.cokurtosis.fourth_moment(w), fourth.fourth_moment(w), rtol=1e-13)


def test_cv_partitions_are_disjoint_and_cover_the_whole_window(monkeypatch):
    x = np.random.default_rng(91).normal(size=(40, 3))
    z = (x - x.mean(0)) / x.std(0)
    original = estimator_module.pair_core
    seen = []

    def inspect(train, test, **kwargs):
        joined = np.vstack((train, test))
        indices = [
            int(np.flatnonzero(np.all(np.isclose(z, row, rtol=1e-13, atol=1e-14), axis=1))[0])
            for row in joined
        ]
        assert sorted(indices) == list(range(len(z)))
        assert len(train) == 26 and len(test) == 14
        assert kwargs["standardized"] is True
        seen.append(indices)
        return original(train, test, **kwargs)

    monkeypatch.setattr(estimator_module, "pair_core", inspect)
    CV(n_splits=3, random_state=65).fit(x)
    assert len(seen) == 3 and seen[0] != seen[1]


def test_reusable_generator_does_not_advance_the_callers_state():
    rng = np.random.default_rng(23)
    state = deepcopy(rng.bit_generator.state)
    estimator = CV(n_splits=2, random_state=rng)
    x = np.random.default_rng(29).normal(size=(32, 3))
    first = estimator.fit(x)
    second = estimator.fit(x)
    assert rng.bit_generator.state == state
    np.testing.assert_array_equal(first.correlation, second.correlation)
    np.testing.assert_array_equal(first.cokurtosis.core, second.cokurtosis.core)


def test_cv_more_assets_than_observations_remains_compact():
    x = np.random.default_rng(634).normal(size=(100, 200)) * 0.01
    model = CV(n_splits=2, random_state=8).fit(x)
    assert model.cokurtosis.basis.shape == (200, 200)
    assert model.cokurtosis.core.shape == (200, 200)
    assert model.cokurtosis.centered is None
    risk = model.portfolio(np.full(200, 1 / 200))
    assert np.isfinite(risk.fourth_moment) and risk.fourth_moment > 0
    assert np.isfinite(risk.variance) and risk.variance > 0


def test_dataframe_labels_and_array_results_match():
    pd = pytest.importorskip("pandas")
    x = np.random.default_rng(12).normal(size=(30, 3))
    frame = pd.DataFrame(x, columns=["AAA", "BBB", "CCC"], index=pd.date_range("2020-01-01", periods=30))
    estimator = CV(n_splits=2, random_state=12)
    labeled, plain = estimator.fit(frame), estimator.fit(x)
    assert labeled.assets == ("AAA", "BBB", "CCC")
    np.testing.assert_array_equal(labeled.covariance, plain.covariance)
    weights = pd.Series([0.5, 0.2, 0.3], index=["CCC", "AAA", "BBB"])
    assert labeled.portfolio(weights) == plain.portfolio([0.2, 0.3, 0.5])


def test_returns_scaling_changes_units_without_changing_dependence():
    rng = np.random.default_rng(93)
    x = rng.normal(size=(45, 4))
    estimator = CV(n_splits=2, random_state=17)
    model, scaled = estimator.fit(x), estimator.fit(x * 100)
    np.testing.assert_allclose(model.correlation, scaled.correlation, atol=2e-12)
    w = np.full(4, 0.25)
    a, b = model.portfolio(w), scaled.portfolio(w)
    np.testing.assert_allclose(b.variance, 100**2 * a.variance, rtol=1e-12)
    np.testing.assert_allclose(b.fourth_moment, 100**4 * a.fourth_moment, rtol=1e-12)
    np.testing.assert_allclose(a.kurtosis, b.kurtosis, rtol=1e-12)


@pytest.mark.parametrize("kwargs", [{"n_splits": 0}, {"n_splits": True}, {"train_fraction": 1}, {"train_fraction": np.nan}, {"device": "gpu"}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CV(**kwargs)


@pytest.mark.parametrize("estimator", [Sample(), CV(n_splits=2)])
def test_invalid_return_windows_are_rejected(estimator):
    with pytest.raises(ValueError):
        estimator.fit(np.ones((20, 3)))
    with pytest.raises(ValueError):
        estimator.fit(np.array([[0.0, 1.0], [2.0, np.nan], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        estimator.fit(np.ones(20))
    with pytest.raises(ValueError):
        CV().fit(np.random.default_rng(7).normal(size=(5, 3)))
