"""Independent contractions, derivatives, scale and alignment checks."""
import numpy as np
import pandas as pd
import pytest

from hoao import CV, Sample


def returns():
    return np.random.default_rng(53).normal(size=(90, 5)) * np.geomspace(.001, .1, 5) + np.arange(5) * .002


@pytest.mark.parametrize("estimator", [Sample(), Sample(even=True), CV(random_state=6)])
def test_dense_contractions_and_derivatives(estimator):
    x = returns()
    model = estimator.fit(x)
    fourth = model.cokurtosis
    w = np.array([.4, -.2, .1, .2, .5])
    dense = fourth.to_dense()
    np.testing.assert_allclose(fourth.fourth_moment(w), np.einsum('ijkl,i,j,k,l', dense, w, w, w, w), rtol=2e-12)
    expected_gradient = 4 * np.einsum('ijkl,j,k,l->i', dense, w, w, w)
    np.testing.assert_allclose(fourth.gradient(w), expected_gradient, rtol=2e-12, atol=1e-18)
    direction = np.array([.2, .1, -.1, .3, .1])
    eps = 1e-5
    finite = (fourth.gradient(w + eps*direction) - fourth.gradient(w - eps*direction)) / (2*eps)
    np.testing.assert_allclose(fourth.hessian_vector_product(w, direction), finite, rtol=2e-8, atol=1e-15)
    scales = np.std(x, axis=0)
    standardized = fourth.to_dense(standardized=True)
    restored = np.einsum('ijkl,i,j,k,l->ijkl', standardized, scales, scales, scales, scales)
    np.testing.assert_allclose(dense, restored, atol=1e-18, rtol=2e-12)


def test_sample_portfolio_direct_returns_and_no_input_mutation():
    x = returns()
    before = x.copy()
    model = Sample().fit(x)
    w = np.array([.4, -.2, .1, .2, .5])
    p = x @ w
    risk = model.portfolio(w)
    assert risk.mean == pytest.approx(p.mean())
    assert risk.variance == pytest.approx(p.var())
    assert risk.fourth_moment == pytest.approx(np.mean((p - p.mean())**4))
    assert risk.kurtosis == pytest.approx(np.mean((p-p.mean())**4) / p.var()**2)
    assert model.reference_variance == pytest.approx(x.mean(axis=1).var())
    np.testing.assert_array_equal(x, before)
    assert not model.covariance.flags.writeable
    assert np.isnan(model.portfolio(np.zeros(5)).kurtosis)


def test_pandas_weights_align_and_export():
    x = pd.DataFrame(returns(), columns=list("ABCDE"))
    model = CV(random_state=5).fit(x)
    w = pd.Series([.3, .1, .2, .4, 0.], index=list("EDCBA"))
    np.testing.assert_allclose(model.portfolio(w).variance, model.portfolio(w.reindex(x.columns).to_numpy()).variance)
    np.testing.assert_allclose(model.cokurtosis.gradient(w), model.cokurtosis.gradient(w.reindex(x.columns).to_numpy()))
    assert model.correlation_estimate.to_pandas().columns.tolist() == list("ABCDE")
    for bad in [w.iloc[:3], pd.Series(np.ones(5), index=list("AACDE")), pd.Series(np.ones(5), index=list("ABCDF"))]:
        with pytest.raises(ValueError, match="labels"):
            model.portfolio(bad)
    with pytest.raises(ValueError, match="labels"):
        Sample().fit(x.to_numpy()).portfolio(w)


def test_dense_export_guard_precedes_allocation():
    k = Sample(even=True).fit_cokurtosis(returns())
    with pytest.raises(MemoryError):
        k.to_dense(max_bytes=1)
    for cap in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            k.to_dense(max_bytes=cap)


@pytest.mark.parametrize("bad", [np.ones((3, 2)), [[1., np.nan], [2., 3.], [4., 5.]], [[1., np.inf], [2., 3.], [4., 5.]]])
def test_invalid_fit_data(bad):
    with pytest.raises(ValueError):
        Sample().fit(bad)


def test_pandas_duplicate_axes_and_complex_inputs_rejected():
    with pytest.raises(ValueError):
        Sample().fit(pd.DataFrame(returns(), columns=list("AACDE")))
    x = pd.DataFrame(returns())
    x.index = [0] * len(x)
    with pytest.raises(ValueError):
        Sample().fit(x)
    with pytest.raises(TypeError):
        Sample().fit(returns().astype(complex))


@pytest.mark.parametrize("label", [None, np.nan, pd.NA, pd.NaT])
def test_missing_asset_labels_rejected(label):
    frame = pd.DataFrame(returns(), columns=[label, "B", "C", "D", "E"])
    for estimator in [Sample(), CV(random_state=4)]:
        with pytest.raises(ValueError, match="missing"):
            estimator.fit(frame)
