"""Allocation checks independent of solver implementation."""
import numpy as np
import pandas as pd
import pytest

from hoao import CV, OptimizationError, Sample


def returns(n=5, t=100):
    rng = np.random.default_rng(53)
    return rng.normal(size=(t, n)) * np.geomspace(.003, .03, n) + rng.normal(size=(t, 1)) * .01


@pytest.mark.parametrize("estimator", [Sample(), Sample(even=True), CV(random_state=2)])
def test_gmv_matches_independent_lagrange_solution(estimator):
    model = estimator.fit(returns())
    solve = np.linalg.solve(model.covariance, np.ones(5))
    expected = solve / solve.sum()
    result = model.optimize()
    np.testing.assert_allclose(result.weights, expected, rtol=1e-10, atol=1e-12)
    assert result.gradient_residual <= 1e-6
    assert result.success and abs(result.weights.sum() - 1) < 1e-12


@pytest.mark.parametrize("estimator", [Sample(), Sample(even=True), CV(random_state=2)])
def test_frontier_stationarity_and_normalization(estimator):
    model = estimator.fit(returns())
    results = model.frontier([0., .1, 1., 10., 100.])
    for gamma, result in zip([0., .1, 1., 10., 100.], results):
        w = result.weights
        gradient = 2 * model.covariance @ w / model.reference_variance
        gradient += gamma * model.cokurtosis.gradient(w) / model.reference_variance**2
        np.testing.assert_allclose(gradient - gradient.mean(), 0, atol=2e-6)
        expected = result.moments.variance/model.reference_variance + gamma*result.moments.fourth_moment/model.reference_variance**2
        assert result.objective == pytest.approx(expected)
        assert result.success
    assert results[-1].moments.fourth_moment <= results[0].moments.fourth_moment + 1e-16


def test_singular_covariance_budget_and_single_asset():
    x = returns(n=3)
    x = np.column_stack((x, x[:, 1]))
    result = Sample().fit(x).optimize()
    assert result.success and result.gradient_residual < 1e-6
    assert result.weights[1] == pytest.approx(result.weights[3])
    single = Sample().fit(x[:, :1]).optimize(100)
    np.testing.assert_array_equal(single.weights, [1.])


def test_portfolio_labels_and_zero_reference():
    model = CV(random_state=2).fit(pd.DataFrame(returns(), columns=list("ABCDE")))
    result = model.optimize(1)
    assert result.weights_to_pandas().index.tolist() == list("ABCDE")
    x = np.linspace(-1, 1, 30)
    zero = Sample().fit(np.column_stack((x, -x)))
    with pytest.raises(ValueError, match="reference variance"):
        zero.optimize()


def test_failure_is_explicit_and_contains_candidate():
    model = Sample().fit(returns(n=12))
    with pytest.raises(OptimizationError) as failure:
        model.optimize(100., max_iter=1)
    assert not failure.value.result.success
    assert failure.value.result.weights.shape == (12,)
    for bad in [-1., np.nan, np.inf]:
        with pytest.raises(ValueError):
            model.optimize(bad)
