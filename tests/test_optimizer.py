"""Independent mathematical oracles, feasibility checks, and failure-path coverage."""

import numpy as np
import pandas as pd
import pytest

from mafinrisk import PortfolioOptimizer


@pytest.fixture
def inputs():
    names = ["A", "B", "C"]
    return (
        pd.Series([0.08, 0.12, 0.16], index=names),
        pd.DataFrame(np.diag([0.04, 0.09, 0.16]), index=names, columns=names),
    )


def test_gmv_matches_inverse_variance_closed_form(inputs):
    mu, cov = inputs
    expected = 1 / np.diag(cov)
    expected /= expected.sum()
    result = PortfolioOptimizer(mu, cov).minimum_variance()
    np.testing.assert_allclose(result.weights, expected, atol=2e-6)


def test_sharpe_matches_tangency_closed_form(inputs):
    mu, cov = inputs
    expected = np.linalg.solve(cov, mu - 0.02)
    expected /= expected.sum()
    result = PortfolioOptimizer(mu, cov, 0.02).maximum_sharpe()
    np.testing.assert_allclose(result.weights, expected, atol=2e-6)


def test_constrained_sharpe_against_dense_two_asset_grid():
    mu = pd.Series([0.07, 0.15], index=["A", "B"])
    cov = pd.DataFrame([[0.0225, 0.006], [0.006, 0.0625]], index=mu.index, columns=mu.index)
    model = PortfolioOptimizer(mu, cov, 0.02, lower=0.2, upper=0.8)
    grid = np.linspace(0.2, 0.8, 20001)
    w = np.column_stack([grid, 1 - grid])
    ratios = (w @ mu.to_numpy() - 0.02) / np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    result = model.maximum_sharpe()
    assert result.sharpe >= ratios.max() - 1e-7
    assert 0.2 - 1e-7 <= result.weights.iloc[0] <= 0.8 + 1e-7


def test_constraints_and_frontier(inputs):
    mu, cov = inputs
    model = PortfolioOptimizer(mu, cov, upper=0.5)
    for p in [model.minimum_variance(), model.maximum_sharpe(), model.minimum_variance(0.13)]:
        assert p.weights.sum() == pytest.approx(1, abs=1e-7)
        assert p.weights.min() >= -1e-7
        assert p.weights.max() <= 0.5 + 1e-7
    assert model.minimum_variance(0.13).expected_return >= 0.13 - 1e-7
    curve = model.efficient_frontier(15)
    assert (np.diff(curve.expected_return) >= -1e-7).all()
    assert (np.diff(curve.volatility) >= -1e-7).all()
    assert curve.iloc[-1].expected_return == pytest.approx(0.14, abs=1e-7)
    np.testing.assert_allclose(curve.filter(like="weight_").sum(axis=1), 1, atol=1e-7)
    with pytest.raises(ValueError, match="feasible maximum"):
        model.minimum_variance(0.2)


def test_asset_specific_bounds(inputs):
    mu, cov = inputs
    lower = pd.Series([0.1, 0.2, 0.1], index=mu.index)
    upper = pd.Series([0.4, 0.7, 0.5], index=mu.index)
    model = PortfolioOptimizer(mu, cov, lower=lower, upper=upper)
    for p in [model.minimum_variance(), model.maximum_sharpe()]:
        assert (p.weights >= lower - 1e-7).all()
        assert (p.weights <= upper + 1e-7).all()


def test_fixed_weights(inputs):
    mu, cov = inputs
    bounds = pd.Series([0.2, 0.3, 0.5], index=mu.index)
    model = PortfolioOptimizer(mu, cov, lower=bounds, upper=bounds)
    np.testing.assert_allclose(model.minimum_variance().weights, bounds, atol=1e-7)
    np.testing.assert_allclose(model.maximum_sharpe().weights, bounds, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upper": 0.2},
        {"lower": 0.4},
        {"lower": -0.1},
        {"upper": 1.1},
        {"upper": float("nan")},
        {"risk_free_rate": float("inf")},
    ],
)
def test_invalid_constraints(inputs, kwargs):
    with pytest.raises(ValueError):
        PortfolioOptimizer(*inputs, **kwargs)


def test_bad_labels_and_covariance(inputs):
    mu, cov = inputs
    with pytest.raises(ValueError, match="labels"):
        PortfolioOptimizer(mu, cov.iloc[::-1])
    with pytest.raises(ValueError, match="positive definite"):
        PortfolioOptimizer(mu, cov * 0)
    bad = cov.copy()
    bad.iloc[0, 1] = 0.2
    with pytest.raises(ValueError, match="symmetric"):
        PortfolioOptimizer(mu, bad)
    bad.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PortfolioOptimizer(mu, bad)


def test_no_positive_excess_return(inputs):
    model = PortfolioOptimizer(*inputs, risk_free_rate=0.20)
    with pytest.raises(ValueError, match="positive excess"):
        model.maximum_sharpe()
    assert np.isfinite(model.minimum_variance().volatility)


def test_constant_expected_returns(inputs):
    mu, cov = inputs
    mu[:] = 0.1
    model = PortfolioOptimizer(mu, cov)
    np.testing.assert_allclose(
        model.maximum_sharpe().weights, model.minimum_variance().weights, atol=2e-6
    )
    assert len(model.efficient_frontier(3)) == 3


def test_evaluate_rejects_invalid_weights(inputs):
    model = PortfolioOptimizer(*inputs)
    for w in [[0.5, 0.5], [0.5, 0.5, 0.5], [-0.1, 0.5, 0.6], [np.nan, 0, 1]]:
        with pytest.raises(ValueError):
            model.evaluate(w)
    with pytest.raises(ValueError, match="labels"):
        model.evaluate(pd.Series([0.2, 0.3, 0.5], index=["C", "B", "A"]))


def test_frontier_rejects_invalid_point_count(inputs):
    for count in [1, 3.5, True]:
        with pytest.raises(ValueError):
            PortfolioOptimizer(*inputs).efficient_frontier(count)
