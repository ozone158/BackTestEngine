"""
Tests for Step 5.2 — Risk attribution.

5.2.7: strategy_returns = 0.5 * market_returns + noise -> beta ≈ 0.5.
       strategy_returns = constant -> beta ≈ 0.
"""

import numpy as np
import pytest

from src.backtest.risk_attribution import RiskAttribution, decompose


def test_beta_half_when_strategy_is_half_market_plus_noise():
    """5.2.7: strategy_returns = 0.5 * market_returns + noise; assert beta ≈ 0.5."""
    np.random.seed(42)
    n = 252
    market = np.random.randn(n) * 0.01
    strategy = 0.5 * market + np.random.randn(n) * 0.005
    out = RiskAttribution.decompose(strategy, factor_returns=market, periods_per_year=252)
    assert abs(out["beta"] - 0.5) < 0.15
    assert out["R2"] > 0.4
    assert "alpha" in out and "residuals" in out
    assert len(out["residuals"]) == n


def test_beta_near_zero_when_strategy_is_constant():
    """5.2.7: strategy_returns = constant; assert beta ≈ 0."""
    n = 100
    market = np.random.randn(n) * 0.01
    strategy = np.full(n, 0.01)
    out = RiskAttribution.decompose(strategy, factor_returns=market, periods_per_year=252)
    assert abs(out["beta"]) < 0.2
    assert out["alpha"] == pytest.approx(0.01 * 252, rel=0.5)


def test_decompose_module_level():
    """decompose() at module level matches RiskAttribution.decompose()."""
    strategy = np.array([0.01, -0.005, 0.02, 0.0])
    market = np.array([0.008, -0.006, 0.015, 0.001])
    out1 = decompose(strategy, factor_returns=market)
    out2 = RiskAttribution.decompose(strategy, factor_returns=market)
    assert out1["alpha"] == out2["alpha"]
    assert out1["beta"] == out2["beta"]
    assert out1["R2"] == out2["R2"]
    np.testing.assert_array_almost_equal(out1["residuals"], out2["residuals"])


def test_residuals_sum_of_squares():
    """Residuals = strategy - (alpha_per_period + beta * market)."""
    np.random.seed(123)
    market = np.random.randn(50) * 0.01
    strategy = 0.3 * market + 0.0001 + np.random.randn(50) * 0.002
    out = RiskAttribution.decompose(strategy, factor_returns=market, periods_per_year=252)
    alpha_ann = out["alpha"]
    beta = out["beta"]
    alpha_per = alpha_ann / 252
    fitted = alpha_per + beta * market
    np.testing.assert_allclose(out["residuals"], strategy - fitted)


def test_factor_matrix_multiple_factors():
    """Multi-factor: factor_matrix (n x k) yields beta array."""
    n = 60
    f1 = np.random.randn(n) * 0.01
    f2 = np.random.randn(n) * 0.005
    strategy = 0.4 * f1 + 0.2 * f2 + np.random.randn(n) * 0.002
    X = np.column_stack([f1, f2])
    out = RiskAttribution.decompose(strategy, factor_matrix=X, periods_per_year=252)
    assert isinstance(out["beta"], np.ndarray)
    assert len(out["beta"]) == 2
    assert abs(out["beta"][0] - 0.4) < 0.2
    assert abs(out["beta"][1] - 0.2) < 0.2


def test_decompose_requires_factor():
    """decompose() raises if neither factor_returns nor factor_matrix provided."""
    with pytest.raises(ValueError, match="factor_returns or factor_matrix"):
        RiskAttribution.decompose([0.01, 0.02])


def test_align_drops_nan():
    """Align drops indices where any series has NaN."""
    strategy = np.array([0.01, np.nan, 0.02, 0.0])
    market = np.array([0.01, 0.0, np.nan, 0.005])
    out = RiskAttribution.decompose(strategy, factor_returns=market)
    assert len(out["residuals"]) == 2
