"""
Tests for Step 5.1 — Performance metrics.

5.1.9: Known return series (constant 0.01); assert Sharpe and total return.
       Synthetic equity with one drawdown; assert max_drawdown.
"""

import numpy as np
import pytest

from src.backtest.performance import (
    PerformanceMetrics,
    calmar_ratio,
    max_drawdown_from_equity,
    sharpe_annual,
    sortino_annual,
    total_return_from_equity,
    total_return_from_returns,
)


def test_total_return_from_equity():
    """5.1.2: total return from equity curve."""
    equity = np.array([100.0, 102.0, 105.0, 103.0])
    assert abs(total_return_from_equity(equity) - 0.03) < 1e-9


def test_total_return_from_returns():
    """5.1.2: total return from return series (1+r1)(1+r2)... - 1."""
    returns = np.array([0.01, 0.01, 0.01])
    expected = 1.01 ** 3 - 1
    assert abs(total_return_from_returns(returns) - expected) < 1e-9


def test_constant_return_series_sharpe_and_total_return():
    """5.1.9: Known return series (e.g. constant 0.01 per period); assert Sharpe and total return."""
    n = 252
    returns = np.full(n, 0.01)
    total_ret = total_return_from_returns(returns)
    assert abs(total_ret - (1.01 ** n - 1)) < 1e-6
    sharpe = sharpe_annual(returns, periods_per_year=252)
    # std of constant is 0 -> sharpe returns 0
    assert sharpe == 0.0
    # Slightly noisy so we get non-zero std
    returns_noisy = 0.01 + np.random.RandomState(42).randn(n) * 0.005
    sharpe_n = sharpe_annual(returns_noisy, periods_per_year=252)
    assert sharpe_n > 0


def test_synthetic_equity_one_drawdown_max_drawdown():
    """5.1.9: Synthetic equity with one drawdown; assert max_drawdown."""
    # Equity: 100 -> 120 (peak) -> 90 (trough) -> 100
    equity = np.array([100.0, 110.0, 120.0, 105.0, 90.0, 95.0, 100.0])
    max_dd = max_drawdown_from_equity(equity)
    # Peak at 120, trough at 90 -> drawdown = (120-90)/120 = 0.25
    assert abs(max_dd - 0.25) < 1e-9


def test_performance_metrics_compute_from_equity():
    """5.1.8: compute() from equity_series returns all keys."""
    equity = np.array([100.0, 102.0, 101.0, 105.0, 103.0])
    out = PerformanceMetrics.compute(equity_series=equity, periods_per_year=252)
    assert "sharpe_annual" in out
    assert "sortino_annual" in out
    assert "calmar" in out
    assert "max_drawdown" in out
    assert "total_return" in out
    assert "num_trades" in out
    assert "win_rate" in out
    assert out["total_return"] == pytest.approx(0.03)
    # Peak 105, trough 103 -> max_drawdown = (105-103)/105
    assert out["max_drawdown"] == pytest.approx(2.0 / 105.0)
    assert out["num_trades"] == 0
    assert out["win_rate"] is None


def test_performance_metrics_compute_from_returns():
    """compute() from returns_series only."""
    returns = np.array([0.02, -0.01, 0.03, 0.0])
    out = PerformanceMetrics.compute(returns_series=returns, periods_per_year=252)
    assert out["total_return"] == pytest.approx(total_return_from_returns(returns))
    assert out["max_drawdown"] >= 0
    assert out["sharpe_annual"] != 0 or np.std(returns) == 0


def test_calmar_zero_drawdown():
    """5.1.6: Handle zero max_drawdown."""
    assert calmar_ratio(0.1, 0.0) == 0.0


def test_sortino_downside_std():
    """5.1.5: Sortino uses downside std (returns below target)."""
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    s = sortino_annual(returns, periods_per_year=252)
    assert s >= 0


def test_trade_stats_from_fills_with_signal_id():
    """5.1.7: num_trades and win_rate from fills with signal_id."""
    # Two trades: signal_id 1 (win), signal_id 2 (loss)
    fills = [
        {"signal_id": 1, "side": "buy", "quantity": 10, "price": 100, "commission": 1},
        {"signal_id": 1, "side": "sell", "quantity": 10, "price": 105, "commission": 1},
        {"signal_id": 2, "side": "sell", "quantity": 5, "price": 50, "commission": 0.5},
        {"signal_id": 2, "side": "buy", "quantity": 5, "price": 55, "commission": 0.5},
    ]
    out = PerformanceMetrics.compute(equity_series=[100, 105, 99], fill_events=fills)
    assert out["num_trades"] == 2
    assert out["win_rate"] == pytest.approx(0.5)


def test_trade_stats_from_fills_without_signal_id():
    """Without signal_id: num_trades = len(fills)//2, win_rate None."""
    from src.backtest.events import FillEvent
    from datetime import datetime, timezone

    utc = timezone.utc
    fills = [
        FillEvent(datetime(2025, 1, 1, tzinfo=utc), symbol="A", side="buy", quantity=10, price=100, commission=0),
        FillEvent(datetime(2025, 1, 1, tzinfo=utc), symbol="B", side="sell", quantity=10, price=80, commission=0),
    ]
    out = PerformanceMetrics.compute(equity_series=[100, 101], fill_events=fills)
    assert out["num_trades"] == 1
    assert out["win_rate"] is None
