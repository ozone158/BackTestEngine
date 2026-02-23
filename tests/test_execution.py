"""
Tests for Step 3.5 — ExecutionHandler.

3.5.7: Unit test: order 10 @ 100, 5 bps slippage, $1 commission → fill price and commission as expected.
"""

from datetime import datetime, timezone

import pytest

from src.backtest.events import SIDE_BUY, SIDE_SELL
from src.backtest.execution import BacktestExecutionHandler


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def test_fill_price_and_commission_buy():
    """Order 10 @ 100, 5 bps slippage, $1 commission → fill price 100.05, commission 1."""
    handler = BacktestExecutionHandler(slippage_bps=5.0, commission_per_trade=1.0)
    ts = _utc(2025, 1, 15, 10, 0)
    fill = handler.execute("AAPL", SIDE_BUY, 10.0, 100.0, ts)

    assert fill.symbol == "AAPL"
    assert fill.side == SIDE_BUY
    assert fill.quantity == 10.0
    assert fill.price == pytest.approx(100.0 * (1.0 + 5.0 / 10_000.0), rel=1e-9)  # 100.05
    assert fill.commission == 1.0
    assert fill.slippage_bps == 5.0
    assert fill.timestamp == ts


def test_fill_price_sell_slippage():
    """Sell: fill price = ref * (1 - slippage_bps/10000)."""
    handler = BacktestExecutionHandler(slippage_bps=5.0)
    ts = _utc(2025, 1, 15)
    fill = handler.execute("MSFT", SIDE_SELL, 5.0, 100.0, ts)

    assert fill.side == SIDE_SELL
    assert fill.price == pytest.approx(99.95, rel=1e-9)
    assert fill.commission == 0.0
    assert fill.slippage_bps == 5.0


def test_commission_per_share():
    """Commission = per_trade + quantity * per_share."""
    handler = BacktestExecutionHandler(commission_per_trade=1.0, commission_per_share=0.05)
    fill = handler.execute("A", SIDE_BUY, 10.0, 50.0, _utc(2025, 1, 1))
    assert fill.commission == pytest.approx(1.0 + 10.0 * 0.05, rel=1e-9)  # 1.5


def test_callable_interface():
    """Handler is callable (symbol, side, quantity, ref_price, timestamp) -> FillEvent."""
    handler = BacktestExecutionHandler(slippage_bps=0.0, commission_per_trade=0.5)
    fill = handler("X", SIDE_SELL, 1.0, 200.0, _utc(2025, 1, 1))
    assert fill.symbol == "X"
    assert fill.side == SIDE_SELL
    assert fill.quantity == 1.0
    assert fill.price == 200.0
    assert fill.commission == 0.5


def test_zero_slippage_no_adjustment():
    """Zero slippage: fill price equals ref price."""
    handler = BacktestExecutionHandler(slippage_bps=0.0)
    fill_buy = handler.execute("A", SIDE_BUY, 1.0, 100.0, _utc(2025, 1, 1))
    fill_sell = handler.execute("A", SIDE_SELL, 1.0, 100.0, _utc(2025, 1, 1))
    assert fill_buy.price == 100.0
    assert fill_sell.price == 100.0


def test_half_spread_bps_buy():
    """Optional bid-ask: buy at ref * (1 + half_spread_bps/10000), then slippage."""
    handler = BacktestExecutionHandler(half_spread_bps=10.0, slippage_bps=5.0)  # 10 bps spread, 5 bps slippage
    fill = handler.execute("A", SIDE_BUY, 1.0, 100.0, _utc(2025, 1, 1))
    # effective_ref = 100 * 1.001 = 100.1, then fill = 100.1 * 1.0005 = 100.15005
    expected = 100.0 * (1.0 + 10.0 / 10_000.0) * (1.0 + 5.0 / 10_000.0)
    assert fill.price == pytest.approx(expected, rel=1e-9)


def test_half_spread_bps_sell():
    """Sell at ref * (1 - half_spread_bps/10000), then slippage."""
    handler = BacktestExecutionHandler(half_spread_bps=10.0, slippage_bps=5.0)
    fill = handler.execute("A", SIDE_SELL, 1.0, 100.0, _utc(2025, 1, 1))
    # effective_ref = 100 * 0.999 = 99.9, fill = 99.9 * 0.9995
    expected = 100.0 * (1.0 - 10.0 / 10_000.0) * (1.0 - 5.0 / 10_000.0)
    assert fill.price == pytest.approx(expected, rel=1e-9)


def test_use_cpp_matches_python():
    """Step 4.4.3: When execution_core is built, use_cpp=True yields same fill as Python."""
    try:
        import execution_core  # noqa: F401
    except ImportError:
        pytest.skip("execution_core not built (need C++ compiler and pybind11)")

    for slippage_bps in (0.0, 5.0, 10.0):
        for commission_per_trade, commission_per_share in [(0.0, 0.0), (1.0, 0.0), (0.5, 0.01)]:
            py_handler = BacktestExecutionHandler(
                slippage_bps=slippage_bps,
                commission_per_trade=commission_per_trade,
                commission_per_share=commission_per_share,
                use_cpp=False,
            )
            cpp_handler = BacktestExecutionHandler(
                slippage_bps=slippage_bps,
                commission_per_trade=commission_per_trade,
                commission_per_share=commission_per_share,
                use_cpp=True,
            )
            ts = _utc(2025, 1, 15, 10, 0)
            for symbol, side, qty, ref in [("AAPL", SIDE_BUY, 10.0, 100.0), ("MSFT", SIDE_SELL, 5.0, 80.0)]:
                f_py = py_handler.execute(symbol, side, qty, ref, ts)
                f_cpp = cpp_handler.execute(symbol, side, qty, ref, ts)
                assert f_py.price == pytest.approx(f_cpp.price, rel=1e-9), (symbol, side)
                assert f_py.commission == pytest.approx(f_cpp.commission, rel=1e-9)
                assert f_py.slippage_bps == f_cpp.slippage_bps


def test_use_cpp_half_spread_matches_python():
    """Step 4.4.3: use_cpp with half_spread_bps matches Python."""
    try:
        import execution_core  # noqa: F401
    except ImportError:
        pytest.skip("execution_core not built")

    handler_py = BacktestExecutionHandler(half_spread_bps=10.0, slippage_bps=5.0, use_cpp=False)
    handler_cpp = BacktestExecutionHandler(half_spread_bps=10.0, slippage_bps=5.0, use_cpp=True)
    ts = _utc(2025, 1, 1)
    for side in (SIDE_BUY, SIDE_SELL):
        f_py = handler_py.execute("A", side, 1.0, 100.0, ts)
        f_cpp = handler_cpp.execute("A", side, 1.0, 100.0, ts)
        assert f_py.price == pytest.approx(f_cpp.price, rel=1e-9)
        assert f_py.commission == pytest.approx(f_cpp.commission, rel=1e-9)


def test_portfolio_integration():
    """Portfolio with BacktestExecutionHandler: fills have correct slippage and commission."""
    from src.backtest.events import SignalEvent
    from src.backtest.portfolio import Portfolio, generate_run_id

    handler = BacktestExecutionHandler(slippage_bps=5.0, commission_per_trade=1.0)
    port = Portfolio(generate_run_id(), 100_000.0, execution_handler=handler)
    port.start_run()

    signal = SignalEvent(
        timestamp=_utc(2025, 1, 15, 10, 0),
        direction="long_spread",
        symbol_a="AAPL",
        symbol_b="MSFT",
        hedge_ratio=1.0,
        size=10.0,
    )
    bar_data = {"AAPL": {"close": 100.0}, "MSFT": {"close": 80.0}}
    fills = port.process_signal(signal, bar_data)

    assert len(fills) == 2
    buy_fill = next(f for f in fills if f.side == SIDE_BUY)
    sell_fill = next(f for f in fills if f.side == SIDE_SELL)
    assert buy_fill.price == pytest.approx(100.0 * (1.0 + 5.0 / 10_000.0), rel=1e-9)
    assert sell_fill.price == pytest.approx(80.0 * (1.0 - 5.0 / 10_000.0), rel=1e-9)
    assert buy_fill.commission == 1.0
    assert sell_fill.commission == 1.0
    assert buy_fill.slippage_bps == 5.0
    assert sell_fill.slippage_bps == 5.0
