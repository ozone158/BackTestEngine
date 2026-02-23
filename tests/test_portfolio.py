"""
Tests for Step 3.4 — Portfolio.

3.4.7: Unit test: one fill buy 10 @ 100 → positions[symbol]=10, cash reduced by 1000 + commission.
       Two legs (pair): assert both positions and cash consistent.
"""

from datetime import datetime, timezone

import pytest

from src.backtest.events import FillEvent, SIDE_BUY, SIDE_SELL, SignalEvent
from src.backtest.portfolio import Portfolio, generate_run_id
from src.data.storage.schema import create_backtest_tables, create_reference_tables, get_engine


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def test_one_fill_buy_updates_positions_and_cash():
    """One fill buy 10 @ 100 → positions[symbol]=10, cash reduced by 1000 + commission."""
    run_id = generate_run_id()
    commission = 1.0

    def mock_execute(symbol, side, quantity, ref_price, timestamp):
        return FillEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=ref_price,
            commission=commission,
            slippage_bps=0.0,
        )

    port = Portfolio(run_id, initial_capital=100_000.0, execution_handler=mock_execute)
    port.start_run()

    signal = SignalEvent(
        timestamp=_utc(2025, 1, 15, 10, 0),
        direction="long_spread",
        symbol_a="AAPL",
        symbol_b="MSFT",
        hedge_ratio=1.2,
        size=10.0,
    )
    bar_data = {"AAPL": {"close": 100.0}, "MSFT": {"close": 80.0}}
    fills = port.process_signal(signal, bar_data)

    assert len(fills) == 2  # two legs
    # Find AAPL buy fill
    aapl_fill = next(f for f in fills if f.symbol == "AAPL")
    assert aapl_fill.side == SIDE_BUY
    assert aapl_fill.quantity == 10.0
    assert aapl_fill.price == 100.0
    assert aapl_fill.commission == commission

    assert port.positions["AAPL"] == 10.0
    assert port.positions["MSFT"] == -12.0  # size * hedge_ratio = 10 * 1.2
    # Cash: started 100_000, spent 10*100 + 1 = 1001 on AAPL, received 12*80 - 1 = 959 on MSFT short
    # So cash = 100_000 - 1001 + 959 = 99_958
    assert port.cash == pytest.approx(100_000.0 - (10 * 100 + commission) + (12 * 80 - commission), rel=1e-9)
    assert len(port.equity_curve) == 1


def test_two_legs_pair_positions_and_cash_consistent():
    """Two legs (pair): both positions and cash consistent."""
    run_id = generate_run_id()

    def mock_execute(symbol, side, quantity, ref_price, timestamp):
        return FillEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=ref_price,
            commission=0.5,
            slippage_bps=0.0,
        )

    port = Portfolio(run_id, initial_capital=50_000.0, execution_handler=mock_execute)
    port.start_run()

    signal = SignalEvent(
        timestamp=_utc(2025, 1, 15, 10, 0),
        direction="long_spread",
        symbol_a="A",
        symbol_b="B",
        hedge_ratio=1.0,
        size=100.0,
    )
    bar_data = {"A": {"close": 50.0}, "B": {"close": 50.0}}
    fills = port.process_signal(signal, bar_data)

    assert len(fills) == 2
    buy_a = next(f for f in fills if f.symbol == "A" and f.side == SIDE_BUY)
    sell_b = next(f for f in fills if f.symbol == "B" and f.side == SIDE_SELL)
    assert buy_a.quantity == 100.0
    assert sell_b.quantity == 100.0
    assert port.positions["A"] == 100.0
    assert port.positions["B"] == -100.0
    # Cash: -100*50 - 0.5 + 100*50 - 0.5 = -1.0
    assert port.cash == pytest.approx(50_000.0 - 1.0, rel=1e-9)


def test_flat_closes_both_legs():
    """Flat signal closes both legs: sell A, buy B to flatten."""
    run_id = generate_run_id()

    def mock_execute(symbol, side, quantity, ref_price, timestamp):
        return FillEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=ref_price,
            commission=0.0,
            slippage_bps=0.0,
        )

    port = Portfolio(run_id, initial_capital=100_000.0, execution_handler=mock_execute)
    port.start_run()

    # Enter long spread
    bar = {"A": {"close": 100.0}, "B": {"close": 100.0}}
    port.process_signal(
        SignalEvent(_utc(2025, 1, 15, 9, 0), direction="long_spread", symbol_a="A", symbol_b="B", hedge_ratio=1.0, size=10.0),
        bar,
    )
    assert port.positions.get("A") == 10.0
    assert port.positions.get("B") == -10.0

    # Flat: close both
    port.process_signal(
        SignalEvent(_utc(2025, 1, 15, 10, 0), direction="flat", symbol_a="A", symbol_b="B", hedge_ratio=1.0, size=0.0),
        bar,
    )
    assert port.positions.get("A", 0) == 0
    assert port.positions.get("B", 0) == 0


def test_process_signal_without_execution_handler_returns_empty():
    """Without execution_handler, process_signal returns [] and does not update state."""
    port = Portfolio(generate_run_id(), 10_000.0)
    port.start_run()
    signal = SignalEvent(_utc(2025, 1, 15), direction="long_spread", symbol_a="A", symbol_b="B", hedge_ratio=1.0, size=5.0)
    fills = port.process_signal(signal, {"A": {"close": 10.0}, "B": {"close": 10.0}})
    assert fills == []
    assert port.positions == {}
    assert port.cash == 10_000.0


def test_start_run_inserts_backtest_runs_and_inits_state():
    """start_run inserts backtest_runs row (when engine given) and inits cash/positions (3.4.6)."""
    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    run_id = generate_run_id()
    port = Portfolio(
        run_id,
        initial_capital=100_000.0,
        strategy_name="ou_pairs",
        pair_id=None,
        start_ts=_utc(2025, 1, 1),
        end_ts=_utc(2025, 1, 31),
        config_json={"entry_k": 2.0},
    )
    port.start_run(engine=engine)

    assert port.cash == 100_000.0
    assert port.positions == {}

    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text("SELECT run_id, strategy_name, start_ts, end_ts FROM backtest_runs WHERE run_id = :rid"), {"rid": run_id}).fetchone()
    assert row is not None
    assert row[1] == "ou_pairs"


def test_flush_equity_curve_inserts_rows():
    """flush_equity_curve batch inserts into backtest_equity (3.4.5)."""
    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    run_id = generate_run_id()
    port = Portfolio(
        run_id,
        initial_capital=50_000.0,
        start_ts=_utc(2025, 1, 1),
        end_ts=_utc(2025, 1, 31),
    )
    port.start_run(engine=engine)
    port._equity_curve = [
        (_utc(2025, 1, 15, 10, 0), 51_000.0, 50_000.0, 1_000.0),
        (_utc(2025, 1, 15, 11, 0), 50_500.0, 49_500.0, 1_000.0),
    ]
    port.flush_equity_curve(engine)

    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ts, equity, cash, positions_value FROM backtest_equity WHERE run_id = :rid ORDER BY ts"), {"rid": run_id}).fetchall()
    assert len(rows) == 2
    assert rows[0][1] == 51_000.0
    assert rows[1][1] == 50_500.0
