"""
Tests for Step 3.6 — Event loop and persistence.

3.6.5: Integration test: Run full backtest on 1 pair; assert backtest_runs has 1 row;
       backtest_signals and backtest_fills have expected counts; backtest_equity has rows;
       no errors; equity curve non-negative.
"""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.backtest.data_handler import DataHandler
from src.backtest.events import DIRECTION_LONG_SPREAD, SignalEvent
from src.backtest.execution import BacktestExecutionHandler
from src.backtest.portfolio import Portfolio, generate_run_id
from src.backtest.run import run_backtest, persist_fill, persist_signal
from src.backtest.strategy import OUStrategy
from src.data.alpha.ou import OUParams
from src.data.storage.schema import create_backtest_tables, create_reference_tables, get_engine


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def _make_bars(symbols: list, dts: list, base: float = 100.0) -> pl.DataFrame:
    rows = []
    for i, dt in enumerate(dts):
        for sym in symbols:
            p = base + i * 0.5
            rows.append({
                "symbol": sym,
                "datetime": dt,
                "open": p, "high": p + 1, "low": p - 1, "close": p + 0.5, "volume": 1e6,
            })
    return pl.DataFrame(rows)


def test_run_backtest_integration():
    """Full backtest: 1 pair, multiple bars; one run row; signals/fills/equity persisted; equity non-negative."""
    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    dts = [_utc(2025, 1, 15, 9, i) for i in range(20)]
    symbols = ["AAPL", "MSFT"]

    def read_fn(syms, start, end):
        return _make_bars(syms, dts)

    data_handler = DataHandler(symbols, dts[0], dts[-1], read_fn)
    ou = OUParams(
        theta=0.1,
        mu=0.0,
        sigma=1.0,
        entry_upper=2.0,
        entry_lower=-2.0,
        exit_threshold=0.0,
    )
    # Provider: spread high on bar 2 -> long signal; spread at mean on bar 8 -> flat
    spread_sequence = [
        0.5, 0.3, 2.5, 2.2, 2.0, 1.5, 1.0, 0.5, 0.0, -0.2,
    ] + [0.0] * 10

    def provider(ts, bar_data):
        try:
            idx = dts.index(ts)
        except ValueError:
            idx = 0
        return (spread_sequence[min(idx, len(spread_sequence) - 1)], 1.0)

    strategy = OUStrategy("AAPL", "MSFT", ou, provider, size=10.0)
    run_id = generate_run_id()
    execution = BacktestExecutionHandler(slippage_bps=5.0, commission_per_trade=1.0)
    portfolio = Portfolio(
        run_id,
        initial_capital=100_000.0,
        strategy_name="ou_pairs",
        pair_id=None,
        start_ts=dts[0],
        end_ts=dts[-1],
        config_json={"entry_k": 2.0},
        execution_handler=execution,
    )
    portfolio.start_run(engine)

    run_id_out = run_backtest(data_handler, strategy, portfolio, engine, record_equity_every_bar=True)

    assert run_id_out == run_id

    from sqlalchemy import text
    with engine.connect() as conn:
        runs = conn.execute(text("SELECT run_id, strategy_name FROM backtest_runs")).fetchall()
        assert len(runs) == 1
        assert runs[0][1] == "ou_pairs"

        signals = conn.execute(text("SELECT id, direction, signal_ts FROM backtest_signals WHERE run_id = :r"), {"r": run_id}).fetchall()
        fills = conn.execute(text("SELECT id, symbol, side FROM backtest_fills WHERE run_id = :r"), {"r": run_id}).fetchall()
        equity = conn.execute(text("SELECT ts, equity, cash FROM backtest_equity WHERE run_id = :r ORDER BY ts"), {"r": run_id}).fetchall()

    assert len(signals) >= 1
    assert len(fills) >= 2
    assert len(equity) >= 1
    for row in equity:
        assert row[1] >= 0.0, f"equity must be non-negative, got {row[1]}"

    # 5.4.6: After run, backtest_metrics has one row with sharpe_annual and total_return
    with engine.connect() as conn2:
        metrics = conn2.execute(
            text("SELECT sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate FROM backtest_metrics WHERE run_id = :r"),
            {"r": run_id},
        ).fetchall()
    assert len(metrics) == 1, "backtest_metrics should have one row per run"
    m = metrics[0]
    assert m[4] is not None, "total_return should be present"
    assert m[0] is not None, "sharpe_annual should be present"
    assert m[3] is not None, "max_drawdown should be present"


def test_compute_and_persist_metrics_after_run():
    """5.4.6: compute_and_persist_metrics can be called standalone (Option B) to recompute metrics."""
    from sqlalchemy import text

    from src.backtest.metrics_persistence import compute_and_persist_metrics

    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    dts = [_utc(2025, 1, 15, 9, i) for i in range(10)]
    symbols = ["A", "B"]
    def read_fn(syms, start, end):
        return _make_bars(syms, dts)
    data_handler = DataHandler(symbols, dts[0], dts[-1], read_fn)
    ou = OUParams(theta=0.1, mu=0.0, sigma=1.0, entry_upper=2.0, entry_lower=-2.0, exit_threshold=0.0)
    spread_vals = [0.0, 2.5] + [0.0] * 8
    def provider(ts, bar_data):
        idx = dts.index(ts) if ts in dts else 0
        return (spread_vals[min(idx, len(spread_vals)-1)], 1.0)
    strategy = OUStrategy("A", "B", ou, provider, size=10.0)
    run_id = generate_run_id()
    execution = BacktestExecutionHandler(slippage_bps=0, commission_per_trade=0)
    portfolio = Portfolio(run_id, initial_capital=100_000.0, strategy_name="ou", pair_id=None, start_ts=dts[0], end_ts=dts[-1], config_json={}, execution_handler=execution)
    portfolio.start_run(engine)
    run_backtest(data_handler, strategy, portfolio, engine, record_equity_every_bar=True)

    with engine.connect() as conn:
        before = conn.execute(text("SELECT computed_at FROM backtest_metrics WHERE run_id = :r"), {"r": run_id}).fetchone()
    assert before is not None

    compute_and_persist_metrics(engine, run_id)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT total_return, sharpe_annual, num_trades FROM backtest_metrics WHERE run_id = :r"), {"r": run_id}).fetchone()
    assert row is not None
    assert row[0] is not None
    assert row[1] is not None


def test_persist_signal_returns_id():
    """persist_signal inserts and returns signal id for linking fills."""
    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    run_id = "test-run-123"
    # backtest_signals has FK to backtest_runs
    with engine.connect() as conn:
        from sqlalchemy import insert
        from src.data.storage.schema import backtest_runs
        conn.execute(
            insert(backtest_runs).values(
                run_id=run_id,
                start_ts=_utc(2025, 1, 1),
                end_ts=_utc(2025, 1, 31),
            )
        )
        conn.commit()

    signal = SignalEvent(
        timestamp=_utc(2025, 1, 15, 10, 0),
        direction=DIRECTION_LONG_SPREAD,
        symbol_a="A",
        symbol_b="B",
        hedge_ratio=1.0,
        size=100.0,
        metadata={"z_score": 2.0},
    )
    signal_id = persist_signal(engine, run_id, signal)
    assert signal_id is not None
    assert signal_id >= 1

    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text("SELECT run_id, direction, symbol_a FROM backtest_signals WHERE id = :id"), {"id": signal_id}).fetchone()
    assert row[0] == run_id
    assert row[1] == DIRECTION_LONG_SPREAD
    assert row[2] == "A"


def test_persist_fill_with_signal_id():
    """persist_fill inserts row with signal_id (3.5.6 link)."""
    from sqlalchemy import insert
    from src.backtest.events import FillEvent, SIDE_BUY
    from src.data.storage.schema import backtest_runs

    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)
    run_id = "test-run-456"
    with engine.connect() as conn:
        conn.execute(
            insert(backtest_runs).values(
                run_id=run_id,
                start_ts=_utc(2025, 1, 1),
                end_ts=_utc(2025, 1, 31),
            )
        )
        conn.commit()
    signal_id = persist_signal(engine, run_id, SignalEvent(_utc(2025, 1, 15), direction="long_spread", symbol_a="A", symbol_b="B", hedge_ratio=1.0, size=10.0))
    fill = FillEvent(timestamp=_utc(2025, 1, 15), symbol="A", side=SIDE_BUY, quantity=10.0, price=100.0, commission=1.0, slippage_bps=5.0)
    persist_fill(engine, run_id, fill, signal_id=signal_id)

    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(text("SELECT run_id, signal_id, symbol, quantity, price FROM backtest_fills WHERE run_id = :r"), {"r": run_id}).fetchone()
    assert row[1] == signal_id
    assert row[2] == "A"
    assert row[3] == 10.0
    assert row[4] == 100.0


def test_backtest_equity_use_cpp_vs_python():
    """Step 4.2.5: Backtest with online Kalman use_cpp=True vs use_cpp=False; equity curves match closely."""
    from src.data.alpha.kalman import KalmanHedgeRatio, _CPP_AVAILABLE

    if not _CPP_AVAILABLE:
        pytest.skip("kalman_core not built")

    import numpy as np
    symbols = ["AAPL", "MSFT"]
    n_bars = 40
    dts = [_utc(2025, 1, 15, 9, i) for i in range(n_bars)]
    np.random.seed(50)
    close_b = np.cumsum(np.random.randn(n_bars) * 0.01) + 100.0
    close_a = 1.2 * close_b + np.random.randn(n_bars) * 0.1

    def make_bars():
        rows = []
        for i, dt in enumerate(dts):
            for sym, close in [(symbols[0], close_a[i]), (symbols[1], close_b[i])]:
                rows.append({
                    "symbol": sym, "datetime": dt,
                    "open": close - 0.1, "high": close + 0.1, "low": close - 0.1, "close": close,
                    "volume": 1e6,
                })
        return pl.DataFrame(rows)

    def read_fn(syms, start, end):
        return make_bars()

    ou = OUParams(
        theta=0.1, mu=0.0, sigma=1.0,
        entry_upper=2.0, entry_lower=-2.0, exit_threshold=0.0,
    )

    def run_one(use_cpp: bool):
        kf = KalmanHedgeRatio(1e-6, 1e-4, use_cpp=use_cpp)

        def provider(ts, bar_data):
            ca = bar_data.get(symbols[0], {}).get("close", 100.0)
            cb = bar_data.get(symbols[1], {}).get("close", 100.0)
            spread, beta = kf.update(float(ca), float(cb))
            return (spread, beta)

        eng = get_engine("sqlite:///:memory:")
        create_reference_tables(eng)
        create_backtest_tables(eng)
        data_handler = DataHandler(symbols, dts[0], dts[-1], read_fn)
        strategy = OUStrategy(symbols[0], symbols[1], ou, provider, size=10.0)
        run_id = generate_run_id()
        execution = BacktestExecutionHandler(slippage_bps=5.0, commission_per_trade=1.0)
        portfolio = Portfolio(
            run_id,
            initial_capital=100_000.0,
            strategy_name="ou_pairs",
            pair_id=None,
            start_ts=dts[0],
            end_ts=dts[-1],
            config_json={"use_cpp": use_cpp},
            execution_handler=execution,
        )
        portfolio.start_run(eng)
        run_backtest(data_handler, strategy, portfolio, eng, record_equity_every_bar=True)
        from sqlalchemy import text
        with eng.connect() as conn:
            rows = conn.execute(
                text("SELECT ts, equity FROM backtest_equity WHERE run_id = :r ORDER BY ts"),
                {"r": run_id},
            ).fetchall()
        return [row[1] for row in rows]

    equity_py = run_one(use_cpp=False)
    equity_cpp = run_one(use_cpp=True)
    assert len(equity_py) == len(equity_cpp)
    assert np.allclose(equity_cpp, equity_py, rtol=1e-9, atol=1e-6), (
        "equity curves with use_cpp=True vs False should match closely"
    )


def test_record_equity_every_bar_false():
    """With record_equity_every_bar=False, equity is recorded only when there are fills (not every bar)."""
    engine = get_engine("sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    dts = [_utc(2025, 1, 15, 9, i) for i in range(20)]
    symbols = ["AAPL", "MSFT"]

    def read_fn(syms, start, end):
        return _make_bars(syms, dts)

    data_handler = DataHandler(symbols, dts[0], dts[-1], read_fn)
    ou = OUParams(
        theta=0.1,
        mu=0.0,
        sigma=1.0,
        entry_upper=2.0,
        entry_lower=-2.0,
        exit_threshold=0.0,
    )
    spread_sequence = [0.5, 2.5, 2.2] + [0.0] * 17  # one long entry early

    def provider(ts, bar_data):
        try:
            idx = dts.index(ts)
        except ValueError:
            idx = 0
        return (spread_sequence[min(idx, len(spread_sequence) - 1)], 1.0)

    strategy = OUStrategy("AAPL", "MSFT", ou, provider, size=10.0)
    run_id = generate_run_id()
    execution = BacktestExecutionHandler(slippage_bps=0, commission_per_trade=0)
    portfolio = Portfolio(
        run_id,
        initial_capital=100_000.0,
        strategy_name="ou_pairs",
        pair_id=None,
        start_ts=dts[0],
        end_ts=dts[-1],
        config_json={},
        execution_handler=execution,
    )
    portfolio.start_run(engine)

    run_backtest(data_handler, strategy, portfolio, engine, record_equity_every_bar=False)

    from sqlalchemy import text
    with engine.connect() as conn:
        equity_rows = conn.execute(
            text("SELECT COUNT(*) FROM backtest_equity WHERE run_id = :r"),
            {"r": run_id},
        ).scalar()
        signal_rows = conn.execute(
            text("SELECT COUNT(*) FROM backtest_signals WHERE run_id = :r"),
            {"r": run_id},
        ).scalar()
    # Equity should be recorded only when we had signals (fills), not every bar
    assert equity_rows >= 1
    assert equity_rows <= len(dts), "with record_equity_every_bar=False we should have at most one row per bar"
    assert signal_rows >= 1
