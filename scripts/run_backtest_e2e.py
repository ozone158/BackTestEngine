"""
Phase 3 E2E: Run a short backtest with DataHandler, OUStrategy, Portfolio, ExecutionHandler,
and persist to DB. Uses in-memory bars by default; optional --root to read from Parquet.

Usage (in-memory, no data required):
  python -m scripts.run_backtest_e2e

Usage (from Parquet bars):
  python -m scripts.run_backtest_e2e --root data --source alpha_vantage --symbols AAPL MSFT --days 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.backtest import (
    BacktestExecutionHandler,
    DataHandler,
    OUStrategy,
    Portfolio,
    generate_run_id,
    run_backtest,
)
from src.data.alpha.ou import OUParams
from src.data.storage.schema import create_backtest_tables, create_reference_tables, get_engine


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 backtest E2E")
    ap.add_argument("--root", default=None, help="Parquet root (if set, read bars from Parquet)")
    ap.add_argument("--source", default="alpha_vantage", help="Partition source when using --root")
    ap.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"], help="Symbols (pair)")
    ap.add_argument("--days", type=int, default=30, help="Days of data when using --root")
    ap.add_argument("--db", default=None, help="Database URL (default: sqlite in-memory)")
    args = ap.parse_args()

    if args.root:
        from src.data.storage.parquet_bars import read_bars
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        def read_fn(syms, s, e):
            return read_bars(args.root, syms, s, e, source=args.source)
        start_ts, end_ts = start, end
    else:
        import polars as pl
        dts = [_utc(2025, 1, 15, 9, i) for i in range(60)]
        def _bars(syms, s, e):
            rows = []
            for i, dt in enumerate(dts):
                for sym in syms:
                    p = 100.0 + i * 0.5
                    rows.append({
                        "symbol": sym, "datetime": dt,
                        "open": p, "high": p + 1, "low": p - 1, "close": p + 0.5, "volume": 1e6,
                    })
            return pl.DataFrame(rows)
        read_fn = _bars
        start_ts, end_ts = dts[0], dts[-1]

    symbols = args.symbols[:2] if len(args.symbols) >= 2 else ["AAPL", "MSFT"]
    engine = get_engine(args.db or "sqlite:///:memory:")
    create_reference_tables(engine)
    create_backtest_tables(engine)

    data_handler = DataHandler(symbols, start_ts, end_ts, read_fn)
    ou = OUParams(theta=0.1, mu=0.0, sigma=1.0, entry_upper=2.0, entry_lower=-2.0, exit_threshold=0.0)
    spread_vals = [0.5] * 5 + [2.5] * 10 + [0.0] * 15 + [-2.5] * 5 + [0.0] * 25

    def provider(ts, bar_data):
        idx = min(len(spread_vals) - 1, getattr(data_handler, "_current_index", 0))
        if 0 <= data_handler._current_index < len(spread_vals):
            idx = data_handler._current_index
        return (spread_vals[idx], 1.0)

    strategy = OUStrategy(symbols[0], symbols[1], ou, provider, size=10.0)
    run_id = generate_run_id()
    execution = BacktestExecutionHandler(slippage_bps=5.0, commission_per_trade=1.0)
    portfolio = Portfolio(
        run_id,
        initial_capital=100_000.0,
        strategy_name="ou_pairs",
        pair_id=None,
        start_ts=start_ts,
        end_ts=end_ts,
        config_json={"entry_k": 2.0},
        execution_handler=execution,
    )
    portfolio.start_run(engine)

    run_backtest(data_handler, strategy, portfolio, engine, record_equity_every_bar=True)

    from sqlalchemy import text
    with engine.connect() as conn:
        r = conn.execute(text("SELECT COUNT(*) FROM backtest_runs WHERE run_id = :r"), {"r": run_id}).scalar()
        s = conn.execute(text("SELECT COUNT(*) FROM backtest_signals WHERE run_id = :r"), {"r": run_id}).scalar()
        f = conn.execute(text("SELECT COUNT(*) FROM backtest_fills WHERE run_id = :r"), {"r": run_id}).scalar()
        e = conn.execute(text("SELECT COUNT(*) FROM backtest_equity WHERE run_id = :r"), {"r": run_id}).scalar()
    print(f"Phase 3 E2E OK: run_id={run_id[:8]}... backtest_runs={r} signals={s} fills={f} equity_rows={e}")


if __name__ == "__main__":
    main()
