"""
Step 4.3 — Benchmark: Python Kalman vs C++ Kalman in same backtest.

Same event loop, Strategy, Portfolio, Execution; only the Kalman implementation
(use_cpp=False vs use_cpp=True) differs. Measures wall-clock over multiple runs
and reports speedup ratio (time_Python / time_C++).

Usage:
  python -m scripts.benchmark_kalman
  python -m scripts.benchmark_kalman --bars 252 --iterations 30
  python -m scripts.benchmark_kalman --fail-on-regression   # exit 1 if C++ slower
  python -m scripts.benchmark_kalman --mode kalman_only   # tight Kalman loop only (per-step)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.backtest import (
    BacktestExecutionHandler,
    DataHandler,
    OUStrategy,
    Portfolio,
    generate_run_id,
    run_backtest,
)
from src.data.alpha.kalman import KalmanHedgeRatio, _CPP_AVAILABLE
from src.data.alpha.ou import OUParams
from src.data.storage.schema import create_backtest_tables, create_reference_tables, get_engine


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def _make_bars(symbols: list[str], n_bars: int, seed: int = 42) -> pl.DataFrame:
    """In-memory bars for one pair: random walk close_a, close_b (Kalman-heavy)."""
    np.random.seed(seed)
    start = _utc(2025, 1, 1, 9, 30)
    dts = [start + timedelta(days=i) for i in range(n_bars)]
    close_b = np.cumsum(np.random.randn(n_bars) * 0.01) + 100.0
    close_a = 1.2 * close_b + np.random.randn(n_bars) * 0.08
    rows = []
    for i, dt in enumerate(dts):
        for sym, close in [(symbols[0], close_a[i]), (symbols[1], close_b[i])]:
            rows.append({
                "symbol": sym,
                "datetime": dt,
                "open": close - 0.1,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1e6,
            })
    return pl.DataFrame(rows)


def run_one_backtest(
    symbols: list[str],
    dts: list,
    bars_df: pl.DataFrame,
    use_cpp: bool,
    engine,
) -> None:
    """Single backtest run (Strategy uses online Kalman with use_cpp)."""

    def read_fn(syms, start, end):
        return bars_df

    kf = KalmanHedgeRatio(1e-6, 1e-4, use_cpp=use_cpp)

    def provider(ts, bar_data):
        ca = bar_data.get(symbols[0], {}).get("close", 100.0)
        cb = bar_data.get(symbols[1], {}).get("close", 100.0)
        spread, beta = kf.update(float(ca), float(cb))
        return (spread, beta)

    ou = OUParams(
        theta=0.1,
        mu=0.0,
        sigma=1.0,
        entry_upper=2.0,
        entry_lower=-2.0,
        exit_threshold=0.0,
    )
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
    portfolio.start_run(engine)
    run_backtest(data_handler, strategy, portfolio, engine, record_equity_every_bar=True)


def benchmark_backtest(n_bars: int, n_iterations: int) -> tuple[float, float | None]:
    """
    Run backtest n_iterations times with Python Kalman, then with C++ Kalman.
    Returns (total_seconds_python, total_seconds_cpp or None if C++ not available).
    """
    symbols = ["AAPL", "MSFT"]
    bars_df = _make_bars(symbols, n_bars)
    dts = bars_df["datetime"].unique().sort().to_list()

    # Python
    engine_py = get_engine("sqlite:///:memory:")
    create_reference_tables(engine_py)
    create_backtest_tables(engine_py)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        run_one_backtest(symbols, dts, bars_df, use_cpp=False, engine=engine_py)
    time_py = time.perf_counter() - t0

    # C++
    if not _CPP_AVAILABLE:
        return time_py, None
    engine_cpp = get_engine("sqlite:///:memory:")
    create_reference_tables(engine_cpp)
    create_backtest_tables(engine_cpp)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        run_one_backtest(symbols, dts, bars_df, use_cpp=True, engine=engine_cpp)
    time_cpp = time.perf_counter() - t0
    return time_py, time_cpp


def benchmark_kalman_only(n_steps: int, n_iterations: int) -> tuple[float, float | None]:
    """
    Tight loop: only Kalman update(price_a, price_b) n_steps per iteration.
    Measures per-step cost (4.3.2). Returns (total_seconds_python, total_seconds_cpp or None).
    """
    np.random.seed(123)
    price_a = np.cumsum(np.random.randn(n_steps) * 0.01) + 100.0
    price_b = np.cumsum(np.random.randn(n_steps) * 0.01) + 100.0

    # Python
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        kf = KalmanHedgeRatio(1e-6, 1e-4, use_cpp=False)
        for i in range(n_steps):
            kf.update(float(price_a[i]), float(price_b[i]))
    time_py = time.perf_counter() - t0

    if not _CPP_AVAILABLE:
        return time_py, None

    # C++
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        kf = KalmanHedgeRatio(1e-6, 1e-4, use_cpp=True)
        for i in range(n_steps):
            kf.update(float(price_a[i]), float(price_b[i]))
    time_cpp = time.perf_counter() - t0
    return time_py, time_cpp


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Step 4.3: Benchmark Python vs C++ Kalman (same backtest or Kalman-only loop)."
    )
    ap.add_argument(
        "--bars",
        type=int,
        default=252,
        help="Number of bars (e.g. 252 = 1 year daily). Default 252.",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of runs per implementation. Default 20.",
    )
    ap.add_argument(
        "--mode",
        choices=("backtest", "kalman_only"),
        default="backtest",
        help="backtest = full backtest with Strategy/Portfolio; kalman_only = tight Kalman loop. Default backtest.",
    )
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if C++ is not faster than Python (ratio < 1.0).",
    )
    args = ap.parse_args()

    if args.mode == "kalman_only":
        n_steps = args.bars
        time_py, time_cpp = benchmark_kalman_only(n_steps, args.iterations)
        total_steps = n_steps * args.iterations
        print(f"Mode: kalman_only (per-step cost)")
        print(f"  Steps per iteration: {n_steps}, iterations: {args.iterations} -> {total_steps} total updates")
        print(f"  Python total: {time_py:.4f} s  -> {time_py / total_steps * 1e6:.2f} us/step")
        if time_cpp is not None:
            print(f"  C++ total:    {time_cpp:.4f} s  -> {time_cpp / total_steps * 1e6:.2f} us/step")
            ratio = time_py / time_cpp
            print(f"  Ratio (Python/C++): {ratio:.2f}x")
            if args.fail_on_regression and ratio < 1.0:
                print("  FAIL: C++ is not faster (regression).")
                return 1
        else:
            print("  C++: not available (kalman_core not built).")
    else:
        time_py, time_cpp = benchmark_backtest(args.bars, args.iterations)
        print(f"Mode: backtest (one pair, online Kalman)")
        print(f"  Bars: {args.bars}, iterations: {args.iterations}")
        print(f"  Python total: {time_py:.4f} s  ({time_py / args.iterations * 1000:.1f} ms/run)")
        if time_cpp is not None:
            print(f"  C++ total:    {time_cpp:.4f} s  ({time_cpp / args.iterations * 1000:.1f} ms/run)")
            ratio = time_py / time_cpp
            print(f"  Ratio (time_Python / time_C++): {ratio:.2f}x  (4.3.3 speedup)")
            if ratio < 1.0:
                print("  Note: C++ slower than Python; I/O or Portfolio may dominate for this size.")
            if args.fail_on_regression and ratio < 1.0:
                print("  FAIL: regression (C++ not faster).")
                return 1
        else:
            print("  C++: not available (kalman_core not built). Run 'pip install -e .' with a C++ compiler.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
