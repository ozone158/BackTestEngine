"""
Step 5.4.4 (Option B) — Recompute and persist backtest metrics for a run.

Reads backtest_equity and backtest_fills for the given run_id, computes
PerformanceMetrics (and optionally RiskAttribution if benchmark provided),
writes/updates backtest_metrics. Allows recomputing metrics without re-running backtest.

Usage:
  python -m scripts.compute_metrics --run-id <run_id> [--db URL]
  python -m scripts.compute_metrics --run-id <run_id> --benchmark returns.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.backtest.metrics_persistence import compute_and_persist_metrics
from src.data.storage.schema import get_engine


def _load_benchmark_returns(path: str) -> list:
    """Load benchmark return series from CSV (single column) or numpy file. Returns list of floats."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark path not found: {path}")
    if path.suffix.lower() == ".csv":
        import csv
        with open(path) as f:
            reader = csv.reader(f)
            next(reader, None)
            return [float(row[0]) for row in reader if row]
    if path.suffix.lower() in (".npy", ".npz"):
        import numpy as np
        data = np.load(path)
        arr = data if isinstance(data, np.ndarray) else data["returns"]
        return arr.ravel().tolist()
    raise ValueError(f"Unsupported benchmark format: {path.suffix}. Use .csv or .npy")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute and persist backtest metrics for a run (5.4.4).")
    ap.add_argument("--run-id", required=True, help="backtest run_id")
    ap.add_argument("--db", default=None, help="Database URL (default: from env or sqlite)")
    ap.add_argument("--benchmark", default=None, help="Path to benchmark returns (CSV or .npy) for alpha/beta")
    ap.add_argument("--periods-per-year", type=float, default=252.0, help="Periods per year for annualization")
    args = ap.parse_args()

    engine = get_engine(args.db)
    benchmark_returns = None
    if args.benchmark:
        benchmark_returns = _load_benchmark_returns(args.benchmark)

    try:
        row = compute_and_persist_metrics(
            engine,
            args.run_id,
            benchmark_returns=benchmark_returns,
            periods_per_year=args.periods_per_year,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Metrics persisted for run_id={args.run_id}")
    print(f"  total_return={row.get('total_return')} sharpe_annual={row.get('sharpe_annual')} max_drawdown={row.get('max_drawdown')}")
    print(f"  num_trades={row.get('num_trades')} win_rate={row.get('win_rate')}")
    if row.get("alpha") is not None:
        print(f"  alpha={row['alpha']} beta={row.get('beta')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
