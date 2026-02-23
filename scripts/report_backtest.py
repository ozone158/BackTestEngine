"""
Step 5.5.1 — Report script: list recent backtest runs and their metrics.

Lists backtest_runs (e.g. last 10); for each run_id loads backtest_metrics and
optionally backtest_equity; prints or exports summary (Sharpe, Sortino, max DD,
total return, alpha, beta). Output: console, CSV, or HTML.

Usage:
  python -m scripts.report_backtest [--limit 10] [--output console|csv|html] [--db URL]
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.engine import Engine

from src.data.storage.schema import backtest_equity, backtest_metrics, backtest_runs, get_engine


def load_recent_runs_with_metrics(
    engine: Engine,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Load recent backtest_runs (by created_at desc) with joined backtest_metrics."""
    with engine.connect() as conn:
        runs = conn.execute(
            select(
                backtest_runs.c.run_id,
                backtest_runs.c.strategy_name,
                backtest_runs.c.pair_id,
                backtest_runs.c.start_ts,
                backtest_runs.c.end_ts,
                backtest_runs.c.created_at,
                backtest_metrics.c.sharpe_annual,
                backtest_metrics.c.sortino_annual,
                backtest_metrics.c.calmar,
                backtest_metrics.c.max_drawdown,
                backtest_metrics.c.total_return,
                backtest_metrics.c.num_trades,
                backtest_metrics.c.win_rate,
                backtest_metrics.c.alpha,
                backtest_metrics.c.beta,
                backtest_metrics.c.computed_at,
            )
            .select_from(backtest_runs.outerjoin(backtest_metrics, backtest_runs.c.run_id == backtest_metrics.c.run_id))
            .order_by(backtest_runs.c.created_at.desc().nullslast())
            .limit(limit)
        ).fetchall()

    return [
        {
            "run_id": r[0],
            "strategy_name": r[1],
            "pair_id": r[2],
            "start_ts": r[3],
            "end_ts": r[4],
            "created_at": r[5],
            "sharpe_annual": r[6],
            "sortino_annual": r[7],
            "calmar": r[8],
            "max_drawdown": r[9],
            "total_return": r[10],
            "num_trades": r[11],
            "win_rate": r[12],
            "alpha": r[13],
            "beta": r[14],
            "computed_at": r[15],
        }
        for r in runs
    ]


def load_equity_curve(engine: Engine, run_id: str) -> List[Tuple[datetime, float]]:
    """Load (ts, equity) for run_id from backtest_equity, ordered by ts. For dashboard charting."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(backtest_equity.c.ts, backtest_equity.c.equity)
            .where(backtest_equity.c.run_id == run_id)
            .order_by(backtest_equity.c.ts)
        ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.2f}"
    if isinstance(v, datetime):
        return v.isoformat()[:19] if v else ""
    return str(v)


def output_console(rows: List[Dict[str, Any]]) -> None:
    """Print table to console."""
    if not rows:
        print("No backtest runs found.")
        return
    cols = ["run_id", "strategy_name", "start_ts", "end_ts", "sharpe_annual", "sortino_annual", "max_drawdown", "total_return", "num_trades", "win_rate", "alpha", "beta"]
    widths = {c: max(len(c), 10) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(r.get(c))))
    h = "  ".join(c[: widths[c]].ljust(widths[c]) for c in cols)
    print(h)
    print("-" * len(h))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def output_csv(rows: List[Dict[str, Any]], stream: Optional[Any] = None) -> None:
    """Write CSV to stream (default stdout)."""
    if stream is None:
        stream = sys.stdout
    if not rows:
        return
    cols = list(rows[0].keys())
    w = csv.DictWriter(stream, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(v) if isinstance(v, (float, datetime)) else v for k, v in r.items()})


def output_html(rows: List[Dict[str, Any]], stream: Optional[Any] = None) -> None:
    """Write simple HTML table to stream (default stdout)."""
    if stream is None:
        stream = sys.stdout
    if not rows:
        stream.write("<p>No backtest runs found.</p>\n")
        return
    cols = ["run_id", "strategy_name", "pair_id", "start_ts", "end_ts", "sharpe_annual", "sortino_annual", "max_drawdown", "total_return", "num_trades", "win_rate", "alpha", "beta"]
    stream.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Backtest runs</title></head><body>\n")
    stream.write("<table border='1' cellpadding='4'>\n<thead><tr>")
    for c in cols:
        stream.write(f"<th>{c}</th>")
    stream.write("</tr></thead>\n<tbody>\n")
    for r in rows:
        stream.write("<tr>")
        for c in cols:
            stream.write(f"<td>{_fmt(r.get(c))}</td>")
        stream.write("</tr>\n")
    stream.write("</tbody></table>\n</body></html>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Report recent backtest runs and metrics (5.5.1).")
    ap.add_argument("--limit", type=int, default=10, help="Max number of runs to list (default 10)")
    ap.add_argument("--output", choices=("console", "csv", "html"), default="console", help="Output format (default console)")
    ap.add_argument("--db", default=None, help="Database URL (default: from env or sqlite)")
    ap.add_argument("--out-file", default=None, help="Write output to file (default stdout for csv/html)")
    args = ap.parse_args()

    engine = get_engine(args.db)
    rows = load_recent_runs_with_metrics(engine, limit=args.limit)

    out_stream = None
    if args.out_file:
        out_stream = open(args.out_file, "w", encoding="utf-8")

    try:
        if args.output == "console":
            output_console(rows)
        elif args.output == "csv":
            output_csv(rows, stream=out_stream or sys.stdout)
        else:
            output_html(rows, stream=out_stream or sys.stdout)
    finally:
        if out_stream:
            out_stream.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
