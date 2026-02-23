"""
Step 5.5.2 — Dashboard: table of backtest runs with key metrics; select a run to view equity curve.

Streamlit app. Run: streamlit run scripts/dashboard.py [-- --db URL]

Usage:
  set PYTHONPATH=e:\\Project\\BackTestEngine
  streamlit run scripts/dashboard.py
  streamlit run scripts/dashboard.py -- --db sqlite:///data/backtest.db
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import streamlit as st

from scripts.report_backtest import load_equity_curve, load_recent_runs_with_metrics
from src.data.storage.schema import get_engine


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.2f}"
    if hasattr(v, "isoformat"):
        return v.isoformat()[:19] if v else ""
    return str(v)


def main() -> None:
    # Parse --db from argv (Streamlit passes script args after --)
    db_url = None
    for i, a in enumerate(sys.argv):
        if a == "--db" and i + 1 < len(sys.argv):
            db_url = sys.argv[i + 1]
            break
    engine = get_engine(db_url)

    st.set_page_config(page_title="Backtest runs", layout="wide")
    st.title("Backtest runs & equity curves")

    limit = st.sidebar.number_input("Max runs", min_value=5, max_value=100, value=20)
    runs: List[Dict[str, Any]] = load_recent_runs_with_metrics(engine, limit=limit)

    if not runs:
        st.info("No backtest runs found. Run a backtest first (e.g. `python -m scripts.run_backtest_e2e`).")
        return

    # Table of runs with key metrics
    display_cols = [
        "run_id", "strategy_name", "pair_id", "start_ts", "end_ts",
        "sharpe_annual", "sortino_annual", "max_drawdown", "total_return",
        "num_trades", "win_rate", "alpha", "beta",
    ]
    rows = [{k: _fmt(r.get(k)) for k in display_cols} for r in runs]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Select run for equity curve
    run_ids = [r["run_id"] for r in runs]
    labels = [f"{r['run_id'][:8]}... {r.get('strategy_name') or '?'} ({r.get('start_ts')})" for r in runs]
    selected_idx = st.selectbox(
        "Select a run to view equity curve",
        range(len(run_ids)),
        format_func=lambda i: labels[i],
    )
    run_id = run_ids[selected_idx]

    equity_data = load_equity_curve(engine, run_id)
    if not equity_data:
        st.warning(f"No equity curve stored for run {run_id[:8]}...")
        return

    ts_list = [t for t, _ in equity_data]
    equity_list = [e for _, e in equity_data]
    import pandas as pd
    df = pd.DataFrame({"equity": equity_list}, index=pd.DatetimeIndex(ts_list))
    st.line_chart(df, use_container_width=True)
    st.caption(f"Run {run_id[:8]}... — {len(equity_data)} points.")


if __name__ == "__main__":
    main()
