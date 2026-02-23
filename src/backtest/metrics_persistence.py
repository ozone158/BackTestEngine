"""
Step 5.4 — Persistence and wiring for backtest metrics.

After a backtest run: load equity and fills from DB, compute PerformanceMetrics
and optionally RiskAttribution (with benchmark); build metrics row and
insert/replace into backtest_metrics (5.4.1–5.4.3).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import delete, select
from sqlalchemy.engine import Engine

from src.backtest.performance import PerformanceMetrics
from src.backtest.risk_attribution import RiskAttribution
from src.data.storage.schema import backtest_equity, backtest_fills, backtest_metrics


def load_equity_for_run(engine: Engine, run_id: str) -> List[float]:
    """Load equity curve for run_id from backtest_equity (ordered by ts). Returns list of equity values."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(backtest_equity.c.equity)
            .where(backtest_equity.c.run_id == run_id)
            .order_by(backtest_equity.c.ts)
        ).fetchall()
    return [float(r[0]) for r in rows]


def load_fills_for_run(engine: Engine, run_id: str) -> List[Dict[str, Any]]:
    """Load fills for run_id from backtest_fills. Returns list of dicts with signal_id, symbol, side, quantity, price, commission."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                backtest_fills.c.signal_id,
                backtest_fills.c.symbol,
                backtest_fills.c.side,
                backtest_fills.c.quantity,
                backtest_fills.c.price,
                backtest_fills.c.commission,
            )
            .where(backtest_fills.c.run_id == run_id)
            .order_by(backtest_fills.c.fill_ts)
        ).fetchall()
    return [
        {
            "signal_id": r[0],
            "symbol": r[1],
            "side": r[2],
            "quantity": r[3],
            "price": r[4],
            "commission": r[5] if r[5] is not None else 0.0,
        }
        for r in rows
    ]


def compute_metrics_for_run(
    engine: Engine,
    run_id: str,
    benchmark_returns: Optional[Union[list, Any]] = None,
    periods_per_year: float = 252.0,
) -> Dict[str, Any]:
    """
    5.4.1–5.4.2: Load equity and fills for run_id; compute PerformanceMetrics and
    optionally RiskAttribution (alpha, beta). Returns full metrics dict suitable
    for backtest_metrics row and metrics_json.
    """
    equity = load_equity_for_run(engine, run_id)
    fills = load_fills_for_run(engine, run_id)

    if not equity:
        row = {
            "sharpe_annual": None,
            "sortino_annual": None,
            "calmar": None,
            "max_drawdown": None,
            "total_return": None,
            "num_trades": 0,
            "win_rate": None,
            "alpha": None,
            "beta": None,
        }
        row["metrics_json"] = json.dumps(row.copy())
        return row

    perf = PerformanceMetrics.compute(
        equity_series=equity,
        fill_events=fills if fills else None,
        periods_per_year=periods_per_year,
    )

    alpha_val = None
    beta_val = None
    if benchmark_returns is not None and len(equity) >= 2:
        import numpy as np
        returns = (np.array(equity[1:]) - np.array(equity[:-1])) / np.where(
            np.array(equity[:-1]) != 0, np.array(equity[:-1]), np.nan
        )
        returns = np.where(np.isfinite(returns), returns, 0.0)
        bench = np.asarray(benchmark_returns, dtype=float).ravel()
        n = min(len(returns), len(bench))
        if n >= 2:
            attr = RiskAttribution.decompose(
                returns[:n],
                factor_returns=bench[:n],
                periods_per_year=periods_per_year,
            )
            alpha_val = attr["alpha"]
            b = attr["beta"]
            if hasattr(b, "__len__") and not isinstance(b, (int, float)):
                beta_val = float(b[0]) if len(b) > 0 else None
            else:
                beta_val = float(b) if b is not None else None

    row = {
        "sharpe_annual": perf["sharpe_annual"],
        "sortino_annual": perf["sortino_annual"],
        "calmar": perf["calmar"],
        "max_drawdown": perf["max_drawdown"],
        "total_return": perf["total_return"],
        "num_trades": perf["num_trades"],
        "win_rate": perf["win_rate"],
        "alpha": alpha_val,
        "beta": beta_val,
    }
    row["metrics_json"] = json.dumps({k: v for k, v in row.items() if k != "metrics_json"})
    return row


def persist_metrics(engine: Engine, run_id: str, metrics_row: Dict[str, Any]) -> None:
    """
    5.4.3: Insert or replace one row in backtest_metrics (policy: replace on conflict).
    metrics_row must contain: run_id, sharpe_annual, sortino_annual, calmar, max_drawdown,
    total_return, num_trades, win_rate, alpha, beta, metrics_json, computed_at.
    """
    from sqlalchemy import insert

    computed_at = datetime.now(timezone.utc)
    payload = {
        "run_id": run_id,
        "sharpe_annual": metrics_row.get("sharpe_annual"),
        "sortino_annual": metrics_row.get("sortino_annual"),
        "calmar": metrics_row.get("calmar"),
        "max_drawdown": metrics_row.get("max_drawdown"),
        "total_return": metrics_row.get("total_return"),
        "num_trades": metrics_row.get("num_trades"),
        "win_rate": metrics_row.get("win_rate"),
        "alpha": metrics_row.get("alpha"),
        "beta": metrics_row.get("beta"),
        "metrics_json": metrics_row.get("metrics_json"),
        "computed_at": computed_at,
    }

    with engine.connect() as conn:
        conn.execute(delete(backtest_metrics).where(backtest_metrics.c.run_id == run_id))
        conn.execute(insert(backtest_metrics).values(**payload))
        conn.commit()


def compute_and_persist_metrics(
    engine: Engine,
    run_id: str,
    benchmark_returns: Optional[Union[list, Any]] = None,
    periods_per_year: float = 252.0,
) -> Dict[str, Any]:
    """
    5.4.1–5.4.4: Load equity/fills, compute metrics (and attribution if benchmark provided),
    persist to backtest_metrics. Returns the metrics dict.
    """
    row = compute_metrics_for_run(
        engine,
        run_id,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
    )
    persist_metrics(engine, run_id, row)
    return row
