"""
Step 3.6 — Event loop and persistence.

Main loop: DataHandler next bar → Strategy (MarketEvent → SignalEvent) → Portfolio
(process_signal → ExecutionHandler → FillEvents) → persist signals and fills;
record equity (per bar or per fill); flush equity curve to backtest_equity.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import insert
from sqlalchemy.engine import Engine

from src.backtest.events import FillEvent, MarketEvent, SignalEvent
from src.backtest.data_handler import DataHandler
from src.backtest.metrics_persistence import compute_and_persist_metrics
from src.backtest.strategy import OUStrategy
from src.backtest.portfolio import Portfolio
from src.data.storage.schema import backtest_fills, backtest_signals


def persist_signal(engine: Engine, run_id: str, signal: SignalEvent) -> Optional[int]:
    """
    Insert one row into backtest_signals; return the generated id for linking fills (3.5.6, 3.6.3).
    """
    metadata_json = json.dumps(signal.metadata) if signal.metadata else None
    with engine.connect() as conn:
        result = conn.execute(
            insert(backtest_signals).values(
                run_id=run_id,
                signal_ts=signal.timestamp,
                direction=signal.direction,
                symbol_a=signal.symbol_a,
                symbol_b=signal.symbol_b,
                hedge_ratio=signal.hedge_ratio,
                size=signal.size,
                metadata_json=metadata_json,
            )
        )
        conn.commit()
        pk = result.inserted_primary_key
        return int(pk[0]) if pk else None


def persist_fill(
    engine: Engine,
    run_id: str,
    fill: FillEvent,
    signal_id: Optional[int] = None,
) -> None:
    """Insert one row into backtest_fills (3.6.3). signal_id links to SignalEvent when available."""
    with engine.connect() as conn:
        conn.execute(
            insert(backtest_fills).values(
                run_id=run_id,
                signal_id=signal_id,
                fill_ts=fill.timestamp,
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission,
                slippage_bps=fill.slippage_bps,
            )
        )
        conn.commit()


def run_backtest(
    data_handler: DataHandler,
    strategy: OUStrategy,
    portfolio: Portfolio,
    engine: Engine,
    *,
    record_equity_every_bar: bool = True,
) -> str:
    """
    Run the event-driven backtest loop (3.6.1).

    1. Portfolio.start_run(engine) already called by caller (with run_id, start_ts, end_ts).
    2. While DataHandler has next bar: get next bar; build MarketEvent; Strategy processes it;
       if SignalEvent: persist signal, Portfolio.process_signal (→ fills), persist each fill;
       record equity snapshot (every bar or only when fills).
    3. Flush equity curve to backtest_equity.

    Returns run_id. Caller must have called portfolio.start_run(engine) before this.
    """
    run_id = portfolio.run_id
    symbols = data_handler.symbols

    while data_handler.has_next():
        bar = data_handler.next_bar()
        if bar is None:
            break
        dt, bar_data = bar
        market_ev = MarketEvent(timestamp=dt, symbols=symbols, bar_data=bar_data)
        signal = strategy.process_market_event(market_ev)

        if signal is not None:
            signal_id = persist_signal(engine, run_id, signal)
            fills = portfolio.process_signal(signal, bar_data)
            for fill in fills:
                persist_fill(engine, run_id, fill, signal_id=signal_id)
        elif record_equity_every_bar:
            # When no signal this bar, record equity so we have one row per bar
            portfolio.record_equity_snapshot(dt, bar_data)

    portfolio.flush_equity_curve(engine)
    # 5.4.1–5.4.4: After run, compute metrics and persist to backtest_metrics
    compute_and_persist_metrics(engine, run_id)
    return run_id
