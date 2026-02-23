"""
Step 3.4 — Portfolio for the event-driven backtesting engine.

Maintains cash, positions; converts SignalEvent to leg orders; calls ExecutionHandler
for fills; updates state on FillEvent; records equity curve; writes backtest_runs and backtest_equity.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import insert
from sqlalchemy.engine import Engine

from src.backtest.events import (
    BarData,
    DIRECTION_FLAT,
    DIRECTION_LONG_SPREAD,
    DIRECTION_SHORT_SPREAD,
    EventQueue,
    FillEvent,
    SIDE_BUY,
    SIDE_SELL,
    SignalEvent,
)
from src.data.storage.schema import backtest_equity, backtest_runs

# Callable: (symbol, side, quantity, ref_price, timestamp) -> FillEvent
ExecuteOrderFn = Callable[[str, str, float, float, datetime], FillEvent]


def _get_close(bar_data: Dict[str, BarData], symbol: str) -> float:
    """Reference price (close) for a symbol from bar_data."""
    if not bar_data or symbol not in bar_data:
        return 0.0
    d = bar_data[symbol]
    return float(d.get("close", 0.0) or 0.0)


class Portfolio:
    """
    Tracks cash and positions; converts SignalEvents to leg orders; updates on FillEvents;
    records equity curve; persists backtest_runs and backtest_equity.
    """

    def __init__(
        self,
        run_id: str,
        initial_capital: float,
        *,
        strategy_name: str = "",
        pair_id: Optional[str] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        config_json: Optional[Dict[str, Any]] = None,
        execution_handler: Optional[ExecuteOrderFn] = None,
    ) -> None:
        self._run_id = run_id
        self._initial_capital = initial_capital
        self._strategy_name = strategy_name
        self._pair_id = pair_id
        self._start_ts = start_ts
        self._end_ts = end_ts
        self._config_json = config_json or {}
        self._execute = execution_handler

        self._cash: float = 0.0
        self._positions: Dict[str, float] = {}
        self._equity_curve: List[Tuple[datetime, float, float, float]] = []  # (ts, equity, cash, positions_value)
        self._started: bool = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, float]:
        return dict(self._positions)

    @property
    def equity_curve(self) -> List[Tuple[datetime, float, float, float]]:
        return list(self._equity_curve)

    def start_run(self, engine: Optional[Engine] = None) -> None:
        """
        Insert backtest_runs row; initialize cash = initial_capital, positions = 0 (3.4.6).
        """
        self._cash = self._initial_capital
        self._positions = {}
        self._equity_curve = []
        self._started = True

        if engine is not None and self._start_ts is not None and self._end_ts is not None:
            with engine.connect() as conn:
                conn.execute(
                    backtest_runs.insert().values(
                        run_id=self._run_id,
                        strategy_name=self._strategy_name or None,
                        pair_id=self._pair_id,
                        start_ts=self._start_ts,
                        end_ts=self._end_ts,
                        config_json=json.dumps(self._config_json) if self._config_json else None,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                conn.commit()

    def process_signal(
        self,
        signal: SignalEvent,
        bar_data: Dict[str, BarData],
        event_queue: Optional[EventQueue] = None,
    ) -> List[FillEvent]:
        """
        Convert SignalEvent to leg orders; call execution_handler for each leg;
        process each FillEvent and optionally put on queue. Returns list of FillEvents (3.4.2–3.4.4).
        """
        if not self._started or self._execute is None:
            return []

        price_a = _get_close(bar_data, signal.symbol_a)
        price_b = _get_close(bar_data, signal.symbol_b)
        size = signal.size or 0.0
        beta = signal.hedge_ratio or 0.0

        # Validate reference prices for long/short: avoid zero or negative fill prices
        if signal.direction in (DIRECTION_LONG_SPREAD, DIRECTION_SHORT_SPREAD):
            if price_a <= 0 or price_b <= 0:
                return []
            if size <= 0:
                return []

        orders: List[Tuple[str, str, float]] = []  # (symbol, side, quantity)

        if signal.direction == DIRECTION_LONG_SPREAD:
            orders = [
                (signal.symbol_a, SIDE_BUY, size),
                (signal.symbol_b, SIDE_SELL, size * beta),
            ]
        elif signal.direction == DIRECTION_SHORT_SPREAD:
            orders = [
                (signal.symbol_a, SIDE_SELL, size),
                (signal.symbol_b, SIDE_BUY, size * beta),
            ]
        elif signal.direction == DIRECTION_FLAT:
            qty_a = self._positions.get(signal.symbol_a, 0.0)
            qty_b = self._positions.get(signal.symbol_b, 0.0)
            if qty_a > 0:
                orders.append((signal.symbol_a, SIDE_SELL, qty_a))
            elif qty_a < 0:
                orders.append((signal.symbol_a, SIDE_BUY, -qty_a))
            if qty_b > 0:
                orders.append((signal.symbol_b, SIDE_SELL, qty_b))
            elif qty_b < 0:
                orders.append((signal.symbol_b, SIDE_BUY, -qty_b))

        fills: List[FillEvent] = []
        ref_prices = {signal.symbol_a: price_a, signal.symbol_b: price_b}
        for symbol, side, qty in orders:
            if qty <= 0:
                continue
            ref_price = ref_prices.get(symbol, 0.0) or _get_close(bar_data, symbol)
            fill = self._execute(symbol, side, qty, ref_price, signal.timestamp)
            fills.append(fill)
            self._process_fill(fill)
            if event_queue is not None:
                event_queue.put(fill)

        if fills:
            self._append_equity(signal.timestamp, bar_data)
        return fills

    def _process_fill(self, fill: FillEvent) -> None:
        """Update positions and cash from FillEvent (3.4.4)."""
        qty = fill.quantity
        signed = qty if fill.side == SIDE_BUY else -qty
        self._positions[fill.symbol] = self._positions.get(fill.symbol, 0.0) + signed

        cost = qty * fill.price + fill.commission
        if fill.side == SIDE_BUY:
            self._cash -= cost
        else:
            self._cash += (qty * fill.price) - fill.commission

    def _append_equity(self, ts: datetime, bar_data: Optional[Dict[str, BarData]] = None) -> None:
        """Record (ts, equity, cash, positions_value) (3.4.5)."""
        positions_value = 0.0
        if bar_data:
            for sym, qty in self._positions.items():
                if qty == 0:
                    continue
                p = _get_close(bar_data, sym) if bar_data else 0.0
                positions_value += qty * p
        equity = self._cash + positions_value
        self._equity_curve.append((ts, equity, self._cash, positions_value))

    def record_equity_snapshot(self, ts: datetime, bar_data: Optional[Dict[str, BarData]] = None) -> None:
        """Append one equity snapshot (e.g. after each bar)."""
        self._append_equity(ts, bar_data)

    def current_equity(self, bar_data: Optional[Dict[str, BarData]] = None) -> float:
        """Current equity (cash + positions at bar closes). For size_provider when generating signals."""
        positions_value = 0.0
        if bar_data:
            for sym, qty in self._positions.items():
                if qty == 0:
                    continue
                p = _get_close(bar_data, sym)
                positions_value += qty * p
        return self._cash + positions_value

    def flush_equity_curve(self, engine: Engine) -> None:
        """Batch insert equity curve into backtest_equity (3.4.5)."""
        if not self._equity_curve:
            return
        rows = [
            {
                "run_id": self._run_id,
                "ts": ts,
                "equity": equity,
                "cash": cash,
                "positions_value": pv,
            }
            for ts, equity, cash, pv in self._equity_curve
        ]
        with engine.connect() as conn:
            conn.execute(insert(backtest_equity), rows)
            conn.commit()


def generate_run_id() -> str:
    """Generate run_id (e.g. UUID)."""
    return str(uuid.uuid4())
