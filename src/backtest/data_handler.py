"""
Step 3.2 — DataHandler for the event-driven backtesting engine.

Loads bars for a backtest range, aligns multi-symbol (pair) bars on datetime,
exposes next_bar() to advance one bar at a time and emit MarketEvent,
and get_latest_bars(symbol, N) with no look-ahead (only bars <= current cursor time).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import polars as pl

from src.backtest.events import BarData, EventQueue, MarketEvent

# Callable: (symbols, start, end) -> pl.DataFrame with columns symbol, datetime, open, high, low, close, volume
ReadBarsFn = Callable[[Sequence[str], datetime, datetime], pl.DataFrame]

BAR_COLS = ["open", "high", "low", "close", "volume"]


def _row_to_bar_data(row: pl.DataFrame) -> Dict[str, Any]:
    """Convert a single-row Polars DataFrame (bar) to BarData dict."""
    d = row.to_dicts()[0]
    return {k: float(d[k]) for k in d if isinstance(d.get(k), (int, float))}


def _align_bars(df: pl.DataFrame, symbols: Sequence[str]) -> List[Tuple[datetime, Dict[str, BarData]]]:
    """
    Align bars on datetime: one row per datetime with all symbols present.
    Drop rows where any symbol is missing. Return list of (datetime, bar_data) sorted by datetime.
    bar_data: symbol -> {open, high, low, close, volume}.
    """
    if df.is_empty() or not symbols:
        return []
    symbol_set = set(symbols)
    cols = [c for c in BAR_COLS if c in df.columns]
    if not cols:
        return []
    out: List[Tuple[datetime, Dict[str, BarData]]] = []
    dts = df["datetime"].unique().sort().to_list()
    for dt in dts:
        sub = df.filter(pl.col("datetime") == dt)
        present = set(sub["symbol"].to_list())
        if not symbol_set.issubset(present):
            continue
        bar_data: Dict[str, BarData] = {}
        for sym in symbols:
            row = sub.filter(pl.col("symbol") == sym)
            if row.height == 0:
                break
            bar_data[sym] = _row_to_bar_data(row.select(cols))
        if len(bar_data) == len(symbols):
            dt_val = dt
            if getattr(dt_val, "tzinfo", None) is None and hasattr(dt_val, "replace"):
                dt_val = dt_val.replace(tzinfo=timezone.utc)
            out.append((dt_val, bar_data))
    return out


class DataHandler:
    """
    Loads bars for a backtest range, aligns by datetime for pairs,
    advances one bar at a time (no look-ahead), emits MarketEvent on advance,
    and provides get_latest_bars(symbol, N) for strategy use.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        start_ts: datetime,
        end_ts: datetime,
        read_bars_fn: ReadBarsFn,
    ) -> None:
        """
        Load bars for the given symbols and range via read_bars_fn(symbols, start_ts, end_ts).
        Bars are aligned on datetime (inner join: only datetimes where all symbols have data).
        """
        self._symbols = tuple(symbols)
        self._start_ts = start_ts
        self._end_ts = end_ts
        df = read_bars_fn(self._symbols, start_ts, end_ts)
        self._aligned_bars: List[Tuple[datetime, Dict[str, BarData]]] = _align_bars(df, self._symbols)
        self._current_index: int = -1  # -1 = before first bar; 0..len-1 = current bar

    @property
    def symbols(self) -> Tuple[str, ...]:
        return self._symbols

    @property
    def current_time(self) -> Optional[datetime]:
        """Current bar datetime (cursor); None if not yet advanced to any bar."""
        if self._current_index < 0 or self._current_index >= len(self._aligned_bars):
            return None
        return self._aligned_bars[self._current_index][0]

    def has_next(self) -> bool:
        """True if there is a next bar to advance to."""
        return self._current_index + 1 < len(self._aligned_bars)

    def next_bar(
        self,
        event_queue: Optional[EventQueue] = None,
    ) -> Optional[Tuple[datetime, Dict[str, BarData]]]:
        """
        Advance to the next bar. Returns (datetime, bar_data) for the new current bar, or None if exhausted.
        If event_queue is provided, emits MarketEvent(timestamp, symbols, bar_data) onto the queue.
        Cursor never goes backward; no peek into future.
        """
        next_index = self._current_index + 1
        if next_index >= len(self._aligned_bars):
            return None
        self._current_index = next_index
        dt, bar_data = self._aligned_bars[self._current_index]
        if event_queue is not None:
            event_queue.put(
                MarketEvent(
                    timestamp=dt,
                    symbols=self._symbols,
                    bar_data=dict(bar_data),
                )
            )
        return (dt, bar_data)

    def get_latest_bars(self, symbol: str, N: int) -> List[Tuple[datetime, BarData]]:
        """
        Return the last N bars for symbol with datetime <= current cursor time.
        If cursor is at T, only bars up to T are visible (no look-ahead).
        Returns list of (datetime, bar_data) ordered from oldest to newest, at most N elements.
        """
        if symbol not in self._symbols or N <= 0:
            return []
        # Bars from index 0 to _current_index (inclusive) are "past and current"
        end_idx = self._current_index + 1
        if end_idx <= 0:
            return []
        start_idx = max(0, end_idx - N)
        result: List[Tuple[datetime, BarData]] = []
        for i in range(start_idx, end_idx):
            dt, bar_data = self._aligned_bars[i]
            if symbol in bar_data:
                result.append((dt, dict(bar_data[symbol])))
        return result
