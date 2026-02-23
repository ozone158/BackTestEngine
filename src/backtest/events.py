"""
Step 3.1 — Event types and queue for the event-driven backtesting engine.

Event model:
- Event: base type with event_type and timestamp (used for ordering and persistence).
- MarketEvent: new bar/tick; DataHandler is the only producer.
- SignalEvent: strategy decision; Strategy is the only producer.
- FillEvent: executed trade; ExecutionHandler is the only producer.

Events are immutable. Processed in a single time-ordered queue; no event uses
data from the future (no event timestamp beyond the latest bar time seen).
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

# --- Event type literals ------------------------------------------------------
EVENT_MARKET = "MARKET"
EVENT_SIGNAL = "SIGNAL"
EVENT_FILL = "FILL"

# --- Direction / side literals (for signals and fills) ------------------------
DIRECTION_LONG_SPREAD = "long_spread"
DIRECTION_SHORT_SPREAD = "short_spread"
DIRECTION_FLAT = "flat"

SIDE_BUY = "buy"
SIDE_SELL = "sell"


# --- Bar data: open, high, low, close, volume (or tick data) ------------------
BarData = Dict[str, Any]  # e.g. {"open": float, "high": float, "low": float, "close": float, "volume": float}


@dataclass(frozen=True)
class Event:
    """Base event: event_type and timestamp. All events carry timestamp for ordering and persistence."""

    timestamp: datetime
    event_type: str

    def __lt__(self, other: Event) -> bool:
        """Order by timestamp for consistent ordering."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class MarketEvent(Event):
    """
    New bar (or tick) available. Payload: symbol(s), datetime, bar/tick data.
    DataHandler is the only producer.
    """

    event_type: str = EVENT_MARKET
    symbols: tuple = ()  # (symbol_a, symbol_b) or (symbol,) for single leg
    bar_data: Optional[Dict[str, BarData]] = None  # symbol -> {open, high, low, close, volume}

    def __post_init__(self) -> None:
        if self.bar_data is None:
            object.__setattr__(self, "bar_data", {})


@dataclass(frozen=True)
class SignalEvent(Event):
    """
    Strategy decision: direction (long_spread | short_spread | flat), symbol_a, symbol_b,
    hedge_ratio, size, optional metadata (z_score, spread, etc.).
    Strategy is the only producer.
    """

    direction: str = DIRECTION_FLAT  # long_spread | short_spread | flat
    symbol_a: str = ""
    symbol_b: str = ""
    hedge_ratio: float = 0.0
    size: float = 0.0  # notional or units
    metadata: Optional[Dict[str, Any]] = None  # z_score, spread, etc.
    event_type: str = EVENT_SIGNAL

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class FillEvent(Event):
    """
    Executed trade. ExecutionHandler is the only producer.
    """

    symbol: str = ""
    side: str = ""  # buy | sell
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    slippage_bps: float = 0.0
    event_type: str = EVENT_FILL


# --- Event queue --------------------------------------------------------------


class EventQueue:
    """
    Single queue for events. Strict rule: events are processed in timestamp order;
    no event may be emitted with a timestamp beyond the latest bar time seen.
    """

    def __init__(self, enforce_time_order: bool = True) -> None:
        self._q: queue.Queue[Event] = queue.Queue()
        self._enforce_time_order = enforce_time_order
        self._latest_bar_time: Optional[datetime] = None

    def put(self, event: Event) -> None:
        if self._enforce_time_order:
            if isinstance(event, MarketEvent):
                self._latest_bar_time = event.timestamp
            elif self._latest_bar_time is not None and event.timestamp > self._latest_bar_time:
                raise ValueError(
                    f"Event timestamp {event.timestamp} is beyond latest bar time {self._latest_bar_time}"
                )
        self._q.put(event)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Event:
        return self._q.get(block=block, timeout=timeout)

    def get_nowait(self) -> Event:
        return self._q.get_nowait()

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def latest_bar_time(self) -> Optional[datetime]:
        return self._latest_bar_time

    def clear_latest_bar_time(self) -> None:
        """Reset latest bar time (e.g. when starting a new run)."""
        self._latest_bar_time = None
