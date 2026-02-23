"""
Tests for Step 3.1 — Event types and queue.

3.1.6: Unit test: create one of each event type; assert fields and ordering by timestamp.
"""

import queue
from datetime import datetime, timezone

import pytest

from src.backtest.events import (
    EVENT_FILL,
    EVENT_MARKET,
    EVENT_SIGNAL,
    DIRECTION_FLAT,
    DIRECTION_LONG_SPREAD,
    DIRECTION_SHORT_SPREAD,
    SIDE_BUY,
    SIDE_SELL,
    Event,
    MarketEvent,
    SignalEvent,
    FillEvent,
    EventQueue,
)


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


# --- Event creation and fields -------------------------------------------------


def test_market_event_fields():
    """Create MarketEvent; assert event_type, timestamp, symbols, bar_data."""
    ts = _utc(2025, 1, 15, 10, 0, 0)
    bar_data = {"AAPL": {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1e6}}
    ev = MarketEvent(timestamp=ts, symbols=("AAPL", "MSFT"), bar_data=bar_data)
    assert ev.event_type == EVENT_MARKET
    assert ev.timestamp == ts
    assert ev.symbols == ("AAPL", "MSFT")
    assert ev.bar_data == bar_data


def test_market_event_default_bar_data():
    """MarketEvent with no bar_data gets empty dict in __post_init__."""
    ts = _utc(2025, 1, 15)
    ev = MarketEvent(timestamp=ts)
    assert ev.event_type == EVENT_MARKET
    assert ev.timestamp == ts
    assert ev.symbols == ()
    assert ev.bar_data == {}


def test_signal_event_fields():
    """Create SignalEvent; assert direction, symbol_a, symbol_b, hedge_ratio, size, metadata."""
    ts = _utc(2025, 1, 15, 10, 5, 0)
    ev = SignalEvent(
        timestamp=ts,
        direction=DIRECTION_LONG_SPREAD,
        symbol_a="AAPL",
        symbol_b="MSFT",
        hedge_ratio=1.2,
        size=10_000.0,
        metadata={"z_score": 2.1, "spread": 0.5},
    )
    assert ev.event_type == EVENT_SIGNAL
    assert ev.timestamp == ts
    assert ev.direction == DIRECTION_LONG_SPREAD
    assert ev.symbol_a == "AAPL"
    assert ev.symbol_b == "MSFT"
    assert ev.hedge_ratio == 1.2
    assert ev.size == 10_000.0
    assert ev.metadata == {"z_score": 2.1, "spread": 0.5}


def test_signal_event_flat_and_default_metadata():
    """SignalEvent with direction=flat and no metadata."""
    ts = _utc(2025, 1, 15)
    ev = SignalEvent(timestamp=ts, direction=DIRECTION_FLAT)
    assert ev.event_type == EVENT_SIGNAL
    assert ev.direction == DIRECTION_FLAT
    assert ev.metadata == {}


def test_fill_event_fields():
    """Create FillEvent; assert symbol, side, quantity, price, commission, slippage_bps."""
    ts = _utc(2025, 1, 15, 10, 10, 0)
    ev = FillEvent(
        timestamp=ts,
        symbol="AAPL",
        side=SIDE_BUY,
        quantity=100.0,
        price=150.25,
        commission=1.0,
        slippage_bps=5.0,
    )
    assert ev.event_type == EVENT_FILL
    assert ev.timestamp == ts
    assert ev.symbol == "AAPL"
    assert ev.side == SIDE_BUY
    assert ev.quantity == 100.0
    assert ev.price == 150.25
    assert ev.commission == 1.0
    assert ev.slippage_bps == 5.0


def test_fill_event_sell():
    """FillEvent with side=sell."""
    ts = _utc(2025, 1, 15)
    ev = FillEvent(timestamp=ts, symbol="MSFT", side=SIDE_SELL, quantity=50.0, price=400.0)
    assert ev.side == SIDE_SELL
    assert ev.event_type == EVENT_FILL


# --- Immutability --------------------------------------------------------------


def test_events_are_immutable():
    """Event types are immutable (frozen dataclasses)."""
    ts = _utc(2025, 1, 15)
    ev = MarketEvent(timestamp=ts, symbols=("A", "B"))
    with pytest.raises(Exception):  # FrozenInstanceError
        ev.timestamp = _utc(2025, 1, 16)  # type: ignore[misc]
    with pytest.raises(Exception):
        ev.symbols = ("X", "Y")  # type: ignore[misc]


# --- Ordering by timestamp -----------------------------------------------------


def test_event_ordering_by_timestamp():
    """Assert ordering by timestamp: earlier < later."""
    t1 = _utc(2025, 1, 15, 9, 0, 0)
    t2 = _utc(2025, 1, 15, 10, 0, 0)
    t3 = _utc(2025, 1, 15, 11, 0, 0)

    m1 = MarketEvent(timestamp=t1)
    m2 = MarketEvent(timestamp=t2)
    s1 = SignalEvent(timestamp=t2, direction=DIRECTION_FLAT)
    f1 = FillEvent(timestamp=t3, symbol="A", side=SIDE_BUY, quantity=1.0, price=1.0)

    assert m1 < m2
    assert m1 < s1
    assert m1 < f1
    assert m2 < f1
    assert not (m2 < s1)  # same timestamp
    assert not (s1 < m2)
    assert s1 < f1


# --- Event queue ---------------------------------------------------------------


def test_event_queue_fifo_and_ordering():
    """Queue returns events in FIFO order; multiple events with timestamps in order."""
    q = EventQueue(enforce_time_order=True)
    t1 = _utc(2025, 1, 15, 9, 0)
    t2 = _utc(2025, 1, 15, 10, 0)
    t3 = _utc(2025, 1, 15, 11, 0)

    q.put(MarketEvent(timestamp=t1))
    q.put(SignalEvent(timestamp=t1, direction=DIRECTION_LONG_SPREAD))
    q.put(MarketEvent(timestamp=t2))
    q.put(FillEvent(timestamp=t2, symbol="A", side=SIDE_BUY, quantity=10.0, price=100.0))
    q.put(MarketEvent(timestamp=t3))

    assert not q.empty()
    assert q.get().timestamp == t1
    assert q.get().timestamp == t1
    assert q.get().timestamp == t2
    assert q.get().timestamp == t2
    assert q.get().timestamp == t3
    assert q.empty()


def test_event_queue_rejects_timestamp_beyond_latest_bar():
    """No event may be emitted with timestamp beyond the latest bar time."""
    q = EventQueue(enforce_time_order=True)
    t_bar = _utc(2025, 1, 15, 10, 0)
    t_later = _utc(2025, 1, 15, 11, 0)

    q.put(MarketEvent(timestamp=t_bar))
    # Signal with same bar time is ok
    q.put(SignalEvent(timestamp=t_bar, direction=DIRECTION_FLAT))
    # Signal with later time than latest bar is rejected
    with pytest.raises(ValueError, match="beyond latest bar time"):
        q.put(SignalEvent(timestamp=t_later, direction=DIRECTION_SHORT_SPREAD))
    # Fill with later time is rejected
    with pytest.raises(ValueError, match="beyond latest bar time"):
        q.put(FillEvent(timestamp=t_later, symbol="A", side=SIDE_BUY, quantity=1.0, price=1.0))


def test_event_queue_accepts_fill_at_same_time_as_bar():
    """FillEvent at same timestamp as latest bar is allowed."""
    q = EventQueue(enforce_time_order=True)
    t = _utc(2025, 1, 15, 10, 0)
    q.put(MarketEvent(timestamp=t))
    q.put(FillEvent(timestamp=t, symbol="AAPL", side=SIDE_BUY, quantity=10.0, price=100.0))
    assert q.qsize() == 2
    assert q.get().event_type == EVENT_MARKET
    assert q.get().event_type == EVENT_FILL


def test_event_queue_first_event_must_be_market_when_enforced():
    """When enforce_time_order is True, first event sets latest_bar_time; non-Market first is allowed but then no future-dated event."""
    q = EventQueue(enforce_time_order=True)
    t1 = _utc(2025, 1, 15, 9, 0)
    t2 = _utc(2025, 1, 15, 10, 0)
    # If we put a Signal first (no Market yet), latest_bar_time is still None, so we can't validate.
    # So we only validate "timestamp beyond latest bar" when latest_bar_time is set.
    q.put(MarketEvent(timestamp=t1))
    assert q.latest_bar_time == t1
    q.put(MarketEvent(timestamp=t2))
    assert q.latest_bar_time == t2


def test_event_queue_no_enforce_allows_any_order():
    """When enforce_time_order is False, any timestamp order is allowed."""
    q = EventQueue(enforce_time_order=False)
    t_early = _utc(2025, 1, 15, 9, 0)
    t_late = _utc(2025, 1, 15, 11, 0)
    q.put(SignalEvent(timestamp=t_late, direction=DIRECTION_LONG_SPREAD))
    q.put(MarketEvent(timestamp=t_early))
    assert q.qsize() == 2
    assert q.get().timestamp == t_late
    assert q.get().timestamp == t_early


def test_event_queue_get_nowait_and_empty():
    """get_nowait and empty()."""
    q = EventQueue(enforce_time_order=False)
    assert q.empty()
    ts = _utc(2025, 1, 15)
    q.put(MarketEvent(timestamp=ts))
    assert not q.empty()
    ev = q.get_nowait()
    assert ev.timestamp == ts
    assert q.empty()
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_clear_latest_bar_time():
    """clear_latest_bar_time resets for new run."""
    q = EventQueue(enforce_time_order=True)
    t = _utc(2025, 1, 15, 10, 0)
    q.put(MarketEvent(timestamp=t))
    assert q.latest_bar_time == t
    q.clear_latest_bar_time()
    assert q.latest_bar_time is None
