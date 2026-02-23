"""
Tests for Step 3.2 — DataHandler.

3.2.6: Unit test: feed 10 bars; advance one by one; get_latest_bars(2) returns
       at most 2 bars and only past bars. Assert no future data.
"""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.backtest.data_handler import DataHandler, _align_bars
from src.backtest.events import EVENT_MARKET, EventQueue, MarketEvent


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def _make_bars_df(symbols: list, dts: list, base_price: float = 100.0) -> pl.DataFrame:
    """Build a long-format bar DataFrame: one row per (datetime, symbol). Close = base_price + bar_index*0.5 + 0.5."""
    rows = []
    for i, dt in enumerate(dts):
        for sym in symbols:
            p = base_price + i * 0.5
            rows.append({
                "symbol": sym,
                "datetime": dt,
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p + 0.5,
                "volume": 1e6,
            })
    return pl.DataFrame(rows)


def test_data_handler_loads_and_advances():
    """DataHandler loads bars and advances one bar at a time."""
    dts = [_utc(2025, 1, 15, 9, i) for i in range(10)]
    symbols = ["AAPL", "MSFT"]

    def read_fn(syms, start, end):
        return _make_bars_df(syms, dts)

    start = _utc(2025, 1, 15, 9, 0)
    end = _utc(2025, 1, 15, 10, 0)
    dh = DataHandler(symbols, start, end, read_fn)
    assert dh.symbols == ("AAPL", "MSFT")
    assert dh.current_time is None
    assert dh.has_next()

    # Advance one by one
    for i in range(10):
        bar = dh.next_bar()
        assert bar is not None
        dt, bar_data = bar
        assert dt == dts[i]
        assert set(bar_data.keys()) == set(symbols)
        assert bar_data["AAPL"]["close"] == 100.0 + i * 0.5 + 0.5
        assert dh.current_time == dts[i]

    assert dh.has_next() is False
    assert dh.next_bar() is None


def test_get_latest_bars_at_most_n_and_only_past():
    """get_latest_bars(symbol, 2) returns at most 2 bars and only bars <= current cursor time."""
    dts = [_utc(2025, 1, 15, 9, i) for i in range(10)]
    symbols = ["AAPL", "MSFT"]

    def read_fn(syms, start, end):
        return _make_bars_df(syms, dts)

    dh = DataHandler(symbols, dts[0], dts[-1], read_fn)

    # Before any advance: no bars visible
    assert dh.get_latest_bars("AAPL", 2) == []
    assert dh.get_latest_bars("MSFT", 5) == []

    # After first advance: 1 bar
    dh.next_bar()
    latest_a = dh.get_latest_bars("AAPL", 2)
    assert len(latest_a) == 1
    assert latest_a[0][0] == dts[0]
    assert latest_a[0][1]["close"] == 100.5

    # After second advance: 2 bars
    dh.next_bar()
    latest_a = dh.get_latest_bars("AAPL", 2)
    assert len(latest_a) == 2
    assert latest_a[0][0] == dts[0]
    assert latest_a[1][0] == dts[1]

    # Ask for 2 when we have 5 bars: still at most 2 (last 2)
    for _ in range(3):
        dh.next_bar()
    latest_a = dh.get_latest_bars("AAPL", 2)
    assert len(latest_a) == 2
    assert latest_a[0][0] == dts[3]
    assert latest_a[1][0] == dts[4]

    # All 10: ask for 20, get 10
    while dh.has_next():
        dh.next_bar()
    latest_a = dh.get_latest_bars("AAPL", 20)
    assert len(latest_a) == 10
    assert latest_a[-1][0] == dts[9]


def test_get_latest_bars_no_future_data():
    """Assert no future data: every bar returned has datetime <= current_time."""
    dts = [_utc(2025, 1, 15, 9, i) for i in range(10)]
    symbols = ["AAPL", "MSFT"]

    def read_fn(syms, start, end):
        return _make_bars_df(syms, dts)

    dh = DataHandler(symbols, dts[0], dts[-1], read_fn)
    for i in range(10):
        dh.next_bar()
        current = dh.current_time
        for sym in symbols:
            latest = dh.get_latest_bars(sym, 10)
            for dt, _ in latest:
                assert dt <= current, f"get_latest_bars returned future bar {dt} > current {current}"


def test_emit_market_event_on_advance():
    """When advancing with event_queue, MarketEvent is emitted with correct timestamp and bar_data."""
    dts = [_utc(2025, 1, 15, 9, 0), _utc(2025, 1, 15, 9, 1)]
    symbols = ["AAPL", "MSFT"]

    def read_fn(syms, start, end):
        return _make_bars_df(syms, dts)

    q = EventQueue(enforce_time_order=True)
    dh = DataHandler(symbols, dts[0], dts[-1], read_fn)

    dh.next_bar(event_queue=q)
    assert not q.empty()
    ev = q.get_nowait()
    assert ev.event_type == EVENT_MARKET
    assert ev.timestamp == dts[0]
    assert ev.symbols == ("AAPL", "MSFT")
    assert "AAPL" in ev.bar_data and "MSFT" in ev.bar_data
    assert ev.bar_data["AAPL"]["close"] == 100.5

    dh.next_bar(event_queue=q)
    ev2 = q.get_nowait()
    assert ev2.timestamp == dts[1]
    assert q.empty()


def test_multi_symbol_alignment():
    """For pairs, only aligned datetimes (both symbols present) are emitted."""
    # One symbol has a missing bar at one datetime
    dts = [_utc(2025, 1, 15, 9, i) for i in range(5)]
    rows = []
    for i, dt in enumerate(dts):
        for sym in ["AAPL", "MSFT"]:
            if sym == "MSFT" and i == 2:
                continue  # missing MSFT at index 2
            p = 100.0 + i
            rows.append({
                "symbol": sym,
                "datetime": dt,
                "open": p, "high": p + 1, "low": p - 1, "close": p + 0.5, "volume": 1e6,
            })
    df = pl.DataFrame(rows)

    def read_fn(syms, start, end):
        return df

    dh = DataHandler(["AAPL", "MSFT"], dts[0], dts[-1], read_fn)
    # Should have 4 aligned bars (datetime at index 2 dropped)
    count = 0
    while dh.next_bar():
        count += 1
    assert count == 4


def test_align_bars_empty():
    """_align_bars with empty df or no symbols returns empty list."""
    df = pl.DataFrame({"symbol": [], "datetime": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
    assert _align_bars(df, ["A", "B"]) == []
    assert _align_bars(_make_bars_df(["A"], [_utc(2025, 1, 1)]), []) == []


def test_get_latest_bars_unknown_symbol_returns_empty():
    """get_latest_bars for symbol not in handler returns []."""
    dts = [_utc(2025, 1, 15, 9, 0)]
    dh = DataHandler(
        ["AAPL", "MSFT"],
        dts[0],
        dts[0],
        lambda syms, s, e: _make_bars_df(syms, dts),
    )
    dh.next_bar()
    assert dh.get_latest_bars("XYZ", 5) == []
    assert dh.get_latest_bars("AAPL", 0) == []
