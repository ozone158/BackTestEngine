"""
Tests for Step 3.3 — Strategy.

3.3.6: Unit test: mock bars and OU params; assert long_spread when z > entry_upper,
       flat when z in exit band. Test state machine (no double entry).
"""

from datetime import datetime, timezone

import pytest

from src.backtest.events import (
    DIRECTION_FLAT,
    DIRECTION_LONG_SPREAD,
    DIRECTION_SHORT_SPREAD,
    EventQueue,
    MarketEvent,
    SignalEvent,
)
from src.backtest.strategy import OUStrategy
from src.data.alpha.ou import OUParams


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


def _ou_params(mu=0.0, sigma=1.0, entry_k=2.0):
    """OU params: entry_upper = mu + k*sigma, entry_lower = mu - k*sigma, exit = mu."""
    return OUParams(
        theta=0.1,
        mu=mu,
        sigma=sigma,
        entry_upper=mu + entry_k * sigma,
        entry_lower=mu - entry_k * sigma,
        exit_threshold=mu,
    )


def test_long_spread_when_spread_above_entry_upper():
    """When spread >= entry_upper and not already long, emit long_spread."""
    ou = _ou_params(mu=0.0, sigma=1.0, entry_k=2.0)  # entry_upper=2, entry_lower=-2
    # Provider returns (spread, beta); spread = 2.5 > entry_upper
    def provider(ts, bar_data):
        return (2.5, 1.2)

    strategy = OUStrategy("AAPL", "MSFT", ou, provider, size=10_000.0)
    ts = _utc(2025, 1, 15, 10, 0)
    bar_data = {"AAPL": {"close": 150.0}, "MSFT": {"close": 100.0}}
    ev = MarketEvent(timestamp=ts, symbols=("AAPL", "MSFT"), bar_data=bar_data)

    out = strategy.process_market_event(ev)
    assert out is not None
    assert out.direction == DIRECTION_LONG_SPREAD
    assert out.symbol_a == "AAPL"
    assert out.symbol_b == "MSFT"
    assert out.hedge_ratio == 1.2
    assert out.size == 10_000.0
    assert out.metadata["spread"] == 2.5
    assert out.metadata["z_score"] == 2.5  # (2.5 - 0) / 1
    assert strategy.position == DIRECTION_LONG_SPREAD


def test_flat_when_spread_in_exit_band_after_long():
    """When long and spread <= exit_threshold (mean), emit flat."""
    ou = _ou_params(mu=0.0, sigma=1.0)  # exit_threshold = 0
    def provider(ts, bar_data):
        return (0.0, 1.0)  # spread at mean

    strategy = OUStrategy("A", "B", ou, provider)
    ts = _utc(2025, 1, 15, 10, 0)
    ev = MarketEvent(timestamp=ts, symbols=("A", "B"), bar_data={"A": {"close": 100}, "B": {"close": 100}})

    # First bar: spread 0, we're flat; no entry (0 is not >= 2). So no signal.
    out1 = strategy.process_market_event(ev)
    assert out1 is None
    assert strategy.position == DIRECTION_FLAT

    # Force long first by using a provider that returns high spread then low
    def provider_sequence(values):
        it = iter(values)
        def fn(ts, bar_data):
            return next(it)
        return fn

    strategy2 = OUStrategy("A", "B", ou, provider_sequence([(2.5, 1.0), (0.0, 1.0)]))
    ev1 = MarketEvent(timestamp=_utc(2025, 1, 15, 9, 0), symbols=("A", "B"), bar_data={"A": {"close": 1}, "B": {"close": 1}})
    ev2 = MarketEvent(timestamp=_utc(2025, 1, 15, 10, 0), symbols=("A", "B"), bar_data={"A": {"close": 1}, "B": {"close": 1}})
    s1 = strategy2.process_market_event(ev1)
    assert s1 is not None and s1.direction == DIRECTION_LONG_SPREAD
    s2 = strategy2.process_market_event(ev2)
    assert s2 is not None and s2.direction == DIRECTION_FLAT
    assert strategy2.position == DIRECTION_FLAT


def test_short_spread_when_spread_below_entry_lower():
    """When spread <= entry_lower and not short, emit short_spread."""
    ou = _ou_params(mu=0.0, sigma=1.0, entry_k=2.0)  # entry_lower = -2
    def provider(ts, bar_data):
        return (-2.5, 1.1)

    strategy = OUStrategy("AAPL", "MSFT", ou, provider)
    ts = _utc(2025, 1, 15, 10, 0)
    ev = MarketEvent(timestamp=ts, symbols=("AAPL", "MSFT"), bar_data={"AAPL": {"close": 90}, "MSFT": {"close": 100}})

    out = strategy.process_market_event(ev)
    assert out is not None
    assert out.direction == DIRECTION_SHORT_SPREAD
    assert strategy.position == DIRECTION_SHORT_SPREAD


def test_no_double_entry_long():
    """State machine: do not emit long_spread again when already long."""
    ou = _ou_params(mu=0.0, sigma=1.0, entry_k=2.0)
    def provider(ts, bar_data):
        return (3.0, 1.0)  # always above entry_upper

    strategy = OUStrategy("A", "B", ou, provider)
    ev = MarketEvent(timestamp=_utc(2025, 1, 15, 10, 0), symbols=("A", "B"), bar_data={"A": {"close": 1}, "B": {"close": 1}})

    s1 = strategy.process_market_event(ev)
    assert s1 is not None and s1.direction == DIRECTION_LONG_SPREAD
    s2 = strategy.process_market_event(ev)
    assert s2 is None  # already long, no duplicate entry
    assert strategy.position == DIRECTION_LONG_SPREAD


def test_no_double_entry_short():
    """State machine: do not emit short_spread again when already short."""
    ou = _ou_params(mu=0.0, sigma=1.0, entry_k=2.0)
    def provider(ts, bar_data):
        return (-3.0, 1.0)

    strategy = OUStrategy("A", "B", ou, provider)
    ev = MarketEvent(timestamp=_utc(2025, 1, 15, 10, 0), symbols=("A", "B"), bar_data={"A": {"close": 1}, "B": {"close": 1}})

    s1 = strategy.process_market_event(ev)
    assert s1 is not None and s1.direction == DIRECTION_SHORT_SPREAD
    s2 = strategy.process_market_event(ev)
    assert s2 is None
    assert strategy.position == DIRECTION_SHORT_SPREAD


def test_signal_emitted_to_queue():
    """When event_queue is passed, SignalEvent is put on queue."""
    ou = _ou_params(mu=0.0, sigma=1.0, entry_k=2.0)
    def provider(ts, bar_data):
        return (2.5, 1.0)

    q = EventQueue(enforce_time_order=False)
    strategy = OUStrategy("A", "B", ou, provider)
    ev = MarketEvent(timestamp=_utc(2025, 1, 15, 10, 0), symbols=("A", "B"), bar_data={"A": {"close": 1}, "B": {"close": 1}})
    strategy.process_market_event(ev, event_queue=q)
    assert not q.empty()
    sig = q.get_nowait()
    assert sig.direction == DIRECTION_LONG_SPREAD
    assert sig.timestamp == ev.timestamp


def test_ignores_market_event_without_both_symbols():
    """If bar_data is missing symbol_a or symbol_b, return None."""
    ou = _ou_params()
    def provider(ts, bar_data):
        return (0.0, 1.0)

    strategy = OUStrategy("AAPL", "MSFT", ou, provider)
    ev = MarketEvent(timestamp=_utc(2025, 1, 15), symbols=("AAPL", "MSFT"), bar_data={"AAPL": {"close": 100}})
    out = strategy.process_market_event(ev)
    assert out is None
