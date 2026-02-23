"""
Step 3.3 — Strategy for the event-driven backtesting engine.

Consumes MarketEvent; uses OU params (entry_upper, entry_lower, exit_threshold)
and spread/β from a provider (precomputed or online Kalman). Emits SignalEvent
when spread crosses entry/exit thresholds. Tracks position state to avoid duplicate entries.

Size can be fixed or from a size_provider (e.g. Kelly or risk-parity based) for PositionSizer integration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from src.backtest.events import (
    BarData,
    DIRECTION_FLAT,
    DIRECTION_LONG_SPREAD,
    DIRECTION_SHORT_SPREAD,
    EVENT_MARKET,
    EventQueue,
    MarketEvent,
    SignalEvent,
)
from src.data.alpha.ou import OUParams

# Provider: (timestamp, bar_data) -> (spread, beta)
SpreadBetaProvider = Callable[[datetime, Dict[str, BarData]], Tuple[float, float]]

# Size provider: (timestamp, bar_data, direction, current_equity | None) -> size (notional or units).
# Used when size_provider is set; current_equity is passed from run loop when available.
SizeProvider = Callable[[datetime, Dict[str, BarData], str, Optional[float]], float]


class OUStrategy:
    """
    Pairs-trading strategy driven by OU entry/exit thresholds (spread units).

    On each MarketEvent: get (spread, β) from spread_beta_provider; compare spread
    to entry_upper, entry_lower, exit_threshold; emit SignalEvent (long_spread,
    short_spread, or flat) and track position to avoid duplicate entries.

    Size: use fixed `size` (default) or pluggable `size_provider` (e.g. Kelly or risk-parity
    via PositionSizer). When size_provider is set, it receives (timestamp, bar_data, direction, current_equity).
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        ou_params: OUParams,
        spread_beta_provider: SpreadBetaProvider,
        *,
        size: float = 0.0,
        size_provider: Optional[SizeProvider] = None,
    ) -> None:
        """
        symbol_a, symbol_b: pair legs (symbol_a < symbol_b per convention).
        ou_params: entry_upper, entry_lower, exit_threshold (spread units), mu, sigma.
        spread_beta_provider: callable(timestamp, bar_data) -> (spread, beta).
        size: fixed notional or units when size_provider is None.
        size_provider: optional (timestamp, bar_data, direction, current_equity) -> size; overrides size when set.
        """
        self._symbol_a = symbol_a
        self._symbol_b = symbol_b
        self._ou = ou_params
        self._spread_beta_provider = spread_beta_provider
        self._size = size
        self._size_provider: Optional[SizeProvider] = size_provider
        self._position: str = DIRECTION_FLAT  # flat | long_spread | short_spread

    @property
    def symbol_a(self) -> str:
        return self._symbol_a

    @property
    def symbol_b(self) -> str:
        return self._symbol_b

    @property
    def position(self) -> str:
        return self._position

    def process_market_event(
        self,
        event: MarketEvent,
        event_queue: Optional[EventQueue] = None,
        current_equity: Optional[float] = None,
    ) -> Optional[SignalEvent]:
        """
        On MarketEvent: get spread and β for current bar; apply OU signal logic;
        optionally emit SignalEvent to queue. Returns the SignalEvent if one was produced.
        Uses only current and past data (provider is responsible for no look-ahead).
        current_equity: optional, for size_provider (e.g. Kelly * equity); pass from run loop when available.
        """
        if event.event_type != EVENT_MARKET:
            return None
        bar_data = event.bar_data or {}
        if self._symbol_a not in bar_data or self._symbol_b not in bar_data:
            return None

        spread, beta = self._spread_beta_provider(event.timestamp, bar_data)
        mu, sigma = self._ou.mu, self._ou.sigma
        z_score = (spread - mu) / sigma if sigma and sigma > 0 else 0.0

        entry_upper = self._ou.entry_upper
        entry_lower = self._ou.entry_lower
        exit_threshold = self._ou.exit_threshold

        new_direction = self._position  # default: no change
        if spread >= entry_upper and self._position != DIRECTION_LONG_SPREAD:
            new_direction = DIRECTION_LONG_SPREAD
        elif spread <= entry_lower and self._position != DIRECTION_SHORT_SPREAD:
            new_direction = DIRECTION_SHORT_SPREAD
        elif self._position == DIRECTION_LONG_SPREAD and spread <= exit_threshold:
            new_direction = DIRECTION_FLAT
        elif self._position == DIRECTION_SHORT_SPREAD and spread >= exit_threshold:
            new_direction = DIRECTION_FLAT

        # Emit only when direction actually changed
        if new_direction == self._position:
            return None
        self._position = new_direction

        if self._size_provider is not None:
            size = self._size_provider(event.timestamp, bar_data, new_direction, current_equity)
        else:
            size = self._size

        signal = SignalEvent(
            timestamp=event.timestamp,
            direction=new_direction,
            symbol_a=self._symbol_a,
            symbol_b=self._symbol_b,
            hedge_ratio=beta,
            size=size,
            metadata={"z_score": z_score, "spread": spread},
        )
        if event_queue is not None:
            event_queue.put(signal)
        return signal
