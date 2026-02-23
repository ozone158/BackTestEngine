"""
Step 3.5 — ExecutionHandler for the event-driven backtesting engine.

Receives order (symbol, side, quantity, ref_price); applies slippage and commission
from config; optionally bid-ask spread; returns FillEvent for Portfolio.

Step 4.4.3: use_cpp=True delegates to execution_core.batch_fill when the extension is available.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.backtest.events import FillEvent, SIDE_BUY, SIDE_SELL

try:
    import execution_core as _execution_core
except ImportError:
    _execution_core = None  # type: ignore


class BacktestExecutionHandler:
    """
    Converts orders to FillEvents with config-driven slippage and commission.
    Implements the ExecuteOrderFn contract: (symbol, side, quantity, ref_price, timestamp) -> FillEvent.
    """

    def __init__(
        self,
        *,
        slippage_bps: float = 0.0,
        commission_per_trade: float = 0.0,
        commission_per_share: float = 0.0,
        half_spread_bps: Optional[float] = None,
        use_cpp: bool = False,
    ) -> None:
        """
        slippage_bps: applied to fill price (buy: ref * (1 + slippage_bps/10000), sell: ref * (1 - slippage_bps/10000)).
        commission_per_trade: fixed commission per order (e.g. $1).
        commission_per_share: commission per share (e.g. $0.01).
        half_spread_bps: optional bid-ask; buy at ref * (1 + half_spread_bps/10000), sell at ref * (1 - half_spread_bps/10000); then slippage applied to that.
        use_cpp: if True and execution_core is built, use C++ batch_fill for fill simulation (Step 4.4.3).
        """
        self._slippage_bps = slippage_bps
        self._commission_per_trade = commission_per_trade
        self._commission_per_share = commission_per_share
        self._half_spread_bps = half_spread_bps
        self._use_cpp = use_cpp and (_execution_core is not None)

    def __call__(
        self,
        symbol: str,
        side: str,
        quantity: float,
        ref_price: float,
        timestamp: datetime,
    ) -> FillEvent:
        """Execute order: apply slippage and commission; return FillEvent (3.5.1–3.5.5)."""
        return self.execute(symbol, side, quantity, ref_price, timestamp)

    def execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        ref_price: float,
        timestamp: datetime,
    ) -> FillEvent:
        """
        Receive order (symbol, side, quantity, ref_price); apply bid-ask if configured,
        then slippage; compute commission; return FillEvent.
        """
        if self._use_cpp:
            half = -1.0 if self._half_spread_bps is None else self._half_spread_bps
            fill_prices, commissions, slippage_bps_list = _execution_core.batch_fill(
                [symbol],
                [side],
                [quantity],
                [ref_price],
                self._slippage_bps,
                self._commission_per_trade,
                self._commission_per_share,
                half,
            )
            return FillEvent(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=fill_prices[0],
                commission=commissions[0],
                slippage_bps=slippage_bps_list[0],
            )

        # Python path
        # Optional bid-ask (3.5.4): buy at ref + half, sell at ref - half
        if self._half_spread_bps is not None and self._half_spread_bps != 0:
            if side == SIDE_BUY:
                effective_ref = ref_price * (1.0 + self._half_spread_bps / 10_000.0)
            else:
                effective_ref = ref_price * (1.0 - self._half_spread_bps / 10_000.0)
        else:
            effective_ref = ref_price

        # Slippage (3.5.2): buy pays more, sell receives less
        if side == SIDE_BUY:
            fill_price = effective_ref * (1.0 + self._slippage_bps / 10_000.0)
        else:
            fill_price = effective_ref * (1.0 - self._slippage_bps / 10_000.0)

        # Commission (3.5.3)
        commission = self._commission_per_trade + quantity * self._commission_per_share

        return FillEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            slippage_bps=self._slippage_bps,
        )
