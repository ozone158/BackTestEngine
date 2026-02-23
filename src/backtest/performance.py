"""
Step 5.1 — Performance metrics.

Accepts equity curve or return series; computes total return, max drawdown,
Sharpe (annualized), Sortino (annualized), Calmar; optional trade stats from
fill events. All from historical data only (no look-ahead).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _as_equity_array(
    equity_series: Union[Sequence[float], np.ndarray, Sequence[Tuple[Any, float]]]
) -> np.ndarray:
    """Extract 1d array of equity values (5.1.1)."""
    arr = np.asarray(equity_series)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.asarray(arr[:, 1], dtype=float)
    return np.asarray(arr, dtype=float)


def _equity_to_returns(equity: np.ndarray) -> np.ndarray:
    """Compute period returns from equity: (equity[t] - equity[t-1]) / equity[t-1]. First value has no return (NaN)."""
    if len(equity) < 2:
        return np.array([])
    ret = (equity[1:] - equity[:-1]) / np.where(equity[:-1] != 0, equity[:-1], np.nan)
    return ret


def total_return_from_equity(equity: np.ndarray) -> float:
    """5.1.2: total_return = (equity[-1] - equity[0]) / equity[0]. Decimal (e.g. 0.05 for 5%)."""
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return float((equity[-1] - equity[0]) / equity[0])


def total_return_from_returns(returns: np.ndarray) -> float:
    """5.1.2: total_return = (1+r1)(1+r2)... - 1 from return series."""
    if len(returns) == 0:
        return 0.0
    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    """5.1.3: peak = running max(equity); drawdown = (peak - equity)/peak; max_drawdown = max(drawdown). Decimal."""
    if len(equity) == 0:
        return 0.0
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    peak = np.where(peak > 0, peak, 1.0)
    drawdown = (peak - equity) / peak
    return float(np.max(drawdown))


def sharpe_annual(
    returns: np.ndarray,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """5.1.4: Mean(returns)/std(returns) * sqrt(periods_per_year). Optional risk-free (annual, decimal). Zero std -> 0."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = np.mean(returns) - rf_per_period
    std = np.std(returns, ddof=1)
    if std <= 0:
        return 0.0
    return float(excess / std * np.sqrt(periods_per_year))


def sortino_annual(
    returns: np.ndarray,
    periods_per_year: float = 252.0,
    target: float = 0.0,
) -> float:
    """5.1.5: Mean(returns) / downside_std(returns) * sqrt(periods_per_year). Downside = returns below target."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < target]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std <= 0:
        return 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """5.1.6: total_return / max_drawdown. Zero max_drawdown -> 0."""
    if max_drawdown <= 0:
        return 0.0
    return float(total_return / max_drawdown)


def _trade_stats_from_fills(
    fill_events: Sequence[Any],
) -> Tuple[int, Optional[float]]:
    """
    5.1.7: num_trades, win_rate from fill events.
    If fills have signal_id, group by it and compute P&L per trade; else num_trades = len(fills)//2, win_rate None.
    """
    if not fill_events:
        return 0, None
    # Support FillEvent (has .symbol, .side, .quantity, .price, .commission) or dict
    def _get(f: Any, key: str) -> Any:
        return getattr(f, key, None) or (f.get(key) if isinstance(f, dict) else None)

    signal_ids = [_get(f, "signal_id") for f in fill_events]
    if all(sid is not None for sid in signal_ids):
        from collections import defaultdict
        by_signal: Dict[int, List[Any]] = defaultdict(list)
        for f, sid in zip(fill_events, signal_ids):
            by_signal[int(sid)].append(f)
        num_trades = len(by_signal)
        wins = 0
        for group in by_signal.values():
            pnl = 0.0
            for f in group:
                side = _get(f, "side")
                qty = float(_get(f, "quantity") or 0)
                price = float(_get(f, "price") or 0)
                comm = float(_get(f, "commission") or 0)
                if str(side).lower() == "buy":
                    pnl -= price * qty + comm
                else:
                    pnl += price * qty - comm
            if pnl > 0:
                wins += 1
        win_rate = wins / num_trades if num_trades else None
        return num_trades, win_rate
    num_trades = len(fill_events) // 2
    return num_trades, None


class PerformanceMetrics:
    """
    Step 5.1.8: Compute performance metrics from equity or returns and optional fills.

    All from historical data only; no look-ahead.
    """

    @staticmethod
    def compute(
        equity_series: Optional[Union[Sequence[float], np.ndarray, Sequence[Tuple[Any, float]]]] = None,
        returns_series: Optional[Union[Sequence[float], np.ndarray]] = None,
        fill_events: Optional[Sequence[Any]] = None,
        periods_per_year: float = 252.0,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute metrics. Provide either equity_series or returns_series (or both; equity takes precedence for return/drawdown).

        Returns dict with: sharpe_annual, sortino_annual, calmar, max_drawdown, total_return,
        num_trades, win_rate (win_rate may be None if fills lack signal_id).
        """
        if equity_series is not None:
            equity = _as_equity_array(equity_series)
            if returns_series is not None:
                returns = np.asarray(returns_series, dtype=float)
            else:
                returns = _equity_to_returns(equity)
            total_ret = total_return_from_equity(equity)
            max_dd = max_drawdown_from_equity(equity)
        elif returns_series is not None:
            returns = np.asarray(returns_series, dtype=float)
            returns = returns[np.isfinite(returns)]
            total_ret = total_return_from_returns(returns)
            # Reconstruct equity for drawdown: start at 1.0
            if len(returns) > 0:
                equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
                max_dd = max_drawdown_from_equity(equity)
            else:
                max_dd = 0.0
        else:
            return {
                "sharpe_annual": 0.0,
                "sortino_annual": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "num_trades": 0,
                "win_rate": None,
            }

        sharpe = sharpe_annual(returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
        sortino = sortino_annual(returns, periods_per_year=periods_per_year)
        calmar = calmar_ratio(total_ret, max_dd)

        num_trades = 0
        win_rate: Optional[float] = None
        if fill_events is not None:
            num_trades, win_rate = _trade_stats_from_fills(fill_events)

        return {
            "sharpe_annual": sharpe,
            "sortino_annual": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "total_return": total_ret,
            "num_trades": num_trades,
            "win_rate": win_rate,
        }
