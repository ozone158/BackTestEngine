"""
Step 5.3 — Position sizing (Kelly / Risk Parity).

Kelly criterion: fraction of capital from edge (win probability) and odds (win/loss ratio).
Half-Kelly or fractional Kelly via configurable fraction. Cap at max_fraction to avoid over-betting.

Risk parity: weights inversely proportional to volatility so each leg contributes equally to risk.
Optional wiring to Strategy/Portfolio in a follow-up (5.3.5).
"""

from __future__ import annotations

from typing import Union

import numpy as np


def kelly(
    edge: float,
    odds: float,
    fraction: float = 1.0,
    max_fraction: float = 0.25,
) -> float:
    """
    5.3.1–5.3.2: Kelly fraction of capital to allocate.

    edge: Win probability p (0 <= p <= 1).
    odds: Win/loss ratio b (e.g. you receive b times stake if you win, lose stake if you lose).
    Kelly f* = p - (1-p)/b = (p*b - q)/b with q=1-p.
    fraction: Multiply full Kelly by this (e.g. 0.5 for half-Kelly). Default 1.0.
    max_fraction: Cap output at this (e.g. 0.25). Default 0.25.

    Returns fraction in [0, max_fraction].
    """
    if edge <= 0 or edge >= 1 or odds <= 0:
        return 0.0
    # f* = edge - (1-edge)/odds
    f_star = edge - (1.0 - edge) / odds
    f_star = max(0.0, f_star)
    f = fraction * f_star
    return float(min(max(0.0, f), max_fraction))


def risk_parity(
    volatilities: Union[list, np.ndarray],
) -> np.ndarray:
    """
    5.3.3: Weights inversely proportional to volatility (each leg contributes equally to risk).

    w_i = (1/sigma_i) / sum(1/sigma_j). Weights sum to 1.
    If any sigma <= 0, that asset gets weight 0 (or use small epsilon); remaining weights renormalized.
    """
    vol = np.asarray(volatilities, dtype=float).ravel()
    n = len(vol)
    if n == 0:
        return np.array([])
    # Avoid division by zero: use max(sigma, 1e-12) or similar
    vol_safe = np.where(vol > 0, vol, np.nan)
    inv = np.where(np.isfinite(vol_safe), 1.0 / vol_safe, 0.0)
    total = np.sum(inv)
    if total <= 0:
        return np.ones(n) / n
    weights = inv / total
    return np.asarray(weights, dtype=float)


class PositionSizer:
    """
    5.3.4: API for position sizing.

    Kelly: kelly(edge, odds, fraction=1.0, max_fraction=0.25) -> float.
    Risk parity: risk_parity(volatilities) -> array of weights summing to 1.
    """

    @staticmethod
    def kelly(
        edge: float,
        odds: float,
        fraction: float = 1.0,
        max_fraction: float = 0.25,
    ) -> float:
        """Kelly fraction of capital. See module-level kelly()."""
        return kelly(edge, odds, fraction=fraction, max_fraction=max_fraction)

    @staticmethod
    def risk_parity(volatilities: Union[list, np.ndarray]) -> np.ndarray:
        """Risk-parity weights. See module-level risk_parity()."""
        return risk_parity(volatilities)
