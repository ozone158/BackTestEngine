"""
Tests for Step 5.3 — Position sizing (Kelly / Risk Parity).

5.3.6: Kelly with known edge/odds; assert fraction in [0, max_fraction].
       Risk parity with [0.1, 0.2] vol; assert weights sum to 1 and lower vol gets higher weight.
"""

import numpy as np
import pytest

from src.backtest.position_sizing import PositionSizer, kelly, risk_parity


def test_kelly_known_edge_odds():
    """5.3.6: Kelly with known edge/odds; fraction in [0, max_fraction]."""
    # edge p=0.6, odds b=2 -> f* = 0.6 - (1-0.6)/2 = 0.6 - 0.2 = 0.4
    f = kelly(edge=0.6, odds=2.0, fraction=1.0, max_fraction=0.5)
    assert abs(f - 0.4) < 1e-9
    assert 0 <= f <= 0.5

    f_capped = kelly(edge=0.6, odds=2.0, fraction=1.0, max_fraction=0.25)
    assert f_capped == 0.25
    assert 0 <= f_capped <= 0.25


def test_kelly_half_kelly():
    """5.3.2: Half-Kelly (fraction=0.5)."""
    # Full Kelly 0.4, half -> 0.2
    f = kelly(edge=0.6, odds=2.0, fraction=0.5, max_fraction=0.5)
    assert abs(f - 0.2) < 1e-9


def test_kelly_zero_or_negative_returns_zero():
    """Kelly returns 0 when edge<=0, edge>=1, or odds<=0."""
    assert kelly(0.0, 2.0) == 0.0
    assert kelly(1.0, 2.0) == 0.0
    assert kelly(0.6, 0.0) == 0.0
    assert kelly(0.6, -1.0) == 0.0


def test_kelly_negative_f_star_clamped_to_zero():
    """When f* < 0 (unfavorable edge), return 0."""
    # p=0.3, b=2 -> f* = 0.3 - 0.7/2 = 0.3 - 0.35 = -0.05
    assert kelly(edge=0.3, odds=2.0, max_fraction=0.5) == 0.0


def test_risk_parity_two_legs():
    """5.3.6: Risk parity with [0.1, 0.2] vol; weights sum to 1; lower vol gets higher weight."""
    w = risk_parity([0.1, 0.2])
    assert len(w) == 2
    assert abs(w.sum() - 1.0) < 1e-9
    # w1 = (1/0.1)/(1/0.1+1/0.2) = 10/15 = 2/3, w2 = 1/3
    assert abs(w[0] - 2.0 / 3.0) < 1e-9
    assert abs(w[1] - 1.0 / 3.0) < 1e-9
    assert w[0] > w[1]


def test_risk_parity_weights_sum_to_one():
    """Weights always sum to 1 for any length."""
    w = risk_parity([0.05, 0.1, 0.15])
    assert abs(w.sum() - 1.0) < 1e-9
    assert len(w) == 3


def test_risk_parity_position_sizer():
    """PositionSizer.risk_parity matches module-level risk_parity."""
    vol = [0.1, 0.2]
    np.testing.assert_array_almost_equal(
        PositionSizer.risk_parity(vol),
        risk_parity(vol),
    )


def test_position_sizer_kelly():
    """PositionSizer.kelly matches module-level kelly."""
    assert PositionSizer.kelly(0.6, 2.0, max_fraction=0.5) == pytest.approx(kelly(0.6, 2.0, max_fraction=0.5))


def test_risk_parity_zero_volatility_handled():
    """If one volatility is 0, use safe inverse (avoid div by zero)."""
    w = risk_parity([0.1, 0.0])
    assert len(w) == 2
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9
