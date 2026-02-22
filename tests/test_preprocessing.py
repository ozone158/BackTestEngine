"""
Tests for preprocessing (Step 1.3): corporate actions, interpolation, outliers.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.preprocessing import (
    CorporateAction,
    adjust,
    detect_outliers,
    interpolate,
    load_corporate_actions,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# 1.3.8: Mock corporate_actions + split
# ---------------------------------------------------------------------------

def test_adjust_split():
    """Unit test: mock corporate_actions; feed bars with one split; assert adj_factor and OHLC after split."""
    bars = pd.DataFrame({
        "symbol": ["AAPL"] * 4,
        "datetime": pd.to_datetime([
            "2024-01-10 10:00:00",
            "2024-01-11 10:00:00",
            "2024-01-16 10:00:00",  # on ex_date
            "2024-01-17 10:00:00",
        ], utc=True),
        "open": [100.0, 101.0, 50.0, 51.0],
        "high": [102.0, 103.0, 52.0, 53.0],
        "low": [99.0, 100.0, 49.0, 50.0],
        "close": [101.0, 102.0, 51.0, 52.0],
        "volume": [1e6, 1e6, 2e6, 2e6],
    })
    # 2:1 split on 2024-01-16 → ratio 2.0; bars on or after get OHLC * 2, volume / 2, adj_factor 2.0
    actions = [
        CorporateAction(
            ex_date=datetime(2024, 1, 16, tzinfo=timezone.utc),
            action_type="split",
            ratio=2.0,
            cash_amount=None,
        ),
    ]
    out = adjust(bars, "AAPL", actions=actions)
    assert (out.loc[out["datetime"] < "2024-01-16", "adj_factor"] == 1.0).all()
    assert (out.loc[out["datetime"] >= "2024-01-16", "adj_factor"] == 2.0).all()
    # Post-split bars (50, 51, 52, 53) * 2 = 100, 102, 104, 106; volume / 2
    assert out.iloc[2]["close"] == 102.0
    assert out.iloc[2]["open"] == 100.0
    assert out.iloc[2]["volume"] == 1e6


def test_adjust_dividend():
    """Test dividend case: subtract cash_amount from OHLC for bars on or after ex_date."""
    bars = pd.DataFrame({
        "symbol": ["AAPL"] * 3,
        "datetime": pd.to_datetime(["2024-01-10 10:00:00", "2024-01-15 10:00:00", "2024-01-20 10:00:00"], utc=True),
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1e6, 1e6, 1e6],
    })
    actions = [
        CorporateAction(
            ex_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            action_type="dividend",
            ratio=None,
            cash_amount=1.0,
        ),
    ]
    out = adjust(bars, "AAPL", actions=actions)
    # First bar unchanged
    assert out.iloc[0]["close"] == 101.0
    # Bars on or after ex_date: subtract 1
    assert out.iloc[1]["close"] == 101.0
    assert out.iloc[2]["close"] == 102.0


def test_interpolation_one_missing_bar():
    """Test interpolation on a series with one missing bar."""
    bars = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-15 11:00:00",
            "2024-01-15 12:00:00",
        ], utc=True),
        "close": [100.0, pd.NA, 104.0],
    })
    out = interpolate(bars, method="linear", max_gap=pd.Timedelta(hours=2))
    assert out["close"].iloc[1] == 102.0
    assert out["missing_filled"].iloc[1] == 1


def test_interpolation_gap_exceeds_max():
    """Gap larger than max_gap is not filled by linear; then forward_fill fills with 100."""
    bars = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-15 11:00:00",
            "2024-01-17 10:00:00",
        ], utc=True),
        "close": [100.0, pd.NA, 104.0],
    })
    out = interpolate(bars, method="linear", max_gap=pd.Timedelta(days=1), forward_fill_beyond=True)
    # Middle row: gap to next valid (01-17) is > 1 day, so linear does not fill; ffill gives 100
    assert out["close"].iloc[1] == 100.0


def test_detect_outliers_spike():
    """Test outlier flag on a spike (z-score)."""
    bars = pd.DataFrame({
        "datetime": pd.to_datetime([f"2024-01-{d:02d} 10:00:00" for d in range(1, 11)], utc=True),
        "close": [100.0, 101.0, 102.0, 101.0, 100.0, 150.0, 101.0, 102.0, 101.0, 100.0],
    })
    out = detect_outliers(bars, method="zscore_levels", threshold=3.0)
    assert out["outlier_flag"].sum() >= 1
    assert out.loc[out["close"] == 150.0, "outlier_flag"].iloc[0] == 1


def test_load_corporate_actions_mock():
    """load_corporate_actions with actions= returns sorted list."""
    actions = [
        CorporateAction(datetime(2024, 1, 20, tzinfo=timezone.utc), "split", 2.0, None),
        CorporateAction(datetime(2024, 1, 10, tzinfo=timezone.utc), "dividend", None, 0.5),
    ]
    out = load_corporate_actions("AAPL", actions=actions)
    assert len(out) == 2
    assert out[0].ex_date <= out[1].ex_date
    assert out[0].action_type == "dividend"


def test_run_pipeline():
    """Pipeline: adjust → interpolate → detect_outliers produces adj_factor and outlier_flag."""
    bars = pd.DataFrame({
        "symbol": ["AAPL"] * 3,
        "datetime": pd.to_datetime(["2024-01-15 10:00:00", "2024-01-16 10:00:00", "2024-01-17 10:00:00"], utc=True),
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1e6, 1e6, 1e6],
    })
    out = run_pipeline(bars, "AAPL", actions=[])
    assert "adj_factor" in out.columns
    assert "outlier_flag" in out.columns
    assert (out["adj_factor"] == 1.0).all()
    assert out["outlier_flag"].isin([0, 1]).all()
