"""
Tests for E2E validation sanity checks (Step 1.5.2).
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.validation import run_cross_check, run_sanity_checks


def _bars_df(symbols=None, dts=None, adj_factor=1.0):
    if symbols is None:
        symbols = ["AAPL", "AAPL", "AAPL"]
    if dts is None:
        dts = pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True)
    return pd.DataFrame({
        "symbol": symbols,
        "datetime": dts,
        "open": [100.0] * len(symbols),
        "high": [101.0] * len(symbols),
        "low": [99.0] * len(symbols),
        "close": [100.5] * len(symbols),
        "volume": [1e6] * len(symbols),
        "adj_factor": [adj_factor] * len(symbols),
        "outlier_flag": [0] * len(symbols),
    })


def test_sanity_checks_empty_df_passes():
    """Empty DataFrame is valid (no errors)."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(pd.DataFrame(), start, end)
    assert errors == []


def test_sanity_checks_valid_bars_pass():
    """Valid bars: no duplicates, ascending, in range, adj_factor > 0, OHLC >= 0."""
    df = _bars_df()
    start = datetime(2024, 1, 14, tzinfo=timezone.utc)
    end = datetime(2024, 1, 18, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert errors == []


def test_sanity_checks_duplicate_symbol_datetime():
    """Duplicate (symbol, datetime) fails."""
    df = _bars_df(
        symbols=["AAPL", "AAPL", "AAPL"],
        dts=pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-16"], utc=True),
    )
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("Duplicate" in e for e in errors)


def test_sanity_checks_datetime_not_strictly_increasing():
    """Datetime not strictly increasing per symbol fails."""
    df = _bars_df(
        dts=pd.to_datetime(["2024-01-16", "2024-01-15", "2024-01-17"], utc=True),
    )
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("strictly increasing" in e for e in errors)


def test_sanity_checks_future_data():
    """Bar datetime after requested end fails."""
    df = _bars_df(dts=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-02-01"], utc=True))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 20, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("after requested end" in e or "future" in e.lower() for e in errors)


def test_sanity_checks_before_start():
    """Bar datetime before requested start fails."""
    df = _bars_df(dts=pd.to_datetime(["2024-01-01", "2024-01-16", "2024-01-17"], utc=True))
    start = datetime(2024, 1, 10, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("before requested start" in e for e in errors)


def test_sanity_checks_adj_factor_zero_or_negative():
    """adj_factor <= 0 fails."""
    df = _bars_df(adj_factor=0.0)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("adj_factor" in e for e in errors)


def test_sanity_checks_ohlc_negative():
    """Negative OHLC fails."""
    df = _bars_df()
    df.loc[1, "close"] = -1.0
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("non-negative" in e or "close" in e for e in errors)


def test_sanity_checks_missing_columns():
    """Missing symbol or datetime returns errors."""
    df = pd.DataFrame({"open": [100.0], "close": [101.0]})
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    errors = run_sanity_checks(df, start, end)
    assert any("symbol" in e or "datetime" in e for e in errors)


def test_cross_check_valid_passes():
    """Cross-check on valid bars returns no issues."""
    df = _bars_df()
    issues = run_cross_check(df, sample_per_symbol=2)
    assert issues == []


def test_cross_check_high_less_than_low():
    """Cross-check flags high < low."""
    df = _bars_df()
    df.loc[0, "high"] = 98.0
    df.loc[0, "low"] = 99.0
    issues = run_cross_check(df, sample_per_symbol=3)
    assert any("high < low" in i for i in issues)


def test_cross_check_adj_factor_non_positive():
    """Cross-check flags adj_factor <= 0."""
    df = _bars_df(adj_factor=0.0)
    issues = run_cross_check(df, sample_per_symbol=2)
    assert any("adj_factor" in i for i in issues)
