"""
Tests for ingestion (Step 1.4): DataSource interface, Alpha Vantage (mock), CSV, symbol registration.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

try:
    import requests  # noqa: F401
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.data.ingestion import (
    CSVDataSource,
    AlphaVantageDataSource,
    DataSource,
    RAW_BAR_COLUMNS,
    normalize_bars_df,
    register_symbol,
)
from src.data.storage import get_engine, create_schema


def test_normalize_bars_df():
    """Normalize preserves schema and UTC."""
    df = pd.DataFrame({
        "date": ["2024-01-15", "2024-01-16"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1e6, 1.1e6],
    })
    out = normalize_bars_df(
        df,
        symbol="AAPL",
        datetime_column="date",
        open_column="open",
        high_column="high",
        low_column="low",
        close_column="close",
        volume_column="volume",
    )
    assert list(out.columns) == RAW_BAR_COLUMNS
    assert (out["symbol"] == "AAPL").all()
    assert out["datetime"].dt.tz is not None
    assert out["close"].tolist() == [101.0, 102.0]


def test_csv_data_source_single_file(tmp_path):
    """CSV DataSource: single file with symbol column; assert normalized schema and UTC."""
    path = tmp_path / "bars.csv"
    path.write_text(
        "symbol,date,open,high,low,close,volume\n"
        "AAPL,2024-01-15,100,102,99,101,1000000\n"
        "AAPL,2024-01-16,101,103,100,102,1100000\n"
    )
    source = CSVDataSource(file_path=path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    df = source.fetch("AAPL", start, end)
    assert list(df.columns) == RAW_BAR_COLUMNS
    assert len(df) == 2
    assert (df["symbol"] == "AAPL").all()
    assert df["datetime"].dt.tz is not None
    assert df["close"].tolist() == [101.0, 102.0]


def test_csv_data_source_per_symbol_file(tmp_path):
    """CSV DataSource: base_path + pattern per symbol."""
    (tmp_path / "AAPL.csv").write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-15,100,102,99,101,1000000\n"
    )
    source = CSVDataSource(base_path=tmp_path, pattern="{symbol}.csv")
    df = source.fetch("AAPL", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc))
    assert list(df.columns) == RAW_BAR_COLUMNS
    assert len(df) == 1
    assert df["close"].iloc[0] == 101.0


def test_csv_data_source_missing_file_returns_empty(tmp_path):
    """CSV DataSource: missing file returns empty DataFrame."""
    source = CSVDataSource(base_path=tmp_path, pattern="{symbol}.csv")
    df = source.fetch("NONEXISTENT", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc))
    assert df.empty
    assert list(df.columns) == RAW_BAR_COLUMNS


def test_alpha_vantage_no_apikey_raises():
    """Alpha Vantage without API key raises."""
    source = AlphaVantageDataSource(apikey="")
    with pytest.raises(ValueError, match="API key"):
        source.fetch("AAPL", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc))


@pytest.mark.skipif(not HAS_REQUESTS, reason="requests not installed")
@patch("src.data.ingestion.alpha_vantage.requests.get")
def test_alpha_vantage_fetch_normalized(mock_get):
    """Alpha Vantage fetch returns normalized schema (mock response)."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "Time Series (Daily)": {
            "2024-01-15": {"1. open": "100", "2. high": "102", "3. low": "99", "4. close": "101", "5. volume": "1000000"},
            "2024-01-16": {"1. open": "101", "2. high": "103", "3. low": "100", "4. close": "102", "5. volume": "1100000"},
        }
    }
    source = AlphaVantageDataSource(apikey="testkey")
    df = source.fetch("AAPL", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc))
    assert list(df.columns) == RAW_BAR_COLUMNS
    assert len(df) == 2
    assert (df["symbol"] == "AAPL").all()
    assert df["datetime"].dt.tz is not None
    assert df["close"].tolist() == [101.0, 102.0]


def test_register_symbol(db_url):
    """Symbol registration: insert if not exists."""
    engine = get_engine(db_url)
    create_schema(engine)
    register_symbol(engine, "AAPL", display_name="Apple Inc.", currency="USD")
    register_symbol(engine, "AAPL")  # no-op second time
    from sqlalchemy import select
    from src.data.storage.schema import symbols
    with engine.connect() as conn:
        row = conn.execute(select(symbols).where(symbols.c.symbol_id == "AAPL")).fetchone()
        assert row is not None
        assert row.display_name == "Apple Inc."
