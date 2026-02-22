"""
Tests for Parquet bars writer and reader (Step 1.2).
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from src.data.storage.parquet_bars import (
    BAR_COLUMNS,
    BAR_SCHEMA,
    OrderingViolationError,
    BarValidationError,
    partition_path,
    read_bars,
    write_bars,
)


@pytest.fixture
def root_path(tmp_path):
    return tmp_path


@pytest.fixture
def sample_bars():
    """One symbol, a few bars, ascending datetime."""
    return pd.DataFrame({
        "symbol": ["AAPL"] * 3,
        "datetime": pd.to_datetime([
            "2024-01-15 14:30:00",
            "2024-01-15 15:00:00",
            "2024-01-15 15:30:00",
        ], utc=True),
        "open": [188.0, 188.5, 189.0],
        "high": [189.0, 189.5, 190.0],
        "low": [187.5, 188.0, 188.5],
        "close": [188.8, 189.2, 189.8],
        "volume": [1e6, 1.1e6, 0.9e6],
        "adj_factor": [1.0, 1.0, 1.0],
        "outlier_flag": [0, 0, 0],
    })


def test_bar_schema_defined():
    """1.2.1: Bar schema has required columns and types."""
    assert "symbol" in BAR_COLUMNS
    assert "datetime" in BAR_COLUMNS
    assert "open" in BAR_COLUMNS
    assert "close" in BAR_COLUMNS
    assert "volume" in BAR_COLUMNS
    assert "adj_factor" in BAR_COLUMNS
    assert "outlier_flag" in BAR_COLUMNS
    assert BAR_SCHEMA.field("datetime").type == pa.timestamp("us", tz="UTC")
    assert BAR_SCHEMA.field("open").type == pa.float64()


def test_partition_path(root_path):
    """1.2.2: Partition path = root/bars/source=X/date=YYYY-MM-DD/."""
    p = partition_path(root_path, {"source": "csv", "date": "2024-01-15"})
    assert p == root_path / "bars" / "source=csv" / "date=2024-01-15"


def test_write_read_one_symbol(root_path, sample_bars):
    """1.2.6: Write a few bars for one symbol, read back, assert equality and order."""
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, sample_bars)
    df = read_bars(
        root_path,
        symbols=["AAPL"],
        start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc),
        source="csv",
    )
    assert len(df) == 3
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        sample_bars.reset_index(drop=True),
        check_dtype=False,
    )
    assert df["datetime"].is_monotonic_increasing
    assert (df["symbol"] == "AAPL").all()


def test_multi_symbol_multi_partition(root_path):
    """1.2.6: Multi-symbol, multi-partition read."""
    bars_1 = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "datetime": pd.to_datetime(["2024-01-15 10:00:00", "2024-01-15 10:00:00"], utc=True),
        "open": [188.0, 380.0],
        "high": [189.0, 381.0],
        "low": [187.0, 379.0],
        "close": [188.5, 380.5],
        "volume": [1e6, 2e6],
        "adj_factor": [1.0, 1.0],
        "outlier_flag": [0, 0],
    })
    bars_2 = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "datetime": pd.to_datetime(["2024-01-16 10:00:00", "2024-01-16 10:00:00"], utc=True),
        "open": [189.0, 381.0],
        "high": [190.0, 382.0],
        "low": [188.0, 380.0],
        "close": [189.5, 381.5],
        "volume": [1.1e6, 2.1e6],
        "adj_factor": [1.0, 1.0],
        "outlier_flag": [0, 0],
    })
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, bars_1)
    write_bars(root_path, {"source": "csv", "date": "2024-01-16"}, bars_2)
    df = read_bars(
        root_path,
        symbols=["AAPL", "MSFT"],
        start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 16, 23, 59, 59, tzinfo=timezone.utc),
        source="csv",
    )
    assert len(df) == 4
    assert set(df["symbol"]) == {"AAPL", "MSFT"}
    # Sorted by (symbol, datetime); datetime is monotonic per symbol, not globally
    assert list(df.sort_values(["symbol", "datetime"])["close"]) == [188.5, 189.5, 380.5, 381.5]


def test_read_empty_range(root_path, sample_bars):
    """1.2.6: Reading with end < min(datetime) returns empty."""
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, sample_bars)
    df = read_bars(
        root_path,
        symbols=["AAPL"],
        start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 10, 23, 59, 59, tzinfo=timezone.utc),
        source="csv",
    )
    assert df.empty
    assert list(df.columns) == BAR_COLUMNS


def test_read_missing_partition_returns_empty(root_path):
    """1.2.5: Missing partitions → empty DataFrame (no error)."""
    df = read_bars(
        root_path,
        symbols=["AAPL"],
        start=datetime(2024, 1, 15, tzinfo=timezone.utc),
        end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        source="csv",
    )
    assert df.empty
    assert list(df.columns) == BAR_COLUMNS


def test_read_column_subset(root_path, sample_bars):
    """1.2.4: Support reading subset of columns (e.g. symbol, datetime, close)."""
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, sample_bars)
    df = read_bars(
        root_path,
        symbols=["AAPL"],
        start=datetime(2024, 1, 15, tzinfo=timezone.utc),
        end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        source="csv",
        columns=["symbol", "datetime", "close"],
    )
    assert list(df.columns) == ["symbol", "datetime", "close"]
    assert len(df) == 3


def test_write_rejects_duplicate_symbol_datetime(root_path, sample_bars):
    """Write path rejects (symbol, datetime) duplicates."""
    bad = pd.concat([sample_bars, sample_bars.iloc[:1]], ignore_index=True)
    with pytest.raises(BarValidationError, match="unique"):
        write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, bad)


def test_write_rejects_non_ascending(root_path):
    """Write path rejects non-ascending datetime per symbol."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "datetime": pd.to_datetime(["2024-01-15 15:00:00", "2024-01-15 14:30:00"], utc=True),
        "open": [189.0, 188.0],
        "high": [190.0, 189.0],
        "low": [188.0, 187.0],
        "close": [189.5, 188.5],
        "volume": [1e6, 1e6],
        "adj_factor": [1.0, 1.0],
        "outlier_flag": [0, 0],
    })
    with pytest.raises(BarValidationError, match="ascending"):
        write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, df)


def test_write_rejects_ordering_violation(root_path, sample_bars):
    """Reject appending older bars after newer (reject_ordering_violations)."""
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, sample_bars)
    # One bar with older datetime (14:00) than existing (14:30, 15:00, 15:30); must be unique (symbol, datetime)
    older = pd.DataFrame({
        "symbol": ["AAPL"],
        "datetime": pd.to_datetime(["2024-01-15 14:00:00"], utc=True),
        "open": [187.0],
        "high": [188.0],
        "low": [186.5],
        "close": [187.8],
        "volume": [1e6],
        "adj_factor": [1.0],
        "outlier_flag": [0],
    })
    with pytest.raises(OrderingViolationError, match="append"):
        write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, older)


def test_read_empty_symbols_returns_empty(root_path, sample_bars):
    """Empty symbols list returns empty DataFrame."""
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, sample_bars)
    df = read_bars(
        root_path,
        symbols=[],
        start=datetime(2024, 1, 15, tzinfo=timezone.utc),
        end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        source="csv",
    )
    assert df.empty


def test_adj_factor_default(root_path):
    """Bars without adj_factor get default 1.0."""
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "datetime": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
        "open": [188.0],
        "high": [189.0],
        "low": [187.0],
        "close": [188.5],
        "volume": [1e6],
        "outlier_flag": [0],
    })
    write_bars(root_path, {"source": "csv", "date": "2024-01-15"}, df)
    out = read_bars(
        root_path, ["AAPL"],
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 16, tzinfo=timezone.utc),
        source="csv",
    )
    assert (out["adj_factor"] == 1.0).all()
