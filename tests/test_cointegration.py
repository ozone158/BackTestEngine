"""
Tests for Step 2.1 — Cointegration pipeline.

2.1.7: Unit test with synthetic cointegrated series (is_cointegrated true)
       and synthetic non-cointegrated series (is_cointegrated false).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.alpha.cointegration import (
    CointegrationResult,
    load_pair_bars,
    persist_cointegration_result,
    run_cointegration_test,
    run_pair_cointegration,
)
from src.data.storage.reference import pair_id
from src.data.storage.schema import create_reference_tables, create_alpha_tables, get_engine, cointegration_results


def _make_aligned_df(close_a: np.ndarray, close_b: np.ndarray, length: int):
    """Build aligned_df with datetime, close_a, close_b."""
    dts = pd.date_range("2020-01-01", periods=length, freq="B", tz=timezone.utc)
    return pd.DataFrame({
        "datetime": dts,
        "close_a": close_a,
        "close_b": close_b,
    })


def test_synthetic_cointegrated_series():
    """
    2.1.7: y = x + noise, both I(1); spread is I(0) -> is_cointegrated true.
    Use small stationary noise so OLS beta ≈ 1 and residual (spread) is clearly I(0).
    """
    np.random.seed(42)
    n = 250
    # I(1): random walk
    x = np.cumsum(np.random.randn(n) * 0.01) + 100
    # y = x + small iid noise so var(z) << var(x) -> OLS beta ≈ 1, spread ≈ -z (I(0))
    z = np.random.randn(n) * 0.002
    y = x + z
    aligned = _make_aligned_df(x, y, n)
    test_ts = aligned["datetime"].iloc[-1]
    pid = "SYN_A_SYN_B"
    result = run_cointegration_test(aligned, pid, test_ts)
    assert result.pair_id == pid
    assert result.test_ts == test_ts
    assert result.adf_statistic is not None
    assert result.adf_pvalue is not None
    assert result.johansen_trace is not None
    assert result.johansen_pvalue is not None
    assert len(result.cointegrating_vector) == 2
    # Cointegrated: both ADF and Johansen should indicate cointegration (Step 2.1.7)
    assert result.is_cointegrated is True, (
        f"Expected cointegrated; adf_pvalue={result.adf_pvalue}, johansen_pvalue={result.johansen_pvalue}"
    )


def test_synthetic_non_cointegrated_series():
    """
    2.1.7: Two independent random walks -> not cointegrated -> is_cointegrated false.
    """
    np.random.seed(123)
    n = 120
    x = np.cumsum(np.random.randn(n) * 0.01) + 100
    y = np.cumsum(np.random.randn(n) * 0.01) + 100  # independent
    aligned = _make_aligned_df(x, y, n)
    test_ts = aligned["datetime"].iloc[-1]
    pid = "RW_A_RW_B"
    result = run_cointegration_test(aligned, pid, test_ts)
    assert result.pair_id == pid
    assert result.is_cointegrated is False, (
        f"Expected not cointegrated; adf_pvalue={result.adf_pvalue}, johansen_pvalue={result.johansen_pvalue}"
    )


def test_run_cointegration_test_returns_all_fields():
    """CointegrationResult has all fields required for persistence."""
    np.random.seed(1)
    n = 80
    x = np.cumsum(np.random.randn(n) * 0.01) + 100
    y = x + np.random.randn(n) * 0.05
    aligned = _make_aligned_df(x, y, n)
    test_ts = aligned["datetime"].iloc[-1]
    result = run_cointegration_test(aligned, "A_B", test_ts)
    assert isinstance(result, CointegrationResult)
    assert isinstance(result.adf_statistic, float)
    assert isinstance(result.adf_pvalue, float)
    assert isinstance(result.johansen_trace, float)
    assert isinstance(result.johansen_pvalue, float)
    assert isinstance(result.cointegrating_vector, list)
    assert isinstance(result.is_cointegrated, bool)


def test_persist_cointegration_result(tmp_path):
    """Persist and read back from cointegration_results."""
    from src.data.storage.schema import symbols, pair_universe

    engine = get_engine(f"sqlite:///{tmp_path / 'coint.db'}")
    create_reference_tables(engine)
    create_alpha_tables(engine)
    with engine.connect() as conn:
        conn.execute(symbols.insert().values(symbol_id="A"))
        conn.execute(symbols.insert().values(symbol_id="B"))
        conn.execute(
            pair_universe.insert().values(pair_id="A_B", symbol_a="A", symbol_b="B")
        )
        conn.commit()
    result = CointegrationResult(
        pair_id="A_B",
        test_ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
        adf_statistic=-3.5,
        adf_pvalue=0.03,
        johansen_trace=25.0,
        johansen_pvalue=0.05,
        cointegrating_vector=[1.0, -1.02],
        is_cointegrated=True,
    )
    persist_cointegration_result(result, engine)
    with engine.connect() as conn:
        row = conn.execute(
            cointegration_results.select().where(
                (cointegration_results.c.pair_id == "A_B")
                & (cointegration_results.c.test_ts == result.test_ts)
            )
        ).fetchone()
    assert row is not None
    assert row.pair_id == "A_B"
    assert row.adf_pvalue == 0.03
    assert row.is_cointegrated is True
    import json
    assert json.loads(row.cointegrating_vector) == [1.0, -1.02]


def test_load_pair_bars_insufficient_data_returns_none(tmp_path):
    """load_pair_bars returns (None, None) when no bars or too few."""
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 31, tzinfo=timezone.utc)
    aligned, test_ts = load_pair_bars(tmp_path, "X", "Y", start, end, min_obs=60)
    assert aligned is None
    assert test_ts is None
