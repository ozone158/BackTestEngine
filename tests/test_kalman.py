"""
Tests for Step 2.2 — Kalman filter (Python).

2.2.8: Unit test with fixed (price_a, price_b) series; compare β and spread to OLS;
       output length equals input length; no NaNs after warm-up.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.alpha.kalman import (
    KalmanHedgeRatio,
    KalmanState,
    run_kalman_on_aligned,
    persist_kalman_params,
    load_kalman_params,
    run_pair_kalman,
)
from src.data.storage.parquet_spreads import read_spreads
from src.data.storage.reference import pair_id


def _make_aligned_df(close_a: np.ndarray, close_b: np.ndarray, length: int) -> pd.DataFrame:
    """Build aligned_df with datetime, close_a, close_b."""
    dts = pd.date_range("2020-01-01", periods=length, freq="B", tz=timezone.utc)
    return pd.DataFrame({
        "datetime": dts,
        "close_a": close_a,
        "close_b": close_b,
    })


def test_kalman_output_length_equals_input():
    """Output length equals input length (2.2.8)."""
    np.random.seed(42)
    n = 80
    b_true = 1.5
    close_b = np.cumsum(np.random.randn(n) * 0.01) + 100
    close_a = b_true * close_b + np.random.randn(n) * 0.1
    aligned = _make_aligned_df(close_a, close_b, n)
    df, _ = run_kalman_on_aligned(aligned, warm_up=15)
    assert len(df) == n
    assert list(df.columns) >= ["datetime", "spread", "beta", "z_score"]


def test_kalman_no_nans_after_warm_up():
    """No NaNs after warm-up (2.2.8)."""
    np.random.seed(43)
    n = 100
    warm_up = 20
    close_b = np.cumsum(np.random.randn(n) * 0.01) + 100
    close_a = 1.3 * close_b + np.random.randn(n) * 0.05
    aligned = _make_aligned_df(close_a, close_b, n)
    df, _ = run_kalman_on_aligned(aligned, warm_up=warm_up)
    after = df.slice(warm_up, df.height - warm_up)
    assert not after["beta"].is_nan().any(), "beta should have no NaN after warm-up"
    assert not after["spread"].is_nan().any(), "spread should have no NaN after warm-up"


def test_kalman_beta_converges_to_ols_reference():
    """Fixed (price_a, price_b) series; Kalman β converges toward OLS β (2.2.8)."""
    np.random.seed(44)
    n = 120
    b_true = 1.5
    close_b = np.cumsum(np.random.randn(n) * 0.01) + 100
    close_a = b_true * close_b + np.random.randn(n) * 0.2
    aligned = _make_aligned_df(close_a, close_b, n)
    ols_beta = np.dot(close_a, close_b) / (np.dot(close_b, close_b) + 1e-20)
    df, _ = run_kalman_on_aligned(aligned, warm_up=30)
    # Last beta should be close to OLS over full sample
    last_beta = float(df["beta"][-1])
    assert abs(last_beta - ols_beta) < 0.15, (
        f"Kalman beta {last_beta} should be close to OLS beta {ols_beta}"
    )
    # Spread at end: price_a - beta*price_b should be small (residual)
    last_spread = float(df["spread"][-1])
    expected_spread = close_a[-1] - last_beta * close_b[-1]
    assert abs(last_spread - expected_spread) < 1e-6


def test_kalman_hedge_ratio_update():
    """KalmanHedgeRatio.update produces β and spread recursively; compare to OLS spread."""
    np.random.seed(45)
    n = 50
    close_b = np.cumsum(np.random.randn(n) * 0.01) + 100
    close_a = 1.2 * close_b + np.random.randn(n) * 0.1
    kf = KalmanHedgeRatio(process_noise=1e-5, measurement_noise=1e-4)
    betas, spreads = [], []
    for i in range(n):
        b, s = kf.update(float(close_a[i]), float(close_b[i]))
        betas.append(b)
        spreads.append(s)
    ols_beta = np.dot(close_a, close_b) / (np.dot(close_b, close_b) + 1e-20)
    assert abs(betas[-1] - ols_beta) < 0.2
    assert abs(spreads[-1] - (close_a[-1] - betas[-1] * close_b[-1])) < 1e-6


def test_persist_and_load_kalman_params(tmp_path):
    """Persist Kalman params and load back (round-trip)."""
    from src.data.storage.schema import create_reference_tables, create_alpha_tables, get_engine
    from src.data.storage.schema import symbols, pair_universe

    engine = get_engine(f"sqlite:///{tmp_path / 'kalman.db'}")
    create_reference_tables(engine)
    create_alpha_tables(engine)
    with engine.connect() as conn:
        conn.execute(symbols.insert().values(symbol_id="A"))
        conn.execute(symbols.insert().values(symbol_id="B"))
        conn.execute(
            pair_universe.insert().values(pair_id="A_B", symbol_a="A", symbol_b="B")
        )
        conn.commit()

    state = KalmanState(beta=1.25, P=0.001)
    persist_kalman_params(
        "A_B",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        state,
        1e-6,
        1e-4,
        engine,
    )
    loaded = load_kalman_params(engine, "A_B")
    assert loaded is not None
    loaded_state, q, r = loaded
    assert abs(loaded_state.beta - 1.25) < 1e-6
    assert abs(loaded_state.P - 0.001) < 1e-6
    assert q == 1e-6
    assert r == 1e-4


def test_run_pair_kalman_writes_parquet_and_returns_df(tmp_path):
    """run_pair_kalman with bars on disk: returns DataFrame and writes Parquet."""
    from src.data.storage import get_engine, write_bars

    # Create ~70 days of daily bars for A and B so we have enough for min_obs=60
    n = 70
    dts = pd.date_range("2020-01-01", periods=n, freq="B", tz=timezone.utc)
    np.random.seed(46)
    close_b = np.cumsum(np.random.randn(n) * 0.01) + 100
    close_a = 1.1 * close_b + np.random.randn(n) * 0.1
    a, b = "SYM_A", "SYM_B"
    bars_a = pd.DataFrame({
        "symbol": [a] * n,
        "datetime": dts,
        "open": close_a - 0.1, "high": close_a + 0.1, "low": close_a - 0.1, "close": close_a,
        "volume": 1e6, "adj_factor": 1.0, "outlier_flag": 0,
    })
    bars_b = pd.DataFrame({
        "symbol": [b] * n,
        "datetime": dts,
        "open": close_b - 0.1, "high": close_b + 0.1, "low": close_b - 0.1, "close": close_b,
        "volume": 1e6, "adj_factor": 1.0, "outlier_flag": 0,
    })
    bars = pd.concat([bars_a, bars_b], ignore_index=True)
    for part_date, grp in bars.groupby(bars["datetime"].dt.date):
        part_key = {"source": "csv", "date": part_date.isoformat()}
        write_bars(tmp_path, part_key, grp)

    start = dts[0].to_pydatetime()
    end = dts[-1].to_pydatetime()
    df = run_pair_kalman(
        tmp_path,
        a,
        b,
        start,
        end,
        write_parquet=True,
        persist_params=False,
        min_obs=60,
    )
    assert df is not None
    assert len(df) == n
    pid = pair_id(a, b)
    read_back = read_spreads(tmp_path, pid)
    assert len(read_back) == n
    assert "spread" in read_back.columns and "beta" in read_back.columns
