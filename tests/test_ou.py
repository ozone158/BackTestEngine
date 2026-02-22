"""
Tests for Step 2.3 — OU process and thresholds.

2.3.7: Unit test with synthetic OU series (known θ, μ, σ); fit and assert parameters close;
       test threshold computation; persist round-trip.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.data.alpha.ou import (
    OUModel,
    OUParams,
    fit_ou,
    load_ou_params,
    persist_ou_params,
    run_pair_ou,
    write_ou_params_parquet,
    DEFAULT_ENTRY_K,
)


def _simulate_ou(n: int, theta: float, mu: float, sigma: float, dt: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate discrete OU series: X_t = μ + φ(X_{t-1} - μ) + ε_t, φ = exp(-θ*dt)."""
    np.random.seed(seed)
    phi = np.exp(-theta * dt)
    var_eps = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta) if theta > 0 else sigma ** 2
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + np.sqrt(var_eps) * np.random.randn()
    return x


def test_synthetic_ou_fit_close_to_true():
    """2.3.7: Synthetic OU series with known θ, μ, σ; fit and assert parameters close."""
    theta_true, mu_true, sigma_true = 0.1, 0.0, 0.5
    n = 500
    spread = _simulate_ou(n, theta_true, mu_true, sigma_true)
    params = fit_ou(spread, entry_k=2.0)
    assert abs(params.mu - mu_true) < 0.15
    assert params.theta > 0
    assert abs(params.theta - theta_true) < 0.1
    assert abs(params.sigma - sigma_true) < 0.3


def test_ou_threshold_computation():
    """2.3.7: Test threshold computation: entry_upper = μ + k*σ, entry_lower = μ - k*σ."""
    spread = _simulate_ou(200, 0.2, 1.0, 0.3, seed=1)
    k = 2.0
    params = fit_ou(spread, entry_k=k)
    assert params.entry_upper == params.mu + k * params.sigma
    assert params.entry_lower == params.mu - k * params.sigma
    assert params.exit_threshold == params.mu


def test_ou_model_api():
    """OUModel.fit() and entry_exit_thresholds() return correct tuple."""
    model = OUModel(entry_k=2.0)
    spread = _simulate_ou(100, 0.15, 0.0, 0.4)
    fitted = model.fit(spread)
    assert fitted is not None
    e_upper, e_lower, exit_t = model.entry_exit_thresholds()
    assert e_upper == fitted.entry_upper
    assert e_lower == fitted.entry_lower
    assert exit_t == fitted.exit_threshold


def test_ou_model_entry_exit_before_fit_raises():
    """entry_exit_thresholds() before fit() raises."""
    model = OUModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.entry_exit_thresholds()


def test_fit_ou_insufficient_observations_raises():
    """fit_ou with fewer than 2 points raises."""
    with pytest.raises(ValueError, match="at least 2"):
        fit_ou(np.array([1.0]))


def test_persist_and_load_ou_params(tmp_path):
    """Persist OU params and load back (round-trip)."""
    from src.data.storage.schema import create_reference_tables, create_alpha_tables, get_engine
    from src.data.storage.schema import symbols, pair_universe

    engine = get_engine(f"sqlite:///{tmp_path / 'ou.db'}")
    create_reference_tables(engine)
    create_alpha_tables(engine)
    with engine.connect() as conn:
        conn.execute(symbols.insert().values(symbol_id="A"))
        conn.execute(symbols.insert().values(symbol_id="B"))
        conn.execute(
            pair_universe.insert().values(pair_id="A_B", symbol_a="A", symbol_b="B")
        )
        conn.commit()

    params = OUParams(
        theta=0.1,
        mu=0.0,
        sigma=0.5,
        entry_upper=1.0,
        entry_lower=-1.0,
        exit_threshold=0.0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    persist_ou_params("A_B", params, engine)
    loaded = load_ou_params(engine, "A_B")
    assert loaded is not None
    assert loaded.theta == 0.1
    assert loaded.mu == 0.0
    assert loaded.entry_upper == 1.0
    assert loaded.entry_lower == -1.0


def test_run_pair_ou_persist_and_parquet(tmp_path):
    """run_pair_ou with persist and write_parquet."""
    from src.data.storage.schema import create_reference_tables, create_alpha_tables, get_engine
    from src.data.storage.schema import symbols, pair_universe

    engine = get_engine(f"sqlite:///{tmp_path / 'ou2.db'}")
    create_reference_tables(engine)
    create_alpha_tables(engine)
    with engine.connect() as conn:
        conn.execute(symbols.insert().values(symbol_id="X"))
        conn.execute(symbols.insert().values(symbol_id="Y"))
        conn.execute(
            pair_universe.insert().values(pair_id="X_Y", symbol_a="X", symbol_b="Y")
        )
        conn.commit()

    spread = _simulate_ou(80, 0.12, 0.5, 0.4, seed=99)
    win_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    win_end = datetime(2024, 3, 1, tzinfo=timezone.utc)
    params = run_pair_ou(
        spread,
        "X_Y",
        window_start=win_start,
        window_end=win_end,
        engine=engine,
        root_path=tmp_path,
        persist=True,
        write_parquet=True,
    )
    assert params.theta > 0
    assert params.window_start == win_start
    assert params.window_end == win_end
    loaded = load_ou_params(engine, "X_Y")
    assert loaded is not None
    assert (tmp_path / "alpha" / "ou" / "pair_id=X_Y" / "params.parquet").exists()
