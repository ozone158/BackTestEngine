"""
Tests for relational schema (Step 1.1).

- Schema creation against a fresh DB.
- FKs and constraints (e.g. pair_universe symbol_a < symbol_b).
- Optional: insert one row per table and read back (Step 1.1.6).
"""

from datetime import datetime, timezone

import pytest

from src.data.storage.schema import (
    ALPHA_TABLES,
    BACKTEST_TABLES,
    REFERENCE_TABLES,
    backtest_equity,
    backtest_fills,
    backtest_metrics,
    backtest_runs,
    backtest_signals,
    corporate_actions,
    cointegration_results,
    create_alpha_tables,
    create_backtest_tables,
    create_reference_tables,
    create_schema,
    drop_schema,
    get_engine,
    kalman_params,
    ou_params,
    pair_universe,
    symbols,
)


@pytest.fixture
def engine(db_url):
    e = get_engine(db_url)
    drop_schema(e)
    create_schema(e)
    yield e
    drop_schema(e)


def test_create_schema(engine):
    """Schema creation runs without error; all tables exist (Steps 1.1.1–1.1.4)."""
    from sqlalchemy import inspect

    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    expected = {
        "symbols",
        "corporate_actions",
        "pair_universe",
        "cointegration_results",
        "kalman_params",
        "ou_params",
        "backtest_runs",
        "backtest_signals",
        "backtest_fills",
        "backtest_equity",
        "backtest_metrics",
    }
    assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"


def test_validate_fresh_db_step_1_1_6(db_url):
    """Step 1.1.6: Run schema creation against a fresh DB; all tables exist."""
    from sqlalchemy import inspect

    engine = get_engine(db_url)
    drop_schema(engine)
    create_schema(engine)
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    required = {
        "symbols",
        "corporate_actions",
        "pair_universe",
        "cointegration_results",
        "kalman_params",
        "ou_params",
        "backtest_runs",
        "backtest_signals",
        "backtest_fills",
        "backtest_equity",
        "backtest_metrics",
    }
    assert required.issubset(table_names), f"Missing tables: {required - table_names}"
    drop_schema(engine)


def test_create_reference_tables_only(db_url):
    """Step 1.1.2: Only reference tables (symbols, corporate_actions, pair_universe) exist."""
    from sqlalchemy import inspect

    engine = get_engine(db_url)
    drop_schema(engine)
    create_reference_tables(engine)
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    reference_names = {t.name for t in REFERENCE_TABLES}
    assert reference_names.issubset(table_names), f"Missing reference tables: {reference_names - table_names}"
    # No alpha or backtest tables
    assert "cointegration_results" not in table_names
    assert "backtest_runs" not in table_names
    drop_schema(engine)


def test_create_alpha_tables_after_reference(db_url):
    """Step 1.1.3: Alpha tables exist after create_reference_tables + create_alpha_tables."""
    from sqlalchemy import inspect

    engine = get_engine(db_url)
    drop_schema(engine)
    create_reference_tables(engine)
    create_alpha_tables(engine)
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    alpha_names = {t.name for t in ALPHA_TABLES}
    assert alpha_names.issubset(table_names), f"Missing alpha tables: {alpha_names - table_names}"
    assert "backtest_runs" not in table_names
    drop_schema(engine)


def test_create_backtest_tables_after_reference(db_url):
    """Step 1.1.4: Backtest tables exist after create_reference_tables + create_backtest_tables."""
    from sqlalchemy import inspect

    engine = get_engine(db_url)
    drop_schema(engine)
    create_reference_tables(engine)
    create_backtest_tables(engine)
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    backtest_names = {t.name for t in BACKTEST_TABLES}
    assert backtest_names.issubset(table_names), f"Missing backtest tables: {backtest_names - table_names}"
    drop_schema(engine)


def test_indexes_exist(engine):
    """Step 1.1.5: Indexes per §7.2 exist after create_schema."""
    from sqlalchemy import inspect

    insp = inspect(engine)
    # Index names we create (dialect may prefix/suffix)
    index_names = set()
    for t in insp.get_table_names():
        for idx in insp.get_indexes(t):
            index_names.add(idx["name"])
    expected_ix = {
        "ix_corporate_actions_symbol_ex_date",
        "ix_pair_universe_symbol_a_b",
        "ix_cointegration_results_pair_test_ts",
        "ix_backtest_signals_run_ts",
        "ix_backtest_fills_run_ts",
        "ix_backtest_equity_run_ts",
        "ix_backtest_runs_start_end",
        "ix_backtest_runs_created_at",
    }
    assert expected_ix.issubset(index_names), f"Missing indexes: {expected_ix - index_names}"


def test_symbols_insert_read(engine):
    """Insert and read one row from symbols."""
    with engine.connect() as conn:
        conn.execute(
            symbols.insert().values(
                symbol_id="AAPL",
                display_name="Apple Inc.",
                asset_class="equity",
                currency="USD",
            )
        )
        conn.commit()
        row = conn.execute(symbols.select().where(symbols.c.symbol_id == "AAPL")).fetchone()
        assert row is not None
        assert row.symbol_id == "AAPL"
        assert row.asset_class == "equity"


def test_pair_universe_symbol_order_constraint(engine):
    """pair_universe enforces symbol_a < symbol_b."""
    with engine.connect() as conn:
        conn.execute(
            symbols.insert().values(
                symbol_id="AAPL",
                display_name="Apple",
                currency="USD",
            )
        )
        conn.execute(
            symbols.insert().values(
                symbol_id="MSFT",
                display_name="Microsoft",
                currency="USD",
            )
        )
        conn.commit()

        # Valid: AAPL < MSFT
        conn.execute(
            pair_universe.insert().values(
                pair_id="AAPL_MSFT",
                symbol_a="AAPL",
                symbol_b="MSFT",
            )
        )
        conn.commit()

        # Invalid: symbol_a > symbol_b should fail (CHECK constraint)
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            conn.execute(
                pair_universe.insert().values(
                    pair_id="MSFT_AAPL",
                    symbol_a="MSFT",
                    symbol_b="AAPL",
                )
            )
            conn.commit()


def test_foreign_keys(engine):
    """Step 1.1.6: Insert one row per table and read back; FKs and constraints behave as expected."""
    with engine.connect() as conn:
        conn.execute(
            symbols.insert().values(symbol_id="SPY", display_name="SPDR S&P 500", currency="USD")
        )
        conn.execute(
            symbols.insert().values(symbol_id="QQQ", display_name="Nasdaq 100", currency="USD")
        )
        conn.execute(
            pair_universe.insert().values(
                pair_id="SPY_QQQ",
                symbol_a="QQQ",
                symbol_b="SPY",
            )
        )
        conn.commit()

        # Insert corporate_actions for existing symbol
        conn.execute(
            corporate_actions.insert().values(
                symbol_id="SPY",
                action_type="dividend",
                ex_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                cash_amount=1.5,
            )
        )
        conn.commit()

        # Insert cointegration_results for existing pair
        conn.execute(
            cointegration_results.insert().values(
                pair_id="SPY_QQQ",
                test_ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
                adf_pvalue=0.03,
                johansen_pvalue=0.02,
                is_cointegrated=True,
            )
        )
        conn.commit()

        # Insert backtest_runs (pair_id nullable)
        conn.execute(
            backtest_runs.insert().values(
                run_id="bt_test_001",
                strategy_name="ou_spread",
                pair_id="SPY_QQQ",
                start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        )
        conn.commit()

        # Insert backtest_signals and backtest_fills
        r = conn.execute(
            backtest_signals.insert().values(
                run_id="bt_test_001",
                signal_ts=datetime(2024, 2, 1, tzinfo=timezone.utc),
                direction="long_spread",
                symbol_a="QQQ",
                symbol_b="SPY",
                hedge_ratio=1.02,
                size=10000.0,
            )
        )
        conn.commit()
        sig_id = r.inserted_primary_key[0]

        conn.execute(
            backtest_fills.insert().values(
                run_id="bt_test_001",
                signal_id=sig_id,
                fill_ts=datetime(2024, 2, 1, tzinfo=timezone.utc),
                symbol="QQQ",
                side="buy",
                quantity=100.0,
                price=400.0,
                commission=1.0,
                slippage_bps=2.0,
            )
        )
        conn.commit()

        conn.execute(
            backtest_equity.insert().values(
                run_id="bt_test_001",
                ts=datetime(2024, 2, 1, tzinfo=timezone.utc),
                equity=100000.0,
                cash=60000.0,
                positions_value=40000.0,
            )
        )
        conn.commit()

        conn.execute(
            backtest_metrics.insert().values(
                run_id="bt_test_001",
                sharpe_annual=1.2,
                total_return=0.05,
                num_trades=10,
                win_rate=0.6,
                computed_at=datetime.now(timezone.utc),
            )
        )
        conn.commit()

    # Read back: one row per table (round-trip)
    with engine.connect() as conn:
        assert conn.execute(symbols.select()).fetchall()
        assert conn.execute(corporate_actions.select()).fetchall()
        assert conn.execute(pair_universe.select()).fetchall()
        assert conn.execute(cointegration_results.select()).fetchall()
        assert conn.execute(backtest_runs.select()).fetchall()
        assert conn.execute(backtest_signals.select()).fetchall()
        assert conn.execute(backtest_fills.select()).fetchall()
        assert conn.execute(backtest_equity.select()).fetchall()
        assert conn.execute(backtest_metrics.select()).fetchall()
