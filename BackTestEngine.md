# BackTestEngine — Project State & Conventions

## DB engine (Step 1.1.1)

- **Choice:** Single schema runs against **SQLite** (default) or **PostgreSQL**.
- **SQLite:** Used for single-user/dev; zero setup; default URL `sqlite:///data/backtest.db`.
- **PostgreSQL:** Use for multi-process or production; set `DATABASE_URL=postgresql://user:pass@host/db`.
- **Schema module:** `src/data/storage/schema.py` — one SQLAlchemy MetaData definition; `create_schema(engine)` creates all tables. Run with:
  - `python -m src.data.storage.schema` (uses `DATABASE_URL` or default SQLite)
  - Or: `from src.data.storage import get_engine, create_schema; create_schema(get_engine())`

## Alpha summary tables (Step 1.1.3)

Tables: `cointegration_results`, `kalman_params`, `ou_params`. Use `create_alpha_tables(engine)` to create only these (after reference tables). Group: `ALPHA_TABLES`.

## Backtest run tables (Step 1.1.4)

Tables: `backtest_runs`, `backtest_signals`, `backtest_fills`, `backtest_equity`, `backtest_metrics`. Use `create_backtest_tables(engine)` to create only these. Group: `BACKTEST_TABLES`.

## Indexes (Step 1.1.5)

Per system_database.md §7.2: `corporate_actions(symbol_id, ex_date)`; `pair_universe(symbol_a, symbol_b)`; `cointegration_results(pair_id, test_ts)`; `backtest_signals(run_id, signal_ts)`; `backtest_fills(run_id, fill_ts)`; `backtest_equity(run_id, ts)`; `backtest_runs(start_ts, end_ts)` and `backtest_runs(created_at)`. PKs cover `symbols(symbol_id)` and `pair_universe(pair_id)`.

## Validation (Step 1.1.6)

Run schema creation against a fresh DB; verify FKs and constraints. Tests: `tests/test_schema.py` — `test_validate_fresh_db_step_1_1_6`, `test_foreign_keys` (insert one row per table and read back), `test_pair_universe_symbol_order_constraint`, `test_indexes_exist`.

## Relational schema (from system_database.md)

All tables live in the same schema module; creation order is handled by SQLAlchemy (dependency order).

| Table | Purpose |
|-------|--------|
| `symbols` | symbol_id (PK), display_name, asset_class, exchange, currency, created_at, updated_at |
| `corporate_actions` | id (PK), symbol_id (FK), action_type, ex_date, recorded_at, ratio, cash_amount, metadata_json |
| `pair_universe` | pair_id (PK), symbol_a (FK), symbol_b (FK), created_at, notes; CHECK(symbol_a < symbol_b) |
| `cointegration_results` | id (PK), pair_id (FK), test_ts, adf_*, johansen_*, cointegrating_vector, is_cointegrated, created_at; UNIQUE(pair_id, test_ts) |
| `kalman_params` | id (PK), pair_id (FK), window_start/end, initial_state, process_noise, measurement_noise, created_at |
| `ou_params` | id (PK), pair_id (FK), window_*, theta, mu, sigma, entry_upper/lower, exit_threshold, created_at |
| `backtest_runs` | run_id (PK), strategy_name, pair_id (FK nullable), start_ts, end_ts, config_json, created_at |
| `backtest_signals` | id (PK), run_id (FK), signal_ts, direction, symbol_a, symbol_b, hedge_ratio, size, metadata_json |
| `backtest_fills` | id (PK), run_id (FK), signal_id (FK nullable), fill_ts, symbol, side, quantity, price, commission, slippage_bps |
| `backtest_equity` | run_id (FK), ts, equity, cash, positions_value; UNIQUE(run_id, ts) |
| `backtest_metrics` | run_id (PK/FK), sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate, alpha, beta, metrics_json, computed_at |

Conventions: timestamps UTC; symbol_id uppercase; pair_id = symbol_a_symbol_b with symbol_a < symbol_b; run_id UUID or bt_date_seq.

## Parquet bars (Step 1.2)

- **Bar schema (1.2.1):** `src/data/storage/parquet_bars.py` — `BAR_SCHEMA` (PyArrow): symbol (string), datetime (timestamp[us], UTC), open, high, low, close, volume (float64), adj_factor (float64, default 1.0), outlier_flag (int8). Bar datetime = bar open time; all UTC.
- **Partition key (1.2.2):** `(source, date)` → `root_path/bars/source={source}/date=YYYY-MM-DD/`. Date format YYYY-MM-DD.
- **Write:** `write_bars(root_path, partition_key, df)` — validates (symbol, datetime) unique and ascending per symbol; rejects appending older bars after newer.
- **Read:** `read_bars(root_path, symbols, start, end, source=None, columns=None)` — predicate pushdown by partition and filters; returns sorted (symbol, datetime). Optional column subset.
- **Edge cases:** Empty range or no symbols → empty DataFrame; missing partitions → empty DataFrame; UTC enforced.

## Preprocessing (Step 1.3)

- **Load corporate actions (1.3.1):** `load_corporate_actions(symbol, engine=None, actions=...)` — query `corporate_actions` by symbol, ordered by ex_date; or pass list for tests. No look-ahead: only ex_date ≤ bar date affects that bar.
- **Split (1.3.2):** For each split, cumulative adj_factor *= ratio; bars on or after ex_date: OHLC *= ratio, volume /= ratio. Applied in ex_date order.
- **Dividend (1.3.3):** For bars on or after ex_date, subtract cash_amount from open, high, low, close (total-return consistency).
- **adj_factor (1.3.4):** Stored on each bar; default 1.0.
- **Interpolation (1.3.5):** `interpolate(bars, method="linear", max_gap=..., forward_fill_beyond=...)` — linear fill for close (and optional OHLC) over max_gap; optional missing_filled flag.
- **Outliers (1.3.6):** `detect_outliers(bars, method="zscore_returns"|"zscore_levels", threshold=3)` — backward-looking only; outlier_flag = 1 for outliers.
- **API (1.3.7):** `adjust(bars, symbol, engine, actions=...)`, `interpolate(...)`, `detect_outliers(...)`, `run_pipeline(bars, symbol, engine, actions=...)`. `Preprocessor(engine)` wraps the same. Pipeline: adjust → interpolate → detect_outliers → output with adj_factor, outlier_flag.

## Data ingestion (Step 1.4)

- **DataSource (abstract):** `fetch(symbol, start, end)` → DataFrame with symbol, datetime, open, high, low, close, volume (UTC). Implementations share this contract.
- **Alpha Vantage (chosen API):** `AlphaVantageDataSource(api_key=...)` or set `ALPHA_VANTAGE_API_KEY`. Uses TIME_SERIES_DAILY. Rate limits: 5 calls/min (free tier), 500/day; throttling and retries on 429/5xx. Raw schema: date, 1. open–5. volume; normalized to UTC.
- **CSV interface:** `CSVDataSource(csv_path=...)` or `CSVDataSource(csv_dir=..., filename_pattern="{symbol}.csv")`. Optional `column_map` (normalized name → CSV header); else inferred from common aliases (Date, Open, Close, etc.). Same `fetch(symbol, start, end)` contract.
- **Symbol registration:** `register_symbol(engine, symbol_id, display_name=..., asset_class=..., exchange=..., currency=...)` — insert if not exists.
- **Pipeline:** `ingest_bars(data_source, symbols, start, end, root_path, source_name, engine=..., preprocess=True)` — fetch → register symbols → preprocess → write_bars per partition.
- **Script:** `PYTHONPATH=. python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 10` (set `ALPHA_VANTAGE_API_KEY`).

## Migrations

New migrations (future): add to this file and to `src/data/storage/schema.py` (or a separate migrations module) so the same schema can be applied to both SQLite and PostgreSQL.
