# BackTestEngine — Project State & Conventions

## DB engine (Step 1.1.1)

- **Choice:** Single schema runs against **SQLite** (default) or **PostgreSQL**.
- **SQLite:** Used for single-user/dev; zero setup; default URL `sqlite:///data/backtest.db`.
- **PostgreSQL:** Use for multi-process or production; set `DATABASE_URL=postgresql://user:pass@host/db`.
- **Schema module:** `src/data/storage/schema.py` — one SQLAlchemy MetaData definition; `create_schema(engine)` creates all tables. Run with:
  - `python -m src.data.storage.schema` (uses `DATABASE_URL` or default SQLite)
  - Or: `from src.data.storage import get_engine, create_schema; create_schema(get_engine())`

## Reference tables first (Step 1.1.2)

Reference tables are: `symbols`, `corporate_actions`, `pair_universe`. They are created first (dependency order). Use `create_reference_tables(engine)` to create only these (e.g. incremental setup). Application logic for `symbol_a < symbol_b`: use `normalize_pair(symbol_a, symbol_b)` → `(a, b)` and `pair_id(symbol_a, symbol_b)` → `"{a}_{b}"` from `src.data.storage.reference` before inserting into `pair_universe`.

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

- **DataSource (abstract):** `src/data/ingestion/base.py` — contract `fetch(symbol, start, end)` → DataFrame with columns `symbol`, `datetime` (UTC), `open`, `high`, `low`, `close`, `volume`. `normalize_bars_df()` for flexible column mapping.
- **Alpha Vantage (1.4.1–1.4.4):** `AlphaVantageDataSource` — TIME_SERIES_DAILY; raw schema: date → "1. open", "2. high", "3. low", "4. close", "5. volume". Rate limits (free tier): 5 requests/minute, 25/day; throttle 12s between requests; retries on 429/5xx. API key: `ALPHA_VANTAGE_API_KEY` or `apikey=` in constructor.
- **CSV interface:** `CSVDataSource(file_path=...)` or `CSVDataSource(base_path=..., pattern="{symbol}.csv")` — reads CSV with flexible column names (symbol, date/datetime, open/O, high/H, low/L, close/C, volume/V); normalizes to same schema and UTC.
- **Symbol registration (1.4.3):** `register_symbol(engine, symbol_id, display_name=..., asset_class=..., exchange=..., currency=...)` — insert if not exists.
- **Integration (1.4.5):** `scripts/ingest_alpha_vantage.py` — fetch symbols from Alpha Vantage → preprocess → write to Parquet (partition source=alpha_vantage); register symbols in DB. Usage: `ALPHA_VANTAGE_API_KEY=key python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 30 --root data`.

## End-to-end validation (Step 1.5)

- **Validation script (1.5.1):** `scripts/validate_e2e.py` — (1) Optionally ingest 2–3 symbols over 5–10 days (same flow as ingest_alpha_vantage). (2) Read back bars with `read_bars(root, symbols, start, end, source=...)`. (3) Run sanity checks and cross-check.
- **Sanity checks (1.5.2):** `src/data/validation.py` — `run_sanity_checks(df, start, end)` asserts: no duplicate (symbol, datetime); datetime strictly increasing per symbol; all bar datetimes in [start, end]; adj_factor present and > 0; OHLC non-negative. Returns list of error messages (empty if pass).
- **Cross-check (1.5.3):** `run_cross_check(df, sample_per_symbol=...)` spot-checks a sample of bars (high ≥ low, adj_factor > 0).
- **Documentation (1.5.4):** `scripts/README.md` — how to run validation, options, expected output. Run: `python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data` (or `--skip-ingest` to validate existing data only).

## Cointegration pipeline (Step 2.1)

- **Input data (2.1.1):** `load_pair_bars(root_path, symbol_a, symbol_b, start, end, source=..., min_obs=60)` — reads bars via `read_bars`, aligns on datetime (inner join), drops missing close; returns `(aligned_df, test_ts)` or `(None, None)` if insufficient observations. `aligned_df` has columns `datetime`, `close_a`, `close_b`.
- **ADF (2.1.2):** Spread formed as `close_a - beta*close_b` with OLS beta over the window; `statsmodels.tsa.stattools.adfuller` with `autolag='AIC'`. Records `adf_statistic`, `adf_pvalue`.
- **Johansen (2.1.3):** `statsmodels.tsa.vector_ar.vecm.coint_johansen` on `(close_a, close_b)`; trace statistic and approximate p-value from critical values; cointegrating vector normalized as `[1, -beta]` (JSON).
- **Decision (2.1.4):** `is_cointegrated = (adf_pvalue < adf_threshold) and (johansen_pvalue < johansen_threshold)`; defaults 0.05 each; configurable via `run_cointegration_test(..., adf_pvalue_threshold=..., johansen_pvalue_threshold=...)`.
- **Persist (2.1.5):** `persist_cointegration_result(result, engine, replace_existing=True)` — insert into `cointegration_results`; replaces existing `(pair_id, test_ts)` when `replace_existing` is True.
- **Single pair:** `run_pair_cointegration(root_path, symbol_a, symbol_b, start, end, engine=..., source=...)` — load, test, optionally persist; returns `CointegrationResult` or `None` if insufficient data.
- **Batch (2.1.6):** `run_batch_cointegration(root_path, engine, start, end, source=..., pair_ids=...)` — for each pair in `pair_universe` (or in `pair_ids`), runs `run_pair_cointegration` and returns list of results. Only pairs with sufficient bar data are included.
- **Tests (2.1.7):** `tests/test_cointegration.py` — synthetic cointegrated series (y = x + noise, both I(1)) → `is_cointegrated` true; synthetic non-cointegrated (two independent random walks) → false; persist round-trip; `load_pair_bars` returns None for insufficient data.

**Acceptance:** CointegrationTest produces ADF and Johansen outputs; pipeline writes `cointegration_results`; only cointegrated pairs (by decision rule) proceed to Kalman/OU.

## Migrations

New migrations (future): add to this file and to `src/data/storage/schema.py` (or a separate migrations module) so the same schema can be applied to both SQLite and PostgreSQL.
