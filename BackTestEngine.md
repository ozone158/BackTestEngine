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

- **Bar schema (1.2.1):** `src/data/storage/parquet_bars.py` — Bar schema uses **Polars** (`pl.Schema`): symbol (Utf8), datetime (Datetime("us", "UTC")), open/high/low/close/volume (Float64), adj_factor (Float64, default 1.0), outlier_flag (Int8). Bar datetime = bar open time; all UTC. `write_bars` accepts Polars or pandas DataFrame; `read_bars` returns `pl.DataFrame`.
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
- **Sanity checks (1.5.2):** `src/data/validation.py` — `run_sanity_checks(df, start, end)` accepts Polars or pandas DataFrame. Asserts: no duplicate (symbol, datetime); datetime strictly increasing per symbol; bars in [start, end]; adj_factor > 0; OHLC non-negative. Returns list of error messages (empty if pass).
- **Cross-check (1.5.3):** `run_cross_check(df, sample_per_symbol=...)` spot-checks a sample of bars (high ≥ low, adj_factor > 0).
- **Documentation (1.5.4):** `scripts/README.md` — how to run validation, options, expected output. Run: `python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data` (or `--skip-ingest` to validate existing data only).

## Cointegration pipeline (Step 2.1)

- **Input data (2.1.1):** `load_pair_bars(root_path, symbol_a, symbol_b, start, end, source=..., min_obs=60)` — reads bars via `read_bars` (Polars), pivots and aligns on datetime; returns `(aligned_pl.DataFrame, test_ts)` or `(None, None)` if insufficient observations. `aligned_df` has columns `datetime`, `close_a`, `close_b`.
- **ADF (2.1.2):** Spread formed as `close_a - beta*close_b` with OLS beta over the window; `statsmodels.tsa.stattools.adfuller` with `autolag='AIC'`. Records `adf_statistic`, `adf_pvalue`.
- **Johansen (2.1.3):** `statsmodels.tsa.vector_ar.vecm.coint_johansen` on `(close_a, close_b)`; trace statistic and approximate p-value from critical values; cointegrating vector normalized as `[1, -beta]` (JSON).
- **Decision (2.1.4):** `is_cointegrated = (adf_pvalue < adf_threshold) and (johansen_pvalue < johansen_threshold)`; defaults 0.05 each; configurable via `run_cointegration_test(..., adf_pvalue_threshold=..., johansen_pvalue_threshold=...)`.
- **Persist (2.1.5):** `persist_cointegration_result(result, engine, replace_existing=True)` — insert into `cointegration_results`; replaces existing `(pair_id, test_ts)` when `replace_existing` is True.
- **Single pair:** `run_pair_cointegration(root_path, symbol_a, symbol_b, start, end, engine=..., source=...)` — load, test, optionally persist; returns `CointegrationResult` or `None` if insufficient data.
- **Batch (2.1.6):** `run_batch_cointegration(root_path, engine, start, end, source=..., pair_ids=...)` — for each pair in `pair_universe` (or in `pair_ids`), runs `run_pair_cointegration` and returns list of results. Only pairs with sufficient bar data are included.
- **Tests (2.1.7):** `tests/test_cointegration.py` — synthetic cointegrated series (y = x + noise, both I(1)) → `is_cointegrated` true; synthetic non-cointegrated (two independent random walks) → false; persist round-trip; `load_pair_bars` returns None for insufficient data.

**Acceptance:** CointegrationTest produces ADF and Johansen outputs; pipeline writes `cointegration_results`; only cointegrated pairs (by decision rule) proceed to Kalman/OU.

## Kalman filter Python (Step 2.2)

- **State-space (2.2.1):** State x = β (hedge ratio). Transition: β random walk (F=1, process noise Q). Observation: z = price_a with H = price_b (so z_pred = β*price_b); measurement noise R. Implemented in `src/data/alpha/kalman.py` — `KalmanHedgeRatio(process_noise, measurement_noise, initial_state=None)`.
- **Initialization (2.2.2):** OLS β over warm-up window (default 20 bars) for initial state and P0; or load from `kalman_params` via `load_kalman_params(engine, pair_id, window_end_before_or_equal=...)` → `(KalmanState, Q, R)`.
- **Recursive update (2.2.3):** `KalmanHedgeRatio.update(price_a, price_b)` → (β, spread). Predict then update; spread = price_a - β*price_b after update. No use of future prices.
- **Z-score (2.2.4):** Expanding mean/std of spread using only past data; appended as `z_score` in output (optional).
- **Output series (2.2.5):** `run_kalman_on_aligned(aligned_df, ...)` → `(pl.DataFrame, final_state)`. Accepts Polars or pandas aligned_df; returns Polars DataFrame with `datetime`, `spread`, `beta`, `z_score`; optional `kalman_state` for debug.
- **Write to Parquet (2.2.6):** `src/data/storage/parquet_spreads.py` — **Polars**: schema `pair_id`, `datetime` (UTC), `spread`, `beta`, `z_score`, `kalman_state`. Partition: `root_path/alpha/spreads/pair_id={pair_id}/`. `write_spreads` / `read_spreads` accept and return `pl.DataFrame` (pandas accepted for write).
- **Persist Kalman params (2.2.7):** `persist_kalman_params(pair_id, window_start, window_end, state, process_noise, measurement_noise, engine)` — insert into `kalman_params` for warm start.
- **Single pair pipeline:** `run_pair_kalman(root_path, symbol_a, symbol_b, start, end, engine=..., write_parquet=True, persist_params=False, min_obs=60)` — uses `load_pair_bars`, runs filter, optionally writes Parquet and persists params; returns spread DataFrame or None.
- **Tests (2.2.8):** `tests/test_kalman.py` — output length equals input length; no NaNs after warm-up; β converges toward OLS reference; `KalmanHedgeRatio.update` recursive; persist/load `kalman_params` round-trip; `run_pair_kalman` with bars on disk writes Parquet and returns DataFrame.

**Acceptance:** KalmanHedgeRatio.update() produces β and spread recursively; spread series written to Parquet; optional kalman_params persistence; no look-ahead.

## OU process and thresholds (Step 2.3)

- **Input (2.3.1):** Spread series (from Kalman or Parquet) over estimation window; require sufficient length (e.g. 60+). Pass 1d array or `df["spread"]` to `fit_ou` / `OUModel.fit`.
- **OU estimation (2.3.2):** Ornstein-Uhlenbeck dX = θ(μ - X)dt + σ dW. Discrete AR(1): X_t = μ + φ(X_{t-1} - μ) + ε_t, φ = exp(-θ). Moment matching: μ = mean(X), φ = lag-1 autocorrelation, θ = -ln(φ), σ = sqrt(2θ·Var(X)). Edge case: θ ≤ 0 (non-mean-reverting) → theta/sigma set 0; caller can reject/flag. Implemented in `src/data/alpha/ou.py` — `_fit_ou_moment_matching`, `fit_ou(spread_series, entry_k=..., exit_threshold_spread=...)`.
- **Entry/exit thresholds (2.3.3):** In spread units: entry_upper = μ + k·σ, entry_lower = μ - k·σ (k configurable, default 2); exit_threshold = μ (or override). `fit_ou(..., entry_k=2.0)` returns `OUParams` with these fields.
- **OUModel API (2.3.4):** `OUModel(entry_k=..., exit_threshold_spread=...)`; `fit(spread_series)` → store theta, mu, sigma and compute thresholds; `entry_exit_thresholds()` → (entry_upper, entry_lower, exit_threshold).
- **Persist (2.3.5):** `persist_ou_params(pair_id, params, engine)` — insert into `ou_params`. `load_ou_params(engine, pair_id, window_end_before_or_equal=...)` — load latest. Optional Parquet: `write_ou_params_parquet(root_path, pair_id, params)` → `data/alpha/ou/pair_id={pair_id}/params.parquet`. `run_pair_ou(spread_series, pair_id, window_start=..., window_end=..., engine=..., root_path=..., persist=..., write_parquet=...)` — fit, optionally persist and write Parquet.
- **Signal spec (2.3.6):** Enter long_spread when spread > entry_upper; enter short_spread when spread < entry_lower; exit (flat) when spread returns to exit_threshold (mean μ). Strategy will implement this logic.
- **Tests (2.3.7):** `tests/test_ou.py` — synthetic OU series (known θ, μ, σ), fit close to true; threshold computation; OUModel API; persist/load round-trip; run_pair_ou with persist and Parquet.

**Acceptance:** OUModel fit returns θ, μ, σ and thresholds; ou_params (and optional Parquet) written; signal spec documented for Strategy.

## Event types and queue (Step 3.1)

- **Base event (3.1.1):** `src/backtest/events.py` — `Event(timestamp, event_type)`; all events carry timestamp for ordering and persistence (signal_ts, fill_ts). Events are immutable (frozen dataclasses).
- **MarketEvent (3.1.2):** `event_type="MARKET"`, timestamp (bar datetime), `symbols` (tuple), `bar_data` (dict: symbol → {open, high, low, close, volume}). DataHandler is the only producer.
- **SignalEvent (3.1.3):** `event_type="SIGNAL"`, timestamp, `direction` (long_spread | short_spread | flat), `symbol_a`, `symbol_b`, `hedge_ratio`, `size`, `metadata` (dict: z_score, spread, etc.). Strategy is the only producer.
- **FillEvent (3.1.4):** `event_type="FILL"`, timestamp (fill_ts), `symbol`, `side` (buy | sell), `quantity`, `price`, `commission`, `slippage_bps`. ExecutionHandler is the only producer.
- **Event queue (3.1.5):** `EventQueue(enforce_time_order=True)` wraps `queue.Queue`. Strict rule: events processed in timestamp order; when enforced, no event may have timestamp beyond the latest bar time (updated on each MarketEvent). `put(event)`, `get()`, `get_nowait()`, `empty()`, `qsize()`, `latest_bar_time`, `clear_latest_bar_time()`.
- **Tests (3.1.6):** `tests/test_events.py` — create one of each event type and assert fields; assert immutability; assert ordering by timestamp; queue FIFO and rejection of events beyond latest bar time.

**Acceptance:** Event types are defined and immutable; queue enforces time ordering in the loop.

## DataHandler (Step 3.2)

- **Load backtest range (3.2.1):** `DataHandler(symbols, start_ts, end_ts, read_bars_fn)` — on init calls `read_bars_fn(symbols, start_ts, end_ts)` (e.g. wrapper around `read_bars(root_path, symbols, start, end, source=...)` from `parquet_bars`). Stores bars in ascending datetime order.
- **Bar iterator (3.2.2):** `next_bar(event_queue=None)` advances cursor to next bar; returns `(datetime, bar_data)` or `None` when exhausted. Cursor never goes backward; no peek into future. `has_next()`, `current_time`.
- **get_latest_bars (3.2.3):** `get_latest_bars(symbol, N)` returns last N bars for symbol with `datetime <= current_time` (list of `(datetime, BarData)`). Only past and current bars visible; no look-ahead.
- **Emit MarketEvent (3.2.4):** When `next_bar(event_queue=q)` is called with an `EventQueue`, creates `MarketEvent(timestamp=bar_datetime, symbols, bar_data)` and puts it on the queue. DataHandler is the driver: each call to next_bar produces one MarketEvent.
- **Multi-symbol alignment (3.2.5):** Bars aligned on datetime: inner join so only datetimes where all requested symbols have data are kept. One MarketEvent per aligned bar time with both legs. Implemented in `_align_bars(df, symbols)`.
- **Tests (3.2.6):** `tests/test_data_handler.py` — feed 10 bars; advance one by one; get_latest_bars(2) returns at most 2 bars and only past bars; assert no future data; emit MarketEvent on advance; multi-symbol alignment drops missing rows.

**Acceptance:** DataHandler loads bars for range, exposes get_latest_bars with no look-ahead, and emits MarketEvents in strict chronological order.

## Strategy (Step 3.3)

- **Consume MarketEvent (3.3.1):** `OUStrategy.process_market_event(event, event_queue=None)` — on each MarketEvent receives (timestamp, bar data for symbol_a, symbol_b). Strategy holds OU params (entry_upper, entry_lower, exit_threshold) and a spread_beta_provider.
- **Spread and z-score (3.3.2):** For current bar, (spread, β) = spread_beta_provider(timestamp, bar_data). Provider can be precomputed (lookup from Parquet) or online Kalman (update with close_a, close_b). z_score = (spread - mu) / sigma (OU μ, σ); included in SignalEvent metadata.
- **Signal logic (3.3.3):** Spread in spread units. If spread >= entry_upper and not long_spread → long_spread. If spread <= entry_lower and not short_spread → short_spread. If long and spread <= exit_threshold → flat; if short and spread >= exit_threshold → flat. Position state (flat | long_spread | short_spread) avoids duplicate entries.
- **Size (3.3.4):** Pass-through config: `OUStrategy(..., size=notional_or_units)`. hedge_ratio (β) attached to every SignalEvent for Portfolio to size legs.
- **Emit SignalEvent (3.3.5):** `SignalEvent(timestamp, direction, symbol_a, symbol_b, hedge_ratio, size, metadata={z_score, spread})`. Put on queue when event_queue is provided.
- **Tests (3.3.6):** `tests/test_strategy.py` — mock bars and OU params; long_spread when spread > entry_upper; flat when spread reverts to exit_threshold; short when spread < entry_lower; no double entry (state machine).

**Acceptance:** Strategy produces SignalEvents consistent with OU thresholds; uses only current and past data.

## Portfolio (Step 3.4)

- **State (3.4.1):** `Portfolio` maintains `cash`, `positions` {symbol: quantity}, `initial_capital`; optionally `equity_curve` list of (ts, equity, cash, positions_value). `start_run(engine=None)` initializes cash = initial_capital, positions = {}.
- **On SignalEvent (3.4.2):** `process_signal(signal, bar_data, event_queue=None)` converts direction to leg orders: long_spread = buy symbol_a (size), sell symbol_b (size × hedge_ratio); short_spread = sell A, buy B; flat = close both legs (sell/buy to flatten). Reference prices from bar_data close. Calls `execution_handler(symbol, side, quantity, ref_price, timestamp)` → FillEvent per leg (one Fill per leg; 3.4.3).
- **On FillEvent (3.4.4):** `_process_fill(fill)` updates positions[symbol] += signed quantity (buy +, sell −); cash −= quantity×price + commission (buy), cash += quantity×price − commission (sell). After each signal’s fills, appends equity snapshot.
- **Equity curve (3.4.5):** `record_equity_snapshot(ts, bar_data)` appends (ts, equity, cash, positions_value). `flush_equity_curve(engine)` batch inserts into `backtest_equity`.
- **Run start (3.4.6):** `start_run(engine)` inserts `backtest_runs` row (run_id, strategy_name, pair_id, start_ts, end_ts, config_json) when engine and start_ts/end_ts are set. `generate_run_id()` returns UUID for run_id.
- **Tests (3.4.7):** `tests/test_portfolio.py` — one fill buy 10 @ 100 → positions[symbol]=10, cash reduced by 1000 + commission; two legs (pair) positions and cash consistent; flat closes both legs; start_run inserts backtest_runs; flush_equity_curve inserts backtest_equity.

**Acceptance:** Portfolio tracks positions and cash; converts signals to orders; updates on fills; records equity curve; writes backtest_runs and backtest_equity.

## ExecutionHandler (Step 3.5)

- **Receive order (3.5.1):** `BacktestExecutionHandler.execute(symbol, side, quantity, ref_price, timestamp)` — input is reference price (e.g. bar close); market order assumed. Handler is callable so Portfolio can use it as `execution_handler`.
- **Slippage (3.5.2):** `slippage_bps` in constructor. fill_price = ref * (1 + slippage_bps/10000) for buy, ref * (1 - slippage_bps/10000) for sell. Stored in FillEvent.slippage_bps for audit.
- **Commission (3.5.3):** `commission_per_trade` (fixed per order) and/or `commission_per_share`. commission = commission_per_trade + quantity * commission_per_share. Added to FillEvent.commission.
- **Bid-ask optional (3.5.4):** `half_spread_bps`: buy at ref * (1 + half_spread_bps/10000), sell at ref * (1 - half_spread_bps/10000); then slippage applied to that effective ref.
- **Emit FillEvent (3.5.5):** Returns FillEvent(fill_ts=timestamp, symbol, side, quantity, price=fill_price, commission, slippage_bps).
- **Link to signal (3.5.6):** When writing backtest_fills (event loop / persistence), set signal_id to the SignalEvent id if available; not part of ExecutionHandler itself.
- **Tests (3.5.7):** `tests/test_execution.py` — order 10 @ 100, 5 bps slippage, $1 commission → fill price and commission as expected; sell slippage; per-share commission; callable interface; optional half_spread_bps; Portfolio integration.

**Acceptance:** ExecutionHandler produces FillEvents with correct price, commission, slippage; config-driven.

## Event loop and persistence (Step 3.6)

- **Main loop (3.6.1):** `run_backtest(data_handler, strategy, portfolio, engine, record_equity_every_bar=True)` — caller must call `portfolio.start_run(engine)` first. While DataHandler has next bar: get (dt, bar_data); build MarketEvent; Strategy.process_market_event → optional SignalEvent; if signal: `persist_signal(engine, run_id, signal)` → signal_id; Portfolio.process_signal(signal, bar_data) → fills; persist each fill with signal_id; when no signal and record_equity_every_bar, record equity snapshot. After loop: Portfolio.flush_equity_curve(engine).
- **Run ID and config (3.6.2):** Caller generates run_id (e.g. `generate_run_id()`), builds Portfolio with run_id, strategy_name, pair_id, start_ts, end_ts, config_json; `start_run(engine)` inserts backtest_runs.
- **Ordering (3.6.3):** Signals and fills persisted in event order (same order as processing); signal_ts and fill_ts non-decreasing. Fill rows get signal_id from persist_signal return (3.5.6 link).
- **Long runs (3.6.4):** Equity curve batch-inserted in flush_equity_curve; for very long runs could write in chunks or to Parquet (run_id, ts, equity, cash, positions_value).
- **Integration test (3.6.5):** `tests/test_run.py` — full backtest on 1 pair, 20 bars; assert backtest_runs 1 row; backtest_signals and backtest_fills expected counts; backtest_equity rows; equity non-negative. Tests for persist_signal (returns id) and persist_fill (signal_id link).

**Acceptance:** End-to-end backtest run completes; all events processed in order; DB has one run with signals, fills, and equity.

## C++ Kalman (Step 4.1)

- **State-space (4.1.1):** Documented in `docs/kalman_state_space.md`: state x = β (scalar), F=1, H=price_b, Q=process_noise, R=measurement_noise; predict then update; initialization and edge cases (price_b=0, first observation).
- **C++ implementation (4.1.2–4.1.4):** `cpp/kalman_hedge_ratio.hpp` + `cpp/kalman_hedge_ratio.cpp` — class `backtest::KalmanHedgeRatio(Q, R, optional initial_beta, optional initial_P)`; `update(price_a, price_b)` returns `(beta, spread)`; sanitization to avoid NaNs/Infs. `cpp/kalman_filter.cpp` is the pybind11 binding only.
- **Build and binding (4.1.5):** setuptools + pybind11; `setup.py` builds extension `kalman_core` from `cpp/kalman_hedge_ratio.cpp` and `cpp/kalman_filter.cpp` (C++17). Install with `pip install -e .` (requires pybind11 and a C++17 compiler). Python: `import kalman_core; k = kalman_core.KalmanFilter(1e-6, 1e-4); beta, spread = k.update(price_a, price_b)`.
- **C++ unit test (4.1.6):** `cpp/kalman_filter_test.cpp` — Google Test; fixed (price_a, price_b) sequence with reference beta/spread from Python; tests: FixedSequenceMatchesReference, FirstObservationInitializes, PriceBZeroReturnsCurrentBetaAndPriceA. Build: `cmake -S cpp -B cpp/build && cmake --build cpp/build` (fetches Googletest via FetchContent; requires CMake 3.14+ and network). Run: `ctest --test-dir cpp/build` or the `kalman_filter_test` executable. See `cpp/README.md`.
- **Python unit test (4.1.7):** `tests/test_kalman.py::test_kalman_cpp_vs_python_same_sequence` — feeds the same (price_a, price_b) sequence to Python `KalmanHedgeRatio` and C++ `KalmanFilter`; asserts `numpy.allclose(beta, spread)` with tolerance 1e-10. Skips if `kalman_core` cannot be imported (e.g. no compiler).

## Python integration and replacement (Step 4.2)

- **Wrapper (4.2.1):** `KalmanHedgeRatio(..., use_cpp=False)` uses Python implementation; `use_cpp=True` delegates to `kalman_core.KalmanFilter` when the extension is available. Same public API: `update(price_a, price_b)` → `(beta, spread)`, `state()` → `KalmanState | None`.
- **Pipeline (4.2.2):** `run_kalman_on_aligned(..., use_cpp=False)` and `run_pair_kalman(..., use_cpp=False)` accept `use_cpp`; when True they use the C++ path. Strategy callers can pass a spread_beta_provider built with `KalmanHedgeRatio(use_cpp=True)` for online Kalman.
- **State (4.2.3):** C++ path: `state()` returns `KalmanState(beta=cpp.beta, P=cpp.P)` when initialized; warm start via `initial_state` in constructor (and OLS warm-up in `run_kalman_on_aligned`). Persist/load `kalman_params` unchanged.
- **Error handling (4.2.4):** Before calling C++, validate `price_a` and `price_b` are finite; on invalid input or C++ exception, raise `ValueError` with a clear message. Test: `test_kalman_use_cpp_rejects_non_finite`.
- **Tests (4.2.5):** `test_run_kalman_on_aligned_use_cpp_matches_python` and `test_run_pair_kalman_use_cpp_matches_python` — same data with `use_cpp=True` and `use_cpp=False`; spread and beta allclose (tolerance 1e-9). `test_backtest_equity_use_cpp_vs_python` — backtest with online Kalman (py vs cpp); equity curves allclose. All skip if `kalman_core` is not built.

## Benchmarking (Step 4.3)

- **Benchmark setup (4.3.1):** `scripts/benchmark_kalman.py` — same backtest (one pair, in-memory bars); configurable bars (default 252 = 1 year daily) and iterations (default 20). Runs N times with Python Kalman then N times with C++ Kalman; same event loop, Strategy, Portfolio, Execution.
- **Profile / per-step (4.3.2):** Optional `--mode kalman_only` runs a tight loop of Kalman updates only (no backtest) to measure per-step cost and compare Python vs C++.
- **Document speedup (4.3.3):** Script prints total time per implementation and ratio (time_Python / time_C++). If ratio &lt; 1, notes that I/O or Portfolio may dominate for that size.
- **CI or script (4.3.4):** `python -m scripts.benchmark_kalman` logs timings; optional `--fail-on-regression` exits with code 1 if C++ is not faster than Python. See `scripts/README.md` for options.

## Execution C++ (Step 4.4)

- **C++ fill simulator (4.4.2):** `cpp/fill_simulator.cpp` — `batch_fill(symbols, sides, quantities, ref_prices, slippage_bps, commission_per_trade, commission_per_share, half_spread_bps)` returns (fill_prices, commissions, slippage_bps_list). Logic matches Python: optional half-spread, then slippage (buy: ref×(1+slippage_bps/1e4), sell: ref×(1−slippage_bps/1e4)), commission = per_trade + quantity×per_share. Built as extension `execution_core` via `setup.py` (pybind11). `half_spread_bps < 0` means not set.
- **Integration (4.4.3):** `BacktestExecutionHandler(..., use_cpp=False)` — when `use_cpp=True` and `execution_core` is importable, `execute()` calls `execution_core.batch_fill` for one order and builds FillEvent from the result. Same API; C++ path used only for fill math.
- **Tests:** `tests/test_execution.py` — `test_use_cpp_matches_python` and `test_use_cpp_half_spread_matches_python` assert C++ and Python fills match (skip if `execution_core` not built).

## Performance metrics (Step 5.1)

- **Input (5.1.1):** `src/backtest/performance.py` — accepts equity curve (array or (ts, equity) pairs) or precomputed return series. If equity: returns = (equity[t] - equity[t-1]) / equity[t-1]; first observation has no return.
- **Total return (5.1.2):** From equity: (equity[-1] - equity[0]) / equity[0]. From returns: (1+r1)(1+r2)... - 1. Stored as decimal.
- **Max drawdown (5.1.3):** Peak = running max(equity); drawdown = (peak - equity) / peak; max_drawdown = max(drawdown). Decimal.
- **Sharpe (5.1.4):** Mean(returns) / std(returns) * sqrt(periods_per_year). Optional risk_free_rate (annual). Zero std → 0.
- **Sortino (5.1.5):** Mean(returns) / downside_std(returns) * sqrt(periods_per_year). Downside std = std of returns below target (default 0).
- **Calmar (5.1.6):** total_return / max_drawdown. Zero max_drawdown → 0.
- **Trade stats (5.1.7):** From optional fill_events: if fills have signal_id, group by signal_id, P&L per trade (buy: -price*qty-comm, sell: +price*qty-comm); num_trades = len(groups), win_rate = count(P&L>0)/num_trades. Without signal_id: num_trades = len(fills)//2, win_rate None.
- **API (5.1.8):** `PerformanceMetrics.compute(equity_series=None, returns_series=None, fill_events=None, periods_per_year=252, risk_free_rate=0.0)` → dict with sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate. Exported from `src.backtest`.
- **Tests (5.1.9):** `tests/test_performance.py` — total return from equity/returns; constant return series (Sharpe, total return); synthetic equity with one drawdown (max_drawdown); compute() from equity/returns; Calmar zero drawdown; Sortino; trade stats with/without signal_id.

## Risk attribution (Step 5.2)

- **Inputs (5.2.1):** `src/backtest/risk_attribution.py` — strategy return series and factor/benchmark return series (same length, aligned); or factor_matrix (n_obs × n_factors). NaN alignment: drop indices where any series has NaN.
- **Regression (5.2.2):** OLS: strategy_returns = alpha + beta * market_returns + error. Alpha = intercept, beta = slope. Pairs strategy should have beta near 0 (market-neutral). NumPy lstsq.
- **PCA (5.2.3):** Optional; not implemented. Default is single- or multi-factor OLS. Document in module.
- **Output (5.2.4):** alpha (annualized via periods_per_year), beta (scalar or array), residuals (series), R².
- **API (5.2.5):** `RiskAttribution.decompose(strategy_returns, factor_returns=None, factor_matrix=None, periods_per_year=252)` → dict with alpha, beta, residuals, R2. Module-level `decompose()` also available. Exported from `src.backtest`.
- **Interpretation (5.2.6):** Documented in module: "Beta near 0 implies market-neutral. Alpha is strategy return not explained by market." Use for monitoring and reporting.
- **Tests (5.2.7):** `tests/test_risk_attribution.py` — strategy = 0.5*market + noise → beta ≈ 0.5; strategy = constant → beta ≈ 0; module-level decompose; residuals = strategy - fitted; factor_matrix multi-factor; ValueError when no factor; align drops NaN.

## Position sizing (Step 5.3)

- **Kelly (5.3.1):** `src/backtest/position_sizing.py` — edge (win probability p), odds (win/loss ratio b). f* = p - (1-p)/b. Output fraction of capital; capped at max_fraction (default 0.25).
- **Half-Kelly (5.3.2):** fraction parameter (default 1.0): f = fraction * f*. E.g. fraction=0.5 for half-Kelly.
- **Risk parity (5.3.3):** Input: volatilities (list or array). w_i = (1/sigma_i) / sum(1/sigma_j). Weights sum to 1. Zero or negative vol gets weight 0; rest renormalized.
- **API (5.3.4):** `PositionSizer.kelly(edge, odds, fraction=1.0, max_fraction=0.25)` → float. `PositionSizer.risk_parity(volatilities)` → array of weights. Module-level `kelly()` and `risk_parity()` also available. Exported from `src.backtest`.
- **Integration (5.3.5):** Optional; Strategy or Portfolio can call PositionSizer in a follow-up. Not wired in Phase 5.
- **Tests (5.3.6):** `tests/test_position_sizing.py` — Kelly with edge=0.6, odds=2 → f=0.4, capped at max_fraction; half-Kelly; zero/negative edge or odds → 0; risk parity [0.1, 0.2] → weights sum 1, lower vol higher weight; PositionSizer matches module-level; zero volatility handled.

## Persistence and wiring (Step 5.4)

- **Trigger (5.4.1):** After backtest run completes, `run_backtest` calls `compute_and_persist_metrics(engine, run_id)`. Loads equity from `backtest_equity` and fills from `backtest_fills`; computes PerformanceMetrics.compute() and optionally RiskAttribution.decompose() when benchmark returns provided; builds metrics row (5.4.2) and inserts/replaces `backtest_metrics` (5.4.3).
- **Build metrics row (5.4.2):** `src/backtest/metrics_persistence.py` — collect sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate, alpha, beta; full dict in metrics_json; computed_at = now().
- **Insert backtest_metrics (5.4.3):** Delete existing row for run_id then INSERT (policy: replace so re-runs update metrics).
- **Where to call (5.4.4):** Option A: inside backtest runner (after loop) — implemented. Option B: `scripts/compute_metrics.py --run-id <id> [--db URL] [--benchmark path]` recomputes and persists without re-running backtest.
- **Benchmark (5.4.5):** Optional `benchmark_returns` (list or array) passed to `compute_metrics_for_run` / `compute_and_persist_metrics`; when provided, strategy returns derived from equity and RiskAttribution.decompose(strategy_returns, factor_returns=benchmark_returns) fills alpha, beta. Script accepts `--benchmark` path to CSV (single column) or .npy.
- **Tests (5.4.6):** `test_run_backtest_integration` asserts backtest_metrics has one row with total_return, sharpe_annual, max_drawdown present. `test_compute_and_persist_metrics_after_run` runs backtest then calls compute_and_persist_metrics again and asserts metrics row updated.

## Reporting (Step 5.5)

- **Report script (5.5.1):** `scripts/report_backtest.py` lists recent `backtest_runs` (by `created_at` desc, limit configurable, default 10). For each run it loads metrics from `backtest_metrics` (left join so runs without metrics still appear). Output: **console** (print table), **CSV**, or **HTML**. Summary fields: run_id, strategy_name, pair_id, start_ts, end_ts, sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate, alpha, beta, created_at, computed_at. CLI: `--limit N`, `--output console|csv|html`, `--db URL`, `--out-file path` (optional file for csv/html).
- **Metrics storage:** Run metadata lives in **backtest_runs**; computed metrics in **backtest_metrics** (one row per run_id). Optional equity series in **backtest_equity** (not required for the report summary).
- **Dashboard (5.5.2):** `scripts/dashboard.py` — Streamlit app: table of recent backtest runs with key metrics; select a run to view equity curve chart. Requires `pip install -e ".[dashboard]"` (streamlit). Run: `streamlit run scripts/dashboard.py` (set PYTHONPATH to project root). Optional `--db URL` after `--` for database.
- **Documentation (5.5.3):** See `scripts/README.md` for report_backtest usage and options.

## Migrations

New migrations (future): add to this file and to `src/data/storage/schema.py` (or a separate migrations module) so the same schema can be applied to both SQLite and PostgreSQL.
