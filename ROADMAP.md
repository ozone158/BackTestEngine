# High-Frequency Statistical Arbitrage — Implementation Roadmap

This document is the **implementation roadmap** for the BackTestEngine: a step-by-step plan, libraries, logic, and sequencing derived from `system.md` (system design) and `system_database.md` (database & storage architecture). No code here—only the plan.

---

## 1. Project Summary

- **Goal:** Production-oriented pairs-trading framework: cointegration → Kalman hedge ratio → OU spread → event-driven backtest → risk/attribution.
- **Principles:** No look-ahead bias; high throughput (columnar storage, async I/O, C++ hot paths); statistical rigor (ADF/Johansen, Kalman, OU).
- **Storage split:** Columnar (Parquet/HDF5) for time series; relational (SQLite/PostgreSQL) for metadata and backtest runs; config in JSON/YAML.

---

## 2. Technology Stack & Libraries

| Layer | Technology | Purpose |
|-------|------------|--------|
| **Language** | Python 3.10+ (3.11+ preferred); C++17 | Main logic; hot-path acceleration |
| **Data I/O** | `pyarrow` | Parquet read/write, partitioning |
| **Alternative columnar** | `tables` or `h5py` | HDF5 if needed |
| **Relational DB** | SQLite (default) or PostgreSQL | Metadata, runs, signals, fills, metrics |
| **Numerics** | `numpy`, `scipy` | ADF, linear algebra, OU fitting |
| **Stats / econometrics** | `statsmodels` | Johansen, OLS, unit-root helpers |
| **Async / APIs** | `aiohttp`, `websockets` | REST/WebSocket ingestion |
| **C++ bindings** | `pybind11` | Expose Kalman, order-matching to Python |
| **Config & validation** | `pydantic` + YAML/JSON | Strategy config, fees, slippage, env |
| **Testing** | `pytest`, `pytest-asyncio` | Unit and integration tests |

---

## 3. High-Level Phases

1. **Phase 1 — Data infrastructure:** Schema, storage backends, one connector, preprocessing. Validate with a small dataset.
2. **Phase 2 — Alpha research:** Cointegration pipeline → Kalman (Python) → OU fit and thresholds. Output: signal spec (when to enter/exit).
3. **Phase 3 — Backtesting engine:** Events → DataHandler → Strategy → Portfolio → ExecutionHandler (Python only). No C++ yet.
4. **Phase 4 — C++ acceleration:** Replace Kalman and order-matching hot paths with pybind11; benchmark.
5. **Phase 5 — Risk & attribution:** Metrics, PCA attribution, Kelly/Risk Parity sizing; wire to engine output and DB.

### 3.1 Detailed plan index (where to find each step)

| Phase | Section | Steps (detailed) |
|-------|---------|------------------|
| **1** | §4.3 | **1.1** Relational schema (1.1.1–1.1.6) · **1.2** Parquet bars writer/reader (1.2.1–1.2.6) · **1.3** Corporate actions & preprocessing (1.3.1–1.3.8) · **1.4** One data connector (1.4.1–1.4.6) · **1.5** End-to-end validation (1.5.1–1.5.4) |
| **2** | §5.2 | **2.1** Cointegration pipeline (2.1.1–2.1.7) · **2.2** Kalman filter Python (2.2.1–2.2.8) · **2.3** OU process & thresholds (2.3.1–2.3.7) |
| **3** | §6.2 | **3.1** Event types & queue (3.1.1–3.1.6) · **3.2** DataHandler (3.2.1–3.2.6) · **3.3** Strategy (3.3.1–3.3.6) · **3.4** Portfolio (3.4.1–3.4.7) · **3.5** ExecutionHandler (3.5.1–3.5.7) · **3.6** Event loop & persistence (3.6.1–3.6.5) |
| **4** | §7.2 | **4.1** C++ Kalman (4.1.1–4.1.7) · **4.2** Python integration (4.2.1–4.2.5) · **4.3** Benchmarking (4.3.1–4.3.4) · **4.4** Optional execution C++ (4.4.1–4.4.4) |
| **5** | §8.2 | **5.1** Performance metrics (5.1.1–5.1.9) · **5.2** Risk attribution (5.2.1–5.2.7) · **5.3** Position sizing (5.3.1–5.3.6) · **5.4** Persistence & wiring (5.4.1–5.4.6) · **5.5** Reporting optional (5.5.1–5.5.3) |

---

## 4. Phase 1 — Data Infrastructure (Detailed Plan)

### 4.1 Storage Schema & Conventions

- **Relational (SQLite/PostgreSQL):** Define and create tables per `system_database.md` §4:  
  `symbols`, `corporate_actions`, `pair_universe`, `cointegration_results`, `kalman_params`, `ou_params`, `backtest_runs`, `backtest_signals`, `backtest_fills`, `backtest_equity`, `backtest_metrics`.
- **Columnar (Parquet):** Define directory layout and schemas:  
  `data/bars/{source}/date=...`, `data/ticks/...`, `data/alpha/spreads/{pair_id}/`, `data/alpha/ou/...` with column specs from §5 (bars: symbol, datetime, OHLCV, adj_factor, outlier_flag; ticks; spread series; OU output).
- **Conventions:** All timestamps UTC, microsecond precision; `symbol_id` uppercase; `pair_id = symbol_a_symbol_b` with `symbol_a < symbol_b`; `run_id` UUID or `bt_date_seq`.

### 4.2 Abstract Interfaces (Design Only)

- **DataSource (abstract):** Contract `fetch(symbol, start, end)` → raw bars or ticks. Implementations will be per venue/source.
- **Preprocessor:** Contract `adjust(series)` (corporate actions), `interpolate(series)` (missing data), `detect_outliers(series)` (e.g. Z-score or Hampel). Output: adjusted OHLCV + metadata, outlier flags.
- **StorageBackend:** Contract `write(partition_key, dataframe)` and `read(symbols, start, end)` for bar/tick data; relational layer for metadata (symbols, corporate_actions, pair_universe).

### 4.3 Detailed Step-by-Step Elaboration (Phase 1)

#### Step 1.1 — Relational schema creation

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 1.1.1 | Choose DB engine | Decide SQLite vs PostgreSQL: SQLite for single-user/dev and zero setup; PostgreSQL for multi-process or production. Create a single schema module or migration script that can run against either. |
| 1.1.2 | Create reference tables first | Create `symbols` (symbol_id PK, display_name, asset_class, exchange, currency, created_at, updated_at). Then `corporate_actions` (id PK, symbol_id FK, action_type, ex_date, recorded_at, ratio, cash_amount, metadata_json). Then `pair_universe` (pair_id PK, symbol_a FK, symbol_b FK, created_at, notes). Enforce `symbol_a < symbol_b` via CHECK or application logic. |
| 1.1.3 | Create alpha summary tables | Create `cointegration_results` (id, pair_id FK, test_ts, adf_statistic, adf_pvalue, johansen_trace, johansen_pvalue, cointegrating_vector TEXT/JSON, is_cointegrated, created_at). Create `kalman_params` (id, pair_id FK, window_start, window_end, initial_state TEXT, process_noise, measurement_noise, created_at). Create `ou_params` (id, pair_id FK, window_start, window_end, theta, mu, sigma, entry_upper, entry_lower, exit_threshold, created_at). |
| 1.1.4 | Create backtest run tables | Create `backtest_runs` (run_id PK, strategy_name, pair_id FK nullable, start_ts, end_ts, config_json, created_at). Create `backtest_signals` (id PK, run_id FK, signal_ts, direction, symbol_a, symbol_b, hedge_ratio, size, metadata_json). Create `backtest_fills` (id PK, run_id FK, signal_id FK nullable, fill_ts, symbol, side, quantity, price, commission, slippage_bps). Create `backtest_equity` (run_id FK, ts, equity, cash, positions_value; unique (run_id, ts)). Create `backtest_metrics` (run_id PK/FK, sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate, alpha, beta, metrics_json, computed_at). |
| 1.1.5 | Add indexes | Indexes per §7.2: symbols(symbol_id); corporate_actions(symbol_id, ex_date); pair_universe(pair_id), (symbol_a, symbol_b); cointegration_results(pair_id, test_ts); backtest_signals(run_id, signal_ts); backtest_fills(run_id, fill_ts); backtest_equity(run_id, ts); backtest_runs(start_ts, end_ts) or created_at. |
| 1.1.6 | Validate | Run schema creation against a fresh DB; verify FKs and constraints; optionally add a small test that inserts one row per table and reads back. |

**Acceptance:** All tables exist; inserts/reads work; FKs and unique constraints behave as expected.

---

#### Step 1.2 — Parquet bars writer and reader

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 1.2.1 | Define bar schema in code | Formalize bar schema: symbol (string), datetime (timestamp[us] UTC), open, high, low, close, volume (float64), adj_factor (float64, default 1.0), outlier_flag (int8, optional). Use PyArrow schema so writer and reader agree. |
| 1.2.2 | Define partition key structure | Partition key = (source, date). Example: `data/bars/source=csv/date=2024-01-15/`. Directory layout must allow listing by source and date for range reads. Document partition naming (e.g. date=YYYY-MM-DD). |
| 1.2.3 | Implement write path | Write function: input = (partition_key dict, DataFrame with bar columns). Convert DataFrame to PyArrow Table; validate (symbol, datetime) unique and datetime ascending per symbol within the batch; write to partition directory as one or more Parquet files. Reject writes that would break ordering (e.g. appending older bars after newer). |
| 1.2.4 | Implement read path | Read function: input = (symbols list, start datetime, end datetime, optional source). List partition directories in date range; read only relevant Parquet files; filter by symbol and datetime range (predicate pushdown); return single combined DataFrame sorted by (symbol, datetime). Support reading only a subset of columns (e.g. symbol, datetime, close) for alpha/backtest. |
| 1.2.5 | Edge cases | Handle empty range (return empty DataFrame). Handle missing partitions (clear error or empty result per policy). Ensure timezone: all datetimes in UTC; document that bar datetime = bar open time. |
| 1.2.6 | Tests | Unit test: write a few bars for one symbol, read back, assert equality and order. Test multi-symbol, multi-partition read. Test that reading with end &lt; min(datetime) returns empty. |

**Acceptance:** Bars written to partition layout; read returns correct bars in ascending order with no future data; predicate pushdown reduces read size when filtering by symbol/date.

---

#### Step 1.3 — Corporate actions and preprocessing

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 1.3.1 | Load corporate actions | For a given symbol, query `corporate_actions` ordered by ex_date ascending. Build an in-memory list of (ex_date, action_type, ratio, cash_amount) for use in adjustment. Only actions with ex_date &lt;= bar date may affect that bar (no look-ahead). |
| 1.3.2 | Split adjustment | For each split: cumulative adj_factor *= ratio (e.g. 2:1 split → ratio 2.0). Apply to all OHLC and optionally volume (divide volume by ratio for constant notional). Apply splits in ex_date order so that bars on or after ex_date get the updated factor. |
| 1.3.3 | Dividend adjustment | For dividends: adjust price series so that returns are total-return consistent. Typical approach: for bars on or after ex_date, subtract (discounted) cash_amount from prices, or use a price-based adj_factor. Document chosen method (e.g. subtract dividend from close, then propagate to OHLC). Apply in ex_date order. |
| 1.3.4 | Output adj_factor | Store the cumulative adj_factor on each bar (so downstream can invert if needed). Default 1.0 for bars before any action. |
| 1.3.5 | Interpolation | Identify missing bars (e.g. expected bar times with no row, or NaN close). Policy: linear interpolation for close (and optionally OHLC) over a max gap (e.g. 1 day); beyond that, leave missing or forward-fill per config. Flag interpolated rows if needed (e.g. missing_filled column). |
| 1.3.6 | Outlier detection | Optional: Z-score on returns or levels (e.g. |close - rolling_mean| / rolling_std &gt; 3) or Hampel filter. Set outlier_flag = 1 for outliers, 0 otherwise. Downstream can filter or downweight. Do not use future data (rolling window must be backward-looking). |
| 1.3.7 | Preprocessor API | Expose adjust(series, symbol), interpolate(series, method, max_gap), detect_outliers(series, method, threshold). adjust uses corporate_actions from DB. Pipeline: fetch raw bars → adjust → interpolate → detect_outliers → output bars with adj_factor and outlier_flag. |
| 1.3.8 | Tests | Unit test: mock corporate_actions; feed bars with one split; assert adj_factor and OHLC after split. Test dividend case. Test interpolation on a series with one missing bar. Test outlier flag on a spike. |

**Acceptance:** Adjusted bars match expected factors; interpolation fills gaps correctly; outlier flags are backward-looking only.

---

#### Step 1.4 — One data connector

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 1.4.1 | Choose first source | Pick one concrete source: e.g. CSV files (symbol, datetime, O, H, L, C, V), or one REST API (e.g. free tier for one exchange). Document the raw schema and rate limits. |
| 1.4.2 | Implement DataSource | Implement fetch(symbol, start, end) returning raw bars (list or DataFrame). Normalize column names to symbol, datetime, open, high, low, close, volume. Normalize datetime to UTC; ensure type consistency (float for prices, timestamp for datetime). |
| 1.4.3 | Symbol registration | Before or after fetch, ensure symbol exists in `symbols`. If not, insert with symbol_id, and optional display_name, asset_class, exchange, currency. Use upsert or "insert if not exists" to avoid duplicates. |
| 1.4.4 | Rate limiting and errors | If API: respect rate limits (backoff or throttle); handle 429/5xx with retries and clear errors. For CSV, handle missing files or empty data gracefully. |
| 1.4.5 | Integration with storage | After fetch, run through Preprocessor (adjust, interpolate, outliers), then pass to StorageBackend.write with appropriate partition_key. One script: fetch for 2–3 symbols over a short range → preprocess → write to Parquet; update symbols in DB. |
| 1.4.6 | Tests | Unit test: mock or fixture CSV; assert normalized schema and UTC. Integration test (optional): small live fetch if API available. |

**Acceptance:** One connector fetches raw data, normalizes to unified schema, registers symbols, and writes preprocessed bars to Parquet.

---

#### Step 1.5 — End-to-end validation

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 1.5.1 | Create validation script | Script: (1) Ingest bars for 2–3 symbols over 5–10 days using the connector. (2) Preprocess and write to Parquet and DB. (3) Read back bars for same symbols and range. |
| 1.5.2 | Sanity checks | Assert no duplicate (symbol, datetime). Assert datetime strictly increasing per symbol. Assert all bar datetimes within requested range (no future data). Assert adj_factor present and &gt; 0. Assert OHLC non-negative (or per policy). |
| 1.5.3 | Cross-check | If possible, spot-check a few bars against source (e.g. one known split date: adj_factor and prices consistent). |
| 1.5.4 | Document | Document how to run the validation and expected output. Add to README or scripts/README. |

**Acceptance:** Validation script runs without error; all sanity checks pass; data is ready for Module 2 and 3.

---

### 4.4 Deliverables (Phase 1)

- Relational DB with all reference and run tables; Parquet layout and read/write APIs.
- One working connector; Preprocessor (adjust, interpolate, outliers); validation script on a small dataset.

---

## 5. Phase 2 — Alpha Research & Signal Generation (Detailed Plan)

### 5.1 Interfaces (Design)

- **CointegrationTest:** `adf(series)`, `johansen(price_matrix)` → test statistics, p-values, cointegrating vectors.
- **KalmanHedgeRatio:** `update(price_a, price_b)` → β, spread, and optional state; recursive, no future data.
- **OUModel:** `fit(spread_series)`, `entry_exit_thresholds()` → θ, μ, σ and entry/exit bands.

### 5.2 Detailed Step-by-Step Elaboration (Phase 2)

#### Step 2.1 — Cointegration pipeline

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 2.1.1 | Obtain input data | Read clean bar data from Module 1 (StorageBackend.read) for symbol_a and symbol_b over window [start, end]. Align on datetime (inner join); drop rows with missing close. Require minimum number of observations (e.g. 60+ bars for ADF/Johansen stability). |
| 2.1.2 | ADF on spread or residuals | Option A: Form spread using a preliminary β (e.g. OLS of price_a on price_b over the window); run ADF on that spread. Option B: Run ADF on residuals from OLS. Use statsmodels ADF implementation; choose lag order (e.g. AIC or fixed). Record adf_statistic and adf_pvalue. Interpretation: reject unit root (stationary) when p &lt; threshold (e.g. 0.05). |
| 2.1.3 | Johansen test | Run Johansen cointegration test on (price_a, price_b) matrix (each column a price series). Obtain trace statistic and p-value; obtain cointegrating vector (normalized, e.g. [1, -β]). Record johansen_trace, johansen_pvalue, and cointegrating_vector (store as JSON array). |
| 2.1.4 | Cointegration decision | Define rule: e.g. is_cointegrated = (adf_pvalue &lt; 0.05) and (johansen_pvalue &lt; 0.05). Document the rule; make threshold configurable. |
| 2.1.5 | Persist results | Insert one row into `cointegration_results`: pair_id, test_ts = end of window (or last bar datetime), adf_statistic, adf_pvalue, johansen_trace, johansen_pvalue, cointegrating_vector (JSON), is_cointegrated, created_at. Ensure (pair_id, test_ts) unique if one test per pair per time. |
| 2.1.6 | Batch screening | Build a small pipeline: for each pair in pair_universe, load bar window, run ADF + Johansen, persist. Optionally support rolling windows (e.g. test_ts every month) for time-varying cointegration. |
| 2.1.7 | Tests | Unit test: synthetic cointegrated series (e.g. y = x + noise, both I(1)); assert is_cointegrated true. Synthetic non-cointegrated (e.g. two random walks); assert false. |

**Acceptance:** CointegrationTest produces ADF and Johansen outputs; pipeline writes `cointegration_results`; only cointegrated pairs proceed to Kalman/OU.

---

#### Step 2.2 — Kalman filter (Python)

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 2.2.1 | State-space formulation | Define state: e.g. x = [β, spread_mean] or just β. Observation: spread = price_a - β*price_b (linear in β). Process model: β evolves (e.g. random walk or constant with process noise). Measurement: observed spread. Document state dimension, transition matrix F, observation matrix H, process noise Q, measurement noise R. |
| 2.2.2 | Initialization | Initial state x0 and covariance P0. Options: (1) OLS β over a short warm-up window; (2) Load from `kalman_params` if available (window_start, window_end, initial_state JSON). Set Q, R from config or from `kalman_params`. |
| 2.2.3 | Recursive update | For each (datetime, price_a, price_b) in time order: (1) Predict: state and covariance time update. (2) Observe: spread_observed = price_a - β_pred*price_b (or use measurement equation). (3) Update: Kalman gain, state update, covariance update. (4) Output: β(t), spread(t) = price_a - β(t)*price_b. No use of future prices. |
| 2.2.4 | Z-score (optional) | For each bar, z_score = (spread(t) - rolling_mean(spread)) / rolling_std(spread) using only past data (expanding or rolling window). Append to output series. |
| 2.2.5 | Output series | Build DataFrame: columns pair_id, datetime, spread, beta, z_score (optional), kalman_state (optional JSON for debug). |
| 2.2.6 | Write to Parquet | Write to `data/alpha/spreads/{pair_id}/` (partition by pair_id; optional date partition). Schema: pair_id, datetime, spread, beta, z_score, kalman_state. |
| 2.2.7 | Optional: persist Kalman params | After warm-up or at end of window, optionally write to `kalman_params`: pair_id, window_start, window_end, initial_state (current state as JSON), process_noise, measurement_noise. Enables warm start for next run or backtest. |
| 2.2.8 | Tests | Unit test: fixed (price_a, price_b) series; compare β and spread to reference (e.g. OLS β and spread). Test that output length equals input length; no NaNs after warm-up. |

**Acceptance:** KalmanHedgeRatio.update() produces β and spread recursively; spread series written to Parquet; optional kalman_params persistence; no look-ahead.

---

#### Step 2.3 — OU process and thresholds

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 2.3.1 | Input | Spread series (from Kalman or from Parquet) over estimation window [window_start, window_end]. Require sufficient length (e.g. 60+ points). |
| 2.3.2 | OU parameter estimation | Ornstein-Uhlenbeck: dX = θ(μ - X)dt + σ dW. Estimate θ (mean-reversion speed), μ (long-run mean), σ (volatility). Methods: (1) MLE (discrete AR(1) representation); (2) Moment matching (variance and autocorrelation). Use scipy/numpy; document formula. Handle edge cases (θ &lt;= 0 → reject or flag). |
| 2.3.3 | Entry/exit thresholds | Entry: when spread is "too far" from μ. In z-score space: entry_upper (e.g. +2) = enter long spread when z &gt; entry_upper; entry_lower (e.g. -2) = enter short spread when z &lt; entry_lower. Exit: exit_threshold (e.g. 0) = close when spread returns to mean (z near 0). Can be in spread units instead: entry_upper = μ + k*σ, entry_lower = μ - k*σ, exit_threshold = μ. Make k configurable. |
| 2.3.4 | OUModel API | fit(spread_series) → store theta, mu, sigma and compute entry_upper, entry_lower, exit_threshold. entry_exit_thresholds() → return (entry_upper, entry_lower, exit_threshold). |
| 2.3.5 | Persist | Insert into `ou_params`: pair_id, window_start, window_end, theta, mu, sigma, entry_upper, entry_lower, exit_threshold, created_at. Optionally write same to Parquet `data/alpha/ou/` (e.g. one row per pair_id and window_end). |
| 2.3.6 | Signal spec document | Document: "Enter long_spread when spread z-score &gt; entry_upper; enter short_spread when z &lt; entry_lower; exit (flat) when z between -exit_threshold and +exit_threshold (e.g. back to mean)." Strategy will implement this logic. |
| 2.3.7 | Tests | Unit test: synthetic OU series (known θ, μ, σ); fit and assert parameters close to true. Test threshold computation. |

**Acceptance:** OUModel fit returns θ, μ, σ and thresholds; ou_params (and optional Parquet) written; signal spec documented for Strategy.

---

### 5.3 Data Flow Summary (Phase 2)

Clean prices (Module 1) → Cointegration screening → Kalman-filtered spread series → OU fit → Parameters and entry/exit bands. Strategy will consume: for each bar time, spread and β from Kalman + OU thresholds to emit long_spread / short_spread / flat.

### 5.4 Deliverables (Phase 2)

- CointegrationTest implementation (ADF + Johansen); pipeline that writes `cointegration_results`.
- KalmanHedgeRatio (Python); spread series written to Parquet; optional `kalman_params` persistence.
- OUModel fit and threshold computation; `ou_params` (and optional Parquet) written. Documented signal spec (when to enter/exit).

---

## 6. Phase 3 — Event-Driven Backtesting Engine (Detailed Plan)

### 6.1 Event Model (Logic)

- **MarketEvent:** New bar (or tick) available; payload: symbol(s), datetime, bar/tick data. Emitted by DataHandler in strict chronological order.
- **SignalEvent:** Strategy decision: direction (long_spread, short_spread, flat), symbol_a, symbol_b, hedge_ratio, size, optional metadata (z-score, spread).
- **FillEvent:** Executed trade: symbol, side, quantity, price, commission, slippage_bps, fill_ts. Emitted by ExecutionHandler.

Events are processed in a single time-ordered queue; no event uses data from the future.

### 6.2 Detailed Step-by-Step Elaboration (Phase 3)

#### Step 3.1 — Event types and queue

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 3.1.1 | Define event base | Define a base event type (e.g. Event with event_type, timestamp). All events carry timestamp = bar/tick time or fill time; used for ordering and for persistence (signal_ts, fill_ts). |
| 3.1.2 | MarketEvent | Fields: event_type="MARKET", timestamp (bar datetime), symbol(s), bar data (open, high, low, close, volume) or tick data. DataHandler is the only producer. |
| 3.1.3 | SignalEvent | Fields: event_type="SIGNAL", timestamp, direction (long_spread | short_spread | flat), symbol_a, symbol_b, hedge_ratio, size (notional or units), metadata (dict: z_score, spread, etc.). Strategy is the only producer. |
| 3.1.4 | FillEvent | Fields: event_type="FILL", timestamp (fill_ts), symbol, side (buy | sell), quantity, price, commission, slippage_bps. ExecutionHandler is the only producer. |
| 3.1.5 | Event queue | Single queue (e.g. queue.Queue or list consumed in order). Strict rule: events are processed in timestamp order; no event is emitted with a timestamp beyond the latest bar time seen. |
| 3.1.6 | Tests | Unit test: create one of each event type; assert fields and ordering by timestamp. |

**Acceptance:** Event types are defined and immutable; queue enforces time ordering in the loop.

---

#### Step 3.2 — DataHandler

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 3.2.1 | Load backtest range | On init: given symbols (e.g. symbol_a, symbol_b for one pair), start_ts, end_ts, call StorageBackend.read(symbols, start_ts, end_ts). Store bars in a structure that allows iteration in ascending datetime order (e.g. DataFrame sorted by datetime, or iterator over (datetime, bar_dict)). |
| 3.2.2 | Bar iterator | Expose a way to "advance" one bar at a time: e.g. next_bar() returns (datetime, bars_dict) for the current bar time, or yield bars in order. Current time cursor must never go backward; no peek into future. |
| 3.2.3 | get_latest_bars(symbol, N) | Return the last N bars for symbol with datetime &lt;= current cursor time. Used by Strategy to compute signals. If cursor is at T, only bars up to T are visible. Implement via sliding window or index into loaded data. |
| 3.2.4 | Emit MarketEvent | When advancing to a new bar, create MarketEvent(timestamp=bar_datetime, symbols, bar_data) and put it on the event queue (or pass to the next component in the loop). DataHandler is the driver of the loop: each call to "next" produces one MarketEvent. |
| 3.2.5 | Multi-symbol alignment | For pairs, bars must be aligned: use same datetime for both symbols (e.g. merge on datetime, drop rows where either symbol is missing). Emit one MarketEvent per aligned bar time with both legs. |
| 3.2.6 | Tests | Unit test: feed 10 bars; advance one by one; get_latest_bars(2) returns at most 2 bars and only past bars. Assert no future data. |

**Acceptance:** DataHandler loads bars for range, exposes get_latest_bars with no look-ahead, and emits MarketEvents in strict chronological order.

---

#### Step 3.3 — Strategy

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 3.3.1 | Consume MarketEvent | On each MarketEvent, Strategy receives (timestamp, bar data for symbol_a, symbol_b). Strategy has access to OU params (entry_upper, entry_lower, exit_threshold) and either precomputed spread series or an online Kalman. |
| 3.3.2 | Spread and z-score at current bar | For current bar: get spread and β (from precomputed Parquet or by running Kalman update online with price_a, price_b). Compute z_score = (spread - mu) / sigma using OU μ, σ, or use rolling stats from past spread only. |
| 3.3.3 | Signal logic | If z_score &gt;= entry_upper and not already long_spread → emit SignalEvent(direction=long_spread). If z_score &lt;= entry_lower and not already short_spread → emit short_spread. If spread reverts (z between -exit_threshold and +exit_threshold) → emit flat. Track current position state (flat, long_spread, short_spread) to avoid duplicate entries. |
| 3.3.4 | Size | Size can be fixed notional, or from PositionSizer (Phase 5) later. For now, pass through config (e.g. fixed units or notional). Attach hedge_ratio (β) to signal so Portfolio can size legs. |
| 3.3.5 | Emit SignalEvent | Create SignalEvent with timestamp=bar datetime, direction, symbol_a, symbol_b, hedge_ratio, size, metadata (z_score, spread). Put on queue or pass to Portfolio. |
| 3.3.6 | Tests | Unit test: mock bars and OU params; assert long_spread when z &gt; entry_upper, flat when z in exit band. Test state machine (no double entry). |

**Acceptance:** Strategy produces SignalEvents consistent with OU thresholds; uses only current and past data.

---

#### Step 3.4 — Portfolio

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 3.4.1 | State | Maintain: cash balance, positions {symbol: quantity}, initial_capital. Optionally: current_equity, equity_curve list [(ts, equity, cash, positions_value)]. |
| 3.4.2 | On SignalEvent | Given SignalEvent(direction, symbol_a, symbol_b, hedge_ratio, size): Convert to leg orders. Long spread = long symbol_a, short symbol_b (sized by hedge_ratio). Short spread = short symbol_a, long symbol_b. Flat = close both legs. Compute target quantities; emit order(s) to ExecutionHandler (e.g. OrderEvent or direct FillEvent simulation). |
| 3.4.3 | Order to fills | Either Portfolio sends OrderEvent to ExecutionHandler and receives FillEvent(s), or ExecutionHandler is called with order details and returns FillEvent(s). Design: one Fill per leg (two fills per pair trade). |
| 3.4.4 | On FillEvent | Update positions[symbol] += signed quantity (buy +, sell -). Update cash -= quantity * price + commission (buy), cash += quantity * price - commission (sell). Optionally append equity snapshot: ts, equity = cash + sum(position * current_price). Current price: use latest bar close from DataHandler or from the bar that triggered the event. |
| 3.4.5 | Equity curve | After each bar (or each fill), record (run_id, ts, equity, cash, positions_value). Batch insert into `backtest_equity` at end or periodically. |
| 3.4.6 | Run start | On backtest start: insert backtest_runs row (run_id, strategy_name, pair_id, start_ts, end_ts, config_json). Initialize cash = initial_capital, positions = 0. |
| 3.4.7 | Tests | Unit test: one fill buy 10 @ 100 → positions[symbol]=10, cash reduced by 1000 + commission. Two legs (pair): assert both positions and cash consistent. |

**Acceptance:** Portfolio tracks positions and cash; converts signals to orders; updates on fills; records equity curve; writes backtest_runs and backtest_equity.

---

#### Step 3.5 — ExecutionHandler

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 3.5.1 | Receive order | Input: symbol, side, quantity, "reference" price (e.g. bar close). Optional: order type (market assumed for backtest). |
| 3.5.2 | Slippage | Apply slippage: e.g. fill_price = ref_price * (1 + slippage_bps/10000) for buy, (1 - slippage_bps/10000) for sell. Slippage_bps from config (fixed or schedule). Store slippage_bps in FillEvent for audit. |
| 3.5.3 | Commission | Compute commission from config (e.g. per share, per trade, or tiered). Add to FillEvent.commission. |
| 3.5.4 | Bid-ask (optional) | If modeling spread: buy at ref + half_spread, sell at ref - half_spread; then apply slippage. |
| 3.5.5 | Emit FillEvent | Create FillEvent(fill_ts=bar time or order time, symbol, side, quantity, price=fill_price, commission, slippage_bps). Return to Portfolio. |
| 3.5.6 | Link to signal | If backtest_signals and backtest_fills are linked: when writing backtest_fills, set signal_id to the id of the SignalEvent that generated this fill (if available). |
| 3.5.7 | Tests | Unit test: order 10 @ 100, 5 bps slippage, $1 commission → fill price and commission as expected. |

**Acceptance:** ExecutionHandler produces FillEvents with correct price, commission, slippage; config-driven.

---

#### Step 3.6 — Event loop and persistence

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 3.6.1 | Main loop | While DataHandler has next bar: (1) Get next bar; DataHandler emits MarketEvent. (2) Strategy processes MarketEvent; may emit SignalEvent(s). (3) Portfolio processes SignalEvent(s); may send order(s) to ExecutionHandler. (4) ExecutionHandler returns FillEvent(s). (5) Portfolio processes FillEvent(s); updates state; appends equity. (6) Persist: append backtest_signals row(s) for each SignalEvent; append backtest_fills row(s) for each FillEvent. After loop: flush equity curve to backtest_equity. |
| 3.6.2 | Run ID and config | At start: generate run_id (UUID or bt_YYYYMMDD_N); load config (capital, fees, slippage, strategy params); insert backtest_runs. Pass run_id to Portfolio and to any persistence layer. |
| 3.6.3 | Ordering guarantee | Ensure backtest_signals and backtest_fills are written in the same order as event processing (signal_ts and fill_ts non-decreasing). |
| 3.6.4 | Long runs | If equity curve is large: write backtest_equity in batches or to Parquet (run_id, ts, equity, cash, positions_value) instead of many relational rows. |
| 3.6.5 | Integration test | Run full backtest on 1 pair, 1 month: assert backtest_runs has 1 row; backtest_signals and backtest_fills have expected counts; backtest_equity has one row per bar or per fill (per policy). No errors; equity curve non-negative. |

**Acceptance:** End-to-end backtest run completes; all events processed in order; DB has one run with signals, fills, and equity.

---

### 6.3 Microstructure Summary

- **Slippage:** Configurable (e.g. fixed bps or schedule from config); apply to fill price.
- **Fees:** Configurable commission schedule (e.g. per share or per trade); store in FillEvent and in `backtest_fills.commission`.
- **Bid-ask:** Optional spread model (e.g. buy at ask, sell at bid) when converting signal to fill price.

### 6.4 Deliverables (Phase 3)

- Event type definitions and event queue; DataHandler (reads from Module 1, emits MarketEvent).
- Strategy (uses Module 2: Kalman spread + OU thresholds to emit SignalEvent).
- Portfolio (positions, cash, equity curve; on_signal, on_fill).
- ExecutionHandler (slippage, fees, FillEvent).
- End-to-end Python backtest run writing to relational tables (and optional Parquet equity). No C++ in this phase.

---

## 7. Phase 4 — C++ Acceleration (Detailed Plan)

### 7.1 Scope

- **Kalman recursion:** Move per-step update (state transition + measurement update) to C++; expose via pybind11 so Strategy or Alpha module calls into C++ for each new (price_a, price_b).
- **Order matching / execution hot path:** If order-matching or fill simulation is a bottleneck, move inner loop to C++ and expose to ExecutionHandler.

### 7.2 Detailed Step-by-Step Elaboration (Phase 4)

#### Step 4.1 — C++ Kalman implementation

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 4.1.1 | Match Python state-space | Document the exact Python Kalman formulation (state dimension, F, H, Q, R, initial state and P0). Replicate the same equations in C++ so that outputs are bit-level comparable for testing. Use double precision. |
| 4.1.2 | C++ class or free functions | Implement one step: input (price_a, price_b), optional (x, P) from previous step; output (beta, spread, updated x, P). Either stateful class (hold x, P internally) or stateless step function with caller-managed state. Stateful is simpler for Python binding: one object per pair, call update(price_a, price_b) repeatedly. |
| 4.1.3 | Initialization | Constructor or init method: accept initial state vector, P0 matrix, Q, R (or scalars). Allow initialization from Python (e.g. from kalman_params or OLS warm-up). |
| 4.1.4 | Numerical stability | Use stable forms (e.g. Joseph form for covariance update if needed); avoid explicit matrix inverses where possible. Check for NaNs/Infs in output. |
| 4.1.5 | Build system | CMake or setuptools with CMake: build shared library. Link with pybind11; expose class as Python module (e.g. kalman_core.KalmanFilter with update(price_a, price_b) returning tuple (beta, spread)). |
| 4.1.6 | Unit test in C++ | Optional: Google Test or similar; feed fixed (price_a, price_b) sequence; assert beta and spread match reference (from Python or hand-computed). |
| 4.1.7 | Python unit test | From Python: same fixed sequence into C++ Kalman and Python Kalman; assert allclose(beta_cpp, beta_py) and allclose(spread_cpp, spread_py). Tolerate small numerical difference (e.g. 1e-10). |

**Acceptance:** C++ Kalman produces same (or very close) beta and spread as Python for the same inputs; no NaNs.

---

#### Step 4.2 — Python integration and replacement

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 4.2.1 | Wrapper interface | In alpha/kalman, introduce a single entry point (e.g. KalmanHedgeRatio) that can use either Python or C++ implementation via config flag (use_cpp=True/False). Same public API: update(price_a, price_b) -> (beta, spread). |
| 4.2.2 | Replace in pipeline | In spread generation script and in Strategy (if Kalman is run online): call the wrapper; when use_cpp=True, delegate to C++ extension. No change to callers beyond config. |
| 4.2.3 | State persistence | If C++ Kalman is stateful: support get_state() and set_state() (or serialize state to dict/bytes) so that warm start and checkpointing still work. Optionally persist to kalman_params as before. |
| 4.2.4 | Error handling | If C++ throws (e.g. bad input), catch in Python and raise a clear exception. Validate inputs (e.g. finite floats) before calling C++. |
| 4.2.5 | Tests | Run full alpha pipeline (one pair, one month) with use_cpp=True and use_cpp=False; compare spread series (should be identical or within tolerance). Run backtest with both; equity curves should match closely. |

**Acceptance:** Alpha and backtest can run with C++ Kalman; results match Python; config toggles implementation.

---

#### Step 4.3 — Benchmarking

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 4.3.1 | Benchmark setup | Same backtest: one pair, 1 year of daily bars (or 1 month of 1-min bars). Run 10–100 times; measure total wall-clock time. Compare: Python Kalman vs C++ Kalman (same event loop, same Strategy/Portfolio/Execution). |
| 4.3.2 | Profile (optional) | If needed, profile Python to confirm Kalman is the bottleneck (e.g. cProfile). Profile C++ with a small loop to measure per-step cost. |
| 4.3.3 | Document speedup | Record ratio (time_Python / time_C++). Target: meaningful speedup (e.g. 2–10x for Kalman-heavy run). If speedup is small, document that other parts (I/O, Portfolio) dominate. |
| 4.3.4 | CI or script | Add a benchmark script or CI step that runs the comparison and logs timings (optional: fail if regression). |

**Acceptance:** Benchmark shows C++ Kalman reduces runtime where Kalman is dominant; no behavioral regression.

---

#### Step 4.4 — Optional: execution hot path in C++

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 4.4.1 | Identify bottleneck | If profiling shows ExecutionHandler or order/fill loop as hot: consider moving fill simulation (slippage, commission, price computation) to C++. |
| 4.4.2 | C++ fill simulator | Input: list of (symbol, side, quantity, ref_price); config (slippage_bps, commission rule). Output: list of (symbol, side, quantity, fill_price, commission, slippage_bps). Implement in C++; expose via pybind11 (e.g. batch_fill(orders, config) -> list of fills). |
| 4.4.3 | Integration | ExecutionHandler calls C++ batch_fill when processing a batch of orders; convert returned fills to FillEvents. Benchmark again. |
| 4.4.4 | Defer if not needed | If execution is not a bottleneck, leave this step for later or skip. |

**Acceptance:** If implemented, fill simulation in C++ matches Python logic and improves throughput when execution is hot.

---

### 7.3 Deliverables (Phase 4)

- C++ library for Kalman (and optionally execution); pybind11 bindings; tests and benchmarks; integrated into Alpha and Engine.

---

## 8. Phase 5 — Risk Management & Performance Attribution (Detailed Plan)

### 8.1 Interfaces (Design)

- **PerformanceMetrics:** `sharpe(returns)`, `sortino(returns)`, `calmar(returns, drawdown)`, plus total return, max drawdown, num_trades, win_rate (annualized where appropriate).
- **RiskAttribution:** `pca_decompose(returns, factor_returns)` or regression-based → alpha, beta, residuals; monitor market neutrality.
- **PositionSizer:** `kelly(edge, odds)` or `risk_parity(volatilities)` → weights for capital allocation.

### 8.2 Detailed Step-by-Step Elaboration (Phase 5)

#### Step 5.1 — Performance metrics

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 5.1.1 | Input: equity or returns | Accept either equity curve (time series of equity) or precomputed return series. If equity: compute returns as (equity[t] - equity[t-1]) / equity[t-1], or log returns. Align on timestamp; handle first value (no return). |
| 5.1.2 | Total return | total_return = (equity[-1] - equity[0]) / equity[0], or (1 + r1)(1 + r2)... - 1 from return series. Store as decimal (e.g. 0.05 for 5%). |
| 5.1.3 | Max drawdown | From equity curve: peak = running max(equity); drawdown = (peak - equity) / peak; max_drawdown = max(drawdown). From returns: reconstruct equity first or compute drawdown from cumulative returns. Output as decimal (e.g. 0.15 for 15% DD). |
| 5.1.4 | Sharpe ratio (annualized) | Mean(returns) / std(returns) * sqrt(periods_per_year). periods_per_year: 252 for daily, 252*390 for 1-min (US equity), etc. Handle zero std (return 0 or NaN per policy). Risk-free rate optional (subtract from mean). |
| 5.1.5 | Sortino ratio (annualized) | Mean(returns) / downside_std(returns) * sqrt(periods_per_year). Downside std = std of negative returns only (or returns below target). |
| 5.1.6 | Calmar ratio | total_return / max_drawdown (over same period), or annualized return / max_drawdown. Handle zero max_drawdown. |
| 5.1.7 | Trade stats | From backtest_fills or from equity/trade log: num_trades (count of round-trip or count of fills/2 for pairs). Win rate = winning_trades / num_trades (define "win" as profitable trade). Optional: avg win, avg loss, profit factor. |
| 5.1.8 | API | PerformanceMetrics.compute(equity_series or returns_series, fill_events optional, periods_per_year) -> dict with sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate. All from historical data only. |
| 5.1.9 | Tests | Unit test: known return series (e.g. constant 0.01 per period); assert Sharpe and total return. Synthetic equity with one drawdown; assert max_drawdown. |

**Acceptance:** PerformanceMetrics produces all listed metrics; annualization correct for given frequency; no look-ahead.

---

#### Step 5.2 — Risk attribution (PCA / regression)

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 5.2.1 | Inputs | Strategy return series (same frequency as backtest). Factor/benchmark return series (e.g. market index) aligned on dates. Optionally multiple factors. |
| 5.2.2 | Regression-based attribution | Regress strategy_returns = alpha + beta * market_returns + error. OLS: alpha (intercept), beta (slope). Alpha = excess return after accounting for market; beta = market exposure. Pairs strategy should have beta near 0 (market-neutral). Use statsmodels or numpy. |
| 5.2.3 | PCA-based (optional) | If multiple factors: PCA on factor returns; regress strategy returns on first few PCs. Interpret loadings as factor exposures. Document which method is default. |
| 5.2.4 | Output | alpha (annualized if inputs are periodic), beta, residual variance or R². Optional: factor loadings, t-stats. |
| 5.2.5 | API | RiskAttribution.decompose(strategy_returns, factor_returns or factor_matrix) -> dict with alpha, beta, residuals (series), R2. |
| 5.2.6 | Interpretation | Document: "Beta near 0 implies market-neutral. Alpha is strategy return not explained by market." Use for monitoring and reporting. |
| 5.2.7 | Tests | Unit test: strategy_returns = 0.5 * market_returns + noise; assert beta ≈ 0.5. Strategy_returns = constant; assert beta ≈ 0. |

**Acceptance:** RiskAttribution returns alpha and beta; useful for checking market neutrality and reporting.

---

#### Step 5.3 — Position sizing (Kelly / Risk Parity)

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 5.3.1 | Kelly criterion | Input: edge (win probability or expected return), odds (e.g. win/loss ratio). Kelly fraction f* = edge - (1-edge)/odds (simplified form) or (p*b - q)/b for win prob p, loss prob q, b = win/loss ratio. Output: fraction of capital to allocate. Cap at max_fraction (e.g. 0.25) to avoid over-betting. |
| 5.3.2 | Half-Kelly or fractional | Often use f = k * f* with k=0.5 for safety. Make k configurable. |
| 5.3.3 | Risk parity | Input: volatilities (or variances) of assets/legs. Weights inversely proportional to volatility so each leg contributes equally to risk. For two legs: w1 = 1/sigma1 / (1/sigma1 + 1/sigma2). Output: weights that sum to 1 (or to target leverage). |
| 5.3.4 | API | PositionSizer.kelly(edge, odds, fraction=1.0, max_fraction=0.25) -> float. PositionSizer.risk_parity(volatilities) -> array of weights. |
| 5.3.5 | Integration point | Strategy or Portfolio can call PositionSizer to get size or weights before emitting signals or sizing legs. Optional in Phase 5: implement module and unit tests; wire into Strategy in a follow-up if desired. |
| 5.3.6 | Tests | Unit test: Kelly with known edge/odds; assert fraction in [0, max_fraction]. Risk parity with [0.1, 0.2] vol; assert weights sum to 1 and lower vol gets higher weight. |

**Acceptance:** Kelly and Risk Parity implemented and tested; optional wiring to Strategy/Portfolio.

---

#### Step 5.4 — Persistence and wiring

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 5.4.1 | Trigger | After backtest run completes (all bars processed, equity curve final), call PerformanceMetrics.compute() and RiskAttribution.decompose() with run's equity and optional benchmark. Get trade stats from backtest_fills or from Portfolio log. |
| 5.4.2 | Build metrics row | Collect: sharpe_annual, sortino_annual, calmar, max_drawdown, total_return, num_trades, win_rate, alpha, beta. Put full dict in metrics_json (JSON string) for extensibility. computed_at = now(). |
| 5.4.3 | Insert backtest_metrics | INSERT into backtest_metrics (run_id PK/FK, ...). One row per run. On conflict (run_id) either replace or fail (policy: replace so re-runs update metrics). |
| 5.4.4 | Where to call | Option A: inside backtest runner (after loop). Option B: separate script that takes run_id, reads backtest_equity and backtest_fills, computes metrics, writes backtest_metrics. Option B allows recomputing metrics without re-running backtest. |
| 5.4.5 | Benchmark data | For attribution, need benchmark returns. Source: same StorageBackend for a benchmark symbol (e.g. SPY), or separate config path. Align dates with backtest period. |
| 5.4.6 | Tests | Integration test: run backtest, then compute metrics and persist; read backtest_metrics for that run_id; assert sharpe_annual and total_return present and sensible. |

**Acceptance:** After each run, backtest_metrics has one row with all core metrics; attribution uses benchmark when provided.

---

#### Step 5.5 — Reporting (optional)

| Sub-step | Action | Elaboration |
|----------|--------|-------------|
| 5.5.1 | Report script | Script: list recent backtest_runs (e.g. last 10); for each run_id load backtest_metrics and optionally backtest_equity; print or export summary (Sharpe, Sortino, max DD, total return, alpha, beta). Output: console, CSV, or HTML. |
| 5.5.2 | Dashboard (optional) | Simple dashboard (e.g. Streamlit or static HTML): table of runs with key metrics; click run -> equity curve chart. Defer if not in scope. |
| 5.5.3 | Document | Document how to run the report and where metrics are stored. |

**Acceptance:** Optional report or dashboard for viewing backtest_metrics and equity.

---

### 8.3 Deliverables (Phase 5)

- PerformanceMetrics (Sharpe, Sortino, Calmar, max drawdown, total return, num_trades, win_rate); RiskAttribution (alpha, beta); PositionSizer (Kelly, Risk Parity).
- Wiring: engine run completion triggers Module 4; results written to `backtest_metrics`. Optional: report or dashboard that reads from `backtest_runs` and `backtest_metrics`.

---

## 9. Directory Layout (Target)

```
BackTestEngine/
├── system.md
├── system_database.md
├── ROADMAP.md                 # This document
├── README.md
├── requirements.txt
├── pyproject.toml             # Optional
├── src/
│   ├── data/                  # Module 1
│   │   ├── ingestion/         # DataSource implementations, REST/WS
│   │   ├── preprocessing/    # adjust, interpolate, outliers
│   │   └── storage/          # Parquet + relational backends
│   ├── alpha/                 # Module 2
│   │   ├── cointegration/     # ADF, Johansen
│   │   ├── kalman/            # Python + C++ bindings
│   │   └── ou_process/        # OU fit, thresholds
│   ├── engine/                # Module 3
│   │   ├── events.py
│   │   ├── data_handler.py
│   │   ├── strategy.py
│   │   ├── portfolio.py
│   │   ├── execution.py
│   │   └── core/              # C++ extension via pybind11
│   └── risk/                  # Module 4
│       ├── metrics.py
│       ├── attribution.py
│       └── sizing.py
├── tests/
├── scripts/                   # Pipelines, backtests, reports
├── config/                    # YAML/JSON
└── data/                      # Parquet/HDF5 (gitignored or sample)
```

---

## 10. Dependency Order (Module Graph)

- **Module 1 (Data)** depends on nothing; consumed by Module 2 and 3.
- **Module 2 (Alpha)** depends on Module 1; consumed by Module 3.
- **Module 3 (Engine)** depends on Module 1 and 2; consumed by Module 4.
- **Module 4 (Risk)** depends on Module 3; output only (reports, metrics).

Implementation order: 1 → 2 → 3 → (C++ for hot paths) → 4.

---

## 11. Data & Integrity Checklist (From system_database.md)

- **Referential:** symbol_id in corporate_actions, pair_universe, bars/ticks exists in `symbols`; pair_id in cointegration/kalman/ou/spreads exists in `pair_universe`; run_id in signals/fills/equity/metrics exists in `backtest_runs`.
- **Temporal:** Bars strictly increasing in datetime per symbol; corporate actions applied only for bars on or after ex_date; signals and fills within run [start_ts, end_ts]; no event uses future data.
- **Validation:** Bars: open/high/low/close and adj_factor policy; pair: symbol_a &lt; symbol_b; config: fees and slippage in config/env, not hardcoded secrets in DB.

---

## 12. Summary Table

| Phase | Focus | Key outputs | Depends on |
|-------|--------|-------------|------------|
| 1 | Data infrastructure | Schema, Parquet + relational, one connector, preprocessing, validation | — |
| 2 | Alpha research | Cointegration, Kalman, OU, signal spec, DB/Parquet persistence | Phase 1 |
| 3 | Backtesting engine | Events, DataHandler, Strategy, Portfolio, ExecutionHandler, run/signals/fills/equity in DB | Phase 1, 2 |
| 4 | C++ acceleration | Kalman (and optionally execution) in C++, pybind11, benchmarks | Phase 2, 3 |
| 5 | Risk & attribution | Metrics, PCA attribution, Kelly/Risk Parity, backtest_metrics | Phase 3 |

---

*This roadmap is the single reference for implementation order, libraries, and logic. Implement in phases; validate each phase before moving to the next. Update as the project evolves.*
