# Database & Storage Architecture: Pairs Trading Framework

## 1. Overview

This document defines the **database structure**, **storage layouts**, **entity relationships**, and **data lifecycle** for the High-Frequency Statistical Arbitrage framework. Storage is split into:

- **Columnar (Parquet / HDF5):** Time series (bars, ticks, spreads, results) — high volume, analytical access.
- **Relational (optional SQLite / PostgreSQL):** Metadata, reference data, backtest runs, and small lookup tables — ACID, relationships, and ad-hoc queries.

All timestamps are stored in **UTC** with microsecond precision unless noted. Symbol and identifier conventions are documented in §6.

---

## 2. Storage Tiers & When to Use What

| Tier | Technology | Use Case | Examples |
|------|------------|----------|----------|
| **Time series** | Apache Parquet (preferred) or HDF5 | OHLCV bars, ticks, spread series, Kalman/OU outputs | `bars/`, `ticks/`, `alpha/` |
| **Metadata & runs** | SQLite (default) or PostgreSQL | Symbols, pairs, corporate actions, backtest configs, run IDs | `symbols`, `backtest_runs`, `cointegration_pairs` |
| **Config / small ref** | JSON/YAML files | Slippage schedules, fee tables, strategy params | `config/` (version-controlled) |

**Rule of thumb:** If it’s a long time series keyed by (symbol, datetime) or (pair_id, datetime), use Parquet/HDF5. If it’s relational (run → signals → fills) or small reference data, use the relational DB. **Columnar API:** Use **Polars** for all Parquet read/write and time-series DataFrames (bars, ticks, spreads); use `scan_parquet` for lazy, predicate-pushdown reads at scale. For ad-hoc SQL over Parquet (e.g. reporting, one-off analytics), **DuckDB** is a popular option (zero-copy, no ETL).

---

## 3. Entity Relationship Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           REFERENCE & METADATA (Relational)                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│  symbols ──┬──< corporate_actions     (1:N)                                       │
│            └──< bar_partition_index    (1:N, optional)                             │
│                                                                                   │
│  pair_universe ──< cointegration_results  (1:N, per test date)                    │
│  pair_universe ──< kalman_params          (1:1 or 1:N by window)                  │
│  pair_universe ──< ou_params              (1:1 or 1:N by window)                  │
│                                                                                   │
│  backtest_runs ──< backtest_signals       (1:N)                                   │
│  backtest_runs ──< backtest_fills         (1:N)                                   │
│  backtest_runs ──< backtest_equity        (1:N, time series)                      │
│  backtest_runs ──< backtest_metrics       (1:1)                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                         TIME SERIES (Columnar: Parquet / HDF5)                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│  bars          (symbol, datetime)  →  open, high, low, close, volume, adj_factor  │
│  ticks         (symbol, datetime)  →  price, size, side, exchange                 │
│  spread_series (pair_id, datetime) →  spread, beta, z_score                       │
│  alpha_output  (pair_id, datetime) →  kalman_state, ou_spread, thresholds        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Relational Schema (Metadata & Runs)

### 4.1 Core Reference Tables

#### `symbols`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `symbol_id` | TEXT (PK) | NOT NULL | Canonical symbol, e.g. `AAPL`, `SPY` |
| `display_name` | TEXT | | Human-readable name |
| `asset_class` | TEXT | | `equity`, `etf`, `future`, etc. |
| `exchange` | TEXT | | Exchange or venue code |
| `currency` | TEXT | | ISO 4217, e.g. `USD` |
| `created_at` | TIMESTAMP | DEFAULT now() | First seen |
| `updated_at` | TIMESTAMP | DEFAULT now() | Last metadata update |

**Notes:** Populated by ingestion; used to validate bar/tick writes and pair universe membership.

---

#### `corporate_actions`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER (PK) | AUTO | Surrogate key |
| `symbol_id` | TEXT (FK) | NOT NULL → symbols | Symbol affected |
| `action_type` | TEXT | NOT NULL | `split`, `dividend`, `rights` |
| `ex_date` | DATE | NOT NULL | Ex-date (effective date) |
| `recorded_at` | TIMESTAMP | | When we ingested |
| `ratio` | REAL | | e.g. split ratio 2.0 for 2:1 |
| `cash_amount` | REAL | | Dividend per share |
| `metadata_json` | TEXT | | Extra (e.g. source URL) |

**Relationships:** Many corporate_actions per symbol. Preprocessing uses this to build adjustment factors; applied in order of `ex_date` to avoid look-ahead.

**Logic:** For splits, `adj_factor` cumulates (multiply by ratio). For dividends, adjustment is price-based (subtract discounted dividend). Store final `adj_factor` on bars (see §5.1).

---

#### `pair_universe`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `pair_id` | TEXT (PK) | NOT NULL | Stable ID, e.g. `AAPL_MSFT`, `SPY_QQQ` |
| `symbol_a` | TEXT (FK) | NOT NULL → symbols | First leg (e.g. spread = price_a - β*price_b) |
| `symbol_b` | TEXT (FK) | NOT NULL → symbols | Second leg |
| `created_at` | TIMESTAMP | | When pair was added |
| `notes` | TEXT | | Optional comment |

**Constraints:** `symbol_a < symbol_b` (lexicographic) to avoid duplicate pairs (AAPL–MSFT vs MSFT–AAPL). `pair_id` can be `symbol_a + '_' + symbol_b`.

---

### 4.2 Alpha Research Outputs (Relational Summary)

#### `cointegration_results`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER (PK) | AUTO | |
| `pair_id` | TEXT (FK) | NOT NULL → pair_universe | |
| `test_ts` | TIMESTAMP | NOT NULL | As-of time of test (no future data) |
| `adf_statistic` | REAL | | ADF test statistic |
| `adf_pvalue` | REAL | | ADF p-value |
| `johansen_trace` | REAL | | Johansen trace stat (if used) |
| `johansen_pvalue` | REAL | | |
| `cointegrating_vector` | TEXT | | JSON: e.g. `[1.0, -β]` |
| `is_cointegrated` | BOOLEAN | | True if passed chosen threshold |
| `created_at` | TIMESTAMP | | |

**Relationships:** Many results per pair (e.g. rolling or periodic re-tests). Unique on `(pair_id, test_ts)` if one test per (pair, time).

**Logic:** Used to screen which pairs are eligible for Kalman/OU and backtest. Only pairs with `is_cointegrated = true` for a given `test_ts` should be traded at that time.

---

#### `kalman_params`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER (PK) | AUTO | |
| `pair_id` | TEXT (FK) | NOT NULL → pair_universe | |
| `window_start` | TIMESTAMP | | Start of estimation window |
| `window_end` | TIMESTAMP | | End (inclusive) |
| `initial_state` | TEXT | | JSON: state vector |
| `process_noise` | REAL | | Q (or scalar) |
| `measurement_noise` | REAL | | R |
| `created_at` | TIMESTAMP | | |

**Logic:** Optional persistence of Kalman initialization per pair/window. Backtest or live can load these to warm-start the filter. Time series of β and spread live in columnar storage (§5.3).

---

#### `ou_params`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER (PK) | AUTO | |
| `pair_id` | TEXT (FK) | NOT NULL → pair_universe | |
| `window_start` | TIMESTAMP | | |
| `window_end` | TIMESTAMP | | |
| `theta` | REAL | | Mean-reversion speed |
| `mu` | REAL | | Long-run mean of spread |
| `sigma` | REAL | | Volatility of spread |
| `entry_upper` | REAL | | Z or spread threshold (long spread) |
| `entry_lower` | REAL | | Short spread |
| `exit_threshold` | REAL | | e.g. back to mean |
| `created_at` | TIMESTAMP | | |

**Logic:** OU parameters and entry/exit bands for a given pair and window. Strategy uses these with the live spread from Kalman to generate signals.

---

### 4.3 Backtest Run & Event Tables

#### `backtest_runs`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `run_id` | TEXT (PK) | NOT NULL | UUID or slug, e.g. `bt_20250217_001` |
| `strategy_name` | TEXT | | Strategy class or config name |
| `pair_id` | TEXT (FK) | → pair_universe | Pair (or NULL if multi-pair) |
| `start_ts` | TIMESTAMP | NOT NULL | Backtest start |
| `end_ts` | TIMESTAMP | NOT NULL | Backtest end |
| `config_json` | TEXT | | Full config (fees, slippage, capital, etc.) |
| `created_at` | TIMESTAMP | | |

**Relationships:** One run has many signals, many fills, one equity series, and one metrics row.

---

#### `backtest_signals`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER (PK) | AUTO | |
| `run_id` | TEXT (FK) | NOT NULL → backtest_runs | |
| `signal_ts` | TIMESTAMP | NOT NULL | Bar/tick time that triggered signal |
| `direction` | TEXT | NOT NULL | `long_spread`, `short_spread`, `flat` |
| `symbol_a` | TEXT | | Leg A (for position) |
| `symbol_b` | TEXT | | Leg B |
| `hedge_ratio` | REAL | | β at signal time |
| `size` | REAL | | Notional or units |
| `metadata_json` | TEXT | | OU z-score, spread value, etc. |

**Logic:** Each row is one signal event. Execution layer consumes these and produces fills. Ordering by `signal_ts` must match event loop order (no look-ahead).

---

#### `backtest_fills`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER (PK) | AUTO | |
| `run_id` | TEXT (FK) | NOT NULL → backtest_runs | |
| `signal_id` | INTEGER (FK) | → backtest_signals | Optional link |
| `fill_ts` | TIMESTAMP | NOT NULL | Simulated fill time |
| `symbol` | TEXT | NOT NULL | Filled symbol |
| `side` | TEXT | NOT NULL | `buy`, `sell` |
| `quantity` | REAL | NOT NULL | Shares/contracts |
| `price` | REAL | NOT NULL | Fill price (after slippage) |
| `commission` | REAL | | Fee |
| `slippage_bps` | REAL | | Slippage in bps (for audit) |

**Logic:** One signal can yield multiple fills (one per leg in pairs). Portfolio and risk modules consume fills to compute PnL and equity curve.

---

#### `backtest_equity`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `run_id` | TEXT (FK) | NOT NULL → backtest_runs | |
| `ts` | TIMESTAMP | NOT NULL | Mark-to-market time |
| `equity` | REAL | NOT NULL | Total equity |
| `cash` | REAL | | Cash balance |
| `positions_value` | REAL | | Mark-to-market positions |

**Notes:** Time series per run. Can be stored in Parquet keyed by `run_id` + `ts` for very long runs; relational is fine for moderate length. Unique on `(run_id, ts)`.

---

#### `backtest_metrics`
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `run_id` | TEXT (PK, FK) | NOT NULL → backtest_runs | One row per run |
| `sharpe_annual` | REAL | | Annualized Sharpe |
| `sortino_annual` | REAL | | Sortino |
| `calmar` | REAL | | Calmar ratio |
| `max_drawdown` | REAL | | Max DD (decimal or %) |
| `total_return` | REAL | | Total return over period |
| `num_trades` | INTEGER | | |
| `win_rate` | REAL | | |
| `alpha` | REAL | | From PCA attribution (if computed) |
| `beta` | REAL | | Market beta |
| `metrics_json` | TEXT | | Full metric set (JSON) |
| `computed_at` | TIMESTAMP | | |

**Logic:** Populated by Module 4 after a run. Single row per run for quick dashboards and comparison.

---

## 5. Columnar Schema (Parquet / HDF5)

All Parquet read/write and time-series DataFrames use **Polars** (dtypes, `read_parquet` / `write_parquet`, `scan_parquet` for lazy reads). Schemas below define column names and types for writer/reader agreement.

### 5.1 Bars (OHLCV)

**Path pattern (Parquet):** `data/bars/{source}/{year}/{month}/` or `data/bars/source={source}/date={YYYY-MM-DD}/`

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Same as `symbols.symbol_id` |
| `datetime` | timestamp[us] (UTC) | Bar open time |
| `open` | float64 | Open price (adjusted if applicable) |
| `high` | float64 | High |
| `low` | float64 | Low |
| `close` | float64 | Close |
| `volume` | float64 | Volume |
| `adj_factor` | float64 | Cumulative adjustment factor (1.0 = no adj) |
| `outlier_flag` | int8 (optional) | 0=normal, 1=outlier (e.g. Z-score/Hampel) |

**Partitioning:** By `source` and by date (year/month or date) to allow predicate pushdown on time range and symbol.  
**Constraints:** `(symbol, datetime)` unique within partition. Bars must be written in ascending `datetime` per symbol to avoid look-ahead when reading.

**Notes:**  
- `adj_factor` is applied to OHLC; volume may be adjusted separately depending on policy.  
- Missing bars (e.g. no trade) can be omitted or filled in preprocessing; if filled, keep a `missing_filled` flag if needed for research.

---

### 5.2 Ticks (Tick-Level)

**Path pattern:** `data/ticks/{source}/date={YYYY-MM-DD}/`

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | |
| `datetime` | timestamp[us] (UTC) | Trade or quote time |
| `price` | float64 | Last/execution price |
| `size` | float64 | Quantity |
| `side` | string (optional) | `buy`, `sell` |
| `exchange` | string (optional) | Venue |

**Partitioning:** By source and date. High volume; prefer Parquet with good compression. Ordering by `(symbol, datetime)` for replay.

---

### 5.3 Spread & Alpha Time Series

**Path pattern:** `data/alpha/spreads/{pair_id}/` or partitioned by date.

| Column | Type | Description |
|--------|------|-------------|
| `pair_id` | string | |
| `datetime` | timestamp[us] (UTC) | Bar time (aligned to bar data) |
| `spread` | float64 | Spread (e.g. price_a - beta * price_b) |
| `beta` | float64 | Kalman hedge ratio at this time |
| `z_score` | float64 (optional) | Normalized spread for thresholds |
| `kalman_state` | string (optional) | JSON state if needed for debugging |

**Relationships:** Logical reference to `pair_universe.pair_id`. One row per (pair_id, datetime). Used by Strategy and for research (OU fit, threshold tuning).

---

### 5.4 OU / Threshold Output (Optional)

**Path:** `data/alpha/ou/{pair_id}/` or single Parquet with `pair_id` column.

| Column | Type | Description |
|--------|------|-------------|
| `pair_id` | string | |
| `window_end` | timestamp[us] | End of estimation window |
| `theta` | float64 | Mean-reversion speed |
| `mu` | float64 | Long-run mean |
| `sigma` | float64 | Volatility |
| `entry_upper` | float64 | |
| `entry_lower` | float64 | |
| `exit_threshold` | float64 | |

Can duplicate or replace relational `ou_params` for bulk research; keep one source of truth (e.g. relational for production, Parquet for batch jobs).

---

## 6. Relationships & Logic Summary

### 6.1 Symbol and Pair Lifecycle

1. **Symbols** are created when first seen by ingestion; corporate_actions are attached by symbol and ex_date.
2. **Pairs** are defined in `pair_universe` (symbol_a, symbol_b); both symbols must exist in `symbols`.
3. **Bars** reference symbols by the same `symbol` string; preprocessing uses `corporate_actions` to set `adj_factor` on bars.

### 6.2 Alpha Pipeline

1. **Cointegration:** Run ADF/Johansen on bar data for each pair up to time T; write `cointegration_results` with `test_ts = T`.
2. **Kalman:** For cointegrated pairs, run Kalman on bars; write `spread_series` (and optionally `kalman_params` for warm start).
3. **OU:** Fit OU on spread series over a window; write `ou_params` (and optionally Parquet OU output). Entry/exit thresholds drive strategy.

### 6.3 Backtest Pipeline

1. Create **backtest_runs** row with run_id, strategy, pair(s), start/end, config.
2. Engine produces **backtest_signals** (ordered by signal_ts) and **backtest_fills** (linked to run_id, optionally signal_id).
3. Portfolio produces **backtest_equity** (run_id, ts, equity, cash, positions_value).
4. Module 4 computes **backtest_metrics** (one row per run_id).

### 6.4 Referential Integrity (Logical)

- Every `symbol_id` in `corporate_actions`, `pair_universe.symbol_a/b`, and bar/tick `symbol` should exist in `symbols` (enforce in app or DB).
- Every `pair_id` in `cointegration_results`, `kalman_params`, `ou_params`, and spread/alpha Parquet should exist in `pair_universe`.
- Every `run_id` in signals, fills, equity, metrics must exist in `backtest_runs`.

---

## 7. Partitioning, Indexing & Performance

### 7.1 Parquet (Polars)

- **API:** Use Polars for all Parquet I/O. `pl.scan_parquet()` with predicate pushdown and `select()` for column pruning; `collect()` only when materialization is needed.
- **Bars:** Partition by `source` and `date` (or year/month). Filter by `symbol` and `datetime` range; use Polars column selection (e.g. only `close`, `volume`) to minimize read size.
- **Ticks:** Partition by date; optional symbol in partition if very high cardinality.
- **Spreads/alpha:** Partition by `pair_id` or by date; keep `pair_id` and `datetime` for range scans.

### 7.2 Relational

- **Indexes:**  
  - `symbols(symbol_id)` — PK.  
  - `corporate_actions(symbol_id, ex_date)`.  
  - `pair_universe(pair_id)`, `(symbol_a, symbol_b)`.  
  - `cointegration_results(pair_id, test_ts)`.  
  - `backtest_signals(run_id, signal_ts)`, `backtest_fills(run_id, fill_ts)`, `backtest_equity(run_id, ts)`.
- **backtest_runs:** Index on `(start_ts, end_ts)` or `created_at` for listing runs.

### 7.3 HDF5 (If Used Instead of Parquet)

- **Structure:** Group per symbol or pair, e.g. `/bars/AAPL`, `/spreads/AAPL_MSFT`. Datasets: `datetime`, `open`, `high`, `low`, `close`, `volume`.  
- **Chunking:** Chunk by time (e.g. 1 day or 1 week) for efficient range reads.  
- **Index:** Keep a small index table (symbol → HDF5 path + row ranges) in relational or a separate index file.

---

## 8. Data Validation & Constraints

| Layer | Rule | Enforcement |
|-------|------|-------------|
| Bars | `datetime` strictly increasing per symbol within partition | Application (and optional DB trigger if bars in DB) |
| Bars | `open, high, low, close >= 0` (or policy) | Application / schema |
| Bars | `adj_factor > 0` | Application |
| Corporate actions | `ex_date` used only for bars on or after that date | Preprocessing logic |
| Signals/Fills | `signal_ts` and `fill_ts` within run’s [start_ts, end_ts] | Application |
| Backtest | No event timestamp beyond data availability | DataHandler + event loop |
| Pair | `symbol_a != symbol_b`, `symbol_a < symbol_b` | DB check or app |

---

## 9. Conventions & Notes

### 9.1 Identifiers

- **symbol_id:** Upper case, no spaces (e.g. `AAPL`, `SPY`).  
- **pair_id:** `{symbol_a}_{symbol_b}` with symbol_a < symbol_b.  
- **run_id:** UUID or `bt_{date}_{sequence}` for traceability.

### 9.2 Time

- All stored times in **UTC**. Convert from exchange local time at ingestion.  
- Bar `datetime` = bar **open** time.  
- Use consistent resolution (e.g. 1min bars → 1min resolution; no mixing with tick timestamps in same series).

### 9.3 Versioning & Reprocessing

- **Schema version:** Store in Parquet metadata or a small `schema_version` table; support read path for at least N-1.  
- **Reprocessing:** Identify bars by (symbol, datetime). Overwrite or append with “version” or “batch_id” if you need to reprocess and keep history.  
- **Backtest runs:** Immutable once completed; new config = new run_id.

### 9.4 Backups & Retention

- Parquet: Incremental backup by partition; retain raw + adjusted bars per retention policy.  
- Relational: Daily dump or WAL archiving; retain backtest_runs and metrics per policy.  
- Corporate actions: Keep full history for re-adjustment.

### 9.5 Security & Access

- Credentials and API keys not in DB; use env or secret manager.  
- Restrict write access to ingestion and backtest services; read-only for research and reporting.

---

## 10. Quick Reference: Where Is What?

| Data | Primary storage | Key(s) | Used by |
|------|-----------------|--------|---------|
| Symbol list | Relational `symbols` | symbol_id | Ingestion, pair universe, validation |
| Corporate actions | Relational `corporate_actions` | symbol_id, ex_date | Preprocessing |
| OHLCV bars | Parquet `bars/` | symbol, datetime | Alpha, backtest DataHandler |
| Ticks | Parquet `ticks/` | symbol, datetime | Optional tick backtest |
| Pairs | Relational `pair_universe` | pair_id | Alpha, backtest |
| Cointegration results | Relational `cointegration_results` | pair_id, test_ts | Alpha screening |
| Spread series | Parquet `alpha/spreads/` | pair_id, datetime | Strategy, OU fit |
| OU params | Relational `ou_params` or Parquet | pair_id, window | Strategy |
| Backtest run | Relational `backtest_runs` | run_id | Engine, reporting |
| Signals | Relational `backtest_signals` | run_id, signal_ts | Engine, audit |
| Fills | Relational `backtest_fills` | run_id, fill_ts | Portfolio, risk |
| Equity curve | Relational `backtest_equity` or Parquet | run_id, ts | Risk, reporting |
| Metrics | Relational `backtest_metrics` | run_id | Module 4, dashboards |

---

*This document is the single source of truth for database and storage design. Update it when adding entities, partitions, or changing relationships.*
