# High-Frequency Statistical Arbitrage: An End-to-End Pairs Trading Framework

A production-oriented framework for identifying, backtesting, and evaluating mean-reverting price relationships (pairs trading) with rigorous statistics, event-driven simulation, and risk-aware performance attribution.

**Design principles:** No look-ahead bias • High throughput (columnar storage, C++ hot paths) • Statistical rigor (cointegration, Kalman hedge ratio, OU spread)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Further Reading](#further-reading)

---

## Architecture Overview

| Module | Purpose |
|--------|--------|
| **1. Data Infrastructure** | Ingestion (REST/CSV), preprocessing (adjustments, interpolation, outliers), columnar storage (Parquet), relational metadata (SQLite/PostgreSQL). |
| **2. Alpha Research** | Cointegration (ADF, Johansen) → Kalman-filtered hedge ratio (β) → OU spread modeling → entry/exit thresholds. |
| **3. Backtesting Engine** | Event-driven loop: DataHandler → Strategy → Portfolio → ExecutionHandler; optional C++ acceleration for Kalman and fill simulation. |
| **4. Risk & Attribution** | Sharpe/Sortino/Calmar, PCA alpha/beta, Kelly/Risk Parity sizing; metrics persisted to DB and reportable via scripts/dashboard. |

Data flow: **Raw sources → Parquet bars → Alpha signals → Backtest (signals/fills/equity) → Metrics & reports.**

---

## Prerequisites

- **Python 3.9+** (3.11+ recommended)
- **C++17 compiler** (optional, for Kalman/execution extensions): MSVC on Windows, `g++` or `clang++` on Linux/macOS
- **Alpha Vantage API key** (optional): only needed for `ingest_alpha_vantage` and `validate_e2e` with ingest; get a free key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

---

## Installation

From the project root:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# Install the package (builds C++ extensions if compiler + pybind11 available)
pip install -e .

# Optional: dev dependencies (pytest, pytest-asyncio)
pip install -e ".[dev]"

# Optional: Streamlit dashboard
pip install -e ".[dashboard]"
```

**Important:** Scripts and the engine expect the project root on `PYTHONPATH` so that `src` and `scripts` are importable. From the project root you can set:

- **Windows (PowerShell):** `$env:PYTHONPATH = (Get-Location).Path`
- **Windows (CMD):** `set PYTHONPATH=%CD%`
- **Linux/macOS/WSL:** `export PYTHONPATH=$(pwd)` or run: `python -m scripts.<script_name> ...` after `pip install -e .` (installing in editable mode adds the project to the path in many environments).

All usage examples below assume you are in the project root with `PYTHONPATH` set (or have installed with `pip install -e .` and run via `python -m scripts.<script>`).

---

## Quick Start

1. **Install:** `pip install -e .` (and optionally `.[dev]` and `.[dashboard]`).
2. **Ingest sample data (e.g. from CSV):**
   ```bash
   python -m scripts.ingest_csv --file path/to/your/bars.csv --root data --max-symbols 5
   ```
   Or use Alpha Vantage (set `ALPHA_VANTAGE_API_KEY` first):
   ```bash
   python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 30 --root data
   ```
3. **Validate pipeline:** `python -m scripts.validate_e2e --root data --source csv --skip-ingest` (adjust `--source` to `alpha_vantage` if you used Alpha Vantage).
4. **Run a backtest:**  
   In-memory (no data on disk):  
   `python -m scripts.run_backtest_e2e`  
   From Parquet:  
   `python -m scripts.run_backtest_e2e --root data --source alpha_vantage --symbols AAPL MSFT --days 30`
5. **View results:** `python -m scripts.report_backtest` or `streamlit run scripts/dashboard.py`.

---

## Detailed Usage

### 1. Data ingestion

#### 1.1 Ingest from CSV

Bars are read from a single CSV (columns such as Date, Symbol, Open, High, Low, Close, Volume), preprocessed, written to Parquet, and symbols registered in the DB.

```bash
# Ingest from CSV; discover up to 10 symbols
python -m scripts.ingest_csv --file SampleData/sample1/sp500_stocks.csv --root data --max-symbols 10

# Ingest specific symbols and date range
python -m scripts.ingest_csv --file SampleData/sample1/sp500_stocks.csv --root data --symbols AOS ABT ABBV --start 2015-01-01 --end 2016-12-31
```

| Option | Default | Description |
|--------|---------|-------------|
| `--file` | (required) | Path to CSV file. |
| `--symbols` | (discover from CSV) | Symbols to ingest. |
| `--max-symbols` | 20 | Max symbols when auto-discovering. |
| `--root` | data | Root directory for Parquet (bars under `root/bars/source=.../date=...`). |
| `--source` | csv | Partition source label. |
| `--start` / `--end` | 1990-01-01 / 2030-12-31 | Date range for filtering bars. |
| `--db` | sqlite:///data/backtest.db | Database URL for symbol registration. |

#### 1.2 Ingest from Alpha Vantage

Fetches daily bars via the Alpha Vantage API, preprocesses them, writes to Parquet, and registers symbols.

- Set `ALPHA_VANTAGE_API_KEY` in the environment.
- Free tier: 5 requests/minute, 25/day; the script throttles automatically.

```bash
# Linux/macOS/WSL
export ALPHA_VANTAGE_API_KEY=your_key_here
python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 30 --root data

# Windows PowerShell
$env:ALPHA_VANTAGE_API_KEY = "your_key_here"
python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 30 --root data
```

| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | (required) | One or more symbols, e.g. `AAPL MSFT`. |
| `--days` | 30 | Number of days of history. |
| `--root` | data | Parquet root path. |
| `--source` | alpha_vantage | Partition source label. |
| `--db` | sqlite:///data/backtest.db | Database URL. |

**Output:** Parquet under `data/bars/source=<source>/date=YYYY-MM-DD/` and rows in the `symbols` table.

---

### 2. End-to-end validation (data pipeline)

Validates: (1) optional ingest, (2) read back from Parquet, (3) sanity checks (no duplicate (symbol, datetime), ascending times, no future data, adj_factor > 0, OHLC non-negative), (4) cross-check.

```bash
# Ingest then validate (needs ALPHA_VANTAGE_API_KEY for ingest)
python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data

# Validate only (existing Parquet; no API key)
python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data --skip-ingest

# With CSV/sample1 and date range
python -m scripts.validate_e2e --root data --source csv --symbols AOS ABT --start 2010-01-01 --end 2016-12-31 --skip-ingest
```

| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | AAPL MSFT | Symbols to validate. |
| `--days` | 7 | Days of history when `--start`/`--end` not set. |
| `--start` / `--end` | (none) | Date range YYYY-MM-DD. |
| `--root` | data | Parquet root. |
| `--source` | alpha_vantage | Partition source. |
| `--db` | default SQLite | Database URL. |
| `--skip-ingest` | off | Skip ingest; only read back and run checks. |

**Success:** Exit code 0 and message: *"Validation complete. Data is ready for Module 2 and 3."*

---

### 3. Running a backtest

The end-to-end backtest script runs the event-driven engine (DataHandler → OUStrategy → Portfolio → ExecutionHandler) and persists runs, signals, fills, and equity to the database.

**In-memory (no Parquet):** Uses synthetic bars; no data directory needed.

```bash
python -m scripts.run_backtest_e2e
```

**From Parquet bars:**

```bash
python -m scripts.run_backtest_e2e --root data --source alpha_vantage --symbols AAPL MSFT --days 30
```

| Option | Default | Description |
|--------|---------|-------------|
| `--root` | None | If set, read bars from Parquet. |
| `--source` | alpha_vantage | Partition source when using `--root`. |
| `--symbols` | AAPL MSFT | Pair of symbols. |
| `--days` | 30 | Days of data when using `--root`. |
| `--db` | in-memory SQLite | Database URL (use a file URL to persist runs). |

**Output:** A run_id and counts for `backtest_runs`, `backtest_signals`, `backtest_fills`, and `backtest_equity`. Use the run_id with `compute_metrics` and `report_backtest`.

---

### 4. Computing and persisting metrics

Recompute performance metrics (and optionally risk attribution) for an existing run from stored equity and fills, then persist to `backtest_metrics`.

```bash
# Recompute and persist metrics for a run
python -m scripts.compute_metrics --run-id <run_id>

# With benchmark returns for alpha/beta
python -m scripts.compute_metrics --run-id <run_id> --benchmark path/to/returns.csv
```

| Option | Description |
|--------|-------------|
| `--run-id` | (Required) Backtest run_id. |
| `--db` | Database URL (default from env or SQLite). |
| `--benchmark` | Path to benchmark returns (CSV one column or .npy) for alpha/beta. |
| `--periods-per-year` | 252 (default) for annualization. |

---

### 5. Reporting backtest results

List recent backtest runs and their metrics (Sharpe, Sortino, max drawdown, total return, alpha, beta, etc.) from `backtest_runs` and `backtest_metrics`.

```bash
# Last 10 runs, console table
python -m scripts.report_backtest

# CSV to stdout or file
python -m scripts.report_backtest --limit 10 --output csv
python -m scripts.report_backtest --output csv --out-file report.csv

# HTML table
python -m scripts.report_backtest --output html --out-file report.html
```

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | 10 | Max runs (ordered by created_at desc). |
| `--output` | console | `console`, `csv`, or `html`. |
| `--db` | default SQLite | Database URL. |
| `--out-file` | (none) | Write output to file (for csv/html). |

---

### 6. Streamlit dashboard

Interactive table of recent runs and equity curve per run. Requires Streamlit: `pip install -e ".[dashboard]"` or `pip install streamlit`.

```bash
streamlit run scripts/dashboard.py
```

Use a specific database:

```bash
streamlit run scripts/dashboard.py -- --db sqlite:///data/backtest.db
```

Use the sidebar to set "Max runs"; select a run in the dropdown to view its equity curve.

---

### 7. Kalman / backtest benchmarking

Compare Python vs C++ Kalman (and full backtest) wall-clock time. Useful to verify C++ acceleration.

```bash
# Default: 252 bars, 20 iterations, full backtest
python -m scripts.benchmark_kalman

# One year daily, 30 runs
python -m scripts.benchmark_kalman --bars 252 --iterations 30

# Kalman-only loop (no Strategy/Portfolio)
python -m scripts.benchmark_kalman --mode kalman_only --bars 5000 --iterations 10

# CI: exit 1 if C++ is slower than Python
python -m scripts.benchmark_kalman --fail-on-regression
```

| Option | Default | Description |
|--------|---------|-------------|
| `--bars` | 252 | Number of bars. |
| `--iterations` | 20 | Runs per implementation. |
| `--mode` | backtest | `backtest` or `kalman_only`. |
| `--fail-on-regression` | off | Exit 1 if C++ is not faster. |

If `kalman_core` is not built, only Python timings are printed.

---

### 8. C++ extensions (optional)

- **Kalman:** Built from `cpp/kalman_hedge_ratio.cpp` and `cpp/kalman_filter.cpp` via `setup.py`. Install with `pip install -e .` (requires pybind11 and a C++17 compiler). Then the engine can use the C++ Kalman path for speed.
- **Execution (fill simulator):** Built from `cpp/fill_simulator.cpp`. When available, `BacktestExecutionHandler(..., use_cpp=True)` uses C++ for fill price/commission; otherwise it falls back to Python. Tests in `tests/test_execution.py` verify parity.

---

## Project Structure

```
BackTestEngine/
├── README.md                 # This file
├── system.md                 # System design and module rundown
├── ROADMAP.md                # Implementation roadmap (phases and steps)
├── system_database.md        # Database and storage schema
├── requirements.txt         # Pip dependencies (also in pyproject.toml)
├── pyproject.toml            # Build metadata, optional deps (dev, dashboard)
├── setup.py                  # C++ extensions (kalman_core, execution_core)
├── src/
│   ├── backtest/             # Engine: events, data_handler, strategy, portfolio, execution, run
│   ├── data/                 # Ingestion, preprocessing, storage (Parquet, schema)
│   └── ...
├── cpp/                      # C++ Kalman and fill simulator (pybind11)
├── scripts/                  # CLI entry points (ingest, validate, backtest, report, dashboard, benchmark)
├── tests/                    # Pytest suite
└── docs/                     # Additional docs (e.g. kalman_state_space.md)
```

Scripts are run as: `python -m scripts.<script_name> [options]`. For full script options and examples, see **scripts/README.md**.

---

## Testing

Run the full test suite from the project root (with `PYTHONPATH` set or after `pip install -e ".[dev]"`):

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Key areas: `tests/test_data_handler.py`, `tests/test_strategy.py`, `tests/test_portfolio.py`, `tests/test_execution.py`, `tests/test_run.py`, `tests/test_performance.py`, `tests/test_risk_attribution.py`, `tests/test_cointegration.py`, `tests/test_kalman.py`, `tests/test_ou.py`, and ingestion/preprocessing/storage tests.

---

## Further Reading

- **system.md** — System design, module breakdown, data flow, and directory layout.
- **ROADMAP.md** — Phased implementation plan and step-by-step tasks.
- **system_database.md** — Relational and columnar schema, tables, and conventions.
- **scripts/README.md** — Detailed script reference and examples.
- **docs/kalman_state_space.md** — Kalman filter state-space description.
