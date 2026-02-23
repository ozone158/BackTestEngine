# Scripts

## ingest_alpha_vantage (Step 1.4.5)

Fetches daily bars from Alpha Vantage → preprocesses (adjust, interpolate, detect_outliers) → writes to Parquet and registers symbols in the DB.

**Requirements**

- `ALPHA_VANTAGE_API_KEY` environment variable (get a free key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key))
- Rate limits (free tier): 5 requests/minute, 25/day — the script throttles automatically

**Usage**

```bash
# From project root, with PYTHONPATH set
set PYTHONPATH=e:\Project\BackTestEngine
set ALPHA_VANTAGE_API_KEY=your_key_here
python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 30 --root data
```

**Options**

| Option     | Default          | Description                          |
|-----------|------------------|--------------------------------------|
| `--symbols` | (required)       | One or more symbols, e.g. `AAPL MSFT` |
| `--days`    | 30               | Number of days of history            |
| `--root`    | data             | Root path for Parquet (bars under `root/bars/source=.../date=...`) |
| `--source`  | alpha_vantage    | Partition source label               |
| `--db`      | (default SQLite) | Database URL for symbol registration |

**Expected output**

- Parquet files under `data/bars/source=alpha_vantage/date=YYYY-MM-DD/`
- Rows in `symbols` for each fetched symbol
- One line per symbol: `SYMBOL: wrote N bars`

---

## ingest_csv

Ingest bars from a single CSV file (e.g. **SampleData/sample1**) → preprocess → write to Parquet and register symbols. Loads the CSV once then processes each symbol in memory. Column names (Date, Symbol, Open, High, Low, Close, Volume) are auto-detected.

**Usage (SampleData/sample1)**

```bash
set PYTHONPATH=e:\Project\BackTestEngine
# Ingest first 10 symbols with data from sample1 (default source=csv)
python -m scripts.ingest_csv --file SampleData/sample1/sp500_stocks.csv --root data --max-symbols 10

# Ingest specific symbols and date range
python -m scripts.ingest_csv --file SampleData/sample1/sp500_stocks.csv --root data --symbols AOS ABT ABBV --start 2015-01-01 --end 2016-12-31
```

**Options**

| Option         | Default       | Description                                      |
|----------------|---------------|--------------------------------------------------|
| `--file`       | (required)    | Path to CSV (e.g. SampleData/sample1/sp500_stocks.csv) |
| `--symbols`    | (discover)    | Symbols to ingest; if omitted, discover from CSV (up to --max-symbols) |
| `--max-symbols`| 20            | When discovering, max symbols to ingest         |
| `--root`       | data          | Parquet root path                               |
| `--source`     | csv           | Partition source label                          |
| `--start` / `--end` | 1990-01-01 / 2030-12-31 | Date range for filtering bars            |

**Validate sample1 after ingest**

```bash
python -m scripts.validate_e2e --root data --source csv --symbols AOS ABT --start 2010-01-01 --end 2016-12-31 --skip-ingest
```

---

## validate_e2e (Step 1.5)

End-to-end validation: (1) optionally ingest bars for 2–3 symbols over a short range, (2) read back from Parquet, (3) run sanity checks and a spot-check.

**Usage**

```bash
# Ingest then validate (requires ALPHA_VANTAGE_API_KEY)
set PYTHONPATH=e:\Project\BackTestEngine
set ALPHA_VANTAGE_API_KEY=your_key_here
python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data
```

```bash
# Validate only (use existing Parquet data; no API key needed)
python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data --skip-ingest
```

**Options**

| Option        | Default        | Description                                      |
|---------------|----------------|--------------------------------------------------|
| `--symbols`   | AAPL MSFT      | Symbols to validate                              |
| `--days`      | 7              | Days of history when --start/--end not set       |
| `--start` / `--end` | (none)   | Optional date range YYYY-MM-DD (e.g. for CSV/sample1) |
| `--root`      | data           | Parquet root path                                |
| `--source`    | alpha_vantage  | Partition source label                           |
| `--db`        | (default)      | Database URL                                     |
| `--skip-ingest` | (off)        | Skip ingest; only read back and run checks       |

**Sanity checks (1.5.2)**

- No duplicate `(symbol, datetime)`
- Datetime strictly increasing per symbol
- All bar datetimes within requested range (no future data)
- `adj_factor` present and > 0
- OHLC non-negative

**Cross-check (1.5.3)**

- Spot-check of a sample of bars: high ≥ low, adj_factor > 0

**Expected output**

- Step 1: Ingest lines (or “Skipping ingest”)
- Step 2: “Read N bars for M symbol(s).”
- Step 3: “All sanity checks passed.”
- Step 4: “Cross-check passed.”
- “Validation complete. Data is ready for Module 2 and 3.”

Exit code 0 if all pass; 1 on failure.

---

## benchmark_kalman (Step 4.3)

Compares wall-clock time: same backtest (or Kalman-only loop) with Python vs C++ Kalman. Logs ratio `time_Python / time_C++` (4.3.3). Optional: fail if C++ is not faster (regression).

**Usage**

```bash
set PYTHONPATH=e:\Project\BackTestEngine
# Default: 252 bars, 20 iterations, full backtest
python -m scripts.benchmark_kalman

# One year daily, 30 runs
python -m scripts.benchmark_kalman --bars 252 --iterations 30

# Per-step cost only (no Strategy/Portfolio)
python -m scripts.benchmark_kalman --mode kalman_only --bars 5000 --iterations 10

# CI: exit 1 if C++ is slower than Python
python -m scripts.benchmark_kalman --fail-on-regression
```

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--bars` | 252 | Number of bars (e.g. 252 = 1 year daily) |
| `--iterations` | 20 | Runs per implementation |
| `--mode` | backtest | `backtest` = full backtest; `kalman_only` = tight Kalman update loop (4.3.2) |
| `--fail-on-regression` | off | Exit 1 if C++ is not faster than Python |

**Expected output**

- Mode, bars, iterations.
- Python total time and per-run (or per-step) time.
- C++ total time and ratio (Python/C++). If ratio &lt; 1, note that I/O/Portfolio may dominate.
- If `kalman_core` is not built, only Python timings are printed.

---

## Execution C++ (Step 4.4)

Fill simulation can use a C++ extension for throughput. Build with `pip install -e .` (requires pybind11 and a C++17 compiler). This builds both `kalman_core` and `execution_core`. Then use `BacktestExecutionHandler(..., use_cpp=True)` in your backtest; when `execution_core` is available, fill price and commission are computed in C++. Same results as Python; tests in `tests/test_execution.py` (`test_use_cpp_matches_python`) verify. If the extension is not built, `use_cpp=True` falls back to Python.

---

## compute_metrics (Step 5.4.4 Option B)

Recompute and persist backtest metrics for a run without re-running the backtest. Reads `backtest_equity` and `backtest_fills`, runs PerformanceMetrics (and optionally RiskAttribution if benchmark provided), writes/updates `backtest_metrics`.

**Usage**

```bash
set PYTHONPATH=e:\Project\BackTestEngine
python -m scripts.compute_metrics --run-id <run_id> [--db URL]
python -m scripts.compute_metrics --run-id <run_id> --benchmark returns.csv
```

**Options**

| Option | Description |
|--------|-------------|
| `--run-id` | (Required) backtest run_id |
| `--db` | Database URL (default: from env or sqlite) |
| `--benchmark` | Path to benchmark returns (CSV one column or .npy) for alpha/beta |
| `--periods-per-year` | 252 (default) for annualization |

---

## report_backtest (Step 5.5)

Lists recent backtest runs and their metrics (Sharpe, Sortino, max drawdown, total return, alpha, beta, etc.). Reads from **backtest_runs** and **backtest_metrics** (left join so runs without metrics still appear).

**Usage**

```bash
set PYTHONPATH=e:\Project\BackTestEngine
# Console table (default), last 10 runs
python -m scripts.report_backtest

# CSV to stdout or file
python -m scripts.report_backtest --limit 10 --output csv
python -m scripts.report_backtest --output csv --out-file report.csv

# HTML table
python -m scripts.report_backtest --output html --out-file report.html
```

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | 10 | Max number of runs to list (ordered by created_at desc) |
| `--output` | console | `console`, `csv`, or `html` |
| `--db` | (default SQLite) | Database URL |
| `--out-file` | (none) | Write output to file (csv/html); default stdout |

---

## dashboard (Step 5.5.2)

Streamlit dashboard: table of recent backtest runs with key metrics; select a run to view its equity curve chart.

**Requirements**

- `pip install -e ".[dashboard]"` (adds Streamlit), or `pip install streamlit`
- Set `PYTHONPATH` to project root so `scripts` and `src` are importable

**Usage**

```bash
set PYTHONPATH=e:\Project\BackTestEngine
streamlit run scripts/dashboard.py
```

To use a specific database:

```bash
streamlit run scripts/dashboard.py -- --db sqlite:///data/backtest.db
```

Use the sidebar to change "Max runs". Click a run in the dropdown to see its equity curve.
