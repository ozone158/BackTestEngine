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
