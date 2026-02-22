"""
Ingest bars from a CSV file → preprocess → write to Parquet and register symbols.

Supports single-file CSVs with columns like Date, Symbol, Open, High, Low, Close, Volume
(e.g. SampleData/sample1/sp500_stocks.csv). Column names are auto-detected.
Loads the CSV once then processes each symbol in memory for speed.

Usage:
  python -m scripts.ingest_csv --file SampleData/sample1/sp500_stocks.csv --root data
  python -m scripts.ingest_csv --file SampleData/sample1/sp500_stocks.csv --symbols AOS AAPL MSFT --root data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.ingestion.csv_source import CSVDataSource
from src.data.ingestion import register_symbol
from src.data.preprocessing import run_pipeline
from src.data.storage import get_engine, create_schema, write_bars
from src.data.storage.parquet_bars import BAR_COLUMNS


def load_and_normalize_csv(csv_path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    """Load CSV once, normalize to raw bar schema, filter by date range. One row per bar; symbol in column."""
    ds = CSVDataSource(file_path=csv_path)
    df = ds._read_csv(csv_path)
    if "datetime" not in df.columns or df.empty:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[(df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))]
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest bars from CSV → Parquet + DB symbols")
    p.add_argument("--file", type=str, required=True, help="Path to CSV (e.g. SampleData/sample1/sp500_stocks.csv)")
    p.add_argument("--symbols", nargs="*", default=None, help="Symbols to ingest (default: discover from CSV, max --max-symbols)")
    p.add_argument("--max-symbols", type=int, default=20, help="When discovering, max symbols to ingest (default: 20)")
    p.add_argument("--root", default="data", help="Root path for Parquet bars")
    p.add_argument("--source", default="csv", help="Partition source label")
    p.add_argument("--db", default=None, help="Database URL (default: sqlite:///data/backtest.db)")
    p.add_argument("--start", default="1990-01-01", help="Start date (YYYY-MM-DD) for fetch range")
    p.add_argument("--end", default="2030-12-31", help="End date (YYYY-MM-DD) for fetch range")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    print("Loading CSV once...")
    full_df = load_and_normalize_csv(path, start_dt, end_dt)
    if full_df.empty:
        print("No data in range.")
        return

    # Require symbol column for multi-symbol CSV
    if "symbol" not in full_df.columns:
        # Single-symbol file: use filename or a default
        full_df["symbol"] = path.stem.upper()
    else:
        full_df["symbol"] = full_df["symbol"].astype(str).str.upper()

    # Drop rows with no valid close
    full_df = full_df.dropna(subset=["close"])
    if full_df.empty:
        print("No rows with valid close.")
        return

    symbols = args.symbols
    if not symbols:
        symbols = full_df["symbol"].unique().tolist()
        if args.max_symbols is not None:
            symbols = symbols[: args.max_symbols]
        print(f"Discovered {len(symbols)} symbols with data (ingesting max {args.max_symbols})")
    else:
        symbols = [s.upper() for s in symbols]

    engine = get_engine(args.db) if args.db else get_engine()
    create_schema(engine)

    for symbol in symbols:
        try:
            bars = full_df[full_df["symbol"] == symbol].copy()
            bars = bars.drop(columns=["symbol"], errors="ignore")
            for c in ["open", "high", "low", "close", "volume"]:
                if c in bars.columns:
                    bars[c] = pd.to_numeric(bars[c], errors="coerce")
                else:
                    bars[c] = float("nan")
            bars = bars.sort_values("datetime").reset_index(drop=True)
            if bars.empty or bars["close"].isna().all():
                print(f"  {symbol}: no data")
                continue
            register_symbol(engine, symbol)
            preprocessed = run_pipeline(bars, symbol, engine, actions=[])
            preprocessed["symbol"] = symbol
            preprocessed = preprocessed.drop(columns=["missing_filled"], errors="ignore")
            preprocessed = preprocessed[[c for c in BAR_COLUMNS if c in preprocessed.columns]]
            for date_str, grp in preprocessed.groupby(preprocessed["datetime"].dt.date):
                partition_key = {"source": args.source, "date": str(date_str)}
                write_bars(args.root, partition_key, grp.reset_index(drop=True))
            print(f"  {symbol}: wrote {len(preprocessed)} bars")
        except Exception as e:
            print(f"  {symbol}: error — {e}")

    print("Done.")


if __name__ == "__main__":
    main()
