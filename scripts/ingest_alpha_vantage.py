"""
Ingest bars from Alpha Vantage → preprocess → write to Parquet and register symbols (Step 1.4.5).

Usage:
  set ALPHA_VANTAGE_API_KEY=your_key
  python -m scripts.ingest_alpha_vantage --symbols AAPL MSFT --days 30 --root data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.data.ingestion import AlphaVantageDataSource, register_symbol
from src.data.preprocessing import run_pipeline
from src.data.storage import get_engine, create_schema, write_bars
from src.data.storage.parquet_bars import BAR_COLUMNS

def main():
    p = argparse.ArgumentParser(description="Ingest Alpha Vantage daily bars → Parquet + DB symbols")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols to fetch (e.g. AAPL MSFT)")
    p.add_argument("--days", type=int, default=30, help="Number of days of history")
    p.add_argument("--root", default="data", help="Root path for Parquet bars")
    p.add_argument("--source", default="alpha_vantage", help="Partition source label")
    p.add_argument("--db", default=None, help="Database URL (default: sqlite:///data/backtest.db)")
    args = p.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    engine = get_engine(args.db) if args.db else get_engine()
    create_schema(engine)

    source = AlphaVantageDataSource()
    for symbol in args.symbols:
        try:
            bars = source.fetch(symbol, start, end)
            if bars.empty:
                print(f"{symbol}: no data")
                continue
            register_symbol(engine, symbol)
            preprocessed = run_pipeline(bars, symbol, engine, actions=[])
            preprocessed["symbol"] = symbol
            preprocessed = preprocessed.drop(columns=["missing_filled"], errors="ignore")
            preprocessed = preprocessed[[c for c in BAR_COLUMNS if c in preprocessed.columns]]
            for date_str, grp in preprocessed.groupby(preprocessed["datetime"].dt.date):
                partition_key = {"source": args.source, "date": str(date_str)}
                write_bars(args.root, partition_key, grp.reset_index(drop=True))
            print(f"{symbol}: wrote {len(preprocessed)} bars")
        except Exception as e:
            print(f"{symbol}: error — {e}")

if __name__ == "__main__":
    main()
