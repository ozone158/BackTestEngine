"""
End-to-end validation (Step 1.5): ingest 2–3 symbols → preprocess → write → read back → sanity checks.

Usage:
  # Ingest then validate (requires ALPHA_VANTAGE_API_KEY for ingest)
  python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data

  # Validate only (use existing Parquet data)
  python -m scripts.validate_e2e --symbols AAPL MSFT --days 7 --root data --skip-ingest
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from src.data.ingestion import AlphaVantageDataSource, register_symbol
from src.data.preprocessing import run_pipeline
from src.data.storage import get_engine, create_schema, read_bars, write_bars
from src.data.storage.parquet_bars import BAR_COLUMNS
from src.data.validation import run_cross_check, run_sanity_checks


def _run_ingest(symbols: list[str], start: datetime, end: datetime, root: str, source: str, db_url: str | None) -> None:
    """Ingest bars for symbols: fetch → register → preprocess → write_bars."""
    engine = get_engine(db_url) if db_url else get_engine()
    create_schema(engine)
    source_ds = AlphaVantageDataSource()
    for symbol in symbols:
        bars = source_ds.fetch(symbol, start, end)
        if bars.empty:
            print(f"  {symbol}: no data (skipped)")
            continue
        register_symbol(engine, symbol)
        preprocessed = run_pipeline(bars, symbol, engine, actions=[])
        preprocessed["symbol"] = symbol
        preprocessed = preprocessed.drop(columns=["missing_filled"], errors="ignore")
        preprocessed = preprocessed[[c for c in BAR_COLUMNS if c in preprocessed.columns]]
        for date_str, grp in preprocessed.groupby(preprocessed["datetime"].dt.date):
            partition_key = {"source": source, "date": str(date_str)}
            write_bars(root, partition_key, grp.reset_index(drop=True))
        print(f"  {symbol}: wrote {len(preprocessed)} bars")


def main() -> int:
    p = argparse.ArgumentParser(description="E2E validation: ingest (optional) → read back → sanity checks")
    p.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT"], help="Symbols (default: AAPL MSFT)")
    p.add_argument("--days", type=int, default=7, help="Days of history when --start/--end not set (default: 7)")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (optional)")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--root", default="data", help="Root path for Parquet bars")
    p.add_argument("--source", default="alpha_vantage", help="Partition source label")
    p.add_argument("--db", default=None, help="Database URL (default: sqlite:///data/backtest.db)")
    p.add_argument("--skip-ingest", action="store_true", help="Skip ingest; only read back and run checks")
    args = p.parse_args()

    if args.start and args.end:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)

    if not args.skip_ingest:
        print("Step 1: Ingesting bars...")
        try:
            _run_ingest(args.symbols, start, end, args.root, args.source, args.db)
        except Exception as e:
            print(f"Ingest failed: {e}", file=sys.stderr)
            return 1
    else:
        print("Step 1: Skipping ingest (--skip-ingest).")

    print("Step 2: Reading back bars...")
    df = read_bars(args.root, args.symbols, start, end, source=args.source)
    if df.empty:
        print("No bars read. Run without --skip-ingest and ensure ALPHA_VANTAGE_API_KEY is set.")
        return 1
    print(f"  Read {len(df)} bars for {df['symbol'].nunique()} symbol(s).")

    print("Step 3: Sanity checks...")
    errors = run_sanity_checks(df, start, end)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        return 1
    print("  All sanity checks passed.")

    print("Step 4: Cross-check (spot-check sample)...")
    cross_issues = run_cross_check(df, sample_per_symbol=3)
    if cross_issues:
        for i in cross_issues:
            print(f"  WARN: {i}", file=sys.stderr)
        return 1
    print("  Cross-check passed.")

    print("Validation complete. Data is ready for Module 2 and 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
