"""
Export abstracts from DuckDB for embedding computation.

Produces a Parquet file with columns: work_id, year, abstract, journal_name, oa_status.
Only works with non-empty abstracts are included.

Usage (on HPC):
    python -m src.nlp.export_abstracts --db-path data/processed/plant_science.duckdb

Then copy to MacBook:
    scp HPC_HOST:~/path/to/abstracts_for_embedding.parquet data/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.storage_monitor import check_storage


OUT_PATH = "data/abstracts_for_embedding.parquet"


def export_abstracts(db_path: str, out_path: str = OUT_PATH):
    """Export all works with abstracts to Parquet."""
    import duckdb

    print("=== Exporting abstracts for embedding ===", flush=True)
    t0 = time.time()

    con = duckdb.connect(db_path, read_only=True)

    query = """
        SELECT
            work_id,
            year,
            abstract,
            journal_name,
            oa_status
        FROM works_clean
        WHERE abstract IS NOT NULL
          AND LENGTH(TRIM(abstract)) > 20
        ORDER BY year, work_id
    """

    print("  Running query...", flush=True)
    df = con.execute(query).df()
    con.close()

    print(f"  {len(df):,} abstracts extracted", flush=True)
    print(f"  Year range: {df['year'].min()}--{df['year'].max()}", flush=True)
    print(f"  Mean abstract length: {df['abstract'].str.len().mean():.0f} chars", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"  Saved to {out_path} ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)

    return df


def main():
    ap = argparse.ArgumentParser(description="Export abstracts for embedding")
    ap.add_argument("--db-path", required=True, help="Path to DuckDB database")
    ap.add_argument("--out-path", default=OUT_PATH, help="Output parquet path")
    args = ap.parse_args()

    check_storage()
    export_abstracts(args.db_path, args.out_path)


if __name__ == "__main__":
    main()
