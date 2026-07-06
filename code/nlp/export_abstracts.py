"""
Export abstracts to a portable parquet file for GPU embedding on MacBook.

Usage:
    python -m src.nlp.export_abstracts --db-path data/processed/plant_science.duckdb
    # Output: data/abstracts_for_embedding.parquet  (~800MB)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_PATH = "data/abstracts_for_embedding.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    check_storage()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='20GB'")

    print("Exporting abstracts to parquet…", flush=True)
    con.execute(f"""
        COPY (
            SELECT work_id,
                   year,
                   title,
                   abstract,
                   journal_name,
                   oa_status,
                   cited_by_count
            FROM works_clean
            WHERE abstract IS NOT NULL
              AND length(abstract) > 100
            ORDER BY cited_by_count DESC NULLS LAST
        ) TO '{args.out}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()

    import os
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out}  ({size_mb:.0f} MB)", flush=True)
    print(f"Done in {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
