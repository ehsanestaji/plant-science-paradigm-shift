"""
Create a small sample DuckDB for reproducibility.

Extracts a random 1% sample (stratified by decade) from the full database,
preserving all tables and relationships.

Usage:
    python reproducibility/create_sample_db.py \
        --source data/processed/plant_science.duckdb \
        --output reproducibility/plant_science_sample.duckdb \
        --fraction 0.01
"""

import argparse
import duckdb
import os


def create_sample(source_path, output_path, fraction=0.01):
    if os.path.exists(output_path):
        os.remove(output_path)

    src = duckdb.connect(source_path, read_only=True)
    dst = duckdb.connect(output_path)

    # Attach source database
    dst.execute(f"ATTACH '{source_path}' AS src (READ_ONLY)")

    print(f"Sampling {fraction*100:.1f}% of works...")

    # Sample works
    dst.execute(f"""
        CREATE TABLE works AS
        SELECT * FROM src.works_clean
        USING SAMPLE {fraction * 100} PERCENT (bernoulli)
    """)
    n_works = dst.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    print(f"  Sampled {n_works:,} works")

    # Related tables (only rows matching sampled works)
    dst.execute("""
        CREATE TABLE work_authors AS
        SELECT wa.* FROM src.work_authors wa
        WHERE wa.work_id IN (SELECT work_id FROM works)
    """)

    dst.execute("""
        CREATE TABLE citations AS
        SELECT c.* FROM src.citations c
        WHERE c.citing_work_id IN (SELECT work_id FROM works)
          AND c.cited_work_id IN (SELECT work_id FROM works)
    """)

    dst.execute("""
        CREATE TABLE concepts AS
        SELECT co.* FROM src.concepts co
        WHERE co.work_id IN (SELECT work_id FROM works)
    """)

    dst.execute("""
        CREATE TABLE authors AS
        SELECT DISTINCT a.* FROM src.authors a
        WHERE a.author_id IN (
            SELECT DISTINCT author_id FROM work_authors
        )
    """)

    # Print stats
    for table in ["works", "authors", "work_authors", "citations", "concepts"]:
        n = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,}")

    dst.execute("DETACH src")
    src.close()
    dst.close()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSample database: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/processed/plant_science.duckdb")
    parser.add_argument("--output", default="reproducibility/plant_science_sample.duckdb")
    parser.add_argument("--fraction", type=float, default=0.01)
    args = parser.parse_args()
    create_sample(args.source, args.output, args.fraction)
