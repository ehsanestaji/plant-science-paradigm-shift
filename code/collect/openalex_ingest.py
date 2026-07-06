"""
Ingest downloaded OpenAlex JSONL files into DuckDB.

Phase 2 of collection: reads compressed JSONL files from data/raw/openalex_api/
and bulk-loads them into the DuckDB database. Uses DuckDB's native JSON reading
for maximum speed.

Usage (SLURM job or login node):
    python -u -m src.collect.openalex_ingest \
        --input-dir data/raw/openalex_api \
        --db-path data/processed/plant_science.duckdb
"""

import os
import sys
import gzip
try:
    import orjson as json  # 2-4x faster JSON parsing (C extension)
    # orjson.loads is API-compatible with json.loads
except ImportError:
    import json
import time
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database, get_stats
from src.utils.storage_monitor import check_storage
from src.utils.checkpointing import checkpoint_save, checkpoint_load, checkpoint_clear


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inverted_index:
        return None
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return None
    max_pos = max(positions.keys())
    words = [positions.get(i, "") for i in range(max_pos + 1)]
    return " ".join(words).strip()


def process_file(filepath: str, con) -> dict:
    """Process a single JSONL.gz file and insert into DuckDB using batch operations."""
    stats = {"works": 0, "authors": 0, "citations": 0, "errors": 0}

    # Accumulate rows in memory for bulk insert
    work_rows = []
    author_rows = []
    work_author_rows = []
    concept_rows = []
    citation_rows = []
    funder_rows = []

    opener = gzip.open if filepath.endswith(".gz") else open

    try:
        fh = opener(filepath, "rb")
    except Exception as e:
        print(f"    [warn] cannot open {filepath}: {e}", flush=True)
        return stats

    try:
        while True:
            try:
                line_bytes = fh.readline()
            except Exception as e:
                print(f"    [warn] gzip read error at {stats['works']:,} works: {e}", flush=True)
                break
            if not line_bytes:
                break
            try:
                work = json.loads(line_bytes)
            except Exception:
                stats["errors"] += 1
                continue

            primary = work.get("primary_location") or {}
            source = primary.get("source") or {}

            work_id = (work.get("id") or "").replace("https://openalex.org/", "")
            doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

            work_rows.append([
                work_id, doi, work.get("title"), abstract,
                work.get("publication_year"), work.get("publication_date"),
                (source.get("id") or "").replace("https://openalex.org/", "") or None,
                source.get("display_name"), source.get("issn_l"),
                (work.get("open_access") or {}).get("oa_status"),
                work.get("cited_by_count", 0),
                len(work.get("referenced_works") or []),
                work.get("type"), work.get("language"),
            ])
            stats["works"] += 1

            # Authors
            for i, authorship in enumerate(work.get("authorships") or []):
                author = authorship.get("author") or {}
                institutions = authorship.get("institutions") or []
                inst = institutions[0] if institutions else {}
                author_id = (author.get("id") or "").replace("https://openalex.org/", "") or f"unk_{work_id}_{i}"
                orcid = (author.get("orcid") or "").replace("https://orcid.org/", "") or None
                author_rows.append([author_id, author.get("display_name"), orcid])
                work_author_rows.append([
                    work_id, author_id, i + 1,
                    authorship.get("is_corresponding", False),
                    (inst.get("id") or "").replace("https://openalex.org/", "") or None,
                    inst.get("display_name"), inst.get("country_code"),
                ])
                stats["authors"] += 1

            # Concepts
            for c in work.get("concepts") or []:
                concept_rows.append([
                    work_id,
                    (c.get("id") or "").replace("https://openalex.org/", ""),
                    c.get("display_name"), c.get("level"), c.get("score"),
                ])

            # Citations
            for ref in work.get("referenced_works") or []:
                cited_id = ref.replace("https://openalex.org/", "")
                citation_rows.append([work_id, cited_id])
                stats["citations"] += 1

            # Funders
            for grant in work.get("grants") or []:
                funder_name = grant.get("funder_display_name")
                funder_id = grant.get("funder", "")
                if isinstance(funder_id, str):
                    funder_id = funder_id.replace("https://openalex.org/", "")
                else:
                    funder_id = None
                funder_rows.append([work_id, funder_id, funder_name, grant.get("award_id")])

            # Flush to DB every 100K works (larger batch = fewer round-trips)
            if stats["works"] % 100000 == 0:
                _flush_to_db(con, work_rows, author_rows, work_author_rows,
                             concept_rows, citation_rows, funder_rows)
                work_rows.clear()
                author_rows.clear()
                work_author_rows.clear()
                concept_rows.clear()
                citation_rows.clear()
                funder_rows.clear()
                print(f"    {stats['works']:,} works processed...", flush=True)

    except Exception as e:
        print(f"    [warn] unexpected error in {filepath}: {e}", flush=True)
    finally:
        try:
            fh.close()
        except Exception:
            pass

    # Final flush
    _flush_to_db(con, work_rows, author_rows, work_author_rows,
                 concept_rows, citation_rows, funder_rows)

    return stats


def _flush_to_db(con, work_rows, author_rows, work_author_rows,
                 concept_rows, citation_rows, funder_rows):
    """Bulk insert accumulated rows into DuckDB via pandas DataFrames.

    pandas → DuckDB is zero-copy (Arrow), orders of magnitude faster than
    executemany for large batches (citations can be millions of rows).
    """
    if not work_rows:
        return

    con.execute("BEGIN TRANSACTION")
    try:
        # Works — register DataFrame, INSERT OR IGNORE via SQL
        df_works = pd.DataFrame(work_rows, columns=[
            'work_id', 'doi', 'title', 'abstract', 'year', 'publication_date',
            'journal_id', 'journal_name', 'journal_issn', 'oa_status',
            'cited_by_count', 'reference_count', 'type', 'language',
        ])
        con.register('_df_works', df_works)
        con.execute("""
            INSERT OR IGNORE INTO works
            (work_id, doi, title, abstract, year, publication_date,
             journal_id, journal_name, journal_issn, oa_status,
             cited_by_count, reference_count, type, language, source)
            SELECT work_id, doi, title, abstract, year, publication_date,
                   journal_id, journal_name, journal_issn, oa_status,
                   cited_by_count, reference_count, type, language, 'openalex'
            FROM _df_works
        """)
        con.unregister('_df_works')

        # Authors
        if author_rows:
            df_authors = pd.DataFrame(author_rows,
                columns=['author_id', 'display_name', 'orcid'])
            con.register('_df_authors', df_authors)
            con.execute("""
                INSERT OR IGNORE INTO authors (author_id, display_name, orcid)
                SELECT DISTINCT author_id, display_name, orcid FROM _df_authors
            """)
            con.unregister('_df_authors')

        # Work-authors
        if work_author_rows:
            df_wa = pd.DataFrame(work_author_rows, columns=[
                'work_id', 'author_id', 'author_position', 'is_corresponding',
                'institution_id', 'institution_name', 'country_code',
            ])
            con.register('_df_wa', df_wa)
            con.execute("""
                INSERT OR IGNORE INTO work_authors
                (work_id, author_id, author_position, is_corresponding,
                 institution_id, institution_name, country_code)
                SELECT * FROM _df_wa
            """)
            con.unregister('_df_wa')

        # Concepts
        if concept_rows:
            df_concepts = pd.DataFrame(concept_rows, columns=[
                'work_id', 'concept_id', 'concept_name', 'level', 'score',
            ])
            con.register('_df_concepts', df_concepts)
            con.execute("""
                INSERT OR IGNORE INTO concepts
                (work_id, concept_id, concept_name, level, score)
                SELECT * FROM _df_concepts
            """)
            con.unregister('_df_concepts')

        # Citations — largest batch; pandas→Arrow is critical here
        if citation_rows:
            df_cit = pd.DataFrame(citation_rows,
                columns=['citing_work_id', 'cited_work_id'])
            con.register('_df_cit', df_cit)
            con.execute("""
                INSERT OR IGNORE INTO citations (citing_work_id, cited_work_id)
                SELECT * FROM _df_cit
            """)
            con.unregister('_df_cit')

        # Funders (no unique constraint — simple append via DataFrame)
        if funder_rows:
            df_funders = pd.DataFrame(funder_rows, columns=[
                'work_id', 'funder_id', 'funder_name', 'award_id',
            ])
            con.register('_df_funders', df_funders)
            con.execute("""
                INSERT INTO funders (work_id, funder_id, funder_name, award_id)
                SELECT * FROM _df_funders
            """)
            con.unregister('_df_funders')

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenAlex JSONL files into DuckDB")
    parser.add_argument("--input-dir", default="data/raw/openalex_api",
                        help="Directory with .jsonl.gz files")
    parser.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate all tables before ingesting (clean start)")
    args = parser.parse_args()

    check_storage()

    if args.reset:
        import duckdb as _duckdb
        print("--reset: dropping all existing tables for clean start...", flush=True)
        con_tmp = _duckdb.connect(args.db_path)
        for tbl in ["works", "authors", "work_authors", "citations", "concepts", "funders", "ingestion_log"]:
            con_tmp.execute(f"DROP TABLE IF EXISTS {tbl}")
        con_tmp.close()
        checkpoint_clear("openalex_ingest")
        print("Tables dropped. Starting fresh.", flush=True)

    con = create_database(args.db_path)

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.jsonl.gz"))
    if not files:
        files = sorted(input_dir.glob("*.jsonl"))
    print(f"Found {len(files)} files in {input_dir}", flush=True)

    # Check for checkpoint
    ckpt = checkpoint_load("openalex_ingest")
    processed_files = set(ckpt.get("processed_files", [])) if ckpt else set()
    if processed_files:
        print(f"Resuming: {len(processed_files)} files already processed", flush=True)

    total_works = 0
    for i, filepath in enumerate(files):
        fname = str(filepath)
        if fname in processed_files:
            continue

        print(f"\n[{i+1}/{len(files)}] Processing {filepath.name} "
              f"({filepath.stat().st_size / 1e6:.0f} MB)...", flush=True)
        t0 = time.time()

        stats = process_file(fname, con)
        elapsed = time.time() - t0
        total_works += stats["works"]

        con.execute("""
            INSERT INTO ingestion_log (source, file_path, records_total, records_matched)
            VALUES ('openalex_api', ?, ?, ?)
        """, [fname, stats["works"], stats["works"]])

        print(f"  {stats['works']:,} works, {stats['authors']:,} authors, "
              f"{stats['citations']:,} citations ({elapsed:.0f}s, {stats['errors']} errors)",
              flush=True)

        processed_files.add(fname)
        checkpoint_save("openalex_ingest", {"processed_files": list(processed_files)})

        if (i + 1) % 3 == 0:
            check_storage()

    checkpoint_clear("openalex_ingest")

    db_stats = get_stats(con)
    print(f"\nIngestion complete! Total works ingested: {total_works:,}", flush=True)
    print(f"Database stats:", flush=True)
    for k, v in db_stats.items():
        print(f"  {k}: {v}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
