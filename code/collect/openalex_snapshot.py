"""
Stream-filter OpenAlex snapshot for plant science works.

The OpenAlex monthly snapshot is ~330GB compressed (1.6TB decompressed).
We NEVER decompress the full dataset. Instead, we stream each compressed
JSON-lines file, filter for plant science concepts, and write matching
records directly to DuckDB. Raw files are deleted after processing.

Usage:
    python -m src.collect.openalex_snapshot \
        --snapshot-dir data/raw/openalex/ \
        --db-path data/processed/plant_science.duckdb \
        --config config/openalex_concepts.json

Peak storage: ~5GB (one compressed file) + growing DuckDB.
"""

import os
import sys
import gzip
import json
import argparse
import time
from pathlib import Path

try:
    import orjson
    def loads(s):
        return orjson.loads(s)
except ImportError:
    def loads(s):
        return json.loads(s)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage
from src.utils.checkpointing import checkpoint_save, checkpoint_load, checkpoint_clear


def load_concept_filter(config_path: str) -> tuple[set, float, bool]:
    """Load concept IDs and minimum score from config."""
    with open(config_path) as f:
        config = json.load(f)

    concept_ids = set()
    for level_key in ["level_1_primary", "level_2_specific", "level_3_narrow"]:
        if level_key in config.get("concepts", {}):
            concept_ids.update(config["concepts"][level_key].keys())

    min_score = config.get("min_score", 0.3)
    require_bio = config.get("require_biology_parent", True)
    bio_id = "https://openalex.org/C86803240"  # Biology

    return concept_ids, min_score, require_bio, bio_id


def load_journal_filter(journal_csv: str) -> set:
    """Load journal ISSNs from CSV."""
    issns = set()
    if not os.path.exists(journal_csv):
        return issns
    with open(journal_csv) as f:
        for line in f:
            if line.startswith("issn"):
                continue
            parts = line.strip().split(",")
            if parts[0]:
                issns.add(parts[0].strip())
    return issns


def matches_plant_science(work: dict, concept_ids: set, min_score: float,
                          require_bio: bool, bio_id: str, journal_issns: set) -> bool:
    """Check if a work matches plant science filters."""
    # Check journal ISSN
    source = work.get("primary_location", {})
    if source:
        src = source.get("source", {})
        if src:
            issn = src.get("issn_l", "")
            if issn and issn in journal_issns:
                return True

    # Check concepts
    concepts = work.get("concepts", [])
    if not concepts:
        concepts = work.get("topics", [])

    has_bio_parent = False
    has_plant_concept = False

    for c in concepts:
        cid = c.get("id", "").replace("https://openalex.org/", "")
        score = c.get("score", 0)

        if c.get("id") == bio_id or cid == "C86803240":
            has_bio_parent = True

        if cid in concept_ids and score >= min_score:
            has_plant_concept = True

    if require_bio:
        return has_plant_concept and has_bio_parent
    return has_plant_concept


def extract_work_record(work: dict) -> dict:
    """Extract structured fields from an OpenAlex work JSON."""
    # Primary location / journal
    primary = work.get("primary_location", {}) or {}
    source = primary.get("source", {}) or {}

    return {
        "work_id": work.get("id", "").replace("https://openalex.org/", ""),
        "doi": work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
        "title": work.get("title"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "journal_id": source.get("id", "").replace("https://openalex.org/", "") if source.get("id") else None,
        "journal_name": source.get("display_name"),
        "journal_issn": source.get("issn_l"),
        "oa_status": work.get("open_access", {}).get("oa_status"),
        "cited_by_count": work.get("cited_by_count", 0),
        "reference_count": len(work.get("referenced_works", [])),
        "type": work.get("type"),
        "language": work.get("language"),
    }


def extract_authors(work: dict) -> list[dict]:
    """Extract author records from an OpenAlex work."""
    authors = []
    for i, authorship in enumerate(work.get("authorships", [])):
        author = authorship.get("author", {}) or {}
        institutions = authorship.get("institutions", []) or []
        inst = institutions[0] if institutions else {}

        authors.append({
            "work_id": work.get("id", "").replace("https://openalex.org/", ""),
            "author_id": author.get("id", "").replace("https://openalex.org/", "") if author.get("id") else f"unknown_{i}",
            "display_name": author.get("display_name"),
            "orcid": author.get("orcid", "").replace("https://orcid.org/", "") if author.get("orcid") else None,
            "author_position": i + 1,
            "is_corresponding": authorship.get("is_corresponding", False),
            "institution_id": inst.get("id", "").replace("https://openalex.org/", "") if inst.get("id") else None,
            "institution_name": inst.get("display_name"),
            "country_code": inst.get("country_code"),
        })
    return authors


def extract_concepts(work: dict) -> list[dict]:
    """Extract concept assignments from an OpenAlex work."""
    work_id = work.get("id", "").replace("https://openalex.org/", "")
    concepts = []
    for c in work.get("concepts", []):
        concepts.append({
            "work_id": work_id,
            "concept_id": c.get("id", "").replace("https://openalex.org/", ""),
            "concept_name": c.get("display_name"),
            "level": c.get("level"),
            "score": c.get("score"),
        })
    return concepts


def extract_citations(work: dict) -> list[tuple]:
    """Extract citation edges (this work -> referenced works)."""
    work_id = work.get("id", "").replace("https://openalex.org/", "")
    refs = []
    for ref in work.get("referenced_works", []):
        cited_id = ref.replace("https://openalex.org/", "")
        refs.append((work_id, cited_id))
    return refs


def extract_funders(work: dict) -> list[dict]:
    """Extract funder information."""
    work_id = work.get("id", "").replace("https://openalex.org/", "")
    funders = []
    for grant in work.get("grants", []):
        funder = grant.get("funder_display_name") or grant.get("funder", "")
        funders.append({
            "work_id": work_id,
            "funder_id": grant.get("funder", "").replace("https://openalex.org/", "") if isinstance(grant.get("funder"), str) else None,
            "funder_name": funder if isinstance(funder, str) else None,
            "award_id": grant.get("award_id"),
        })
    return funders


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    # inverted_index: {"word": [pos1, pos2, ...], ...}
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return None
    max_pos = max(positions.keys())
    words = [positions.get(i, "") for i in range(max_pos + 1)]
    return " ".join(words).strip()


def process_snapshot_file(filepath: str, con, concept_ids: set,
                          min_score: float, require_bio: bool,
                          bio_id: str, journal_issns: set) -> dict:
    """Process a single compressed OpenAlex snapshot file."""
    stats = {"total": 0, "matched": 0, "errors": 0}

    opener = gzip.open if filepath.endswith(".gz") else open

    with opener(filepath, "rb") as f:
        for line_bytes in f:
            stats["total"] += 1
            try:
                work = loads(line_bytes)
            except Exception:
                stats["errors"] += 1
                continue

            if not matches_plant_science(work, concept_ids, min_score,
                                         require_bio, bio_id, journal_issns):
                continue

            stats["matched"] += 1

            # Extract and insert
            record = extract_work_record(work)
            con.execute("""
                INSERT OR IGNORE INTO works
                (work_id, doi, title, abstract, year, publication_date,
                 journal_id, journal_name, journal_issn, oa_status,
                 cited_by_count, reference_count, type, language, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'openalex')
            """, [record["work_id"], record["doi"], record["title"],
                  record["abstract"], record["year"], record["publication_date"],
                  record["journal_id"], record["journal_name"], record["journal_issn"],
                  record["oa_status"], record["cited_by_count"],
                  record["reference_count"], record["type"], record["language"]])

            # Authors
            for auth in extract_authors(work):
                con.execute("""
                    INSERT OR IGNORE INTO authors (author_id, display_name, orcid)
                    VALUES (?, ?, ?)
                """, [auth["author_id"], auth["display_name"], auth["orcid"]])
                con.execute("""
                    INSERT OR IGNORE INTO work_authors
                    (work_id, author_id, author_position, is_corresponding,
                     institution_id, institution_name, country_code)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [auth["work_id"], auth["author_id"], auth["author_position"],
                      auth["is_corresponding"], auth["institution_id"],
                      auth["institution_name"], auth["country_code"]])

            # Concepts
            for concept in extract_concepts(work):
                con.execute("""
                    INSERT OR IGNORE INTO concepts
                    (work_id, concept_id, concept_name, level, score)
                    VALUES (?, ?, ?, ?, ?)
                """, [concept["work_id"], concept["concept_id"],
                      concept["concept_name"], concept["level"], concept["score"]])

            # Citations
            for citing, cited in extract_citations(work):
                con.execute("""
                    INSERT OR IGNORE INTO citations (citing_work_id, cited_work_id)
                    VALUES (?, ?)
                """, [citing, cited])

            # Funders
            for funder in extract_funders(work):
                con.execute("""
                    INSERT INTO funders (work_id, funder_id, funder_name, award_id)
                    VALUES (?, ?, ?, ?)
                """, [funder["work_id"], funder["funder_id"],
                      funder["funder_name"], funder["award_id"]])

            # Progress
            if stats["matched"] % 10000 == 0:
                print(f"    {stats['matched']:,} matched / {stats['total']:,} total",
                      flush=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter OpenAlex snapshot for plant science")
    parser.add_argument("--snapshot-dir", required=True, help="Directory with OpenAlex .gz files")
    parser.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    parser.add_argument("--config", default="config/openalex_concepts.json")
    parser.add_argument("--journals", default="config/journal_list.csv")
    args = parser.parse_args()

    # Storage check
    check_storage()

    # Load filters
    concept_ids, min_score, require_bio, bio_id = load_concept_filter(args.config)
    journal_issns = load_journal_filter(args.journals)
    print(f"Filters: {len(concept_ids)} concept IDs, {len(journal_issns)} journal ISSNs, min_score={min_score}")

    # Open database
    con = create_database(args.db_path)

    # Find all snapshot files
    snapshot_dir = Path(args.snapshot_dir)
    files = sorted(snapshot_dir.glob("**/*.gz"))
    if not files:
        files = sorted(snapshot_dir.glob("**/*.jsonl"))
    print(f"Found {len(files)} snapshot files in {snapshot_dir}")

    # Check for checkpoint
    ckpt = checkpoint_load("openalex_filter")
    processed_files = set(ckpt.get("processed_files", [])) if ckpt else set()
    if processed_files:
        print(f"Resuming: {len(processed_files)} files already processed")

    # Process each file
    total_matched = 0
    for i, filepath in enumerate(files):
        fname = str(filepath)
        if fname in processed_files:
            continue

        print(f"\n[{i+1}/{len(files)}] Processing {filepath.name} ...")
        t0 = time.time()

        stats = process_snapshot_file(fname, con, concept_ids, min_score,
                                       require_bio, bio_id, journal_issns)
        elapsed = time.time() - t0
        total_matched += stats["matched"]

        # Log
        con.execute("""
            INSERT INTO ingestion_log (source, file_path, records_total, records_matched)
            VALUES ('openalex', ?, ?, ?)
        """, [fname, stats["total"], stats["matched"]])

        print(f"  {stats['matched']:,} matched / {stats['total']:,} total "
              f"({elapsed:.0f}s, {stats['errors']} errors)")

        # Checkpoint
        processed_files.add(fname)
        checkpoint_save("openalex_filter", {
            "processed_files": list(processed_files),
            "total_matched": total_matched,
        })

        # Storage check every 5 files
        if (i + 1) % 5 == 0:
            check_storage()

    # Done
    checkpoint_clear("openalex_filter")
    from src.db.schema import get_stats
    stats = get_stats(con)
    print(f"\nComplete! Database stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    con.close()


if __name__ == "__main__":
    main()
