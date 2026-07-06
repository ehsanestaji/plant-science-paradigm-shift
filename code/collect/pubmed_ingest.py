"""
Phase 2c: PubMed deduplication and ingestion into DuckDB.

Match strategy (in order):
  1. DOI match      — exact, case-insensitive (~70-80% of PubMed records)
  2. Title+year     — normalized title + same year (catches DOI-less records)
  3. New insert     — no OpenAlex counterpart; inserted with source='pubmed'

For matched works: sets pmid, adds MeSH terms.
For new works:     full insert (title/abstract/authors/MeSH).

Usage:
    python -m src.collect.pubmed_ingest \
        --input-dir data/raw/pubmed_api \
        --db-path   data/processed/plant_science.duckdb
"""

import os
import re
import sys
import gzip
import json
import time
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.checkpointing import checkpoint_save, checkpoint_load
from src.utils.storage_monitor import check_storage

try:
    import orjson
    def _loads(b): return orjson.loads(b)
except ImportError:
    def _loads(b): return json.loads(b)


# ── Schema additions ──────────────────────────────────────────────────

MESH_DDL = """
CREATE TABLE IF NOT EXISTS mesh_terms (
    work_id  VARCHAR NOT NULL,
    term     VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mesh_work ON mesh_terms(work_id);
"""


def _ensure_schema(con):
    con.execute(MESH_DDL)


# ── Title normalisation ───────────────────────────────────────────────

_RE_STRIP = re.compile(r"[^a-z0-9 ]")
_RE_WS    = re.compile(r"\s+")

_MONTHS = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}

def norm_date(d: str | None) -> str | None:
    """Normalize PubMed date strings to YYYY-MM-DD or None.
    Handles: '1990-Dec-17', '2023-Mar', '2023', '2023-03-15',
             '1999-2000' (year-range → first year), '1990 Dec-17'.
    """
    if not d:
        return None
    s = str(d).strip()
    # Replace abbreviated month names with numbers
    for name, num in _MONTHS.items():
        s = s.replace(name, num)
    parts = s.replace("/", "-").replace(" ", "-").split("-")
    try:
        year = int(parts[0])
        if not (1000 <= year <= 2100):
            return None
        if len(parts) >= 2:
            month = int(parts[1])
            if not (1 <= month <= 12):
                # e.g. "1999-2000" — second part is a year, not a month
                return f"{year:04d}-01-01"
            if len(parts) >= 3:
                day = int(parts[2])
                if not (1 <= day <= 31):
                    return f"{year:04d}-{month:02d}-01"
                return f"{year:04d}-{month:02d}-{day:02d}"
            return f"{year:04d}-{month:02d}-01"
        return f"{year:04d}-01-01"
    except (ValueError, IndexError):
        pass
    return None


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    return _RE_WS.sub(" ", _RE_STRIP.sub("", t.lower())).strip()


# ── File loader ───────────────────────────────────────────────────────

def load_pubmed_file(filepath: Path) -> list[dict]:
    """Read a gzip JSONL file, return list of record dicts."""
    records = []
    with gzip.open(filepath, "rb") as fh:
        while True:
            try:
                line = fh.readline()
            except Exception:
                break
            if not line:
                break
            try:
                records.append(_loads(line))
            except Exception:
                continue
    return records


# ── Main ingest function ──────────────────────────────────────────────

def process_file(filepath: Path, con) -> dict:
    stats = dict(doi_match=0, title_match=0, new_insert=0,
                 mesh_added=0, errors=0)

    records = load_pubmed_file(filepath)
    if not records:
        return stats

    # ── Build staging DataFrame ───────────────────────────────────────
    rows = []
    for r in records:
        doi   = r.get("doi") or None
        pmid  = r.get("pmid") or None
        title = r.get("title") or None
        if not pmid:
            continue
        rows.append({
            "pmid":             pmid,
            "doi":              doi.lower().strip() if doi else None,
            "title":            title,
            "title_norm":       norm_title(title),
            "year":             r.get("year"),
            "abstract":         r.get("abstract"),
            "publication_date": r.get("publication_date"),
            "journal_name":     r.get("journal_name"),
            "journal_issn":     r.get("journal_issn"),
            "language":         r.get("language"),
            "authors":          r.get("authors") or [],
            "mesh_terms":       r.get("mesh_terms") or [],
        })

    if not rows:
        return stats

    df = pd.DataFrame(rows)
    matched_work_ids = {}   # pmid → work_id

    # ── Step 1: DOI match ─────────────────────────────────────────────
    doi_df = df[df["doi"].notna()][["pmid", "doi"]].copy()
    if not doi_df.empty:
        con.register("_pm_doi", doi_df)
        doi_matches = con.execute("""
            SELECT w.work_id, p.pmid
            FROM works w
            JOIN _pm_doi p ON lower(w.doi) = p.doi
        """).df()
        con.unregister("_pm_doi")
        doi_matches = doi_matches.drop_duplicates(subset=["work_id"])
        for _, row in doi_matches.iterrows():
            matched_work_ids[row["pmid"]] = row["work_id"]
        stats["doi_match"] = len(doi_matches)

    # ── Step 2: Title + year match for unmatched ──────────────────────
    unmatched_df = df[~df["pmid"].isin(matched_work_ids)].copy()
    unmatched_df = unmatched_df[
        unmatched_df["title_norm"].str.len() > 10
    ]
    if not unmatched_df.empty:
        title_df = unmatched_df[["pmid", "title_norm", "year"]].copy()
        title_df = title_df[title_df["year"].notna()]
        if not title_df.empty:
            con.register("_pm_title", title_df)
            title_matches = con.execute("""
                SELECT w.work_id, p.pmid
                FROM works w
                JOIN _pm_title p
                  ON w.year = p.year
                 AND regexp_replace(lower(w.title), '[^a-z0-9 ]', ' ', 'g') = p.title_norm
                WHERE w.title IS NOT NULL
            """).df()
            con.unregister("_pm_title")
            title_matches = title_matches.drop_duplicates(subset=["work_id"])
            for _, row in title_matches.iterrows():
                if row["pmid"] not in matched_work_ids:
                    matched_work_ids[row["pmid"]] = row["work_id"]
            stats["title_match"] = len(title_matches)

    # ── Step 3: Update matched works with PMID ────────────────────────
    if matched_work_ids:
        update_df = pd.DataFrame(
            [(wid, pmid) for pmid, wid in matched_work_ids.items()],
            columns=["work_id", "pmid"],
        ).drop_duplicates(subset=["work_id"])  # one PMID per work_id
        con.register("_pm_update", update_df)
        con.execute("""
            UPDATE works
            SET pmid = u.pmid
            FROM _pm_update u
            WHERE works.work_id = u.work_id
              AND works.pmid IS NULL
        """)
        con.unregister("_pm_update")

    # ── Step 4: Insert new works ──────────────────────────────────────
    new_df = df[~df["pmid"].isin(matched_work_ids)].copy()
    new_work_rows = []
    new_author_rows = []
    new_wa_rows = []

    for _, r in new_df.iterrows():
        pmid  = r["pmid"]
        wid   = f"PM{pmid}"
        new_work_rows.append((
            wid, None, r["title"], r["abstract"],
            r["year"], norm_date(r["publication_date"]),
            None, r["journal_name"], r["journal_issn"],
            None, None, None, None, r["language"],
            "pubmed", pmid,
        ))
        matched_work_ids[pmid] = wid

        for i, auth in enumerate(r["authors"]):
            name = auth.get("name") or ""
            if not name:
                continue
            aid = f"PMA{pmid}_{i}"
            new_author_rows.append((aid, name, None))
            new_wa_rows.append((wid, aid, i + 1, False, None,
                                auth.get("affiliation"), None))

    if new_work_rows:
        nw = pd.DataFrame(new_work_rows, columns=[
            "work_id", "doi", "title", "abstract", "year",
            "publication_date", "journal_id", "journal_name",
            "journal_issn", "oa_status", "cited_by_count",
            "reference_count", "type", "language", "source", "pmid",
        ])
        con.register("_new_works", nw)
        con.execute("""
            INSERT OR IGNORE INTO works
            (work_id, doi, title, abstract, year, publication_date,
             journal_id, journal_name, journal_issn, oa_status,
             cited_by_count, reference_count, type, language, source, pmid)
            SELECT work_id, doi, title, abstract, year, publication_date,
                   journal_id, journal_name, journal_issn, oa_status,
                   cited_by_count, reference_count, type, language, source, pmid
            FROM _new_works
        """)
        con.unregister("_new_works")
        stats["new_insert"] = len(new_work_rows)

    if new_author_rows:
        da = pd.DataFrame(new_author_rows,
                          columns=["author_id", "display_name", "orcid"])
        dwa = pd.DataFrame(new_wa_rows, columns=[
            "work_id", "author_id", "author_position", "is_corresponding",
            "institution_id", "institution_name", "country_code",
        ])
        con.register("_new_auth", da)
        con.register("_new_wa", dwa)
        con.execute("INSERT OR IGNORE INTO authors (author_id, display_name, orcid) SELECT * FROM _new_auth")
        con.execute("INSERT OR IGNORE INTO work_authors SELECT * FROM _new_wa")
        con.unregister("_new_auth")
        con.unregister("_new_wa")

    # ── Step 5: Insert MeSH terms ─────────────────────────────────────
    mesh_rows = []
    for _, r in df.iterrows():
        wid = matched_work_ids.get(r["pmid"])
        if wid:
            for term in r["mesh_terms"]:
                if term:
                    mesh_rows.append((wid, term))

    if mesh_rows:
        dm = pd.DataFrame(mesh_rows, columns=["work_id", "term"])
        con.register("_mesh", dm)
        con.execute("INSERT INTO mesh_terms SELECT * FROM _mesh")
        con.unregister("_mesh")
        stats["mesh_added"] = len(mesh_rows)

    return stats


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/raw/pubmed_api")
    ap.add_argument("--db-path",   default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    con = create_database(args.db_path)
    _ensure_schema(con)

    files = sorted(Path(args.input_dir).glob("pubmed_plant_*.jsonl.gz"))
    total_files = len(files)
    print(f"Found {total_files} PubMed files to process", flush=True)

    ckpt          = checkpoint_load("pubmed_ingest") or {}
    processed     = set(ckpt.get("processed_files", []))
    totals        = ckpt.get("totals", dict(doi_match=0, title_match=0,
                                            new_insert=0, mesh_added=0))

    t_start = time.time()

    for i, fpath in enumerate(files, 1):
        fname = fpath.name
        if fname in processed:
            print(f"[{i}/{total_files}] Skipping {fname} (already done)", flush=True)
            continue

        fsize_mb = fpath.stat().st_size / 1_048_576
        print(f"[{i}/{total_files}] {fname} ({fsize_mb:.0f} MB)...", flush=True)

        t0    = time.time()
        stats = process_file(fpath, con)
        elapsed = int(time.time() - t0)

        for k in totals:
            totals[k] += stats.get(k, 0)

        print(f"  doi_match={stats['doi_match']:,}  "
              f"title_match={stats['title_match']:,}  "
              f"new={stats['new_insert']:,}  "
              f"mesh={stats['mesh_added']:,}  "
              f"({elapsed}s)", flush=True)

        processed.add(fname)
        checkpoint_save("pubmed_ingest", dict(
            processed_files=list(processed), totals=totals,
        ))

    elapsed_total = int(time.time() - t_start)
    check_storage()

    print(f"\nDeduplication complete! ({elapsed_total}s total)", flush=True)
    print(f"  DOI matches:   {totals['doi_match']:,}", flush=True)
    print(f"  Title matches: {totals['title_match']:,}", flush=True)
    print(f"  New inserts:   {totals['new_insert']:,}", flush=True)
    print(f"  MeSH terms:    {totals['mesh_added']:,}", flush=True)

    db_size = Path(args.db_path).stat().st_size / 1_073_741_824
    print(f"  DB size:       {db_size:.1f} GB", flush=True)

    con.close()


if __name__ == "__main__":
    main()
