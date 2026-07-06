"""
Analysis K: Research Question Dynamics.

Has plant science shifted from "what is" to "how does" to "can we engineer"?
Classifies abstracts by epistemic question type using regex patterns, tracks
proportions over time, and tests citation advantage by question type.

Output → results/novel/
  question_type_by_year.csv
  question_complexity_by_year.csv
  question_citation_advantage.csv
  question_type_by_country.csv

Usage:
    python -m src.novel.question_dynamics --db-path data/processed/plant_science.duckdb
"""

import argparse
import re
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/novel"
BATCH_SIZE = 500_000

# ── Epistemic type classifiers ───────────────────────────────────────────────

QUESTION_TYPES = {
    "descriptive": re.compile(
        r"\b(what is|what are|characteriz|identif[iy]|describ|catalog|profil|"
        r"document|inventory|survey|we report|here we report)\b", re.I),
    "mechanistic": re.compile(
        r"\b(how does|how do|mechanism|pathway|regulat|signal|mediat|"
        r"underl[yi]|modulat|interact|crosstalk|cross-talk)\b", re.I),
    "comparative": re.compile(
        r"\b(compar|differ(ence|ent)|between .{0,30} and|versus|vs\.|"
        r"relative to|contrast)\b", re.I),
    "predictive": re.compile(
        r"\b(predict|forecast|model[li]|simulat|project|estimat|"
        r"prognostic)\b", re.I),
    "methodological": re.compile(
        r"\b(method|protocol|technique|new approach|develop.{0,10}(tool|pipeline|"
        r"method|assay)|optimiz|high-throughput|workflow)\b", re.I),
    "applied": re.compile(
        r"\b(improv|enhanc|engineer|breed|toleran|resist|yield|"
        r"biofortif|crop improvement|field performance|agronomic)\b", re.I),
    "integrative": re.compile(
        r"\b(integrat|systems|multi-omics|multiomics|network analysis|"
        r"holistic|comprehensive|meta-analy|systematic review)\b", re.I),
}


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def classify_questions(text: str) -> dict:
    """Classify abstract by epistemic types. Returns dict of booleans."""
    result = {}
    for qtype, pattern in QUESTION_TYPES.items():
        result[f"is_{qtype}"] = bool(pattern.search(text))
    # Question complexity = number of distinct types
    result["complexity"] = sum(1 for v in result.values() if v is True)
    return result


def fetch_and_classify(con) -> pd.DataFrame:
    """Fetch all abstracts in batches and classify by question type."""
    print("Counting abstracts…", flush=True)
    total = con.execute("""
        SELECT COUNT(*) FROM works_clean
        WHERE abstract IS NOT NULL AND length(abstract) > 100 AND year >= 1970
    """).fetchone()[0]
    print(f"  {total:,} abstracts to process", flush=True)

    all_rows = []
    offset = 0
    batch_num = 0

    while offset < total:
        batch_num += 1
        t0 = time.time()
        df = con.execute(f"""
            SELECT work_id, year, abstract, journal_name, oa_status, cited_by_count
            FROM works_clean
            WHERE abstract IS NOT NULL AND length(abstract) > 100 AND year >= 1970
            ORDER BY work_id
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).df()

        if len(df) == 0:
            break

        results = df["abstract"].apply(classify_questions).apply(pd.Series)
        df = pd.concat([df.drop(columns=["abstract"]), results], axis=1)
        all_rows.append(df)

        elapsed = time.time() - t0
        print(f"  Batch {batch_num}: {len(df):,} rows ({elapsed:.0f}s)", flush=True)
        offset += BATCH_SIZE

    return pd.concat(all_rows, ignore_index=True)


def fetch_country(con, work_ids: list) -> pd.DataFrame:
    """Get first-author country for work_ids."""
    ids_df = pd.DataFrame({"work_id": work_ids})
    con.register("_q_ids", ids_df)
    df = con.execute("""
        SELECT DISTINCT ON (h.work_id) h.work_id, wa.country_code
        FROM _q_ids h
        JOIN work_authors wa ON h.work_id = wa.work_id
        WHERE wa.author_position = 1 AND wa.country_code IS NOT NULL
    """).df()
    con.unregister("_q_ids")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='40GB'")
    con.execute("SET threads=8")

    df = fetch_and_classify(con)
    print(f"\nClassified {len(df):,} papers", flush=True)

    qtypes = [c.replace("is_", "") for c in df.columns if c.startswith("is_")]
    for qt in qtypes:
        col = f"is_{qt}"
        pct = df[col].mean() * 100
        print(f"  {qt}: {df[col].sum():,} ({pct:.1f}%)", flush=True)

    # ── By year ──────────────────────────────────────────────────────────────
    year_records = []
    for year, grp in df.groupby("year"):
        row = {"year": year, "n_papers": len(grp)}
        for qt in qtypes:
            col = f"is_{qt}"
            row[f"n_{qt}"] = int(grp[col].sum())
            row[f"pct_{qt}"] = round(grp[col].mean() * 100, 2)
        year_records.append(row)
    by_year = pd.DataFrame(year_records)
    by_year.to_csv(f"{OUT_DIR}/question_type_by_year.csv", index=False)

    # ── Question complexity by year ──────────────────────────────────────────
    complexity_by_year = (df.groupby("year")
                          .agg(mean_complexity=("complexity", "mean"),
                               median_complexity=("complexity", "median"),
                               n_papers=("work_id", "count"))
                          .reset_index())
    complexity_by_year.to_csv(f"{OUT_DIR}/question_complexity_by_year.csv", index=False)

    # ── Citation advantage by type ───────────────────────────────────────────
    records = []
    for qt in qtypes:
        col = f"is_{qt}"
        cites = df.loc[df[col], "cited_by_count"].dropna()
        records.append({
            "question_type": qt,
            "n_papers": len(cites),
            "median_citations": float(cites.median()) if len(cites) > 0 else 0,
            "mean_citations": float(cites.mean()) if len(cites) > 0 else 0,
        })
    pd.DataFrame(records).to_csv(
        f"{OUT_DIR}/question_citation_advantage.csv", index=False)

    # ── By country (top 30) ──────────────────────────────────────────────────
    print("Fetching first-author countries…", flush=True)
    countries = fetch_country(con, df["work_id"].tolist())
    df_c = df.merge(countries, on="work_id", how="left")

    country_records = []
    valid_countries = (df_c[df_c["country_code"].notna()]
                       .groupby("country_code")["work_id"].count()
                       .nlargest(30).index)
    for cc in valid_countries:
        grp = df_c[df_c["country_code"] == cc]
        row = {"country_code": cc, "n_papers": len(grp)}
        for qt in qtypes:
            col = f"is_{qt}"
            row[f"pct_{qt}"] = round(grp[col].mean() * 100, 2)
        country_records.append(row)
    pd.DataFrame(country_records).to_csv(
        f"{OUT_DIR}/question_type_by_country.csv", index=False)

    con.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Research Question Dynamics Summary ===")
    print(f"  Total papers: {len(df):,}")
    print(f"  Mean complexity: {df['complexity'].mean():.2f} types/paper")
    print(f"\n  Citation advantage by question type:")
    for r in records:
        print(f"    {r['question_type']:<18} n={r['n_papers']:>8,}  "
              f"median={r['median_citations']:>6.0f}  "
              f"mean={r['mean_citations']:>8.1f}")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis K complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
