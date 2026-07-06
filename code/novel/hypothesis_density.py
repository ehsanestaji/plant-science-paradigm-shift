"""
Analysis F: Hypothesis Density Mapping.

Is plant science becoming less hypothesis-driven as omics/big-data approaches
dominate? Classifies abstracts as hypothesis-driven vs descriptive using regex
patterns, then tracks trends by year, country, journal, and OA status.

Output → results/novel/
  hypothesis_density_by_year.csv
  hypothesis_density_by_country.csv
  hypothesis_citation_advantage.csv
  hypothesis_density_by_journal_top50.csv

Usage:
    python -m src.novel.hypothesis_density --db-path data/processed/plant_science.duckdb
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

# ── Regex patterns ───────────────────────────────────────────────────────────

HYPOTHESIS_PATTERNS = {
    "explicit_hypothesis": re.compile(
        r"\b(we hypothesiz|our hypothesis|we propose that|we postulate|"
        r"it is hypothesized|it was hypothesized)\b", re.I),
    "tested_whether": re.compile(
        r"\b(we tested whether|we tested if|we examined whether|"
        r"we investigated whether|we asked whether)\b", re.I),
    "prediction": re.compile(
        r"\b(we predict(ed)?|we expected|it was predicted|our prediction)\b", re.I),
    "aimed_to_test": re.compile(
        r"\b(aimed to test|designed to test|set out to test|sought to test)\b", re.I),
}

DESCRIPTIVE_PATTERNS = {
    "characterize": re.compile(
        r"\b(we characteriz|we describe|we report|we identified|we profiled|"
        r"here we report|here we describe)\b", re.I),
    "survey": re.compile(
        r"\b(we surveyed|we catalogu|we inventori|we screened|we mapped)\b", re.I),
}


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def classify_abstract(text: str) -> dict:
    """Classify a single abstract. Returns dict with boolean flags."""
    is_hyp = any(p.search(text) for p in HYPOTHESIS_PATTERNS.values())
    is_desc = any(p.search(text) for p in DESCRIPTIVE_PATTERNS.values())
    return {"is_hypothesis": is_hyp, "is_descriptive": is_desc}


def fetch_and_classify(con) -> pd.DataFrame:
    """Fetch all abstracts in batches and classify."""
    print("Counting abstracts…", flush=True)
    total = con.execute("""
        SELECT COUNT(*) FROM works_clean
        WHERE abstract IS NOT NULL AND length(abstract) > 100 AND year >= 1960
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
            WHERE abstract IS NOT NULL AND length(abstract) > 100 AND year >= 1960
            ORDER BY work_id
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """).df()

        if len(df) == 0:
            break

        results = df["abstract"].apply(classify_abstract).apply(pd.Series)
        df = pd.concat([df.drop(columns=["abstract"]), results], axis=1)
        all_rows.append(df)

        elapsed = time.time() - t0
        print(f"  Batch {batch_num}: {len(df):,} rows ({elapsed:.0f}s)", flush=True)
        offset += BATCH_SIZE

    return pd.concat(all_rows, ignore_index=True)


def fetch_country(con, work_ids: list) -> pd.DataFrame:
    """Get first-author country for a list of work_ids."""
    ids_df = pd.DataFrame({"work_id": work_ids})
    con.register("_hyp_ids", ids_df)
    df = con.execute("""
        SELECT DISTINCT ON (h.work_id) h.work_id, wa.country_code
        FROM _hyp_ids h
        JOIN work_authors wa ON h.work_id = wa.work_id
        WHERE wa.author_position = 1 AND wa.country_code IS NOT NULL
    """).df()
    con.unregister("_hyp_ids")
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
    print(f"  Hypothesis-driven: {df['is_hypothesis'].sum():,} "
          f"({df['is_hypothesis'].mean()*100:.1f}%)", flush=True)
    print(f"  Descriptive: {df['is_descriptive'].sum():,} "
          f"({df['is_descriptive'].mean()*100:.1f}%)", flush=True)

    # ── By year ──────────────────────────────────────────────────────────────
    by_year = (df.groupby("year")
               .agg(n_papers=("work_id", "count"),
                    n_hypothesis=("is_hypothesis", "sum"),
                    n_descriptive=("is_descriptive", "sum"))
               .reset_index())
    by_year["pct_hypothesis"] = by_year["n_hypothesis"] / by_year["n_papers"] * 100
    by_year["pct_descriptive"] = by_year["n_descriptive"] / by_year["n_papers"] * 100
    by_year.to_csv(f"{OUT_DIR}/hypothesis_density_by_year.csv", index=False)
    print(f"  Saved hypothesis_density_by_year.csv", flush=True)

    # ── By country ───────────────────────────────────────────────────────────
    print("Fetching first-author countries…", flush=True)
    countries = fetch_country(con, df["work_id"].tolist())
    df_c = df.merge(countries, on="work_id", how="left")
    by_country = (df_c[df_c["country_code"].notna()]
                  .groupby("country_code")
                  .agg(n_papers=("work_id", "count"),
                       n_hypothesis=("is_hypothesis", "sum"),
                       n_descriptive=("is_descriptive", "sum"))
                  .reset_index())
    by_country["pct_hypothesis"] = by_country["n_hypothesis"] / by_country["n_papers"] * 100
    by_country = by_country.sort_values("n_papers", ascending=False)
    by_country.to_csv(f"{OUT_DIR}/hypothesis_density_by_country.csv", index=False)

    # ── By journal (top 50) ──────────────────────────────────────────────────
    by_journal = (df[df["journal_name"].notna()]
                  .groupby("journal_name")
                  .agg(n_papers=("work_id", "count"),
                       n_hypothesis=("is_hypothesis", "sum"),
                       pct_hypothesis=("is_hypothesis", "mean"))
                  .reset_index())
    by_journal["pct_hypothesis"] *= 100
    by_journal = by_journal.nlargest(50, "n_papers")
    by_journal.to_csv(f"{OUT_DIR}/hypothesis_density_by_journal_top50.csv", index=False)

    # ── Citation advantage ───────────────────────────────────────────────────
    hyp_cites = df.loc[df["is_hypothesis"], "cited_by_count"].dropna()
    desc_cites = df.loc[df["is_descriptive"], "cited_by_count"].dropna()
    neither_cites = df.loc[~df["is_hypothesis"] & ~df["is_descriptive"],
                           "cited_by_count"].dropna()

    records = []
    for label, vals in [("hypothesis", hyp_cites), ("descriptive", desc_cites),
                        ("neither", neither_cites)]:
        records.append({
            "group": label,
            "n_papers": len(vals),
            "median_citations": float(vals.median()),
            "mean_citations": float(vals.mean()),
        })

    # Mann-Whitney test: hypothesis vs descriptive
    if len(hyp_cites) > 0 and len(desc_cites) > 0:
        stat, pval = sp_stats.mannwhitneyu(hyp_cites, desc_cites, alternative="two-sided")
        records[0]["mannwhitney_U"] = float(stat)
        records[0]["mannwhitney_p"] = float(pval)

    pd.DataFrame(records).to_csv(
        f"{OUT_DIR}/hypothesis_citation_advantage.csv", index=False)

    con.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Hypothesis Density Summary ===")
    print(f"  Total papers: {len(df):,}")
    print(f"  Hypothesis-driven: {df['is_hypothesis'].sum():,} "
          f"({df['is_hypothesis'].mean()*100:.1f}%)")
    print(f"  Descriptive: {df['is_descriptive'].sum():,} "
          f"({df['is_descriptive'].mean()*100:.1f}%)")
    for r in records:
        print(f"  {r['group']}: median={r['median_citations']:.0f}, "
              f"mean={r['mean_citations']:.1f} citations")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis F complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
