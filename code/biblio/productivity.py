"""
Themes 3 & 4: Citation Dynamics, Author Productivity, Team Size.

Produces CSVs in results/citations/ for:
  - Q14: Citation distribution (Price's law)
  - Q15: Most-cited papers
  - Q16: Citation half-life
  - Q20: Lotka's law (author productivity)
  - Q21: Team size evolution
  - Q25: Big team vs small team impact

Usage:
    python -m src.biblio.productivity --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR_CIT = "results/citations"
OUT_DIR_NET = "results/networks"


def _ensure_dirs():
    os.makedirs(OUT_DIR_CIT, exist_ok=True)
    os.makedirs(OUT_DIR_NET, exist_ok=True)


def q14_citation_distribution(con):
    """Citation count distribution — test Price's law (power law)."""
    print("Q14: Citation distribution...", flush=True)
    df = con.execute("""
        SELECT cited_by_count, COUNT(*) AS n_papers
        FROM works_clean
        WHERE cited_by_count IS NOT NULL AND cited_by_count >= 0
        GROUP BY cited_by_count
        ORDER BY cited_by_count
    """).df()
    df.to_csv(f"{OUT_DIR_CIT}/citation_distribution.csv", index=False)

    # Power law fit on tail (citations >= 10)
    tail = df[df["cited_by_count"] >= 10].copy()
    if len(tail) > 5:
        x = np.log10(tail["cited_by_count"].values.astype(float))
        y = np.log10(tail["n_papers"].values.astype(float))
        try:
            coeffs = np.polyfit(x, y, 1)
            alpha = -coeffs[0]
            print(f"  Power law exponent (alpha): {alpha:.2f}", flush=True)
            pd.DataFrame([{"alpha": alpha, "intercept": coeffs[1]}]).to_csv(
                f"{OUT_DIR_CIT}/power_law_fit.csv", index=False)
        except Exception as e:
            print(f"  Power law fit failed: {e}", flush=True)

    # Basic stats
    stats = con.execute("""
        SELECT COUNT(*) AS total,
               AVG(cited_by_count) AS mean_cit,
               MEDIAN(cited_by_count) AS median_cit,
               MAX(cited_by_count) AS max_cit,
               PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY cited_by_count) AS p90,
               PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY cited_by_count) AS p99
        FROM works_clean
        WHERE cited_by_count IS NOT NULL
    """).df()
    stats.to_csv(f"{OUT_DIR_CIT}/citation_stats.csv", index=False)
    print(f"  Mean: {stats['mean_cit'].iloc[0]:.1f}, "
          f"Median: {stats['median_cit'].iloc[0]:.0f}, "
          f"Max: {stats['max_cit'].iloc[0]:,.0f}", flush=True)


def q15_most_cited(con):
    """Top-100 most cited papers."""
    print("Q15: Most cited papers...", flush=True)
    df = con.execute("""
        SELECT work_id, doi, title, year, journal_name,
               cited_by_count, oa_status
        FROM works_clean
        WHERE cited_by_count IS NOT NULL
        ORDER BY cited_by_count DESC
        LIMIT 100
    """).df()
    df.to_csv(f"{OUT_DIR_CIT}/top100_cited.csv", index=False)
    print(f"  #1: '{df.iloc[0]['title'][:60]}...' "
          f"({df.iloc[0]['cited_by_count']:,} citations)", flush=True)


def q16_citation_halflife(con):
    """Citation half-life: how quickly papers accumulate citations by cohort."""
    print("Q16: Citation half-life...", flush=True)
    # Mean citations by publication year and age-at-citation
    # We use cited_by_count as a proxy (age = current_year - pub_year)
    df = con.execute("""
        SELECT year,
               AVG(cited_by_count) AS mean_citations,
               MEDIAN(cited_by_count) AS median_citations,
               COUNT(*) AS n
        FROM works_clean
        WHERE cited_by_count IS NOT NULL AND year >= 1960
        GROUP BY year ORDER BY year
    """).df()
    df["age"] = 2024 - df["year"]
    df.to_csv(f"{OUT_DIR_CIT}/citations_by_cohort.csv", index=False)
    print(f"  {len(df)} cohorts", flush=True)


def q20_lotka(con):
    """Lotka's law: author productivity distribution."""
    print("Q20: Lotka's law...", flush=True)
    df = con.execute("""
        SELECT n_papers, COUNT(*) AS n_authors FROM (
            SELECT author_id, COUNT(*) AS n_papers
            FROM work_authors
            WHERE author_id != 'A9999999999'
            GROUP BY author_id
        ) GROUP BY n_papers
        ORDER BY n_papers
    """).df()
    df.to_csv(f"{OUT_DIR_NET}/lotka_distribution.csv", index=False)

    # Fit power law: n_authors = C * n_papers^(-alpha)
    if len(df) > 5:
        x = np.log10(df["n_papers"].values.astype(float))
        y = np.log10(df["n_authors"].values.astype(float))
        try:
            coeffs = np.polyfit(x, y, 1)
            alpha = -coeffs[0]
            print(f"  Lotka exponent: {alpha:.2f} (classic = 2.0)", flush=True)
            pd.DataFrame([{"lotka_alpha": alpha}]).to_csv(
                f"{OUT_DIR_NET}/lotka_fit.csv", index=False)
        except Exception as e:
            print(f"  Lotka fit failed: {e}", flush=True)

    total_authors = df["n_authors"].sum()
    one_paper = df[df["n_papers"] == 1]["n_authors"].iloc[0] if len(df) > 0 else 0
    print(f"  {total_authors:,} authors, {one_paper:,} ({100*one_paper/total_authors:.1f}%) "
          f"with 1 paper", flush=True)


def q21_team_size(con):
    """Team size (authors per paper) evolution over time."""
    print("Q21: Team size evolution...", flush=True)
    df = con.execute("""
        SELECT w.year,
               AVG(team.n_authors) AS mean_team_size,
               MEDIAN(team.n_authors) AS median_team_size,
               MAX(team.n_authors) AS max_team_size,
               COUNT(*) AS n_papers
        FROM works_clean w
        JOIN (
            SELECT work_id, COUNT(*) AS n_authors
            FROM work_authors
            GROUP BY work_id
        ) team ON w.work_id = team.work_id
        WHERE w.year >= 1960
        GROUP BY w.year ORDER BY w.year
    """).df()
    df.to_csv(f"{OUT_DIR_NET}/team_size_by_year.csv", index=False)
    print(f"  Latest mean team size: {df.iloc[-1]['mean_team_size']:.1f}", flush=True)


def q25_team_size_impact(con):
    """Big team vs small team: mean citations by team size bucket."""
    print("Q25: Team size vs impact...", flush=True)
    df = con.execute("""
        SELECT
            CASE
                WHEN n_authors = 1 THEN 'solo'
                WHEN n_authors <= 3 THEN 'small (2-3)'
                WHEN n_authors <= 6 THEN 'medium (4-6)'
                WHEN n_authors <= 10 THEN 'large (7-10)'
                ELSE 'mega (11+)'
            END AS team_bucket,
            AVG(w.cited_by_count) AS mean_citations,
            MEDIAN(w.cited_by_count) AS median_citations,
            COUNT(*) AS n_papers
        FROM works_clean w
        JOIN (
            SELECT work_id, COUNT(*) AS n_authors
            FROM work_authors GROUP BY work_id
        ) team ON w.work_id = team.work_id
        WHERE w.year BETWEEN 2000 AND 2022
          AND w.cited_by_count IS NOT NULL
        GROUP BY team_bucket
    """).df()
    df.to_csv(f"{OUT_DIR_CIT}/team_size_impact.csv", index=False)
    for _, row in df.iterrows():
        print(f"  {row['team_bucket']}: mean={row['mean_citations']:.1f}, "
              f"n={row['n_papers']:,}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    con = create_database(args.db_path)

    t0 = time.time()
    q14_citation_distribution(con)
    q15_most_cited(con)
    q16_citation_halflife(con)
    q20_lotka(con)
    q21_team_size(con)
    q25_team_size_impact(con)

    elapsed = int(time.time() - t0)
    print(f"\nThemes 3-4 (Citations/Productivity) complete ({elapsed}s)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
