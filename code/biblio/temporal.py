"""
Theme 1: Growth & Publication Dynamics.

Produces CSVs in results/temporal/ for:
  - Q1: Papers per year (overall + by top subfield concepts)
  - Q2: Subfield growth drivers
  - Q3: Doubling time (exponential + logistic fit)
  - Q4: OA type composition over time
  - Q5: Language diversity over time
  - Q6: Article type distribution over time

Usage:
    python -m src.biblio.temporal --db-path data/processed/plant_science.duckdb
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

OUT_DIR = "results/temporal"


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def q1_growth_by_year(con):
    """Overall publication count per year."""
    print("Q1: Growth by year...", flush=True)
    df = con.execute("""
        SELECT year, COUNT(*) AS n_papers
        FROM works_clean
        GROUP BY year ORDER BY year
    """).df()
    df.to_csv(f"{OUT_DIR}/papers_per_year.csv", index=False)
    print(f"  {len(df)} years, total {df['n_papers'].sum():,} papers", flush=True)
    return df


def q2_subfield_growth(con):
    """Papers per year by top-level concept (level 0 and 1)."""
    print("Q2: Subfield growth...", flush=True)

    # Get top-15 level-1 concepts by total papers
    top_concepts = con.execute("""
        SELECT c.concept_name, COUNT(*) AS n
        FROM concepts c
        JOIN works_clean w ON c.work_id = w.work_id
        WHERE c.level = 1 AND c.concept_name IS NOT NULL
        GROUP BY c.concept_name
        ORDER BY n DESC
        LIMIT 15
    """).df()

    concept_names = top_concepts["concept_name"].tolist()
    print(f"  Top concepts: {concept_names[:5]}...", flush=True)

    # Year × concept matrix
    df = con.execute("""
        SELECT w.year, c.concept_name, COUNT(*) AS n
        FROM concepts c
        JOIN works_clean w ON c.work_id = w.work_id
        WHERE c.level = 1
          AND c.concept_name IN (
              SELECT concept_name FROM (
                  SELECT concept_name, COUNT(*) AS n
                  FROM concepts WHERE level = 1 AND concept_name IS NOT NULL
                  GROUP BY concept_name ORDER BY n DESC LIMIT 15
              )
          )
        GROUP BY w.year, c.concept_name
        ORDER BY w.year
    """).df()
    df.to_csv(f"{OUT_DIR}/subfield_by_year.csv", index=False)
    top_concepts.to_csv(f"{OUT_DIR}/top_concepts.csv", index=False)
    print(f"  {len(df)} rows", flush=True)


def q3_doubling_time(con, growth_df):
    """Fit exponential and logistic growth models."""
    print("Q3: Doubling time analysis...", flush=True)

    df = growth_df[growth_df["year"] >= 1960].copy()
    x = df["year"].values - 1960  # normalize
    y = df["n_papers"].values.astype(float)

    results = {}

    # Exponential fit: y = a * exp(b * x)
    try:
        def exp_model(x, a, b):
            return a * np.exp(b * x)
        popt, _ = curve_fit(exp_model, x, y, p0=[1000, 0.05], maxfev=5000)
        doubling_time = np.log(2) / popt[1]
        results["exp_a"] = popt[0]
        results["exp_b"] = popt[1]
        results["doubling_time_years"] = doubling_time
        print(f"  Exponential doubling time: {doubling_time:.1f} years", flush=True)
    except Exception as e:
        print(f"  Exponential fit failed: {e}", flush=True)

    # Logistic fit: y = K / (1 + exp(-r*(x - x0)))
    try:
        def logistic_model(x, K, r, x0):
            return K / (1 + np.exp(-r * (x - x0)))
        popt_l, _ = curve_fit(logistic_model, x, y,
                              p0=[y.max() * 2, 0.05, 40], maxfev=10000)
        results["logistic_K"] = popt_l[0]
        results["logistic_r"] = popt_l[1]
        results["logistic_x0"] = popt_l[2] + 1960
        print(f"  Logistic carrying capacity: {popt_l[0]:,.0f}, "
              f"inflection: {popt_l[2] + 1960:.0f}", flush=True)
    except Exception as e:
        print(f"  Logistic fit failed: {e}", flush=True)

    pd.DataFrame([results]).to_csv(f"{OUT_DIR}/growth_model_fits.csv", index=False)


def q4_oa_composition(con):
    """OA type breakdown per year."""
    print("Q4: OA composition over time...", flush=True)
    df = con.execute("""
        SELECT year,
               COALESCE(oa_status, 'unknown') AS oa_status,
               COUNT(*) AS n
        FROM works_clean
        WHERE year >= 1990
        GROUP BY year, oa_status
        ORDER BY year, oa_status
    """).df()
    df.to_csv(f"{OUT_DIR}/oa_by_year.csv", index=False)
    print(f"  {len(df)} rows", flush=True)


def q5_language_diversity(con):
    """Language distribution per year."""
    print("Q5: Language diversity...", flush=True)
    df = con.execute("""
        SELECT year, language, COUNT(*) AS n
        FROM works_clean
        WHERE language IS NOT NULL AND year >= 1960
        GROUP BY year, language
        ORDER BY year, n DESC
    """).df()
    df.to_csv(f"{OUT_DIR}/language_by_year.csv", index=False)

    # Shannon diversity index per year
    div = []
    for year, grp in df.groupby("year"):
        counts = grp["n"].values
        total = counts.sum()
        p = counts / total
        H = -np.sum(p * np.log(p))
        div.append({"year": year, "shannon_H": H, "n_languages": len(counts), "total": total})
    pd.DataFrame(div).to_csv(f"{OUT_DIR}/language_diversity.csv", index=False)
    print(f"  {len(div)} years", flush=True)


def q6_article_types(con):
    """Article type distribution per year."""
    print("Q6: Article types...", flush=True)
    df = con.execute("""
        SELECT year,
               COALESCE(type, 'unknown') AS article_type,
               COUNT(*) AS n
        FROM works_clean
        WHERE year >= 1960
        GROUP BY year, article_type
        ORDER BY year, n DESC
    """).df()
    df.to_csv(f"{OUT_DIR}/article_types_by_year.csv", index=False)
    print(f"  {len(df)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    con = create_database(args.db_path)

    t0 = time.time()
    growth_df = q1_growth_by_year(con)
    q2_subfield_growth(con)
    q3_doubling_time(con, growth_df)
    q4_oa_composition(con)
    q5_language_diversity(con)
    q6_article_types(con)

    elapsed = int(time.time() - t0)
    print(f"\nTheme 1 complete ({elapsed}s)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
