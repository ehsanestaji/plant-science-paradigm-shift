"""
Organism collaboration patterns: team size, international collaboration, and
geographic concentration per organism label.

Outputs
-------
results/paper_a/supplementary/organism_collaboration.csv
    organism, year, mean_team_size, intl_collab_rate
results/paper_a/supplementary/organism_geographic.csv
    organism, country_code, paper_count, share
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.db.schema import create_database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LABELS = [
    "arabidopsis", "rice", "wheat", "maize", "soybean", "tomato",
    "barley", "cotton", "potato", "tobacco",
    "other_crop", "other_model_organism", "non_specific",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism collaboration analysis")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load organism classifications
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")
    print(f"Loading classifications from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label", "confidence"])
    clf = clf[clf["predicted_label"].isin(VALID_LABELS)]
    print(f"  {len(clf):,} rows loaded ({clf['predicted_label'].nunique()} labels)", flush=True)

    # ------------------------------------------------------------------
    # 2. Open DuckDB and register organism labels as a temp table
    # ------------------------------------------------------------------
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    print("\nRegistering organism labels as temp table ...", flush=True)
    con.register("clf", clf)
    con.execute("CREATE TEMP TABLE org_labels AS SELECT work_id, predicted_label FROM clf")

    # ------------------------------------------------------------------
    # 3. Team size and international collaboration per paper
    # ------------------------------------------------------------------
    print("Computing team size and international collaboration per paper ...", flush=True)
    paper_collab = con.execute("""
        SELECT
            wa.work_id,
            COUNT(DISTINCT wa.author_id)   AS team_size,
            COUNT(DISTINCT wa.country_code) AS n_countries
        FROM work_authors wa
        WHERE wa.work_id IN (SELECT work_id FROM org_labels)
        GROUP BY wa.work_id
    """).fetchdf()
    print(f"  {len(paper_collab):,} papers with author data", flush=True)

    # ------------------------------------------------------------------
    # 4. Per organism x year: mean team size and intl collab rate
    # ------------------------------------------------------------------
    print("Querying per-organism-year collaboration statistics ...", flush=True)

    # Register paper_collab as temp table for efficient join
    con.register("paper_collab_df", paper_collab)
    con.execute("CREATE TEMP TABLE paper_collab AS SELECT * FROM paper_collab_df")

    collab_df = con.execute("""
        SELECT
            o.predicted_label            AS organism,
            w.year,
            AVG(pc.team_size)            AS mean_team_size,
            AVG(CASE WHEN pc.n_countries > 1 THEN 1.0 ELSE 0.0 END) AS intl_collab_rate,
            COUNT(*)                     AS n_papers
        FROM org_labels o
        JOIN works_clean w  ON o.work_id = w.work_id
        JOIN paper_collab pc ON o.work_id = pc.work_id
        WHERE w.year BETWEEN 1990 AND 2024
        GROUP BY o.predicted_label, w.year
        ORDER BY o.predicted_label, w.year
    """).fetchdf()
    print(f"  {len(collab_df):,} organism-year rows", flush=True)

    # ------------------------------------------------------------------
    # 5. Geographic concentration: paper count per organism per country
    # ------------------------------------------------------------------
    print("\nQuerying geographic concentration per organism ...", flush=True)
    geo_df = con.execute("""
        SELECT
            o.predicted_label  AS organism,
            wa.country_code,
            COUNT(DISTINCT o.work_id) AS paper_count
        FROM org_labels o
        JOIN work_authors wa ON o.work_id = wa.work_id
        WHERE wa.country_code IS NOT NULL
          AND wa.country_code <> ''
        GROUP BY o.predicted_label, wa.country_code
        ORDER BY o.predicted_label, paper_count DESC
    """).fetchdf()
    print(f"  {len(geo_df):,} organism-country rows", flush=True)

    con.close()

    # ------------------------------------------------------------------
    # 6. Compute share: paper_count / total papers for that organism
    # ------------------------------------------------------------------
    total_org = (
        geo_df.groupby("organism")["paper_count"]
        .sum()
        .rename("total_papers")
    )
    geo_df = geo_df.join(total_org, on="organism")
    geo_df["share"] = geo_df["paper_count"] / geo_df["total_papers"]
    geo_df = geo_df.drop(columns=["total_papers"])

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    collab_path = os.path.join(args.out_dir, "supplementary", "organism_collaboration.csv")
    geo_path    = os.path.join(args.out_dir, "supplementary", "organism_geographic.csv")

    collab_df[["organism", "year", "mean_team_size", "intl_collab_rate"]].to_csv(
        collab_path, index=False
    )
    print(f"\nWrote {collab_path}  ({len(collab_df):,} rows)", flush=True)

    geo_df.to_csv(geo_path, index=False)
    print(f"Wrote {geo_path}  ({len(geo_df):,} rows)", flush=True)

    # ------------------------------------------------------------------
    # 8. Summary printouts
    # ------------------------------------------------------------------
    print("\n=== Top 3 countries per organism ===", flush=True)
    for organism in sorted(geo_df["organism"].unique()):
        top3 = (
            geo_df[geo_df["organism"] == organism]
            .sort_values("paper_count", ascending=False)
            .head(3)
        )
        countries = ", ".join(
            f"{r['country_code']} ({r['share']*100:.1f}%)"
            for _, r in top3.iterrows()
        )
        print(f"  {organism:<28} {countries}", flush=True)

    print("\n=== Overall international collaboration rates per organism ===", flush=True)
    overall_intl = (
        collab_df.groupby("organism")
        .apply(lambda g: np.average(g["intl_collab_rate"], weights=g["n_papers"]))
        .sort_values(ascending=False)
    )
    for org, rate in overall_intl.items():
        print(f"  {org:<28} {rate*100:6.1f}%", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
