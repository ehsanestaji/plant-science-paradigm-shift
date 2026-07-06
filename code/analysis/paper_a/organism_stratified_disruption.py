"""
Organism-stratified disruption index analysis.

Re-stratifies the CD index results (results/novel/cd_index.csv) by organism
type to reveal which organisms drive disruptive vs. consolidating science.

Outputs
-------
results/paper_a/supplementary/organism_disruption_index.csv
    organism, decade, mean_cd, median_cd, n_papers

results/paper_a/supplementary/organism_top_disruptors.csv
    work_id, organism, cd_score, title, year  (top 50 per organism)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def year_to_decade(year: pd.Series) -> pd.Series:
    """Map a year series to decade labels like '1990s', '2000s', etc."""
    decade_start = (year // 10) * 10
    return decade_start.astype(str) + "s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism-stratified disruption index")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    cd_path = "results/novel/cd_index.csv"
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")

    print(f"Loading CD index from {cd_path} ...", flush=True)
    cd = pd.read_csv(cd_path, usecols=["work_id", "year", "cd_index", "title"])
    print(f"  {len(cd):,} rows", flush=True)

    print(f"Loading organism labels from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label"])
    print(f"  {len(clf):,} rows", flush=True)

    # ------------------------------------------------------------------
    # 2. Merge
    # ------------------------------------------------------------------
    df = cd.merge(clf, on="work_id", how="inner")
    df = df.rename(columns={"predicted_label": "organism"})
    df = df[df["organism"].isin(VALID_LABELS)]
    df = df.dropna(subset=["cd_index", "year"])
    df["year"] = df["year"].astype(int)
    df["decade"] = year_to_decade(df["year"])
    print(f"\nAfter merge: {len(df):,} rows", flush=True)

    # ------------------------------------------------------------------
    # 3. Mean / median CD per organism per decade
    # ------------------------------------------------------------------
    print("Computing per-organism per-decade statistics ...", flush=True)
    agg = (
        df.groupby(["organism", "decade"])["cd_index"]
        .agg(mean_cd="mean", median_cd="median", n_papers="count")
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. Kruskal-Wallis test across organisms
    # ------------------------------------------------------------------
    groups = [
        grp["cd_index"].dropna().values
        for _, grp in df.groupby("organism")
        if len(grp) >= 5
    ]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis H={kw_stat:.4f}, p={kw_p:.4e}", flush=True)
    else:
        print("\nNot enough groups for Kruskal-Wallis test", flush=True)

    # ------------------------------------------------------------------
    # 5. Mean CD per organism (overall)
    # ------------------------------------------------------------------
    print("\n=== Mean CD index per organism ===", flush=True)
    org_means = (
        df.groupby("organism")["cd_index"]
        .agg(mean_cd="mean", n_papers="count")
        .sort_values("mean_cd", ascending=False)
    )
    for org, row in org_means.iterrows():
        print(f"  {org:<28}  mean_cd={row['mean_cd']:+.4f}  n={row['n_papers']:,}", flush=True)

    # ------------------------------------------------------------------
    # 6. Top 50 most disruptive per organism
    # ------------------------------------------------------------------
    print("\nSelecting top 50 disruptors per organism ...", flush=True)
    top_rows = []
    for organism, grp in df.groupby("organism"):
        top = grp.nlargest(50, "cd_index")[["work_id", "organism", "cd_index", "title", "year"]]
        top_rows.append(top)
    top_df = pd.concat(top_rows, ignore_index=True)
    top_df = top_df.rename(columns={"cd_index": "cd_score"})

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    agg_path = os.path.join(args.out_dir, "supplementary", "organism_disruption_index.csv")
    top_path = os.path.join(args.out_dir, "supplementary", "organism_top_disruptors.csv")

    agg.to_csv(agg_path, index=False)
    print(f"\nWrote {agg_path}  ({len(agg):,} rows)", flush=True)

    top_df.to_csv(top_path, index=False)
    print(f"Wrote {top_path}  ({len(top_df):,} rows)", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
