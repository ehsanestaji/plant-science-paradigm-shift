"""
Organism-stratified sleeping beauty analysis.

Re-stratifies sleeping_beauty_scores.csv by our more granular organism labels
(overriding the coarser `crop` column already in that file).

Outputs
-------
results/paper_a/supplementary/organism_sleeping_beauties.csv
    organism, n_sleeping_beauties, mean_sleep_duration, mean_awakening_year

results/paper_a/supplementary/organism_sleeping_beauty_detail.csv
    work_id, organism, beauty_score, sleep_years, awakening_year
    (top 20 per organism by B score)
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
    ap = argparse.ArgumentParser(description="Organism-stratified sleeping beauty analysis")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    sb_path = "results/novel/sleeping_beauty_scores.csv"
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")

    print(f"Loading sleeping beauty scores from {sb_path} ...", flush=True)
    sb = pd.read_csv(
        sb_path,
        usecols=["work_id", "pub_year", "crop", "title", "total_citations",
                 "B", "awakening_year", "sleep_years", "c_at_awakening"],
    )
    print(f"  {len(sb):,} sleeping beauties loaded", flush=True)

    print(f"Loading organism labels from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label"])
    print(f"  {len(clf):,} classified papers", flush=True)

    # ------------------------------------------------------------------
    # 2. Merge — use our organism labels, ignore the coarser `crop` column
    # ------------------------------------------------------------------
    df = sb.merge(clf, on="work_id", how="inner")
    df = df.rename(columns={"predicted_label": "organism"})
    df = df[df["organism"].isin(VALID_LABELS)]
    df = df.dropna(subset=["B"])
    print(f"\nAfter merge: {len(df):,} rows with organism labels", flush=True)

    # ------------------------------------------------------------------
    # 3. Summary per organism
    # ------------------------------------------------------------------
    print("Computing per-organism summary ...", flush=True)
    summary = (
        df.groupby("organism")
        .agg(
            n_sleeping_beauties=("work_id", "count"),
            mean_sleep_duration=("sleep_years", "mean"),
            mean_awakening_year=("awakening_year", "mean"),
        )
        .reset_index()
        .sort_values("n_sleeping_beauties", ascending=False)
    )

    # ------------------------------------------------------------------
    # 4. Top 20 per organism by beauty score B
    # ------------------------------------------------------------------
    print("Selecting top 20 per organism by beauty score ...", flush=True)
    detail_rows = []
    for organism, grp in df.groupby("organism"):
        top = grp.nlargest(20, "B")[
            ["work_id", "organism", "B", "sleep_years", "awakening_year"]
        ]
        detail_rows.append(top)
    detail_df = pd.concat(detail_rows, ignore_index=True)
    detail_df = detail_df.rename(columns={"B": "beauty_score"})

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n=== Sleeping beauties per organism ===", flush=True)
    for _, row in summary.iterrows():
        print(
            f"  {row['organism']:<28}  n={row['n_sleeping_beauties']:>5}  "
            f"mean_sleep={row['mean_sleep_duration']:.1f}yr  "
            f"mean_awaken={row['mean_awakening_year']:.1f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    summary_path = os.path.join(
        args.out_dir, "supplementary", "organism_sleeping_beauties.csv"
    )
    detail_path = os.path.join(
        args.out_dir, "supplementary", "organism_sleeping_beauty_detail.csv"
    )

    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}  ({len(summary):,} rows)", flush=True)

    detail_df.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path}  ({len(detail_df):,} rows)", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
