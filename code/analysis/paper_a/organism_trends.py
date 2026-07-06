"""
Organism prevalence trends, 1990-2024.

Analyses the share, growth rates, and rank changes of 13 organism categories
across 2.77M plant science papers.

Outputs
-------
results/paper_a/main/organism_timeseries.csv
results/paper_a/main/organism_growth_rates.csv
results/paper_a/supplementary/organism_rank_changes.csv
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

DECADES = {
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2024),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cagr(start_count: float, end_count: float, n_years: int) -> float:
    """Return CAGR, or NaN when calculation is not possible."""
    if n_years <= 0 or start_count <= 0 or end_count <= 0:
        return float("nan")
    return (end_count / start_count) ** (1.0 / n_years) - 1.0


def growth_rates(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-organism CAGR and absolute growth for each decade.

    Parameters
    ----------
    ts : DataFrame with columns [year, organism, paper_count]
    """
    rows = []
    for decade_name, (d_start, d_end) in DECADES.items():
        decade_data = ts[(ts["year"] >= d_start) & (ts["year"] <= d_end)]
        for organism in ts["organism"].unique():
            sub = decade_data[decade_data["organism"] == organism].sort_values("year")
            if sub.empty:
                continue
            # Use first and last year that have papers
            sub_nonzero = sub[sub["paper_count"] > 0]
            if sub_nonzero.empty:
                cagr = float("nan")
                abs_growth = 0
            else:
                first_row = sub_nonzero.iloc[0]
                last_row = sub_nonzero.iloc[-1]
                n_years = int(last_row["year"]) - int(first_row["year"])
                cagr = compute_cagr(
                    first_row["paper_count"], last_row["paper_count"], n_years
                )
                abs_growth = int(last_row["paper_count"]) - int(first_row["paper_count"])
            rows.append(
                {
                    "organism": organism,
                    "decade": decade_name,
                    "cagr": cagr,
                    "absolute_growth": abs_growth,
                }
            )
    return pd.DataFrame(rows, columns=["organism", "decade", "cagr", "absolute_growth"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism trends analysis")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "main"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load organism classifications
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")
    print(f"Loading classifications from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label", "confidence"])
    print(f"  {len(clf):,} rows loaded", flush=True)

    # Confidence distribution summary
    print("\nConfidence distribution:", flush=True)
    conf_bins = [0.0, 0.5, 0.7, 0.9, 0.95, 1.0]
    counts, edges = np.histogram(clf["confidence"].dropna(), bins=conf_bins)
    for lo, hi, n in zip(edges[:-1], edges[1:], counts):
        print(f"  [{lo:.2f}, {hi:.2f}): {n:,}", flush=True)

    # ------------------------------------------------------------------
    # 2. Load year information from DuckDB
    # ------------------------------------------------------------------
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    print("\nQuerying work years from database ...", flush=True)
    years_df = con.execute(
        "SELECT work_id, year FROM works_clean WHERE year BETWEEN 1990 AND 2024"
    ).fetchdf()
    print(f"  {len(years_df):,} works in 1990-2024", flush=True)
    con.close()

    # ------------------------------------------------------------------
    # 3. Join classifications with years
    # ------------------------------------------------------------------
    df = clf.merge(years_df, on="work_id", how="inner")
    print(f"\nAfter join: {len(df):,} classified works with year data", flush=True)

    # Drop rows with missing year (should be none after the WHERE clause, but
    # guard against edge cases)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)]
    df = df[df["predicted_label"].isin(VALID_LABELS)]

    # ------------------------------------------------------------------
    # 4. Annual paper count per organism
    # ------------------------------------------------------------------
    print("\nComputing annual counts ...", flush=True)
    annual = (
        df.groupby(["year", "predicted_label"])
        .size()
        .reset_index(name="paper_count")
        .rename(columns={"predicted_label": "organism"})
    )

    # Ensure every (year, organism) combination exists (fill zeros)
    all_years = pd.RangeIndex(1990, 2025)
    idx = pd.MultiIndex.from_product(
        [all_years, VALID_LABELS], names=["year", "organism"]
    )
    annual = (
        annual.set_index(["year", "organism"])
        .reindex(idx, fill_value=0)
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 5. Market share per year
    # ------------------------------------------------------------------
    total_per_year = annual.groupby("year")["paper_count"].sum().rename("total")
    annual = annual.join(total_per_year, on="year")
    annual["share_pct"] = np.where(
        annual["total"] > 0,
        annual["paper_count"] / annual["total"] * 100.0,
        0.0,
    )
    annual = annual.drop(columns=["total"])

    # ------------------------------------------------------------------
    # 6. Growth rates
    # ------------------------------------------------------------------
    print("Computing growth rates ...", flush=True)
    growth_df = growth_rates(annual)

    # ------------------------------------------------------------------
    # 7. Rank per organism per year
    # ------------------------------------------------------------------
    print("Computing ranks ...", flush=True)
    annual["rank"] = annual.groupby("year")["paper_count"].rank(
        method="min", ascending=False
    ).astype(int)
    rank_df = annual[["year", "organism", "rank"]].copy()

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    ts_path = os.path.join(args.out_dir, "main", "organism_timeseries.csv")
    gr_path = os.path.join(args.out_dir, "main", "organism_growth_rates.csv")
    rk_path = os.path.join(args.out_dir, "supplementary", "organism_rank_changes.csv")

    ts_out = annual[["year", "organism", "paper_count", "share_pct"]]
    ts_out.to_csv(ts_path, index=False)
    print(f"\nWrote {ts_path}  ({len(ts_out):,} rows)", flush=True)

    growth_df.to_csv(gr_path, index=False)
    print(f"Wrote {gr_path}  ({len(growth_df):,} rows)", flush=True)

    rank_df.to_csv(rk_path, index=False)
    print(f"Wrote {rk_path}  ({len(rank_df):,} rows)", flush=True)

    # ------------------------------------------------------------------
    # 9. Summary printout
    # ------------------------------------------------------------------
    print("\n=== Summary: total papers per organism (1990-2024) ===", flush=True)
    totals = (
        annual.groupby("organism")["paper_count"]
        .sum()
        .sort_values(ascending=False)
    )
    for org, cnt in totals.items():
        print(f"  {org:<28} {cnt:>10,}", flush=True)

    print("\n=== Inflection points (largest year-on-year share change) ===", flush=True)
    annual_sorted = annual.sort_values(["organism", "year"])
    annual_sorted["share_delta"] = annual_sorted.groupby("organism")["share_pct"].diff()
    inflections = (
        annual_sorted.dropna(subset=["share_delta"])
        .reindex(annual_sorted["share_delta"].abs().sort_values(ascending=False).index)
        .head(10)[["organism", "year", "share_pct", "share_delta"]]
    )
    print(inflections.to_string(index=False), flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
