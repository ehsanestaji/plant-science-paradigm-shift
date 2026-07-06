"""
Organism-stratified method diffusion analysis.

The existing method_adoption_timeseries.csv is already aggregated by
year×method and lacks work_ids.  This script re-detects methods from
paper abstracts stored in DuckDB and stratifies the results by organism.

Outputs
-------
results/paper_a/main/method_diffusion_by_organism.csv
    method, organism, year, paper_count, cumulative_share

results/paper_a/supplementary/method_adoption_lag.csv
    method, organism, first_adoption_year, lag_vs_arabidopsis_years,
    current_penetration_pct
"""
import argparse
import os
import re
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

METHODS = {
    "CRISPR":       re.compile(r'\bcrispr\b',                         re.IGNORECASE),
    "single-cell":  re.compile(r'\bsingle.cell\b',                    re.IGNORECASE),
    "pangenomics":  re.compile(r'\b(pangenome|pan.genome|pangenomic)\b', re.IGNORECASE),
}

YEAR_MIN = 1990
YEAR_MAX = 2024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_methods(abstract: str) -> list[str]:
    """Return list of method names found in abstract."""
    if not isinstance(abstract, str):
        return []
    return [name for name, pat in METHODS.items() if pat.search(abstract)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism-stratified method diffusion")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "main"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load organism labels
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")
    print(f"Loading organism labels from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label"])
    clf = clf[clf["predicted_label"].isin(VALID_LABELS)]
    print(f"  {len(clf):,} classified papers", flush=True)

    # ------------------------------------------------------------------
    # 2. Query abstracts from DuckDB using REGEXP_MATCHES for efficiency
    # ------------------------------------------------------------------
    print(f"\nOpening DuckDB: {args.db_path} ...", flush=True)
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    # Use server-side regexp filtering to pull only papers that mention at
    # least one method — much smaller result set to load into Python.
    crispr_pat    = r'(?i)\bcrispr\b'
    sc_pat        = r'(?i)\bsingle.cell\b'
    pan_pat       = r'(?i)\b(pangenome|pan.genome|pangenomic)\b'

    print("Querying abstracts (server-side regexp pre-filter) ...", flush=True)
    sql = f"""
        SELECT work_id, year, abstract
        FROM works_clean
        WHERE year BETWEEN {YEAR_MIN} AND {YEAR_MAX}
          AND abstract IS NOT NULL
          AND (
              regexp_matches(abstract, '{crispr_pat}')
              OR regexp_matches(abstract, '{sc_pat}')
              OR regexp_matches(abstract, '{pan_pat}')
          )
    """
    abstracts = con.execute(sql).fetchdf()
    con.close()
    print(f"  {len(abstracts):,} papers pre-filtered by method keywords", flush=True)

    # ------------------------------------------------------------------
    # 3. Join with organism labels
    # ------------------------------------------------------------------
    df = abstracts.merge(clf, on="work_id", how="inner")
    df = df.rename(columns={"predicted_label": "organism"})
    df["year"] = df["year"].astype(int)
    print(f"  {len(df):,} rows after organism join", flush=True)

    # ------------------------------------------------------------------
    # 4. Detect methods per paper (Python-side fine-grained check)
    # ------------------------------------------------------------------
    print("Detecting methods per paper ...", flush=True)
    method_rows = []
    for _, row in df.iterrows():
        for method in detect_methods(row["abstract"]):
            method_rows.append({
                "work_id":  row["work_id"],
                "year":     row["year"],
                "organism": row["organism"],
                "method":   method,
            })

    if not method_rows:
        print("No method mentions found — check regex patterns.", flush=True)
        return

    mdf = pd.DataFrame(method_rows)
    # De-duplicate: one paper can only count once per method
    mdf = mdf.drop_duplicates(subset=["work_id", "method"])
    print(f"  {len(mdf):,} paper×method pairs", flush=True)

    # ------------------------------------------------------------------
    # 5. Paper count per method × organism × year
    # ------------------------------------------------------------------
    counts = (
        mdf.groupby(["method", "organism", "year"])
        .size()
        .reset_index(name="paper_count")
    )

    # Total organism papers per year (denominator for cumulative share)
    # Use full classified set merged with year info from the abstracts query.
    # We need total organism papers per year across ALL papers, not just those
    # with method mentions.  Re-query the totals.
    print("Querying total organism paper counts per year ...", flush=True)
    con2 = create_database(args.db_path, read_only=True)
    con2.execute("SET memory_limit='60GB'")
    con2.execute("SET threads=8")
    year_df = con2.execute(
        f"SELECT work_id, year FROM works_clean "
        f"WHERE year BETWEEN {YEAR_MIN} AND {YEAR_MAX}"
    ).fetchdf()
    con2.close()

    totals_df = (
        clf.merge(year_df, on="work_id", how="inner")
        .groupby(["predicted_label", "year"])
        .size()
        .reset_index(name="total_org_papers")
        .rename(columns={"predicted_label": "organism"})
    )
    totals_df["year"] = totals_df["year"].astype(int)

    counts = counts.merge(totals_df, on=["organism", "year"], how="left")

    # Cumulative share: cumulative method papers / total organism papers that year
    counts = counts.sort_values(["method", "organism", "year"])
    counts["cumulative_count"] = counts.groupby(["method", "organism"])["paper_count"].cumsum()
    counts["cumulative_share"] = np.where(
        counts["total_org_papers"] > 0,
        counts["cumulative_count"] / counts["total_org_papers"],
        np.nan,
    )
    diffusion = counts[["method", "organism", "year", "paper_count", "cumulative_share"]].copy()

    # ------------------------------------------------------------------
    # 6. First adoption year per organism per method
    # ------------------------------------------------------------------
    print("Computing adoption lags vs Arabidopsis ...", flush=True)
    first_adoption = (
        mdf.groupby(["method", "organism"])["year"]
        .min()
        .reset_index(name="first_adoption_year")
    )

    # Lag vs Arabidopsis
    arab_first = (
        first_adoption[first_adoption["organism"] == "arabidopsis"]
        .set_index("method")["first_adoption_year"]
    )
    first_adoption["lag_vs_arabidopsis_years"] = first_adoption.apply(
        lambda r: (
            r["first_adoption_year"] - arab_first[r["method"]]
            if r["method"] in arab_first.index
            else np.nan
        ),
        axis=1,
    )

    # Current penetration: latest year count / total org papers in latest year
    latest_year = mdf["year"].max()
    latest_counts = counts[counts["year"] == latest_year][
        ["method", "organism", "paper_count", "total_org_papers"]
    ].copy()
    latest_counts["current_penetration_pct"] = np.where(
        latest_counts["total_org_papers"] > 0,
        latest_counts["paper_count"] / latest_counts["total_org_papers"] * 100.0,
        np.nan,
    )

    lag_df = first_adoption.merge(
        latest_counts[["method", "organism", "current_penetration_pct"]],
        on=["method", "organism"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print("\n=== First adoption year per method × organism ===", flush=True)
    for method in sorted(METHODS.keys()):
        sub = first_adoption[first_adoption["method"] == method].sort_values(
            "first_adoption_year"
        )
        print(f"\n  {method}:", flush=True)
        for _, row in sub.iterrows():
            lag = row["lag_vs_arabidopsis_years"]
            lag_str = f"lag={lag:+.0f}yr" if not np.isnan(lag) else "lag=N/A"
            print(
                f"    {row['organism']:<28}  first={int(row['first_adoption_year'])}  {lag_str}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    diff_path = os.path.join(args.out_dir, "main", "method_diffusion_by_organism.csv")
    lag_path  = os.path.join(args.out_dir, "supplementary", "method_adoption_lag.csv")

    diffusion.to_csv(diff_path, index=False)
    print(f"\nWrote {diff_path}  ({len(diffusion):,} rows)", flush=True)

    lag_df.to_csv(lag_path, index=False)
    print(f"Wrote {lag_path}  ({len(lag_df):,} rows)", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
