"""
Paper A — Paradigm Shift Analysis.

Examines the shift from fundamental to applied research in plant science,
stratified by organism type.

Outputs → results/paper_a/
  main/paradigm_shift_by_organism.csv
      year, organism, fundamental_count, applied_count, applied_share
  supplementary/paradigm_citation_impact.csv
      paradigm, organism, median_citations, mean_citations, n_papers
  supplementary/paradigm_oa_interaction.csv
      paradigm, oa_status, count, share

Usage:
    python -m src.analysis.paper_a.paradigm_shift \\
        --db-path data/processed/plant_science.duckdb
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
from src.utils.storage_monitor import check_storage

# Short label constants to avoid repeated long strings
LABEL_FUNDAMENTAL = "fundamental basic science research"
LABEL_APPLIED = "applied translational agricultural research"

YEAR_MIN = 1990
YEAR_MAX = 2024

OUT_MAIN = "results/paper_a/main"
OUT_SUPP = "results/paper_a/supplementary"


def _ensure_dirs():
    os.makedirs(OUT_MAIN, exist_ok=True)
    os.makedirs(OUT_SUPP, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_classifications(classifications_dir: str) -> pd.DataFrame:
    """Load paradigm and organism CSVs and merge on work_id."""
    print("Loading classification CSVs...", flush=True)

    paradigm_path = os.path.join(classifications_dir, "paper_a_paradigm.csv")
    organism_path = os.path.join(classifications_dir, "paper_a_organism.csv")

    paradigm_df = pd.read_csv(paradigm_path)
    organism_df = pd.read_csv(organism_path)

    print(f"  Paradigm CSV: {len(paradigm_df):,} rows", flush=True)
    print(f"  Organism CSV: {len(organism_df):,} rows", flush=True)

    # Rename to avoid column name clashes after merge
    paradigm_df = paradigm_df.rename(columns={
        "predicted_label": "paradigm_label",
        "confidence": "paradigm_confidence",
    })
    organism_df = organism_df.rename(columns={
        "predicted_label": "organism_label",
        "confidence": "organism_confidence",
    })

    merged = paradigm_df.merge(organism_df, on="work_id", how="inner")
    print(f"  Merged: {len(merged):,} rows (inner join on work_id)", flush=True)
    return merged


def fetch_works_metadata(con, work_ids: list) -> pd.DataFrame:
    """Pull year, oa_status, cited_by_count from works_clean for the given IDs."""
    print("Fetching works metadata from DuckDB...", flush=True)

    # Register as a temporary table for efficient filtering
    id_df = pd.DataFrame({"work_id": work_ids})
    con.register("_tmp_ids", id_df)

    df = con.execute("""
        SELECT w.work_id, w.year, w.oa_status, w.cited_by_count
        FROM works_clean w
        INNER JOIN _tmp_ids t ON w.work_id = t.work_id
        WHERE w.year BETWEEN ? AND ?
    """, [YEAR_MIN, YEAR_MAX]).df()

    con.unregister("_tmp_ids")
    print(f"  {len(df):,} works with year {YEAR_MIN}-{YEAR_MAX}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_master_df(classifications: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join classifications with DuckDB metadata and filter to study window."""
    df = classifications.merge(metadata, on="work_id", how="inner")
    print(f"  Master dataframe: {len(df):,} rows after join with metadata", flush=True)
    return df


def paradigm_shift_by_organism(df: pd.DataFrame) -> pd.DataFrame:
    """Year × organism counts for fundamental vs applied, plus applied_share."""
    print("Computing paradigm shift by organism...", flush=True)

    fund_df = df[df["paradigm_label"] == LABEL_FUNDAMENTAL]
    appl_df = df[df["paradigm_label"] == LABEL_APPLIED]

    fund_counts = (
        fund_df.groupby(["year", "organism_label"])
               .size()
               .reset_index(name="fundamental_count")
    )
    appl_counts = (
        appl_df.groupby(["year", "organism_label"])
               .size()
               .reset_index(name="applied_count")
    )

    result = fund_counts.merge(
        appl_counts, on=["year", "organism_label"], how="outer"
    ).fillna(0)

    result["fundamental_count"] = result["fundamental_count"].astype(int)
    result["applied_count"] = result["applied_count"].astype(int)
    total = result["fundamental_count"] + result["applied_count"]
    result["applied_share"] = np.where(total > 0, result["applied_count"] / total, np.nan)

    result = result.rename(columns={"organism_label": "organism"})
    result = result.sort_values(["year", "organism"]).reset_index(drop=True)

    print(f"  {len(result):,} year x organism rows", flush=True)
    return result


def paradigm_citation_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Median and mean citations per paradigm per organism."""
    print("Computing paradigm citation impact...", flush=True)

    df_cit = df.dropna(subset=["cited_by_count", "paradigm_label", "organism_label"]).copy()

    agg = (
        df_cit.groupby(["paradigm_label", "organism_label"])
              .agg(
                  median_citations=("cited_by_count", "median"),
                  mean_citations=("cited_by_count", "mean"),
                  n_papers=("work_id", "count"),
              )
              .reset_index()
              .rename(columns={"paradigm_label": "paradigm", "organism_label": "organism"})
              .sort_values(["paradigm", "organism"])
              .reset_index(drop=True)
    )

    print(f"  {len(agg):,} paradigm x organism rows", flush=True)
    return agg


def paradigm_oa_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Paradigm × OA cross-tabulation: count and share within each paradigm."""
    print("Computing paradigm x OA interaction...", flush=True)

    df_oa = df.dropna(subset=["paradigm_label", "oa_status"]).copy()
    df_oa["oa_status"] = df_oa["oa_status"].fillna("unknown")

    counts = (
        df_oa.groupby(["paradigm_label", "oa_status"])
             .size()
             .reset_index(name="count")
    )

    paradigm_totals = counts.groupby("paradigm_label")["count"].transform("sum")
    counts["share"] = counts["count"] / paradigm_totals

    result = (
        counts.rename(columns={"paradigm_label": "paradigm"})
              .sort_values(["paradigm", "oa_status"])
              .reset_index(drop=True)
    )

    print(f"  {len(result):,} paradigm x oa_status rows", flush=True)
    return result


def run_chi2_test(df: pd.DataFrame):
    """Chi-square test of independence: organism x paradigm contingency table."""
    print("Running chi-square test (organism x paradigm)...", flush=True)

    # Filter to the two main paradigm labels only
    df_filtered = df[df["paradigm_label"].isin([LABEL_FUNDAMENTAL, LABEL_APPLIED])].copy()

    contingency = pd.crosstab(df_filtered["organism_label"], df_filtered["paradigm_label"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"  Chi2 = {chi2:.2f}, p = {p_value:.3e}, df = {dof}", flush=True)
    return chi2, p_value, dof, contingency


def print_key_findings(df: pd.DataFrame, chi2: float, p_value: float,
                       dof: int, contingency: pd.DataFrame):
    """Print human-readable summary of key findings."""
    print("\n" + "=" * 60, flush=True)
    print("KEY FINDINGS", flush=True)
    print("=" * 60, flush=True)

    # Overall applied share
    df_labeled = df[df["paradigm_label"].isin([LABEL_FUNDAMENTAL, LABEL_APPLIED])]
    total = len(df_labeled)
    n_applied = (df_labeled["paradigm_label"] == LABEL_APPLIED).sum()
    overall_applied_share = n_applied / total if total > 0 else 0.0

    print(f"Total labeled papers: {total:,}", flush=True)
    print(f"Overall applied share: {overall_applied_share:.3f} "
          f"({n_applied:,} applied / {total:,} total)", flush=True)
    print(f"\nChi-square test (organism x paradigm):", flush=True)
    print(f"  chi2 = {chi2:.4f}", flush=True)
    print(f"  p-value = {p_value:.4e}", flush=True)
    print(f"  degrees of freedom = {dof}", flush=True)

    # Per-organism applied shares
    print("\nPer-organism applied share:", flush=True)
    org_totals = contingency.sum(axis=1)
    if LABEL_APPLIED in contingency.columns:
        org_applied = contingency[LABEL_APPLIED]
        org_share = (org_applied / org_totals).sort_values(ascending=False)
        print(f"  {'Organism':<30} {'Applied share':>14} {'N papers':>10}", flush=True)
        print("  " + "-" * 56, flush=True)
        for org, share in org_share.items():
            n = org_totals[org]
            print(f"  {org:<30} {share:>14.3f} {n:>10,}", flush=True)
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Paradigm shift analysis (fundamental vs applied) by organism"
    )
    ap.add_argument(
        "--db-path",
        default="data/processed/plant_science.duckdb",
        help="Path to the DuckDB database",
    )
    ap.add_argument(
        "--classifications-dir",
        default="data/processed/classifications",
        help="Directory containing paper_a_paradigm.csv and paper_a_organism.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: results/paper_a/{main,supplementary})",
    )
    args = ap.parse_args()

    # Allow out-dir override for both outputs
    global OUT_MAIN, OUT_SUPP
    if args.out_dir:
        OUT_MAIN = os.path.join(args.out_dir, "main")
        OUT_SUPP = os.path.join(args.out_dir, "supplementary")

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    # --- Load & merge classification CSVs ------------------------------------
    classifications = load_classifications(args.classifications_dir)

    # --- Fetch metadata from DuckDB ------------------------------------------
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=8")

    metadata = fetch_works_metadata(con, classifications["work_id"].tolist())
    con.close()

    # --- Build master dataframe ----------------------------------------------
    df = build_master_df(classifications, metadata)

    # --- Analyses ------------------------------------------------------------
    shift_df = paradigm_shift_by_organism(df)
    citation_df = paradigm_citation_impact(df)
    oa_df = paradigm_oa_interaction(df)
    chi2, p_value, dof, contingency = run_chi2_test(df)

    # --- Save outputs --------------------------------------------------------
    shift_path = os.path.join(OUT_MAIN, "paradigm_shift_by_organism.csv")
    citation_path = os.path.join(OUT_SUPP, "paradigm_citation_impact.csv")
    oa_path = os.path.join(OUT_SUPP, "paradigm_oa_interaction.csv")

    shift_df.to_csv(shift_path, index=False)
    citation_df.to_csv(citation_path, index=False)
    oa_df.to_csv(oa_path, index=False)

    print(f"\nSaved: {shift_path}")
    print(f"Saved: {citation_path}")
    print(f"Saved: {oa_path}")

    # --- Console highlights --------------------------------------------------
    print_key_findings(df, chi2, p_value, dof, contingency)

    elapsed = int(time.time() - t0)
    print(f"\nParadigm shift analysis complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
