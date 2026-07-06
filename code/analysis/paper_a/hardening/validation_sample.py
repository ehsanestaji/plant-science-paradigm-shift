"""
G2 — Stratified 500-paper classifier validation sample.

Generates a CSV template for manual domain-expert review of organism and
paradigm predictions.  A companion scorer (validation_scorer.py) will
compute accuracy / F1 / kappa once the reviewer fills in the ground-truth
columns.

Usage
-----
python3 -u -m src.analysis.paper_a.hardening.validation_sample \
    --db-path data/processed/plant_science.duckdb
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ORGANISM_CSV = PROJECT_ROOT / "data/processed/classifications/paper_a_organism.csv"
PARADIGM_CSV = PROJECT_ROOT / "data/processed/classifications/paper_a_paradigm.csv"
OUT_CSV = (
    PROJECT_ROOT
    / "results/paper_a/supplementary/hardening/classifier_validation_sample.csv"
)

TOTAL_SAMPLE = 500
MIN_PER_LABEL = 10
SEED = 42


def load_classifications():
    print("Loading organism classifications …")
    org = pd.read_csv(ORGANISM_CSV, dtype={"work_id": str})
    print(f"  Organism rows: {len(org):,}")
    print("Loading paradigm classifications …")
    par = pd.read_csv(PARADIGM_CSV, dtype={"work_id": str})
    print(f"  Paradigm rows: {len(par):,}")
    return org, par


def compute_stratum_sizes(org: pd.DataFrame) -> pd.Series:
    """Proportional allocation with floor of MIN_PER_LABEL, scaled to TOTAL_SAMPLE."""
    counts = org["predicted_label"].value_counts()
    proportional = (counts / counts.sum() * TOTAL_SAMPLE).clip(lower=MIN_PER_LABEL)
    # rescale to total
    proportional = (proportional / proportional.sum() * TOTAL_SAMPLE).round().astype(int)
    # Fix rounding drift
    diff = TOTAL_SAMPLE - proportional.sum()
    if diff != 0:
        idx = proportional.idxmax() if diff < 0 else proportional.idxmin()
        proportional[idx] += diff
    return proportional


def query_titles_abstracts(work_ids: list, db_path: str) -> pd.DataFrame:
    import duckdb
    ids_str = ", ".join(f"'{w}'" for w in work_ids)
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        f"""
        SELECT work_id,
               year,
               title,
               abstract
        FROM works_clean
        WHERE work_id IN ({ids_str})
        """
    ).df()
    con.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate classifier validation sample")
    parser.add_argument("--db-path", required=True, help="Path to plant_science.duckdb")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=TOTAL_SAMPLE)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    org, par = load_classifications()

    # Build merged frame (inner join — both classifiers must have the paper)
    merged = org.merge(par, on="work_id", suffixes=("_org", "_par"))
    print(f"\nMerged (both classifiers): {len(merged):,} rows")

    stratum_sizes = compute_stratum_sizes(merged.rename(columns={"predicted_label_org": "predicted_label"}))

    print("\nStratum allocation:")
    for label, n in stratum_sizes.items():
        print(f"  {label:<30} → {n}")

    # Stratified sample
    frames = []
    for label, n in stratum_sizes.items():
        pool = merged[merged["predicted_label_org"] == label]
        if len(pool) == 0:
            print(f"  WARNING: no papers for label '{label}', skipping")
            continue
        n_draw = min(n, len(pool))
        chosen_idx = rng.choice(len(pool), size=n_draw, replace=False)
        frames.append(pool.iloc[chosen_idx])

    sample = pd.concat(frames, ignore_index=True)

    # Oversample confidence=0.7 check
    n_low_conf = (sample["confidence_org"] < 0.9).sum()
    n_high_conf = (sample["confidence_org"] >= 0.9).sum()
    print(f"\nConfidence breakdown in sample:")
    print(f"  confidence < 0.9 (multi-match): {n_low_conf}")
    print(f"  confidence = 1.0 (unambiguous): {n_high_conf}")

    # Fetch titles + abstracts from DuckDB
    print("\nQuerying DuckDB for titles and abstracts …")
    meta = query_titles_abstracts(sample["work_id"].tolist(), args.db_path)
    sample = sample.merge(meta, on="work_id", how="left")

    # Year coverage check
    if "year" in sample.columns:
        year_counts = sample["year"].value_counts().sort_index()
        missing_years = [y for y in range(1990, 2025) if y not in year_counts.index]
        thin_years = [y for y in range(1990, 2025) if year_counts.get(y, 0) < 5]
        print(f"\nYear coverage (1990–2024): {len(year_counts)} distinct years")
        if thin_years:
            print(f"  Years with <5 papers: {thin_years[:20]}")

    # Build output CSV
    out = pd.DataFrame(
        {
            "work_id": sample["work_id"],
            "year": sample.get("year", pd.NA),
            "title": sample.get("title", "").str[:150].fillna(""),
            "abstract_snippet": sample.get("abstract", "").str[:300].fillna(""),
            "predicted_organism": sample["predicted_label_org"],
            "organism_confidence": sample["confidence_org"],
            "predicted_paradigm": sample["predicted_label_par"],
            "paradigm_confidence": sample["confidence_par"],
            "true_organism": "",
            "true_paradigm": "",
            "reviewer_notes": "",
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out):,} rows → {OUT_CSV}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
