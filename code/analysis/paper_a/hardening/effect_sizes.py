"""
G6 — Effect sizes: Cohen's d for CD disruption, Cramér's V for paradigm × organism.

Outputs:
  results/paper_a/supplementary/hardening/effect_sizes.csv
  results/paper_a/supplementary/hardening/cramersv_paradigm_organism.csv

Usage
-----
python3 -u -m src.analysis.paper_a.hardening.effect_sizes \
    --db-path data/processed/plant_science.duckdb
"""
import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DISRUPTION_CSV = PROJECT_ROOT / "results/paper_a/supplementary/organism_disruption_index.csv"
PARADIGM_SHIFT = PROJECT_ROOT / "results/paper_a/main/paradigm_shift_by_organism.csv"
ORGANISM_CSV = PROJECT_ROOT / "data/processed/classifications/paper_a_organism.csv"
PARADIGM_CSV = PROJECT_ROOT / "data/processed/classifications/paper_a_paradigm.csv"
OUT_DIR = PROJECT_ROOT / "results/paper_a/supplementary/hardening"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (signed: a − b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_sd


def magnitude_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "small"
    if ad < 0.5:
        return "medium"
    return "large"


def magnitude_v(v: float) -> str:
    if v < 0.1:
        return "small"
    if v < 0.3:
        return "medium"
    return "large"


def main():
    parser = argparse.ArgumentParser(description="Compute effect sizes")
    parser.add_argument("--db-path", required=True, help="Path to plant_science.duckdb (unused, kept for consistency)")
    args = parser.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Analysis A: Cohen's d for CD disruption ───────────────────────────────
    print("Loading organism disruption index …")
    dis = pd.read_csv(DISRUPTION_CSV)
    print(f"  {len(dis):,} rows, columns: {dis.columns.tolist()}")

    # Build organism-level arrays of mean_cd (weighted by n_papers)
    organisms = sorted(dis["organism"].unique())
    org_cd: dict = {}
    org_n: dict = {}
    for org in organisms:
        sub = dis[dis["organism"] == org]
        # Expand: repeat mean_cd by n_papers for each decade
        cd_vals = np.repeat(sub["mean_cd"].values, sub["n_papers"].values.astype(int))
        org_cd[org] = cd_vals
        org_n[org] = int(sub["n_papers"].sum())

    print(f"\nComputing Cohen's d for {len(organisms)} organisms ({len(organisms)*(len(organisms)-1)//2} pairs) …")
    d_rows = []
    for org_a, org_b in combinations(organisms, 2):
        a_vals = org_cd[org_a]
        b_vals = org_cd[org_b]
        d = cohens_d(a_vals, b_vals)
        d_rows.append(
            {
                "organism_a": org_a,
                "organism_b": org_b,
                "cohens_d": round(d, 4) if not np.isnan(d) else np.nan,
                "abs_d": round(abs(d), 4) if not np.isnan(d) else np.nan,
                "magnitude": magnitude_d(d) if not np.isnan(d) else "NA",
                "n_a": org_n[org_a],
                "n_b": org_n[org_b],
            }
        )

    effect_df = pd.DataFrame(d_rows).sort_values("abs_d", ascending=False)
    effect_path = OUT_DIR / "effect_sizes.csv"
    effect_df.to_csv(effect_path, index=False)
    print(f"Wrote {len(effect_df)} pairs → {effect_path}")

    print("\nTop 10 effect sizes:")
    print(effect_df.head(10)[["organism_a", "organism_b", "cohens_d", "magnitude"]].to_string(index=False))

    # ── Analysis B: Cramér's V for paradigm × organism ────────────────────────
    print("\nLoading organism and paradigm classifications …")
    org_cls = pd.read_csv(ORGANISM_CSV, dtype={"work_id": str})
    par_cls = pd.read_csv(PARADIGM_CSV, dtype={"work_id": str})
    merged = org_cls.merge(par_cls, on="work_id", suffixes=("_org", "_par"))
    print(f"  Merged: {len(merged):,} rows")

    contingency = pd.crosstab(
        merged["predicted_label_org"],
        merged["predicted_label_par"],
    )
    chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency)
    N = int(contingency.values.sum())
    r, c = contingency.shape
    v = np.sqrt(chi2_val / (N * (min(r, c) - 1)))
    mag = magnitude_v(v)

    cramers_df = pd.DataFrame(
        [
            {
                "test": "paradigm_x_organism",
                "chi2": round(chi2_val, 1),
                "df": dof,
                "p_value": p_val,
                "n": N,
                "cramers_v": round(v, 4),
                "magnitude": mag,
            }
        ]
    )
    cramers_path = OUT_DIR / "cramersv_paradigm_organism.csv"
    cramers_df.to_csv(cramers_path, index=False)
    print(f"Wrote → {cramers_path}")

    print("\n=== Cramér's V result ===")
    print(f"  χ² = {chi2_val:.1f}, df = {dof}, p = {p_val:.2e}")
    print(f"  N  = {N:,}")
    print(f"  V  = {v:.4f} → {mag}")

    print("\n=== Manuscript insertion (effect sizes section) ===")
    large_pairs = effect_df[effect_df["magnitude"] == "large"]
    print(f"Large Cohen's d pairs: {len(large_pairs)}")
    if len(large_pairs):
        print(large_pairs[["organism_a", "organism_b", "cohens_d"]].to_string(index=False))
    print(f"\nCramér's V = {v:.3f} ({mag} effect) for paradigm × organism contingency.")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
