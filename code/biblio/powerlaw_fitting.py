"""
Rigorous power-law fitting using the Clauset et al. (2009) methodology.

Fits power-law distributions to:
  - Citation counts (Price's law / Zipf): works_clean.cited_by_count
  - Author productivity (Lotka's law): papers per author via work_authors

For each distribution, the script:
  1. Estimates xmin and alpha via MLE (discrete=True, automatic xmin)
  2. Reports alpha, xmin, sigma, 95% CI, KS statistic
  3. Runs likelihood ratio tests vs lognormal, exponential,
     stretched_exponential, and truncated_power_law alternatives

Outputs:
  results/bibliometrics/powerlaw_fits.csv       — one row per distribution
  results/bibliometrics/powerlaw_comparison.csv — LR tests against alternatives

Usage:
    python -m src.biblio.powerlaw_fitting --db-path data/processed/plant_science.duckdb

Reference:
    Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009).
    Power-law distributions in empirical data. SIAM Review, 51(4), 661-703.
    https://doi.org/10.1137/070710111
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import powerlaw  # pip install powerlaw  (Alstott et al. 2014, wraps Clauset 2009)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/bibliometrics"

ALTERNATIVE_DISTRIBUTIONS = [
    "lognormal",
    "exponential",
    "stretched_exponential",
    "truncated_power_law",
]

# 95 % confidence interval half-width multiplier (1.96 * sigma)
Z95 = 1.96


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_citation_counts(con) -> np.ndarray:
    """Return array of per-paper citation counts (>= 1)."""
    print("Fetching citation counts from works_clean...", flush=True)
    df = con.execute("""
        SELECT cited_by_count
        FROM works_clean
        WHERE cited_by_count IS NOT NULL
          AND cited_by_count >= 1
    """).df()
    data = df["cited_by_count"].values.astype(int)
    print(f"  {len(data):,} papers with >= 1 citation", flush=True)
    return data


def fetch_author_productivity(con) -> np.ndarray:
    """Return array of papers-per-author counts (>= 1)."""
    print("Fetching author productivity (papers per author)...", flush=True)
    df = con.execute("""
        SELECT COUNT(*) AS n_papers
        FROM work_authors wa
        JOIN works_clean  wc ON wa.work_id = wc.work_id
        WHERE wa.author_id != 'A9999999999'
        GROUP BY wa.author_id
    """).df()
    data = df["n_papers"].values.astype(int)
    print(f"  {len(data):,} authors", flush=True)
    return data


# ── Power-law fitting ─────────────────────────────────────────────────────────

def fit_powerlaw(data: np.ndarray, label: str) -> dict:
    """
    Fit a power-law to *data* using MLE with automatic xmin selection.

    Returns a dict with fit statistics and LR comparison results.
    """
    print(f"\nFitting power-law: {label}...", flush=True)

    # --- Primary fit ---
    fit = powerlaw.Fit(data, discrete=True, verbose=False)
    alpha   = fit.power_law.alpha
    xmin    = fit.power_law.xmin
    sigma   = fit.power_law.sigma          # standard error of alpha
    ci_lo   = alpha - Z95 * sigma
    ci_hi   = alpha + Z95 * sigma

    # KS statistic (D attribute — KS() method has a bug in powerlaw 2.0.0)
    ks_stat = fit.power_law.D

    n_tail  = int((data >= xmin).sum())
    frac_in_tail = n_tail / len(data)

    print(f"  alpha={alpha:.4f}, sigma={sigma:.4f}, xmin={xmin:.0f}", flush=True)
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)
    print(f"  KS={ks_stat:.4f}, n_tail={n_tail:,} ({100*frac_in_tail:.1f}% of data)", flush=True)

    fit_row = {
        "distribution":    label,
        "n_total":         len(data),
        "xmin":            xmin,
        "alpha":           alpha,
        "sigma":           sigma,
        "ci_lo_95":        ci_lo,
        "ci_hi_95":        ci_hi,
        "ks_statistic":    ks_stat,
        "n_tail":          n_tail,
        "frac_in_tail":    frac_in_tail,
    }

    # --- Likelihood ratio tests vs alternative distributions ---
    comparison_rows = []
    for alt in ALTERNATIVE_DISTRIBUTIONS:
        try:
            R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            # R > 0 → power_law is preferred; R < 0 → alternative preferred
            comparison_rows.append({
                "distribution":     label,
                "alternative":      alt,
                "loglikelihood_ratio": R,
                "p_value":          p,
                "preferred":        "power_law" if R > 0 else alt,
            })
            direction = ">" if R > 0 else "<"
            print(f"  vs {alt:30s}: LR={R:+.3f}  p={p:.4f}  ({direction} power_law)", flush=True)
        except Exception as exc:
            print(f"  vs {alt}: comparison failed — {exc}", flush=True)
            comparison_rows.append({
                "distribution":     label,
                "alternative":      alt,
                "loglikelihood_ratio": float("nan"),
                "p_value":          float("nan"),
                "preferred":        "error",
            })

    return fit_row, comparison_rows


# ── Output helpers ────────────────────────────────────────────────────────────

def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Clauset et al. (2009) power-law fitting for citation and productivity data"
    )
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb",
                    help="Path to DuckDB database (default: data/processed/plant_science.duckdb)")
    ap.add_argument("--memory-limit", default="60GB",
                    help="DuckDB memory limit (default: 60GB)")
    ap.add_argument("--threads", type=int, default=8,
                    help="DuckDB thread count (default: 8)")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    # Open database (read-only — 25 GB shared resource)
    con = create_database(args.db_path, read_only=True)
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")

    # --- Fetch data ---
    citations_data    = fetch_citation_counts(con)
    productivity_data = fetch_author_productivity(con)
    con.close()

    # --- Fit power-laws ---
    all_fit_rows    = []
    all_comp_rows   = []

    for label, data in [("citations", citations_data),
                         ("author_productivity", productivity_data)]:
        fit_row, comp_rows = fit_powerlaw(data, label)
        all_fit_rows.append(fit_row)
        all_comp_rows.extend(comp_rows)

    # --- Save results ---
    fits_path = f"{OUT_DIR}/powerlaw_fits.csv"
    comp_path = f"{OUT_DIR}/powerlaw_comparison.csv"

    fits_df = pd.DataFrame(all_fit_rows)
    comp_df = pd.DataFrame(all_comp_rows)

    fits_df.to_csv(fits_path, index=False)
    comp_df.to_csv(comp_path, index=False)

    # --- Summary printout ---
    elapsed = int(time.time() - t0)
    print("\n" + "=" * 60, flush=True)
    print("POWER-LAW FITTING SUMMARY (Clauset et al. 2009)", flush=True)
    print("=" * 60, flush=True)
    for _, row in fits_df.iterrows():
        print(f"\n  {row['distribution'].upper()}", flush=True)
        print(f"    alpha      = {row['alpha']:.4f}  (sigma={row['sigma']:.4f})", flush=True)
        print(f"    95% CI     = [{row['ci_lo_95']:.4f}, {row['ci_hi_95']:.4f}]", flush=True)
        print(f"    xmin       = {row['xmin']:.0f}", flush=True)
        print(f"    KS stat    = {row['ks_statistic']:.4f}", flush=True)
        print(f"    n_tail     = {int(row['n_tail']):,}  ({100*row['frac_in_tail']:.1f}% of data)", flush=True)

    print(f"\nOutputs written to:", flush=True)
    print(f"  {fits_path}", flush=True)
    print(f"  {comp_path}", flush=True)
    print(f"\nPower-law fitting complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
