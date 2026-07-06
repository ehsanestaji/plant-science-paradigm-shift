"""
Analysis H: Methodological Diffusion Half-Life.

How fast do new methods penetrate plant science? Fits logistic S-curves to
method adoption time series and computes diffusion half-life (years from 10%
to 90% of peak adoption). Tests whether newer methods diffuse faster.

Output → results/novel/
  method_adoption_timeseries.csv
  method_diffusion_metrics.csv

Usage:
    python -m src.novel.method_diffusion --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/novel"

METHODS = {
    "PCR": {
        "keywords": ["%polymerase chain reaction%", "% pcr %"],
        "origin_year": 1985,
    },
    "Flow cytometry": {
        "keywords": ["%flow cytometry%", "%flow cytometric%"],
        "origin_year": 1975,
    },
    "Confocal microscopy": {
        "keywords": ["%confocal%"],
        "origin_year": 1985,
    },
    "Microarray": {
        "keywords": ["%microarray%", "%gene chip%"],
        "origin_year": 1995,
    },
    "Proteomics": {
        "keywords": ["%proteom%"],
        "origin_year": 1997,
    },
    "Metabolomics": {
        "keywords": ["%metabolom%"],
        "origin_year": 2000,
    },
    "RNA-seq": {
        "keywords": ["%rna-seq%", "%rnaseq%", "%rna sequencing%"],
        "origin_year": 2008,
    },
    "Machine learning": {
        "keywords": ["%machine learning%", "%deep learning%", "%neural network%"],
        "origin_year": 2010,
    },
    "GBS/RAD-seq": {
        "keywords": ["%genotyping-by-sequencing%", "% gbs %", "%rad-seq%", "%radseq%"],
        "origin_year": 2011,
    },
    "CRISPR": {
        "keywords": ["%crispr%", "%cas9%", "%cas12%", "%cas13%"],
        "origin_year": 2012,
    },
    "Single-cell": {
        "keywords": ["%single-cell%", "%single cell sequenc%", "%scrna%"],
        "origin_year": 2015,
    },
    "Pangenomics": {
        "keywords": ["%pangenome%", "%pan-genome%"],
        "origin_year": 2015,
    },
    "Long-read sequencing": {
        "keywords": ["%nanopore%", "%pacbio%", "%long-read%", "%long read sequenc%"],
        "origin_year": 2014,
    },
}


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def _logistic(t, K, r, t_mid):
    """Logistic growth model."""
    return K / (1.0 + np.exp(-r * (t - t_mid)))


def fetch_method_timeseries(con) -> pd.DataFrame:
    """Fetch yearly paper counts per method."""
    print("Fetching method adoption time series…", flush=True)
    all_rows = []
    for method, info in METHODS.items():
        patterns = info["keywords"]
        title_where = " OR ".join(f"lower(title) LIKE '{p}'" for p in patterns)
        abstract_where = " OR ".join(f"lower(abstract) LIKE '{p}'" for p in patterns)
        df = con.execute(f"""
            SELECT year, COUNT(*) AS n_papers
            FROM works_clean
            WHERE year >= 1970
              AND (({title_where})
                   OR (abstract IS NOT NULL AND ({abstract_where})))
            GROUP BY year
            ORDER BY year
        """).df()
        df["method"] = method
        df["origin_year"] = info["origin_year"]
        all_rows.append(df)
        peak = df["n_papers"].max() if len(df) > 0 else 0
        print(f"  {method}: {df['n_papers'].sum():,} total, peak {peak:,}/yr", flush=True)
    return pd.concat(all_rows, ignore_index=True)


def fit_scurves(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Fit logistic S-curves and compute diffusion metrics."""
    print("Fitting logistic S-curves…", flush=True)
    records = []

    for method in ts_df["method"].unique():
        mdf = ts_df[ts_df["method"] == method].sort_values("year")
        origin = mdf["origin_year"].iloc[0]
        years = mdf["year"].values.astype(float)
        counts = mdf["n_papers"].values.astype(float)

        # Cumulative count for S-curve fitting
        cumulative = np.cumsum(counts)

        rec = {
            "method": method,
            "origin_year": int(origin),
            "total_papers": int(counts.sum()),
            "peak_annual": int(counts.max()),
            "peak_year": int(years[np.argmax(counts)]),
        }

        # Fit logistic to cumulative
        if len(years) >= 5 and cumulative[-1] > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    K_init = cumulative[-1] * 1.2
                    r_init = 0.3
                    t_mid_init = years[np.searchsorted(cumulative, cumulative[-1] / 2)]
                    popt, _ = curve_fit(
                        _logistic, years, cumulative,
                        p0=[K_init, r_init, t_mid_init],
                        maxfev=10000,
                        bounds=([0, 0.01, years[0]], [K_init * 5, 2.0, years[-1] + 20])
                    )
                    K, r, t_mid = popt

                    # Compute t_10 and t_90
                    t_10 = t_mid - np.log(9) / r
                    t_90 = t_mid + np.log(9) / r
                    halflife = t_90 - t_10

                    # R-squared
                    predicted = _logistic(years, *popt)
                    ss_res = np.sum((cumulative - predicted) ** 2)
                    ss_tot = np.sum((cumulative - cumulative.mean()) ** 2)
                    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    rec.update({
                        "K": round(K, 0),
                        "r": round(r, 4),
                        "t_mid": round(t_mid, 1),
                        "t_10": round(t_10, 1),
                        "t_90": round(t_90, 1),
                        "diffusion_halflife": round(halflife, 1),
                        "r_squared": round(r_sq, 4),
                    })
                    print(f"  {method}: halflife={halflife:.1f}yr, "
                          f"R²={r_sq:.3f}", flush=True)
            except Exception as e:
                print(f"  {method}: fit failed ({e})", flush=True)

        records.append(rec)

    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='20GB'")
    con.execute("SET threads=8")

    ts_df = fetch_method_timeseries(con)
    ts_df.to_csv(f"{OUT_DIR}/method_adoption_timeseries.csv", index=False)

    metrics = fit_scurves(ts_df)
    metrics.to_csv(f"{OUT_DIR}/method_diffusion_metrics.csv", index=False)

    con.close()

    # Spearman correlation: origin_year vs diffusion_halflife
    valid = metrics.dropna(subset=["diffusion_halflife"])
    if len(valid) >= 4:
        rho, pval = spearmanr(valid["origin_year"], valid["diffusion_halflife"])
        print(f"\n  Spearman(origin_year, halflife): ρ={rho:.3f}, p={pval:.4f}")
        if rho < 0:
            print("  → Newer methods diffuse FASTER (negative correlation)")
        else:
            print("  → No evidence that newer methods diffuse faster")

    print(f"\n=== Method Diffusion Summary ===")
    for _, r in metrics.sort_values("diffusion_halflife", na_position="last").iterrows():
        hl = f"{r['diffusion_halflife']:.1f}yr" if pd.notna(r.get('diffusion_halflife')) else "N/A"
        print(f"  {r['method']:<25} origin={r['origin_year']} halflife={hl}")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis H complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
