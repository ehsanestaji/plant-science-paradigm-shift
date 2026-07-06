"""
G9 — Logistic growth model sensitivity to end-year cutoff.

Re-fits logistic vs. exponential models with corpus truncated at each year
2015–2024 (10 truncation points).  Also bootstraps the inflection year t₀
on the full 1990–2024 series.

Outputs:
  results/paper_a/supplementary/hardening/logistic_sensitivity.csv
  results/paper_a/supplementary/hardening/logistic_bootstrap_inflection.csv

Usage
-----
python3 -u -m src.analysis.paper_a.hardening.sensitivity_logistic \
    --db-path data/processed/plant_science.duckdb
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = PROJECT_ROOT / "results/paper_a/supplementary/hardening"

N_BOOT = 1_000
SEED = 42
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Model definitions ─────────────────────────────────────────────────────────

def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def exponential(t, a, b):
    return a * np.exp(b * t)


def aic(n, sse, k_params):
    """AIC for nonlinear regression: AIC = n*ln(SSE/n) + 2*k."""
    return n * np.log(sse / n + 1e-12) + 2 * k_params


def fit_logistic(t, y):
    K0 = float(y.max()) * 1.5
    t0_0 = float(t[np.argmax(np.gradient(y))])
    r0 = 0.1
    try:
        popt, _ = curve_fit(
            logistic, t, y,
            p0=[K0, r0, t0_0],
            bounds=([0, 0, t.min() - 5], [K0 * 10, 2.0, t.max() + 10]),
            maxfev=10_000,
        )
        y_pred = logistic(t, *popt)
        sse = float(np.sum((y - y_pred) ** 2))
        return popt, sse
    except Exception:
        return None, None


def fit_exponential(t, y):
    t_rel = t - t.min()
    try:
        popt, _ = curve_fit(
            exponential, t_rel, y,
            p0=[float(y[0]), 0.05],
            maxfev=10_000,
        )
        y_pred = exponential(t_rel, *popt)
        sse = float(np.sum((y - y_pred) ** 2))
        return popt, sse
    except Exception:
        return None, None


def load_annual_counts(db_path: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        """
        SELECT year, count(*) AS n_papers
        FROM works_clean
        WHERE year BETWEEN 1990 AND 2024
        GROUP BY year
        ORDER BY year
        """
    ).df()
    con.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Logistic sensitivity analysis")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    t0_wall = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annual publication counts from DuckDB …")
    counts = load_annual_counts(args.db_path)
    print(f"  {len(counts)} years, {counts['n_papers'].sum():,} total papers")

    t_full = counts["year"].values.astype(float)
    y_full = counts["n_papers"].values.astype(float)

    # ── Truncation sensitivity ────────────────────────────────────────────────
    print("\nFitting logistic and exponential models for each truncation year …")
    rows = []
    for trunc_year in range(2015, 2025):
        mask = counts["year"] <= trunc_year
        t_sub = t_full[mask]
        y_sub = y_full[mask]
        n = len(t_sub)

        log_popt, log_sse = fit_logistic(t_sub, y_sub)
        exp_popt, exp_sse = fit_exponential(t_sub, y_sub)

        if log_popt is not None and exp_popt is not None:
            aic_log = aic(n, log_sse, k_params=3)
            aic_exp = aic(n, exp_sse, k_params=2)
            delta = aic_exp - aic_log  # positive ⟹ logistic preferred
            if delta > 2:
                preferred = "logistic"
            elif delta < -2:
                preferred = "exponential"
            else:
                preferred = "ambiguous"
            K_fit, r_fit, t0_fit = log_popt
        else:
            aic_log = aic_exp = delta = np.nan
            K_fit = r_fit = t0_fit = np.nan
            preferred = "fit_failed"

        rows.append(
            {
                "truncation_year": trunc_year,
                "K_fit": round(K_fit, 1) if not np.isnan(K_fit) else np.nan,
                "t0_fit": round(t0_fit, 3) if not np.isnan(t0_fit) else np.nan,
                "r_fit": round(r_fit, 5) if not np.isnan(r_fit) else np.nan,
                "aic_logistic": round(aic_log, 2) if not np.isnan(aic_log) else np.nan,
                "aic_exponential": round(aic_exp, 2) if not np.isnan(aic_exp) else np.nan,
                "delta_aic": round(delta, 2) if not np.isnan(delta) else np.nan,
                "preferred_model": preferred,
            }
        )
        print(f"  {trunc_year}: {preferred} (ΔAIC={delta:.1f})" if not np.isnan(delta) else f"  {trunc_year}: fit failed")

    sens_df = pd.DataFrame(rows)
    n_logistic_pref = (sens_df["preferred_model"] == "logistic").sum()
    robustness_pct = n_logistic_pref / len(sens_df) * 100
    print(f"\nLogistic preferred in {n_logistic_pref}/{len(sens_df)} truncation years ({robustness_pct:.0f}%)")

    sens_path = OUT_DIR / "logistic_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"Wrote → {sens_path}")

    # ── Bootstrap inflection point ────────────────────────────────────────────
    print(f"\nBootstrapping inflection point t₀ ({args.n_boot} iterations) …")
    t0_boots = []
    for i in range(args.n_boot):
        idx = rng.integers(0, len(t_full), size=len(t_full))
        t_b = t_full[idx]
        y_b = y_full[idx]
        # Sort by year for sensible fitting
        sort_order = np.argsort(t_b)
        t_b = t_b[sort_order]
        y_b = y_b[sort_order]
        popt, sse = fit_logistic(t_b, y_b)
        if popt is not None:
            t0_boots.append(popt[2])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.n_boot}")

    t0_boots = np.array(t0_boots)
    t0_point = float(sens_df[sens_df["truncation_year"] == 2024]["t0_fit"].values[0])
    t0_ci_lo = float(np.percentile(t0_boots, 2.5))
    t0_ci_hi = float(np.percentile(t0_boots, 97.5))

    boot_df = pd.DataFrame({"iteration": range(1, len(t0_boots) + 1), "t0_boot": t0_boots})
    summary_row = pd.DataFrame(
        [{"iteration": "summary", "t0_boot": t0_point,
          "t0_ci_lower": t0_ci_lo, "t0_ci_upper": t0_ci_hi}]
    )
    boot_out = pd.concat([boot_df, summary_row], ignore_index=True)

    boot_path = OUT_DIR / "logistic_bootstrap_inflection.csv"
    boot_out.to_csv(boot_path, index=False)
    print(f"Wrote → {boot_path}")

    print("\n=== Robustness summary ===")
    print(f"Truncation sensitivity: logistic preferred in {robustness_pct:.0f}% of cutoff years.")
    print(f"Bootstrap inflection year: t₀ = {t0_point:.1f} [95% CI: {t0_ci_lo:.1f}–{t0_ci_hi:.1f}]")
    print(f"\nElapsed: {time.time() - t0_wall:.1f}s")


if __name__ == "__main__":
    main()
