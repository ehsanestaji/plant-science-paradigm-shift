"""
G7 — Post-hoc power analysis for the Kruskal-Wallis CD disruption test.

Observed: H = 12.9, p = 0.17, k = 10 groups.
Computes η², achieved power at current N and at 2×/5×/10× N, and
the minimum detectable effect at 80% power.

Output:
  results/paper_a/supplementary/hardening/cd_power_analysis.csv

Usage
-----
python3 -u -m src.analysis.paper_a.hardening.power_analysis \
    --db-path data/processed/plant_science.duckdb
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DISRUPTION_CSV = PROJECT_ROOT / "results/paper_a/supplementary/organism_disruption_index.csv"
OUT_DIR = PROJECT_ROOT / "results/paper_a/supplementary/hardening"

# Observed test statistics
H_OBS = 12.9
K_GROUPS = 10
ALPHA = 0.05


def power_kw(n_total: int, eta2: float, k: int, alpha: float = 0.05) -> float:
    """
    Approximate power for Kruskal-Wallis using noncentral chi-square.
    ncp ≈ n_total * eta2  (from Cohen 1988 for H test)
    The critical value is from the central chi-square with df = k-1.
    """
    df = k - 1
    crit = stats.chi2.ppf(1 - alpha, df=df)
    ncp = n_total * eta2 * (k - 1) / (1 - eta2 + 1e-12)
    power = 1 - stats.ncx2.cdf(crit, df=df, nc=ncp)
    return float(power)


def min_detectable_eta2(n_total: int, k: int, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """Binary search for the smallest η² that gives target_power at given N."""
    def objective(eta2):
        return power_kw(n_total, eta2, k, alpha) - target_power

    try:
        return brentq(objective, 1e-6, 1 - 1e-6)
    except ValueError:
        return np.nan


def main():
    parser = argparse.ArgumentParser(description="Post-hoc power analysis for KW test")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB (unused, consistency)")
    args = parser.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading organism disruption index …")
    dis = pd.read_csv(DISRUPTION_CSV)

    # Aggregate total N per organism across all decades
    org_n = dis.groupby("organism")["n_papers"].sum()
    N_obs = int(org_n.sum())
    k = len(org_n)
    print(f"  Organisms: {k}, Total N (organism-decades pooled): {N_obs:,}")

    # Observed η²
    # For Kruskal-Wallis: η² ≈ (H - k + 1) / (N - k)
    eta2_obs = max(0.0, (H_OBS - K_GROUPS + 1) / (N_obs - K_GROUPS))
    print(f"  H = {H_OBS}, k = {K_GROUPS}, N = {N_obs}")
    print(f"  η² (observed) = {eta2_obs:.6f}")

    multipliers = [1.0, 2.0, 5.0, 10.0]
    rows = []
    for mult in multipliers:
        n_eff = int(N_obs * mult)
        pwr = power_kw(n_eff, eta2_obs, K_GROUPS, ALPHA)
        mde = min_detectable_eta2(n_eff, K_GROUPS, ALPHA, 0.80)
        rows.append(
            {
                "n_multiplier": mult,
                "total_n": n_eff,
                "effect_size_eta2": round(eta2_obs, 6),
                "achieved_power": round(pwr, 4),
                "min_detectable_eta2": round(mde, 6) if not np.isnan(mde) else np.nan,
            }
        )

    result_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "cd_power_analysis.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nWrote → {out_path}")
    print(result_df.to_string(index=False))

    # Find the multiplier needed for 80% power
    pwr_at_1x = rows[0]["achieved_power"]
    pwr_80_mult = None
    for r in rows:
        if r["achieved_power"] >= 0.80:
            pwr_80_mult = r["n_multiplier"]
            break

    print("\n=== One-paragraph power analysis summary (for manuscript) ===")
    print(
        f"Post-hoc power analysis for the Kruskal-Wallis test on CD disruption "
        f"(H\\,=\\,{H_OBS}, k\\,=\\,{K_GROUPS} groups) was conducted using the "
        f"noncentral chi-square approximation. "
        f"The observed effect size is η²\\,=\\,{eta2_obs:.4f}. "
        f"At the observed sample size (N\\,=\\,{N_obs:,}), "
        f"the test achieves {pwr_at_1x*100:.1f}\\% power to detect this effect at α\\,=\\,0.05. "
    )
    if pwr_80_mult:
        n_80 = int(N_obs * pwr_80_mult)
        print(
            f"A sample {pwr_80_mult:.0f}× larger (N\\,≈\\,{n_80:,}) would be required "
            f"to reach 80\\% power. "
        )
    else:
        # Find continuous estimate
        try:
            def obj_mult(m):
                return power_kw(int(N_obs * m), eta2_obs, K_GROUPS, ALPHA) - 0.80
            m80 = brentq(obj_mult, 1.0, 100.0)
            n80 = int(N_obs * m80)
            print(
                f"A sample {m80:.1f}× larger (N\\,≈\\,{n80:,}) would be required "
                f"to reach 80\\% power. "
            )
        except ValueError:
            print("Power does not reach 80% within 100× the current N.")
    print(
        "These results support treating organism-level CD differences as "
        "suggestive rather than definitive pending a larger study."
    )

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
