"""
Analysis E: Concept Half-Life & Zombie Concepts.

For each of the top-200 concepts by usage volume, fit an exponential decay
model n(t) = n_peak * exp(-lambda * t) to post-peak annual paper counts.
Compute half-life = ln(2) / lambda, then categorise each concept as:
  Rising Star | Evergreen | Declining | Dead (fast decay) | Zombie (resurrected)

Output → results/novel/
  concept_usage_by_year.csv
  concept_lifecycle.csv
  zombie_concepts_detail.csv

Usage:
    python -m src.novel.concept_lifecycle --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/novel"
TOP_N_CONCEPTS = 200
MIN_YEAR = 1960


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


# ── SQL ──────────────────────────────────────────────────────────────────────

def fetch_concept_usage(con) -> pd.DataFrame:
    print("Fetching concept usage by year (top 200 concepts)…", flush=True)
    df = con.execute(f"""
        WITH top_concepts AS (
            SELECT concept_id
            FROM concepts
            GROUP BY concept_id
            ORDER BY COUNT(*) DESC
            LIMIT {TOP_N_CONCEPTS}
        )
        SELECT c.concept_id,
               c.concept_name,
               c.level,
               w.year,
               COUNT(*) AS n_papers
        FROM concepts c
        JOIN works_clean w ON c.work_id = w.work_id
        WHERE c.concept_id IN (SELECT concept_id FROM top_concepts)
          AND w.year >= {MIN_YEAR}
        GROUP BY c.concept_id, c.concept_name, c.level, w.year
        ORDER BY c.concept_id, w.year
    """).df()
    print(f"  {len(df):,} rows, {df['concept_id'].nunique()} concepts", flush=True)
    return df


# ── Lifecycle fitting ─────────────────────────────────────────────────────────

def _exp_decay(t, lam):
    return np.exp(-lam * t)


def fit_concept_lifecycle(concept_df: pd.DataFrame) -> dict:
    """Fit post-peak exponential decay for a single concept time series."""
    df = concept_df.sort_values("year").copy()
    peak_idx = df["n_papers"].idxmax()
    peak_year = int(df.loc[peak_idx, "year"])
    peak_count = int(df.loc[peak_idx, "n_papers"])

    post = df[df["year"] >= peak_year].copy()
    post["t"] = post["year"] - peak_year

    result = {
        "peak_year": peak_year,
        "peak_count": peak_count,
        "lambda": None,
        "half_life": None,
        "status": "still_rising",
        "is_zombie": False,
        "zombie_resurrection_year": None,
    }

    # Need at least 5 post-peak years to fit
    if len(post) < 5:
        return result

    # Check if still growing (no clear peak in last 5 years)
    recent = df[df["year"] >= df["year"].max() - 5]["n_papers"]
    if peak_year >= df["year"].max() - 3:
        result["status"] = "still_rising"
        return result

    y = (post["n_papers"].values / peak_count).clip(1e-6)
    t = post["t"].values

    try:
        popt, _ = curve_fit(
            _exp_decay, t, y, p0=[0.05],
            bounds=(0, 2), maxfev=5000
        )
        lam = float(popt[0])
        half_life = np.log(2) / lam if lam > 1e-6 else np.inf
        result["lambda"] = round(lam, 5)
        result["half_life"] = round(half_life, 1) if np.isfinite(half_life) else None

        if lam < 0.005:
            result["status"] = "evergreen"
        elif lam < 0.03:
            result["status"] = "declining"
        else:
            result["status"] = "dead"

    except Exception:
        result["status"] = "evergreen"
        return result

    # ── Zombie detection ──────────────────────────────────────────────────────
    # Zombie: peak before 2010, then a trough < 50% of peak,
    #         then a recovery > 80% of peak
    if peak_year <= 2010 and len(post) >= 10:
        vals = post["n_papers"].values
        years = post["year"].values
        trough_idx = int(np.argmin(vals[1:])) + 1   # skip year 0
        trough_val = vals[trough_idx]
        trough_year = int(years[trough_idx])

        if trough_val < 0.5 * peak_count:
            # Is there a recovery AFTER the trough?
            after_trough = vals[trough_idx + 1:]
            if len(after_trough) > 0 and after_trough.max() > 0.8 * peak_count:
                resur_rel = int(np.argmax(after_trough)) + trough_idx + 1
                result["is_zombie"] = True
                result["zombie_resurrection_year"] = int(years[resur_rel])
                result["status"] = "zombie"

    return result


def classify_all_concepts(usage_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = usage_df.groupby(["concept_id", "concept_name", "level"])
    for (cid, cname, level), grp in groups:
        r = fit_concept_lifecycle(grp)
        rows.append({
            "concept_id": cid,
            "concept_name": cname,
            "level": int(level),
            **r
        })
    df = pd.DataFrame(rows)

    # Final category label
    def label(row):
        if row["is_zombie"]:
            return "Zombie"
        s = row["status"]
        if s == "still_rising":
            return "Rising Star"
        if s == "evergreen":
            return "Evergreen"
        if s == "dead":
            hl = row["half_life"]
            return "Dead" if (hl is not None and hl < 10) else "Fast Declining"
        return "Declining"

    df["category"] = df.apply(label, axis=1)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=4")

    usage_df = fetch_concept_usage(con)
    con.close()

    # Save raw usage
    usage_df.to_csv(f"{OUT_DIR}/concept_usage_by_year.csv", index=False)
    print(f"  Saved concept_usage_by_year.csv ({len(usage_df):,} rows)", flush=True)

    # Fit lifecycle
    print("Fitting concept lifecycles…", flush=True)
    lifecycle_df = classify_all_concepts(usage_df)
    lifecycle_df.to_csv(f"{OUT_DIR}/concept_lifecycle.csv", index=False)

    # Summary
    counts = lifecycle_df["category"].value_counts()
    print("\n=== Concept Category Summary ===")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")

    # Zombie detail
    zombies = lifecycle_df[lifecycle_df["is_zombie"]].copy()
    if len(zombies):
        zombies.to_csv(f"{OUT_DIR}/zombie_concepts_detail.csv", index=False)
        print(f"\nZombie concepts ({len(zombies)}):")
        for _, r in zombies.iterrows():
            print(f"  {r['concept_name']}  peak={r['peak_year']}  "
                  f"resurrection={r['zombie_resurrection_year']}")
    else:
        print("\nNo zombie concepts detected (try relaxing thresholds).")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis E complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
