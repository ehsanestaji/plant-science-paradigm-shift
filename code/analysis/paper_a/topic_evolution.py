"""
Topic evolution and diversity analysis for plant science macro-themes.

Tracks which macro-themes are growing, stable, or declining over time and
measures field fragmentation via Shannon entropy per year.

Inputs
------
results/paper_a/supplementary/paper_macro_themes.csv  — work_id, macro_theme_id, topic_id
results/paper_a/main/macro_themes.csv                 — macro_theme_id, name, n_docs, ...
data/processed/plant_science.duckdb                   — works_clean view (work_id, year)

Outputs
-------
results/paper_a/main/topic_growth_decline.csv
results/paper_a/supplementary/topic_diversity_by_year.csv
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

DECADES = {
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2024),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Topic evolution and diversity analysis for macro-themes."
    )
    p.add_argument(
        "--db-path",
        default="data/processed/plant_science.duckdb",
        help="Path to the DuckDB database",
    )
    p.add_argument(
        "--out-dir",
        default="results/paper_a",
        help="Root output directory (main/ and supplementary/ inside)",
    )
    return p.parse_args()


def shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy (bits) from an array of non-negative counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    root = Path(os.getcwd())
    out_dir = root / args.out_dir
    main_dir = out_dir / "main"
    supp_dir = out_dir / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    log("=== topic_evolution.py ===")
    log(f"db_path:  {args.db_path}")
    log(f"out_dir:  {out_dir}")
    log("")

    # ------------------------------------------------------------------
    # 1. Load paper_macro_themes.csv
    # ------------------------------------------------------------------
    paper_themes_path = supp_dir / "paper_macro_themes.csv"
    log(f"Loading {paper_themes_path} ...")
    paper_themes = pd.read_csv(
        paper_themes_path,
        dtype={"work_id": str, "macro_theme_id": int, "topic_id": int},
    )
    log(f"  {len(paper_themes):,} rows loaded.")

    # ------------------------------------------------------------------
    # 2. Load macro_themes.csv for theme names
    # ------------------------------------------------------------------
    macro_path = main_dir / "macro_themes.csv"
    log(f"Loading {macro_path} ...")
    macro_themes = pd.read_csv(
        macro_path,
        dtype={"macro_theme_id": int, "name": str},
        usecols=["macro_theme_id", "name"],
    )
    log(f"  {len(macro_themes):,} themes loaded.")
    theme_name_map: dict[int, str] = dict(
        zip(macro_themes["macro_theme_id"], macro_themes["name"])
    )

    # ------------------------------------------------------------------
    # 3. Load year data from DuckDB
    # ------------------------------------------------------------------
    log("Connecting to DuckDB and querying work years ...")
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")
    years_df = con.execute(
        "SELECT work_id, year FROM works_clean WHERE year BETWEEN 1990 AND 2024"
    ).fetchdf()
    con.close()
    log(f"  {len(years_df):,} works in 1990-2024.")

    # ------------------------------------------------------------------
    # 4. Join paper_macro_themes with year data
    # ------------------------------------------------------------------
    log("Joining paper themes with year data ...")
    df = paper_themes.merge(years_df, on="work_id", how="inner")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)]
    log(f"  {len(df):,} rows after join and year filter.")

    # Exclude outlier macro_theme_id == -1 from theme analyses (kept for
    # entropy only when needed — see below)
    df_valid = df[df["macro_theme_id"] != -1].copy()
    log(f"  {len(df_valid):,} rows excluding outlier macro-theme (-1).")

    # ------------------------------------------------------------------
    # 5. Count papers per macro_theme_id per year
    # ------------------------------------------------------------------
    log("Computing annual paper counts per macro-theme ...")
    annual = (
        df_valid.groupby(["macro_theme_id", "year"])
        .size()
        .reset_index(name="paper_count")
    )

    # ------------------------------------------------------------------
    # 6. Compute decade counts and growth rates
    # ------------------------------------------------------------------
    log("Computing decade counts and growth rates ...")

    all_themes = sorted(df_valid["macro_theme_id"].unique().tolist())
    decade_rows = []

    for theme_id in all_themes:
        theme_annual = annual[annual["macro_theme_id"] == theme_id]
        for decade_name, (d_start, d_end) in DECADES.items():
            decade_data = theme_annual[
                (theme_annual["year"] >= d_start) & (theme_annual["year"] <= d_end)
            ]
            decade_count = int(decade_data["paper_count"].sum())

            # Growth rate: (last_year_count - first_year_count) / first_year_count
            # Use first and last year with papers in the decade
            nonzero = decade_data[decade_data["paper_count"] > 0].sort_values("year")
            if len(nonzero) >= 2:
                start_count = float(nonzero.iloc[0]["paper_count"])
                end_count = float(nonzero.iloc[-1]["paper_count"])
                growth_rate = (end_count - start_count) / start_count
            else:
                growth_rate = float("nan")

            decade_rows.append(
                {
                    "macro_theme_id": theme_id,
                    "name": theme_name_map.get(theme_id, str(theme_id)),
                    "decade": decade_name,
                    "paper_count": decade_count,
                    "growth_rate": growth_rate,
                }
            )

    decade_df = pd.DataFrame(
        decade_rows,
        columns=["macro_theme_id", "name", "decade", "paper_count", "growth_rate"],
    )

    # ------------------------------------------------------------------
    # 7. Classify trend labels
    # ------------------------------------------------------------------
    log("Classifying trend labels ...")

    total_per_theme = (
        df_valid.groupby("macro_theme_id").size().rename("total_papers")
    )

    before_2005 = (
        df_valid[df_valid["year"] < 2005]
        .groupby("macro_theme_id")
        .size()
        .rename("before_2005")
    )

    after_2015 = (
        df_valid[df_valid["year"] > 2015]
        .groupby("macro_theme_id")
        .size()
        .rename("after_2015")
    )

    trend_df = (
        pd.DataFrame({"total_papers": total_per_theme})
        .join(before_2005, how="left")
        .join(after_2015, how="left")
        .fillna(0)
    )
    trend_df["frac_before_2005"] = trend_df["before_2005"] / trend_df["total_papers"]
    trend_df["frac_after_2015"] = trend_df["after_2015"] / trend_df["total_papers"]

    def classify_trend(row: pd.Series) -> str:
        if row["frac_after_2015"] > 0.50:
            return "emerging"
        if row["frac_before_2005"] > 0.50:
            return "declining"
        return "stable"

    trend_df["trend_label"] = trend_df.apply(classify_trend, axis=1)
    trend_map: dict[int, str] = trend_df["trend_label"].to_dict()

    decade_df["trend_label"] = decade_df["macro_theme_id"].map(trend_map)

    # ------------------------------------------------------------------
    # 8. Save topic_growth_decline.csv
    # ------------------------------------------------------------------
    out_growth = main_dir / "topic_growth_decline.csv"
    decade_df.to_csv(out_growth, index=False)
    log(f"Wrote {out_growth}  ({len(decade_df):,} rows)")

    # ------------------------------------------------------------------
    # 9. Shannon entropy per year (exclude macro_theme_id == -1)
    # ------------------------------------------------------------------
    log("Computing Shannon entropy per year ...")

    entropy_rows = []
    for year in range(1990, 2025):
        year_counts = annual[annual["year"] == year]["paper_count"].values
        h = shannon_entropy(year_counts)
        n_active = int((year_counts > 0).sum())
        entropy_rows.append(
            {"year": year, "shannon_entropy": h, "n_active_topics": n_active}
        )

    entropy_df = pd.DataFrame(entropy_rows, columns=["year", "shannon_entropy", "n_active_topics"])

    out_entropy = supp_dir / "topic_diversity_by_year.csv"
    entropy_df.to_csv(out_entropy, index=False)
    log(f"Wrote {out_entropy}  ({len(entropy_df):,} rows)")

    # ------------------------------------------------------------------
    # 10. Summary printout
    # ------------------------------------------------------------------
    print("\n" + "=" * 72, flush=True)
    print("TREND CLASSIFICATION SUMMARY", flush=True)
    print("=" * 72, flush=True)

    trend_counts = trend_df["trend_label"].value_counts()
    for label in ["emerging", "stable", "declining"]:
        n = trend_counts.get(label, 0)
        print(f"  {label:<12} {n:>4} macro-themes", flush=True)

    # Top 5 emerging (highest frac_after_2015)
    print("\nTop 5 EMERGING macro-themes (highest fraction of papers after 2015):", flush=True)
    emerging = (
        trend_df[trend_df["trend_label"] == "emerging"]
        .sort_values("frac_after_2015", ascending=False)
        .head(5)
    )
    for mid, row in emerging.iterrows():
        name = theme_name_map.get(mid, str(mid))
        print(
            f"  [{mid:>2}] {name:<45} {row['frac_after_2015']:.1%} after 2015",
            flush=True,
        )

    # Top 5 declining (highest frac_before_2005)
    print("\nTop 5 DECLINING macro-themes (highest fraction of papers before 2005):", flush=True)
    declining = (
        trend_df[trend_df["trend_label"] == "declining"]
        .sort_values("frac_before_2005", ascending=False)
        .head(5)
    )
    for mid, row in declining.iterrows():
        name = theme_name_map.get(mid, str(mid))
        print(
            f"  [{mid:>2}] {name:<45} {row['frac_before_2005']:.1%} before 2005",
            flush=True,
        )

    # Entropy trend (first vs last 5 years)
    early_entropy = entropy_df[entropy_df["year"] <= 1994]["shannon_entropy"].mean()
    late_entropy = entropy_df[entropy_df["year"] >= 2020]["shannon_entropy"].mean()
    print(
        f"\nEntropy trend:  1990-1994 mean={early_entropy:.3f} bits  |"
        f"  2020-2024 mean={late_entropy:.3f} bits"
        f"  ({'increasing' if late_entropy > early_entropy else 'decreasing'} diversity)",
        flush=True,
    )
    print("=" * 72, flush=True)

    log(f"Done ({int(time.time() - t0)}s)")


if __name__ == "__main__":
    main()
