"""
Summary statistics table for Paper A.

Reads all result CSVs from the Paper A analysis pipeline and extracts key
numbers for the paper text, abstract, and results section.

Outputs
-------
results/paper_a/supplementary/summary_statistics.csv
    columns: metric, value, organism, note
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOCAL_ORGANISMS = [
    "arabidopsis", "rice", "wheat", "maize", "soybean", "tomato",
    "barley", "cotton", "potato", "tobacco",
    "other_crop", "other_model_organism", "non_specific",
]

LATEST_YEAR = 2024
LATEST_5_START = 2020


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(metric: str, value, organism: str = "", note: str = "") -> dict:
    """Return a single summary row dict."""
    if isinstance(value, float):
        value = round(value, 6)
    return {"metric": metric, "value": str(value), "organism": organism, "note": note}


def _load(path: Path) -> pd.DataFrame | None:
    """Load CSV, return None with a warning if absent."""
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Per-organism stats
# ---------------------------------------------------------------------------

def extract_organism_stats(results_dir: Path) -> list[dict]:
    rows: list[dict] = []

    ts   = _load(results_dir / "main"  / "organism_timeseries.csv")
    gr   = _load(results_dir / "main"  / "organism_growth_rates.csv")
    par  = _load(results_dir / "main"  / "paradigm_shift_by_organism.csv")
    dis  = _load(results_dir / "supplementary" / "organism_disruption_index.csv")
    coll = _load(results_dir / "supplementary" / "organism_collaboration.csv")

    for org in FOCAL_ORGANISMS:
        # --- Total papers (all time) ---
        if ts is not None:
            sub = ts[ts.organism == org]
            total = int(sub.paper_count.sum())
            rows.append(_row("total_papers_all_time", total, org))

            # Total papers 2020-2024
            recent = int(sub[sub.year >= 2020].paper_count.sum())
            rows.append(_row("total_papers_2020_2024", recent, org, "years 2020-2024"))

            # Share in latest year
            latest_row = sub[sub.year == LATEST_YEAR]
            if not latest_row.empty:
                share = round(float(latest_row.share_pct.iloc[0]), 4)
                rows.append(_row("share_pct_latest_year", share, org,
                                 f"share_pct in {LATEST_YEAR}"))

        # --- CAGR by decade ---
        if gr is not None:
            sub_gr = gr[gr.organism == org]
            for decade in ("2010s", "2020s"):
                d_row = sub_gr[sub_gr.decade == decade]
                if not d_row.empty:
                    cagr = round(float(d_row.cagr.iloc[0]), 6)
                    rows.append(_row(f"cagr_{decade}", cagr, org))

        # --- Applied share latest decade ---
        if par is not None:
            sub_par = par[(par.organism == org) & (par.year >= 2020)]
            if not sub_par.empty:
                total_f = sub_par.fundamental_count.sum()
                total_a = sub_par.applied_count.sum()
                if total_f + total_a > 0:
                    app_share = round(total_a / (total_f + total_a), 6)
                    rows.append(_row("applied_share_2020s", app_share, org,
                                     "applied / (fundamental + applied), 2020-2024"))

        # --- Mean disruption index (all decades) ---
        if dis is not None:
            sub_dis = dis[dis.organism == org]
            if not sub_dis.empty:
                # weighted mean across decades
                total_n = sub_dis.n_papers.sum()
                if total_n > 0:
                    wmean = float((sub_dis.mean_cd * sub_dis.n_papers).sum() / total_n)
                    rows.append(_row("mean_disruption_index", round(wmean, 6), org,
                                     "weighted mean CD index across decades"))

        # --- Team size and intl collab (latest 5 years) ---
        if coll is not None:
            sub_coll = coll[(coll.organism == org) & (coll.year >= LATEST_5_START)]
            if not sub_coll.empty:
                mean_ts = round(float(sub_coll.mean_team_size.mean()), 4)
                mean_ic = round(float(sub_coll.intl_collab_rate.mean()), 4)
                rows.append(_row("mean_team_size_latest5", mean_ts, org,
                                 f"mean of mean_team_size, {LATEST_5_START}-{LATEST_YEAR}"))
                rows.append(_row("intl_collab_rate_latest5", mean_ic, org,
                                 f"mean of intl_collab_rate, {LATEST_5_START}-{LATEST_YEAR}"))

    return rows


# ---------------------------------------------------------------------------
# Field-level stats
# ---------------------------------------------------------------------------

def extract_field_stats(results_dir: Path) -> list[dict]:
    rows: list[dict] = []

    ts   = _load(results_dir / "main" / "organism_timeseries.csv")
    mt   = _load(results_dir / "main" / "macro_themes.csv")
    tgd  = _load(results_dir / "main" / "topic_growth_decline.csv")
    div  = _load(results_dir / "supplementary" / "topic_diversity_by_year.csv")
    drift = _load(results_dir / "main" / "semantic_drift_centroids.csv")

    # Total papers analysed
    if ts is not None:
        total_field = int(ts.paper_count.sum())
        rows.append(_row("total_papers_field", total_field, "",
                         "sum across all organisms and years"))

    # Number of macro-themes (excluding noise cluster -1)
    if mt is not None:
        n_themes = int((mt.macro_theme_id >= 0).sum())
        rows.append(_row("n_macro_themes", n_themes, "",
                         "macro themes, excluding outlier cluster -1"))

    # Emerging / stable / declining counts (latest decade = 2020s)
    if tgd is not None:
        latest_dec = tgd[tgd.decade == "2020s"] if "decade" in tgd.columns else tgd
        if not latest_dec.empty:
            vc = latest_dec.trend_label.value_counts()
            for label in ("emerging", "stable", "declining"):
                rows.append(_row(f"n_themes_{label}_2020s", int(vc.get(label, 0)), "",
                                 "count of macro-themes with that trend in 2020s"))

    # Shannon entropy: first vs last decade
    if div is not None:
        first_dec = div[div.year <= 1999].shannon_entropy.mean()
        last_dec  = div[div.year >= 2020].shannon_entropy.mean()
        rows.append(_row("shannon_entropy_1990s_mean", round(float(first_dec), 6), "",
                         "mean Shannon entropy 1990-1999"))
        rows.append(_row("shannon_entropy_2020s_mean", round(float(last_dec), 6), "",
                         "mean Shannon entropy 2020-2024"))
        rows.append(_row("shannon_entropy_delta", round(float(last_dec - first_dec), 6), "",
                         "2020s mean minus 1990s mean"))

    # Semantic convergence / divergence (non-arabidopsis organisms, latest window)
    if drift is not None:
        non_ara = drift[
            (drift.organism != "arabidopsis") &
            (drift.distance_to_arabidopsis.notna())
        ]
        if not non_ara.empty:
            latest_w = non_ara.window_start.max()
            latest_drift = non_ara[non_ara.window_start == latest_w]
            if not latest_drift.empty:
                most_conv = latest_drift.loc[latest_drift.distance_to_arabidopsis.idxmin()]
                most_div  = latest_drift.loc[latest_drift.distance_to_arabidopsis.idxmax()]
                rows.append(_row("most_convergent_organism",
                                 most_conv.organism, "",
                                 f"smallest distance_to_arabidopsis in window {latest_w}-{int(most_conv.window_end)}"))
                rows.append(_row("most_convergent_distance",
                                 round(float(most_conv.distance_to_arabidopsis), 6), "",
                                 f"distance for {most_conv.organism}"))
                rows.append(_row("most_divergent_organism",
                                 most_div.organism, "",
                                 f"largest distance_to_arabidopsis in window {latest_w}-{int(most_div.window_end)}"))
                rows.append(_row("most_divergent_distance",
                                 round(float(most_div.distance_to_arabidopsis), 6), "",
                                 f"distance for {most_div.organism}"))

    return rows


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def extract_stat_tests(results_dir: Path) -> list[dict]:
    rows: list[dict] = []

    par = _load(results_dir / "main" / "paradigm_shift_by_organism.csv")
    dis = _load(results_dir / "supplementary" / "organism_disruption_index.csv")

    # Chi-square: paradigm (fundamental vs applied) × organism
    # Aggregate over all years; exclude non_specific and pseudo-categories for clarity
    if par is not None:
        excl = {"non_specific", "other_crop", "other_model_organism"}
        sub = par[~par.organism.isin(excl)]
        contingency = (
            sub.groupby("organism")[["fundamental_count", "applied_count"]]
            .sum()
        )
        if contingency.shape[0] >= 2 and contingency.values.min() >= 0:
            chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency.values)
            rows.append(_row("chi2_paradigm_x_organism", round(float(chi2), 4), "",
                             f"chi-square, paradigm × organism contingency, dof={dof}"))
            rows.append(_row("chi2_paradigm_x_organism_p", round(float(p_chi2), 8), "",
                             "p-value"))
            rows.append(_row("chi2_paradigm_x_organism_dof", int(dof), "",
                             "degrees of freedom"))

    # Kruskal-Wallis: disruption index across organisms
    if dis is not None:
        excl = {"non_specific", "other_crop", "other_model_organism"}
        sub = dis[~dis.organism.isin(excl)]
        groups = [
            grp.mean_cd.dropna().values
            for _, grp in sub.groupby("organism")
            if grp.mean_cd.notna().sum() >= 2
        ]
        if len(groups) >= 2:
            h_stat, p_kw = stats.kruskal(*groups)
            rows.append(_row("kruskal_wallis_disruption_H", round(float(h_stat), 4), "",
                             "Kruskal-Wallis H, mean_cd across organisms (decade-level)"))
            rows.append(_row("kruskal_wallis_disruption_p", round(float(p_kw), 8), "",
                             "p-value"))
            rows.append(_row("kruskal_wallis_disruption_n_groups", len(groups), "",
                             "number of organism groups tested"))

    return rows


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    width = 80
    print("\n" + "=" * width)
    print("  PAPER A — SUMMARY STATISTICS")
    print("=" * width)

    sections = {
        "FIELD-LEVEL": [
            "total_papers_field", "n_macro_themes",
            "n_themes_emerging_2020s", "n_themes_stable_2020s", "n_themes_declining_2020s",
            "shannon_entropy_1990s_mean", "shannon_entropy_2020s_mean", "shannon_entropy_delta",
            "most_convergent_organism", "most_convergent_distance",
            "most_divergent_organism",  "most_divergent_distance",
        ],
        "STATISTICAL TESTS": [
            "chi2_paradigm_x_organism", "chi2_paradigm_x_organism_p",
            "chi2_paradigm_x_organism_dof",
            "kruskal_wallis_disruption_H", "kruskal_wallis_disruption_p",
            "kruskal_wallis_disruption_n_groups",
        ],
    }

    for section, metrics in sections.items():
        print(f"\n  {section}")
        print("  " + "-" * (width - 2))
        for m in metrics:
            sub = df[df.metric == m]
            if sub.empty:
                continue
            val = sub.iloc[0]["value"]
            note = sub.iloc[0]["note"]
            note_str = f"  ({note})" if note else ""
            print(f"  {m:<45s} {val}{note_str}")

    print(f"\n  PER-ORGANISM STATS")
    print("  " + "-" * (width - 2))

    per_org_metrics = [
        "total_papers_all_time", "total_papers_2020_2024", "share_pct_latest_year",
        "cagr_2010s", "cagr_2020s", "applied_share_2020s",
        "mean_disruption_index", "mean_team_size_latest5", "intl_collab_rate_latest5",
    ]
    header = f"  {'organism':<24s}" + "".join(f"{m[:12]:>14s}" for m in per_org_metrics)
    print(header)
    print("  " + "-" * max(len(header) - 2, width - 2))

    for org in FOCAL_ORGANISMS:
        org_rows = df[df.organism == org]
        vals = []
        for m in per_org_metrics:
            sub = org_rows[org_rows.metric == m]
            vals.append(sub.iloc[0]["value"] if not sub.empty else "—")
        print(f"  {org:<24s}" + "".join(f"{v:>14s}" for v in vals))

    print("\n" + "=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract summary statistics from Paper A result CSVs."
    )
    p.add_argument(
        "--results-dir",
        default="results/paper_a",
        help="Path to paper_a results root (default: results/paper_a)",
    )
    return p.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()
    results_dir = Path(args.results_dir)

    print(f"[summary_tables] results_dir = {results_dir.resolve()}")

    rows: list[dict] = []

    print("[1/3] Per-organism statistics …")
    rows.extend(extract_organism_stats(results_dir))

    print("[2/3] Field-level statistics …")
    rows.extend(extract_field_stats(results_dir))

    print("[3/3] Statistical tests …")
    rows.extend(extract_stat_tests(results_dir))

    df = pd.DataFrame(rows, columns=["metric", "value", "organism", "note"])

    out_dir = results_dir / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_statistics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df):,} rows → {out_path}")

    print_summary(df)

    elapsed = time.time() - t0
    print(f"[summary_tables] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
