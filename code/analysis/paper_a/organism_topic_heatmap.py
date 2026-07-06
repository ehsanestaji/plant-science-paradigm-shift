"""
Organism × macro-theme cross-tabulation with enrichment scores.

Cross-tabulates organism type × macro-theme to identify which organisms
dominate which research topics.  Enrichment is log2(observed / expected)
where expected comes from the row-marginal × column-marginal / grand-total
null model.

Outputs
-------
results/paper_a/main/organism_topic_heatmap.csv
    organism, macro_theme_id, macro_theme_name, count, enrichment_log2
results/paper_a/supplementary/organism_topic_detail.csv
    organism, macro_theme_id, decade, count, enrichment
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

VALID_LABELS = [
    "arabidopsis", "rice", "wheat", "maize", "soybean", "tomato",
    "barley", "cotton", "potato", "tobacco",
    "other_crop", "other_model_organism", "non_specific",
]

DECADES = {
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2024),
}

ENRICHMENT_CLIP = 4.0  # log2 units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_enrichment(ct: pd.DataFrame) -> pd.DataFrame:
    """
    Given a crosstab (organisms × macro_theme_id), return a long-form
    DataFrame with columns [organism, macro_theme_id, count, enrichment_log2].

    Expected count = row_total * col_total / grand_total.
    enrichment_log2 = log2(observed / expected), clipped to [-4, 4].
    """
    grand = ct.values.sum()
    row_totals = ct.sum(axis=1).values[:, None]      # (n_org, 1)
    col_totals = ct.sum(axis=0).values[None, :]      # (1, n_theme)
    expected = (row_totals * col_totals) / grand

    # log2 enrichment; guard against zero observed / zero expected
    with np.errstate(divide="ignore", invalid="ignore"):
        enrich = np.where(
            (ct.values > 0) & (expected > 0),
            np.log2(ct.values / expected),
            np.where(ct.values == 0, -ENRICHMENT_CLIP, 0.0),
        )
    enrich = np.clip(enrich, -ENRICHMENT_CLIP, ENRICHMENT_CLIP)

    # Melt to long form
    ct_long = ct.copy()
    ct_long.columns.name = "macro_theme_id"
    ct_long.index.name = "organism"
    long = ct_long.stack().reset_index(name="count")

    enrich_df = pd.DataFrame(
        enrich, index=ct.index, columns=ct.columns
    )
    enrich_df.columns.name = "macro_theme_id"
    enrich_df.index.name = "organism"
    enrich_long = enrich_df.stack().reset_index(name="enrichment_log2")

    result = long.merge(enrich_long, on=["organism", "macro_theme_id"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism × macro-theme heatmap")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--themes-dir", default="results/paper_a")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "main"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load organism classifications
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")
    print(f"Loading organism classifications from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label", "confidence"])
    clf = clf[clf["predicted_label"].isin(VALID_LABELS)]
    print(f"  {len(clf):,} rows after label filtering", flush=True)

    # ------------------------------------------------------------------
    # 2. Load paper → macro-theme assignments
    # ------------------------------------------------------------------
    themes_path = os.path.join(args.themes_dir, "supplementary", "paper_macro_themes.csv")
    print(f"Loading paper macro-themes from {themes_path} ...", flush=True)
    paper_themes = pd.read_csv(themes_path, usecols=["work_id", "macro_theme_id"])
    print(f"  {len(paper_themes):,} rows loaded", flush=True)

    # ------------------------------------------------------------------
    # 3. Load macro-theme names
    # ------------------------------------------------------------------
    macro_path = os.path.join(args.themes_dir, "main", "macro_themes.csv")
    print(f"Loading macro-theme names from {macro_path} ...", flush=True)
    macro_meta = pd.read_csv(macro_path, usecols=["macro_theme_id", "name"])
    theme_name = dict(zip(macro_meta["macro_theme_id"], macro_meta["name"]))
    print(f"  {len(macro_meta):,} macro-themes found", flush=True)

    # ------------------------------------------------------------------
    # 4. Merge and exclude outliers (macro_theme_id == -1)
    # ------------------------------------------------------------------
    df = clf.merge(paper_themes, on="work_id", how="inner")
    before = len(df)
    df = df[df["macro_theme_id"] != -1]
    print(
        f"\nMerged: {before:,} rows → {len(df):,} after excluding outlier theme (-1)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 5. Cross-tabulation and enrichment (overall)
    # ------------------------------------------------------------------
    print("\nComputing overall cross-tabulation ...", flush=True)
    ct = pd.crosstab(df["predicted_label"], df["macro_theme_id"])
    enrich_df = compute_enrichment(ct)
    enrich_df = enrich_df.rename(columns={"organism": "organism"})

    # Attach macro-theme names
    enrich_df["macro_theme_name"] = enrich_df["macro_theme_id"].map(theme_name).fillna("")

    # Reorder columns
    heatmap_out = enrich_df[
        ["organism", "macro_theme_id", "macro_theme_name", "count", "enrichment_log2"]
    ].copy()
    heatmap_out = heatmap_out.sort_values(
        ["organism", "enrichment_log2"], ascending=[True, False]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Print top-3 enriched topics per organism
    # ------------------------------------------------------------------
    print("\n=== Top 3 enriched macro-themes per organism ===", flush=True)
    for org in sorted(heatmap_out["organism"].unique()):
        sub = (
            heatmap_out[heatmap_out["organism"] == org]
            .query("count > 0")
            .sort_values("enrichment_log2", ascending=False)
            .head(3)
        )
        print(f"\n  {org}:", flush=True)
        for _, row in sub.iterrows():
            print(
                f"    theme {int(row['macro_theme_id']):>3d}  "
                f"enrich={row['enrichment_log2']:+.2f}  "
                f"n={int(row['count']):,}  "
                f"\"{row['macro_theme_name'][:60]}\"",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 7. Save main heatmap output
    # ------------------------------------------------------------------
    main_path = os.path.join(args.out_dir, "main", "organism_topic_heatmap.csv")
    heatmap_out.to_csv(main_path, index=False)
    print(f"\nWrote {main_path}  ({len(heatmap_out):,} rows)", flush=True)

    # ------------------------------------------------------------------
    # 8. Temporal enrichment (decade × organism × macro-theme)
    # ------------------------------------------------------------------
    print("\nLoading year data from DuckDB for temporal analysis ...", flush=True)
    try:
        con = create_database(args.db_path, read_only=True)
        con.execute("SET memory_limit='60GB'")
        con.execute("SET threads=8")
        years_df = con.execute(
            "SELECT work_id, year FROM works_clean WHERE year BETWEEN 1990 AND 2024"
        ).fetchdf()
        con.close()
        print(f"  {len(years_df):,} works with year data", flush=True)
        has_years = True
    except Exception as exc:
        print(f"  WARNING: could not load year data ({exc}); skipping temporal output",
              flush=True)
        has_years = False

    if has_years:
        df_temp = df.merge(years_df, on="work_id", how="inner")
        df_temp = df_temp.dropna(subset=["year"])
        df_temp["year"] = df_temp["year"].astype(int)

        # Assign decade label
        def year_to_decade(y):
            for label, (lo, hi) in DECADES.items():
                if lo <= y <= hi:
                    return label
            return None

        df_temp["decade"] = df_temp["year"].map(year_to_decade)
        df_temp = df_temp.dropna(subset=["decade"])

        print("Computing decade-level enrichment ...", flush=True)
        detail_rows = []
        for decade_label in DECADES:
            sub = df_temp[df_temp["decade"] == decade_label]
            if sub.empty:
                continue
            ct_d = pd.crosstab(sub["predicted_label"], sub["macro_theme_id"])
            enr_d = compute_enrichment(ct_d)
            enr_d["decade"] = decade_label
            detail_rows.append(enr_d)

        if detail_rows:
            detail_df = pd.concat(detail_rows, ignore_index=True)
            detail_df = detail_df.rename(
                columns={"enrichment_log2": "enrichment"}
            )[["organism", "macro_theme_id", "decade", "count", "enrichment"]]
            detail_df = detail_df.sort_values(
                ["organism", "decade", "enrichment"], ascending=[True, True, False]
            ).reset_index(drop=True)

            detail_path = os.path.join(
                args.out_dir, "supplementary", "organism_topic_detail.csv"
            )
            detail_df.to_csv(detail_path, index=False)
            print(f"Wrote {detail_path}  ({len(detail_df):,} rows)", flush=True)
        else:
            print("  No decade data produced; skipping detail output.", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
