"""
Analysis B: Concept Marriage Detection.

For every pair of broad concepts (level ≤ 1, score ≥ 0.4) compute their
annual Jaccard similarity.  A concept "marriage" is the first year the
Jaccard crossed a threshold.  Cluster marriage years to find intellectual
epochs.

Output → results/novel/
  concept_jaccard_by_year.csv   (large: concept_a, concept_b, year, jaccard, …)
  concept_marriages.csv         (one row per pair that ever married)
  concept_epochs.csv            (epoch summary)

Usage:
    python -m src.novel.concept_marriage --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/novel"
MARRIAGE_THRESHOLD = 0.02
MIN_YEAR = 1960
MAX_LEVEL = 1
MIN_SCORE = 0.4


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


# ── SQL ──────────────────────────────────────────────────────────────────────

def compute_jaccard_by_year(con) -> pd.DataFrame:
    print("Computing yearly concept co-occurrence (self-join)…", flush=True)
    print("  This may take 15-30 min on the full dataset.", flush=True)

    df = con.execute(f"""
        WITH
        concept_year AS (
            SELECT c.concept_id,
                   c.concept_name,
                   c.level,
                   w.year,
                   c.work_id
            FROM concepts c
            JOIN works_clean w ON c.work_id = w.work_id
            WHERE c.level <= {MAX_LEVEL}
              AND c.score >= {MIN_SCORE}
              AND w.year >= {MIN_YEAR}
        ),
        -- count papers per concept per year (needed for union)
        concept_totals AS (
            SELECT concept_id, year, COUNT(DISTINCT work_id) AS n_total
            FROM concept_year
            GROUP BY concept_id, year
        ),
        -- co-occurrences: both concepts in same paper
        pairs AS (
            SELECT a.concept_id   AS concept_a,
                   a.concept_name AS name_a,
                   b.concept_id   AS concept_b,
                   b.concept_name AS name_b,
                   a.year,
                   COUNT(DISTINCT a.work_id) AS n_both
            FROM concept_year a
            JOIN concept_year b
                ON a.work_id = b.work_id
               AND a.concept_id < b.concept_id
            GROUP BY a.concept_id, a.concept_name, b.concept_id, b.concept_name, a.year
        )
        SELECT
            p.concept_a, p.name_a,
            p.concept_b, p.name_b,
            p.year,
            p.n_both,
            ta.n_total AS n_a,
            tb.n_total AS n_b,
            (ta.n_total + tb.n_total - p.n_both) AS n_union,
            p.n_both * 1.0 / NULLIF(ta.n_total + tb.n_total - p.n_both, 0) AS jaccard
        FROM pairs p
        JOIN concept_totals ta ON ta.concept_id = p.concept_a AND ta.year = p.year
        JOIN concept_totals tb ON tb.concept_id = p.concept_b AND tb.year = p.year
        ORDER BY p.concept_a, p.concept_b, p.year
    """).df()

    print(f"  {len(df):,} rows, "
          f"{df.groupby(['concept_a','concept_b']).ngroups:,} concept pairs", flush=True)
    return df


# ── Marriage detection ────────────────────────────────────────────────────────

def find_marriages(jaccard_df: pd.DataFrame) -> pd.DataFrame:
    print("Finding marriage years…", flush=True)
    records = []
    for (ca, na, cb, nb), grp in jaccard_df.groupby(
            ["concept_a", "name_a", "concept_b", "name_b"]):
        grp_s = grp.sort_values("year")
        above = grp_s[grp_s["jaccard"] >= MARRIAGE_THRESHOLD]
        if len(above) == 0:
            continue
        marriage_year = int(above["year"].min())
        peak_row = grp_s.loc[grp_s["jaccard"].idxmax()]
        peak_year = int(peak_row["year"])
        peak_j = float(peak_row["jaccard"])
        n_years_above = len(above)

        # Divorce: was there a sustained period AFTER marriage where it dropped back?
        post_marriage = grp_s[grp_s["year"] >= marriage_year]
        divorced = False
        if len(post_marriage) >= 10:
            last_5 = post_marriage.tail(5)["jaccard"].mean()
            if last_5 < MARRIAGE_THRESHOLD * 0.5:
                divorced = True

        records.append({
            "concept_a": ca, "name_a": na,
            "concept_b": cb, "name_b": nb,
            "marriage_year": marriage_year,
            "peak_year": peak_year,
            "peak_jaccard": round(peak_j, 4),
            "n_years_above_threshold": n_years_above,
            "divorced": divorced,
        })
    df = pd.DataFrame(records).sort_values("marriage_year")
    print(f"  {len(df):,} concept marriages found", flush=True)
    return df


def assign_epochs(marriages_df: pd.DataFrame, n_epochs: int = 6) -> pd.DataFrame:
    """Cluster marriage years into intellectual epochs."""
    if len(marriages_df) == 0:
        return marriages_df

    epoch_edges = [1960, 1975, 1985, 1995, 2005, 2015, 2030]
    epoch_labels = [
        "1960s–1974 (Classical Era)",
        "1975–1984 (Molecular Biology)",
        "1985–1994 (Genomics Dawn)",
        "1995–2004 (Bioinformatics)",
        "2005–2014 (Omics Revolution)",
        "2015–2024 (AI & Climate Era)",
    ]

    def epoch(year):
        for i in range(len(epoch_edges) - 1):
            if epoch_edges[i] <= year < epoch_edges[i + 1]:
                return epoch_labels[i]
        return epoch_labels[-1]

    marriages_df["epoch"] = marriages_df["marriage_year"].apply(epoch)
    return marriages_df


def build_epoch_summary(marriages_df: pd.DataFrame) -> pd.DataFrame:
    if len(marriages_df) == 0:
        return pd.DataFrame()
    rows = []
    for epoch, grp in marriages_df.groupby("epoch"):
        top5 = grp.nlargest(5, "peak_jaccard")[["name_a", "name_b", "peak_jaccard"]]
        examples = "; ".join(
            f"{r.name_a} ↔ {r.name_b} ({r.peak_jaccard:.3f})"
            for _, r in top5.iterrows()
        )
        rows.append({
            "epoch": epoch,
            "n_marriages": len(grp),
            "mean_peak_jaccard": round(grp["peak_jaccard"].mean(), 4),
            "top_marriages": examples,
        })
    return pd.DataFrame(rows).sort_values("epoch")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='80GB'")
    con.execute("SET threads=8")

    jaccard_df = compute_jaccard_by_year(con)
    con.close()

    jaccard_df.to_csv(f"{OUT_DIR}/concept_jaccard_by_year.csv", index=False)
    print(f"  Saved concept_jaccard_by_year.csv", flush=True)

    marriages_df = find_marriages(jaccard_df)
    marriages_df = assign_epochs(marriages_df)
    marriages_df.to_csv(f"{OUT_DIR}/concept_marriages.csv", index=False)

    epoch_df = build_epoch_summary(marriages_df)
    epoch_df.to_csv(f"{OUT_DIR}/concept_epochs.csv", index=False)

    print("\n=== Intellectual Epochs ===")
    for _, r in epoch_df.iterrows():
        print(f"  {r['epoch']}: {r['n_marriages']} marriages")
        print(f"    Top: {r['top_marriages'][:120]}")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis B complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
