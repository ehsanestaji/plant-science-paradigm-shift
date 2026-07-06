"""
Organism citation patterns: per-year citation statistics and cross-citation matrix.

For each organism label, computes median/mean citations by year and the
cross-citation matrix (how much each organism cites every other organism).

Outputs
-------
results/paper_a/supplementary/organism_citations.csv
    organism, year, median_citations, mean_citations, n_papers
results/paper_a/supplementary/organism_cross_citation.csv
    citing_organism, cited_organism, n_citations, share
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism citation patterns analysis")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load organism classifications
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")
    print(f"Loading classifications from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label", "confidence"])
    clf = clf[clf["predicted_label"].isin(VALID_LABELS)]
    print(f"  {len(clf):,} rows loaded ({clf['predicted_label'].nunique()} labels)", flush=True)

    # ------------------------------------------------------------------
    # 2. Open DuckDB and register organism labels as a temp table
    # ------------------------------------------------------------------
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    print("\nRegistering organism labels as temp table ...", flush=True)
    con.execute("CREATE TEMP TABLE org_labels AS SELECT * FROM clf")
    con.register("clf", clf)
    # Re-create after register (register must precede CREATE TEMP TABLE)
    con.execute("DROP TABLE IF EXISTS org_labels")
    con.execute("CREATE TEMP TABLE org_labels AS SELECT work_id, predicted_label FROM clf")

    # ------------------------------------------------------------------
    # 3. Median/mean citations per organism per year
    # ------------------------------------------------------------------
    print("Querying citation statistics per organism per year ...", flush=True)
    citations_df = con.execute("""
        SELECT
            o.predicted_label  AS organism,
            w.year,
            MEDIAN(w.cited_by_count)  AS median_citations,
            AVG(w.cited_by_count)     AS mean_citations,
            COUNT(*)                  AS n_papers
        FROM org_labels o
        JOIN works_clean w ON o.work_id = w.work_id
        WHERE w.year BETWEEN 1990 AND 2024
        GROUP BY o.predicted_label, w.year
        ORDER BY o.predicted_label, w.year
    """).fetchdf()
    print(f"  {len(citations_df):,} organism-year rows", flush=True)

    # ------------------------------------------------------------------
    # 4. Cross-citation matrix
    # ------------------------------------------------------------------
    print("\nQuerying cross-citation matrix ...", flush=True)
    cross_df = con.execute("""
        SELECT
            o1.predicted_label AS citing_organism,
            o2.predicted_label AS cited_organism,
            COUNT(*)           AS n_citations
        FROM citations c
        JOIN org_labels o1 ON c.citing_work_id = o1.work_id
        JOIN org_labels o2 ON c.cited_work_id  = o2.work_id
        GROUP BY o1.predicted_label, o2.predicted_label
        ORDER BY o1.predicted_label, n_citations DESC
    """).fetchdf()
    print(f"  {len(cross_df):,} organism-pair rows", flush=True)

    con.close()

    # ------------------------------------------------------------------
    # 5. Compute share: n_citations / total citations from that organism
    # ------------------------------------------------------------------
    total_citing = (
        cross_df.groupby("citing_organism")["n_citations"]
        .sum()
        .rename("total_from_citing")
    )
    cross_df = cross_df.join(total_citing, on="citing_organism")
    cross_df["share"] = cross_df["n_citations"] / cross_df["total_from_citing"]
    cross_df = cross_df.drop(columns=["total_from_citing"])

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    cit_path = os.path.join(args.out_dir, "supplementary", "organism_citations.csv")
    cross_path = os.path.join(args.out_dir, "supplementary", "organism_cross_citation.csv")

    citations_df.to_csv(cit_path, index=False)
    print(f"\nWrote {cit_path}  ({len(citations_df):,} rows)", flush=True)

    cross_df.to_csv(cross_path, index=False)
    print(f"Wrote {cross_path}  ({len(cross_df):,} rows)", flush=True)

    # ------------------------------------------------------------------
    # 7. Summary printouts
    # ------------------------------------------------------------------
    print("\n=== Top 10 cross-citation pairs ===", flush=True)
    top_pairs = cross_df.sort_values("n_citations", ascending=False).head(10)
    print(top_pairs[["citing_organism", "cited_organism", "n_citations", "share"]].to_string(index=False), flush=True)

    print("\n=== Self-citation rates per organism ===", flush=True)
    self_cit = cross_df[cross_df["citing_organism"] == cross_df["cited_organism"]].copy()
    self_cit = self_cit.sort_values("share", ascending=False)
    for _, row in self_cit.iterrows():
        print(f"  {row['citing_organism']:<28} {row['share']*100:6.1f}%  ({int(row['n_citations']):,} citations)", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
