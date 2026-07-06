"""
Organism-stratified concept marriage analysis.

concept_marriages.csv has no work_ids.  This script attributes each marriage
to organisms by finding which organism most frequently uses concept_a among
papers in the DuckDB `concepts` table.  To keep the analysis tractable, only
the top 500 marriages by peak_jaccard are processed.

Outputs
-------
results/paper_a/supplementary/organism_concept_marriages.csv
    organism, n_marriages, marriages_per_1000_papers, decade

results/paper_a/supplementary/organism_top_marriages.csv
    concept_a, concept_b, organism, year
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

TOP_N_MARRIAGES = 500


def year_to_decade(year: pd.Series) -> pd.Series:
    decade_start = (year // 10) * 10
    return decade_start.astype(str) + "s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Organism-stratified concept marriages")
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--classifications-dir", default="data/processed/classifications")
    ap.add_argument("--out-dir", default="results/paper_a")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "supplementary"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load concept marriages — keep top 500 by peak_jaccard
    # ------------------------------------------------------------------
    marriages_path = "results/novel/concept_marriages.csv"
    print(f"Loading concept marriages from {marriages_path} ...", flush=True)
    marriages = pd.read_csv(
        marriages_path,
        usecols=["concept_a", "name_a", "concept_b", "name_b",
                 "marriage_year", "peak_year", "peak_jaccard", "epoch"],
    )
    print(f"  {len(marriages):,} marriages total", flush=True)

    top_marriages = marriages.nlargest(TOP_N_MARRIAGES, "peak_jaccard").reset_index(drop=True)
    print(f"  Keeping top {TOP_N_MARRIAGES} by peak_jaccard", flush=True)

    # ------------------------------------------------------------------
    # 2. Load organism labels
    # ------------------------------------------------------------------
    clf_path = os.path.join(args.classifications_dir, "paper_a_organism.csv")
    print(f"Loading organism labels from {clf_path} ...", flush=True)
    clf = pd.read_csv(clf_path, usecols=["work_id", "predicted_label"])
    clf = clf[clf["predicted_label"].isin(VALID_LABELS)]
    print(f"  {len(clf):,} classified papers", flush=True)

    # ------------------------------------------------------------------
    # 3. Load concept-paper associations from DuckDB
    #    Attribute each marriage to the organism that most frequently uses
    #    concept_a (simpler and faster than the full two-concept join).
    # ------------------------------------------------------------------
    print(f"\nOpening DuckDB: {args.db_path} ...", flush=True)
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    # Get the unique concept_a IDs we care about
    concept_a_ids = top_marriages["concept_a"].unique().tolist()
    ids_literal = ", ".join(f"'{c}'" for c in concept_a_ids)

    print(f"Querying concepts table for {len(concept_a_ids)} concept_a IDs ...", flush=True)
    sql = f"""
        SELECT work_id, concept_id
        FROM concepts
        WHERE concept_id IN ({ids_literal})
    """
    concept_works = con.execute(sql).fetchdf()
    con.close()
    print(f"  {len(concept_works):,} concept×paper rows fetched", flush=True)

    # ------------------------------------------------------------------
    # 4. Join concept_works with organism labels
    # ------------------------------------------------------------------
    cw_org = concept_works.merge(clf, on="work_id", how="inner")
    cw_org = cw_org.rename(columns={"predicted_label": "organism"})
    print(f"  {len(cw_org):,} rows after organism join", flush=True)

    # Most frequent organism per concept_a
    concept_organism = (
        cw_org.groupby(["concept_id", "organism"])
        .size()
        .reset_index(name="n_papers")
    )
    dominant_organism = (
        concept_organism.sort_values("n_papers", ascending=False)
        .groupby("concept_id")
        .first()
        .reset_index()[["concept_id", "organism"]]
        .rename(columns={"concept_id": "concept_a"})
    )

    # ------------------------------------------------------------------
    # 5. Attribute each marriage to its dominant organism
    # ------------------------------------------------------------------
    top_marriages = top_marriages.merge(dominant_organism, on="concept_a", how="left")
    top_marriages["organism"] = top_marriages["organism"].fillna("non_specific")
    top_marriages["decade"] = year_to_decade(top_marriages["marriage_year"].astype(int))

    # ------------------------------------------------------------------
    # 6. Count marriages per organism, normalise per 1000 papers
    # ------------------------------------------------------------------
    org_total_papers = clf["predicted_label"].value_counts().rename("total_papers")

    per_organism_decade = (
        top_marriages.groupby(["organism", "decade"])
        .size()
        .reset_index(name="n_marriages")
    )
    # Also compute overall per organism for the normalisation
    per_organism = (
        top_marriages.groupby("organism")
        .size()
        .reset_index(name="n_marriages_total")
    )
    per_organism = per_organism.join(org_total_papers, on="organism")
    per_organism["marriages_per_1000_papers"] = np.where(
        per_organism["total_papers"] > 0,
        per_organism["n_marriages_total"] / per_organism["total_papers"] * 1000.0,
        np.nan,
    )

    # Merge the per-1000 rate back into the decade-level frame
    summary_df = per_organism_decade.merge(
        per_organism[["organism", "marriages_per_1000_papers"]],
        on="organism",
        how="left",
    )

    # ------------------------------------------------------------------
    # 7. Top marriages detail
    # ------------------------------------------------------------------
    top_detail = top_marriages[
        ["concept_a", "concept_b", "organism", "marriage_year"]
    ].rename(columns={"marriage_year": "year"})

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n=== Concept marriages per organism ===", flush=True)
    for _, row in (
        per_organism.sort_values("n_marriages_total", ascending=False).iterrows()
    ):
        print(
            f"  {row['organism']:<28}  n={row['n_marriages_total']:>4}  "
            f"per_1000={row['marriages_per_1000_papers']:.3f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    summary_path = os.path.join(
        args.out_dir, "supplementary", "organism_concept_marriages.csv"
    )
    detail_path = os.path.join(
        args.out_dir, "supplementary", "organism_top_marriages.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}  ({len(summary_df):,} rows)", flush=True)

    top_detail.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path}  ({len(top_detail):,} rows)", flush=True)

    print(f"\nDone ({int(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
