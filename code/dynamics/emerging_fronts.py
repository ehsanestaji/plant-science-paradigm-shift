"""
Theme 6 (partial) + Theme 7: Keyword & MeSH Dynamics.

CPU-only analyses (no embeddings needed):
  - Q35: Technology adoption S-curves (CRISPR, single-cell, AI, etc.)
  - Q36: MeSH term evolution — trending terms
  - Q38: Model organism trends
  - Q40: Knowledge flow via concept citations
  - Q46: Interdisciplinarity index
  - Q47: Climate change in plant science

Usage:
    python -m src.dynamics.emerging_fronts --db-path data/processed/plant_science.duckdb
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

OUT_DIR = "results/dynamics"

# Keywords to track for technology S-curves
TECH_KEYWORDS = {
    "CRISPR": ["crispr", "cas9", "cas12", "cas13", "genome editing", "gene editing"],
    "Single-cell": ["single-cell", "single cell", "scrna", "sc-rna"],
    "Machine learning": ["machine learning", "deep learning", "neural network",
                         "random forest", "artificial intelligence"],
    "Pangenomics": ["pangenome", "pan-genome", "pangenomic"],
    "Long-read sequencing": ["nanopore", "pacbio", "long-read", "long read"],
    "Metabolomics": ["metabolomics", "metabolome", "metabonomics"],
    "Proteomics": ["proteomics", "proteome", "mass spectrometry"],
    "Microbiome": ["microbiome", "microbiota", "metagenom"],
    "Climate change": ["climate change", "global warming", "climate adaptation",
                       "drought stress", "heat stress"],
    "Epigenetics": ["epigenetic", "dna methylation", "histone modification",
                    "chromatin remodeling"],
}

# Model organisms to track via title/abstract keywords
MODEL_ORGANISMS = {
    "Arabidopsis": ["arabidopsis"],
    "Rice": ["rice", "oryza sativa"],
    "Maize": ["maize", "zea mays", "corn"],
    "Wheat": ["wheat", "triticum"],
    "Tomato": ["tomato", "solanum lycopersicum"],
    "Tobacco": ["tobacco", "nicotiana"],
    "Soybean": ["soybean", "glycine max"],
    "Barley": ["barley", "hordeum vulgare"],
    "Cotton": ["cotton", "gossypium"],
    "Potato": ["potato", "solanum tuberosum"],
}


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def q35_tech_scurves(con):
    """Technology adoption S-curves via title/abstract keyword search."""
    print("Q35: Technology adoption S-curves...", flush=True)
    results = []

    for tech_name, keywords in TECH_KEYWORDS.items():
        # Build LIKE conditions
        conditions = " OR ".join(
            [f"lower(title) LIKE '%{kw}%' OR lower(abstract) LIKE '%{kw}%'"
             for kw in keywords]
        )
        df = con.execute(f"""
            SELECT year, COUNT(*) AS n_papers
            FROM works_clean
            WHERE year >= 1990 AND ({conditions})
            GROUP BY year ORDER BY year
        """).df()
        df["technology"] = tech_name
        results.append(df)
        total = df["n_papers"].sum()
        peak_year = df.loc[df["n_papers"].idxmax(), "year"] if len(df) > 0 else "N/A"
        print(f"  {tech_name}: {total:,} papers, peak year: {peak_year}", flush=True)

    all_df = pd.concat(results, ignore_index=True)
    all_df.to_csv(f"{OUT_DIR}/tech_scurves.csv", index=False)

    # Also compute as % of total papers per year
    totals = con.execute("""
        SELECT year, COUNT(*) AS total FROM works_clean
        WHERE year >= 1990 GROUP BY year
    """).df()
    merged = all_df.merge(totals, on="year")
    merged["pct"] = 100.0 * merged["n_papers"] / merged["total"]
    merged.to_csv(f"{OUT_DIR}/tech_scurves_pct.csv", index=False)


def q36_mesh_trends(con):
    """MeSH term evolution — fastest growing terms."""
    print("Q36: MeSH term trends...", flush=True)

    # Check if mesh_terms table exists
    try:
        count = con.execute("SELECT COUNT(*) FROM mesh_terms").fetchone()[0]
        if count == 0:
            print("  No MeSH terms found, skipping", flush=True)
            return
    except Exception:
        print("  mesh_terms table not found, skipping", flush=True)
        return

    # MeSH counts by year (top 100 terms)
    df = con.execute("""
        SELECT m.term, w.year, COUNT(*) AS n
        FROM mesh_terms m
        JOIN works_clean w ON m.work_id = w.work_id
        WHERE w.year >= 2000
          AND m.term IN (
              SELECT term FROM (
                  SELECT term, COUNT(*) AS total
                  FROM mesh_terms GROUP BY term
                  ORDER BY total DESC LIMIT 100
              )
          )
        GROUP BY m.term, w.year
        ORDER BY m.term, w.year
    """).df()
    df.to_csv(f"{OUT_DIR}/mesh_by_year.csv", index=False)

    # Growth rate: compare 2019-2023 vs 2010-2014
    growth = con.execute("""
        WITH recent AS (
            SELECT m.term, COUNT(*) AS n_recent
            FROM mesh_terms m JOIN works_clean w ON m.work_id = w.work_id
            WHERE w.year BETWEEN 2019 AND 2023
            GROUP BY m.term HAVING COUNT(*) >= 50
        ),
        earlier AS (
            SELECT m.term, COUNT(*) AS n_earlier
            FROM mesh_terms m JOIN works_clean w ON m.work_id = w.work_id
            WHERE w.year BETWEEN 2010 AND 2014
            GROUP BY m.term HAVING COUNT(*) >= 10
        )
        SELECT r.term, r.n_recent, e.n_earlier,
               (r.n_recent * 1.0 / e.n_earlier) AS growth_ratio
        FROM recent r
        JOIN earlier e ON r.term = e.term
        ORDER BY growth_ratio DESC
        LIMIT 50
    """).df()
    growth.to_csv(f"{OUT_DIR}/mesh_growth_top50.csv", index=False)
    print(f"  Top growing MeSH: {growth.iloc[0]['term']} "
          f"({growth.iloc[0]['growth_ratio']:.1f}x)", flush=True)


def q38_model_organisms(con):
    """Model organism paper counts over time."""
    print("Q38: Model organism trends...", flush=True)
    results = []

    for organism, keywords in MODEL_ORGANISMS.items():
        conditions = " OR ".join(
            [f"lower(title) LIKE '%{kw}%'" for kw in keywords]
        )
        df = con.execute(f"""
            SELECT year, COUNT(*) AS n_papers
            FROM works_clean
            WHERE year >= 1980 AND ({conditions})
            GROUP BY year ORDER BY year
        """).df()
        df["organism"] = organism
        results.append(df)
        total = df["n_papers"].sum()
        print(f"  {organism}: {total:,} papers", flush=True)

    all_df = pd.concat(results, ignore_index=True)
    all_df.to_csv(f"{OUT_DIR}/model_organisms.csv", index=False)


def q40_knowledge_flow(con):
    """Cross-field knowledge flow via concept co-citation."""
    print("Q40: Knowledge flow between fields...", flush=True)

    # Level-0 concept citation flow: if paper A (with concept X) cites paper B
    # (with concept Y), that's a knowledge flow from Y → X
    df = con.execute("""
        SELECT src.concept_name AS source_field,
               dst.concept_name AS target_field,
               COUNT(*) AS flow_weight
        FROM citations cit
        JOIN concepts src ON cit.cited_work_id = src.work_id AND src.level = 0
        JOIN concepts dst ON cit.citing_work_id = dst.work_id AND dst.level = 0
        WHERE src.concept_name != dst.concept_name
        GROUP BY src.concept_name, dst.concept_name
        ORDER BY flow_weight DESC
        LIMIT 200
    """).df()
    df.to_csv(f"{OUT_DIR}/knowledge_flow.csv", index=False)
    print(f"  {len(df)} field-to-field flows", flush=True)
    if len(df) > 0:
        print(f"  Strongest: {df.iloc[0]['source_field']} → "
              f"{df.iloc[0]['target_field']} ({df.iloc[0]['flow_weight']:,})", flush=True)


def q46_interdisciplinarity(con):
    """Interdisciplinarity index: concept diversity per paper over time."""
    print("Q46: Interdisciplinarity trend...", flush=True)
    df = con.execute("""
        SELECT w.year,
               AVG(cd.n_concepts) AS mean_concepts,
               AVG(cd.n_l0_concepts) AS mean_l0_concepts,
               COUNT(*) AS n_papers
        FROM works_clean w
        JOIN (
            SELECT work_id,
                   COUNT(DISTINCT concept_id) AS n_concepts,
                   COUNT(DISTINCT CASE WHEN level = 0 THEN concept_id END) AS n_l0_concepts
            FROM concepts
            GROUP BY work_id
        ) cd ON w.work_id = cd.work_id
        WHERE w.year >= 1960
        GROUP BY w.year ORDER BY w.year
    """).df()
    df.to_csv(f"{OUT_DIR}/interdisciplinarity.csv", index=False)
    print(f"  Latest mean concepts/paper: {df.iloc[-1]['mean_concepts']:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    con = create_database(args.db_path)

    t0 = time.time()
    q35_tech_scurves(con)
    q36_mesh_trends(con)
    q38_model_organisms(con)
    q40_knowledge_flow(con)
    q46_interdisciplinarity(con)

    elapsed = int(time.time() - t0)
    print(f"\nDynamics analysis complete ({elapsed}s)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
