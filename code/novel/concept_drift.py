"""
Analysis J: Concept Neighborhood Drift.

How have concept meanings shifted over time? CPU-only proxy for semantic drift
via co-occurrence neighborhood analysis. For each target concept, in 5-year
windows, computes the top-50 co-occurring concepts and measures cosine
similarity between consecutive windows and vs the earliest window.

Output → results/novel/
  concept_neighborhood_by_window.csv
  concept_drift_scores.csv
  concept_drift_summary.csv

Usage:
    python -m src.novel.concept_drift --db-path data/processed/plant_science.duckdb
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

TARGET_CONCEPTS = [
    "Epigenetics",
    "Gene expression",
    "Genomics",
    "Biotechnology",
    "Biodiversity",
    "Climate change",
    "Agroforestry",
    "Food security",
    "Systems biology",
    "Phenotype",
    "Bioinformatics",
    "Machine learning",
    "Microbiome",
    "Organic farming",
    "Proteomics",
]

WINDOW_SIZE = 5  # years per window
MIN_YEAR = 1975
MAX_YEAR = 2024
TOP_N_NEIGHBORS = 50
MIN_SCORE = 0.3
MAX_LEVEL = 2


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def _cosine_sim(v1, v2):
    """Cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


def get_available_concepts(con) -> list:
    """Find which target concepts actually exist in the database."""
    print("Checking which target concepts exist…", flush=True)
    available = []
    for concept in TARGET_CONCEPTS:
        row = con.execute(f"""
            SELECT COUNT(*) FROM concepts
            WHERE lower(concept_name) = lower('{concept}')
              AND score >= {MIN_SCORE}
        """).fetchone()
        if row[0] > 0:
            available.append(concept)
            print(f"  ✓ {concept}: {row[0]:,} annotations", flush=True)
        else:
            print(f"  ✗ {concept}: not found", flush=True)
    return available


def fetch_neighborhood(con, concept: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Get top co-occurring concepts for a target concept in a time window."""
    df = con.execute(f"""
        WITH target_papers AS (
            SELECT c.work_id
            FROM concepts c
            JOIN works_clean w ON c.work_id = w.work_id
            WHERE lower(c.concept_name) = lower('{concept}')
              AND c.score >= {MIN_SCORE}
              AND w.year BETWEEN {start_year} AND {end_year}
        )
        SELECT c2.concept_name AS neighbor, COUNT(*) AS co_count
        FROM target_papers tp
        JOIN concepts c2 ON tp.work_id = c2.work_id
        WHERE lower(c2.concept_name) != lower('{concept}')
          AND c2.score >= {MIN_SCORE}
          AND c2.level <= {MAX_LEVEL}
        GROUP BY c2.concept_name
        ORDER BY co_count DESC
        LIMIT {TOP_N_NEIGHBORS}
    """).df()
    return df


def compute_drift(neighborhoods: dict, concept: str) -> tuple:
    """Compute drift scores for a concept across time windows."""
    windows = sorted(neighborhoods.keys())
    if len(windows) < 2:
        return [], {}

    # Build vocabulary from all windows
    all_neighbors = set()
    for w in windows:
        all_neighbors.update(neighborhoods[w]["neighbor"].tolist())
    vocab = sorted(all_neighbors)
    vocab_idx = {n: i for i, n in enumerate(vocab)}

    # Build vectors
    vectors = {}
    for w in windows:
        v = np.zeros(len(vocab))
        df = neighborhoods[w]
        for _, row in df.iterrows():
            if row["neighbor"] in vocab_idx:
                v[vocab_idx[row["neighbor"]]] = row["co_count"]
        # Normalize
        total = v.sum()
        if total > 0:
            v = v / total
        vectors[w] = v

    # Compute pairwise drift
    drift_records = []
    first_window = windows[0]

    for i, w in enumerate(windows):
        # Similarity to first window
        sim_to_first = _cosine_sim(vectors[first_window], vectors[w])

        # Similarity to previous window
        if i > 0:
            prev_w = windows[i - 1]
            sim_to_prev = _cosine_sim(vectors[prev_w], vectors[w])
        else:
            sim_to_prev = 1.0

        drift_records.append({
            "concept": concept,
            "window": w,
            "sim_to_first": round(sim_to_first, 4),
            "sim_to_previous": round(sim_to_prev, 4),
            "cumulative_drift": round(1 - sim_to_first, 4),
        })

    # Summary
    total_drift = 1 - _cosine_sim(vectors[first_window], vectors[windows[-1]])
    max_single = max(1 - r["sim_to_previous"] for r in drift_records[1:]) if len(drift_records) > 1 else 0
    max_drift_window = None
    if len(drift_records) > 1:
        max_idx = np.argmax([1 - r["sim_to_previous"] for r in drift_records[1:]]) + 1
        max_drift_window = drift_records[max_idx]["window"]

    summary = {
        "concept": concept,
        "n_windows": len(windows),
        "total_drift": round(total_drift, 4),
        "max_single_window_drift": round(max_single, 4),
        "max_drift_window": max_drift_window,
    }

    return drift_records, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=8")

    available = get_available_concepts(con)
    if not available:
        print("No target concepts found in database.", flush=True)
        con.close()
        return

    # Generate time windows
    windows = []
    y = MIN_YEAR
    while y + WINDOW_SIZE - 1 <= MAX_YEAR:
        windows.append((y, y + WINDOW_SIZE - 1))
        y += WINDOW_SIZE

    print(f"\nProcessing {len(available)} concepts × {len(windows)} windows…",
          flush=True)

    all_neighborhoods = []
    all_drift = []
    all_summaries = []

    for concept in available:
        t1 = time.time()
        print(f"\n  {concept}:", flush=True)
        neighborhoods = {}

        for start, end in windows:
            w_label = f"{start}-{end}"
            nbr_df = fetch_neighborhood(con, concept, start, end)
            if len(nbr_df) > 0:
                neighborhoods[w_label] = nbr_df
                nbr_df = nbr_df.copy()
                nbr_df["concept"] = concept
                nbr_df["window"] = w_label
                all_neighborhoods.append(nbr_df)

        drift_records, summary = compute_drift(neighborhoods, concept)
        all_drift.extend(drift_records)
        if summary:
            all_summaries.append(summary)
            print(f"    total_drift={summary['total_drift']:.4f}, "
                  f"max_single={summary['max_single_window_drift']:.4f} "
                  f"at {summary['max_drift_window']} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    con.close()

    # Save results
    if all_neighborhoods:
        nbr_df = pd.concat(all_neighborhoods, ignore_index=True)
        nbr_df.to_csv(f"{OUT_DIR}/concept_neighborhood_by_window.csv", index=False)

    if all_drift:
        drift_df = pd.DataFrame(all_drift)
        drift_df.to_csv(f"{OUT_DIR}/concept_drift_scores.csv", index=False)

    if all_summaries:
        sum_df = pd.DataFrame(all_summaries).sort_values("total_drift", ascending=False)
        sum_df.to_csv(f"{OUT_DIR}/concept_drift_summary.csv", index=False)

    # Summary
    print("\n=== Concept Drift Summary ===")
    if all_summaries:
        sum_df = pd.DataFrame(all_summaries).sort_values("total_drift", ascending=False)
        for _, r in sum_df.iterrows():
            print(f"  {r['concept']:<25} drift={r['total_drift']:.4f}  "
                  f"max_shift={r['max_single_window_drift']:.4f} "
                  f"at {r['max_drift_window']}")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis J complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
