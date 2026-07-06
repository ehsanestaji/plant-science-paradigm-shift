"""
Paper A — Semantic Drift Analysis.

Measures how the semantic content of each organism community shifts over time
using SPECTER2 embeddings. Computes centroid embeddings per organism per 5-year
window and tracks cosine distances.

Inputs:
  data/processed/classifications/paper_a_organism.csv  — work_id, predicted_label, confidence
  data/processed/embeddings/specter2_embeddings.npy    — (2,770,088, 768) float16
  data/processed/embeddings/specter2_work_ids.npy      — string object array

Outputs:
  results/paper_a/main/semantic_drift_centroids.csv
      organism, window_start, window_end, drift_from_previous, distance_to_arabidopsis
  results/paper_a/supplementary/organism_similarity_matrix.csv
      organism_a, organism_b, decade, cosine_similarity

Usage:
    python -m src.analysis.paper_a.semantic_drift \\
        --db-path data/processed/plant_science.duckdb \\
        --embeddings-path data/processed/embeddings/specter2_embeddings.npy \\
        --work-ids-path data/processed/embeddings/specter2_work_ids.npy
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YEAR_MIN = 1990
YEAR_MAX = 2024

# 5-year windows: (window_start, window_end) inclusive
WINDOWS = [
    (1990, 1994),
    (1995, 1999),
    (2000, 2004),
    (2005, 2009),
    (2010, 2014),
    (2015, 2019),
    (2020, 2024),
]

# Decades for similarity matrix
DECADES = {
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2024),
}

ARABIDOPSIS_LABEL = "arabidopsis"

OUT_MAIN = "results/paper_a/main"
OUT_SUPP = "results/paper_a/supplementary"


def _ensure_dirs():
    os.makedirs(OUT_MAIN, exist_ok=True)
    os.makedirs(OUT_SUPP, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_organism_classifications(classifications_dir: str) -> pd.DataFrame:
    """Load organism CSV and return work_id → organism mapping."""
    path = os.path.join(classifications_dir, "paper_a_organism.csv")
    print(f"Loading organism classifications from {path} ...", flush=True)
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows", flush=True)
    return df[["work_id", "predicted_label"]].rename(columns={"predicted_label": "organism"})


def load_embeddings_and_index(embeddings_path: str, work_ids_path: str):
    """Load the full SPECTER2 embedding array and build a work_id → index mapping.

    Returns
    -------
    embeddings : np.ndarray, shape (N, 768), float16 (kept as-is until used)
    work_id_to_idx : dict[str, int]
    """
    print(f"Loading work_ids from {work_ids_path} ...", flush=True)
    work_ids = np.load(work_ids_path, allow_pickle=True)
    print(f"  {len(work_ids):,} work IDs", flush=True)

    work_id_to_idx = {wid: i for i, wid in enumerate(work_ids)}

    print(f"Loading embeddings from {embeddings_path} ...", flush=True)
    t0 = time.time()
    embeddings = np.load(embeddings_path)
    elapsed = int(time.time() - t0)
    print(
        f"  Loaded shape={embeddings.shape}, dtype={embeddings.dtype} ({elapsed}s)",
        flush=True,
    )
    return embeddings, work_id_to_idx


def fetch_work_years(con, work_ids: list) -> pd.DataFrame:
    """Pull year from works_clean for the given work IDs."""
    print("Fetching work years from DuckDB ...", flush=True)
    id_df = pd.DataFrame({"work_id": work_ids})
    con.register("_tmp_ids", id_df)
    df = con.execute("""
        SELECT w.work_id, w.year
        FROM works_clean w
        INNER JOIN _tmp_ids t ON w.work_id = t.work_id
        WHERE w.year BETWEEN ? AND ?
    """, [YEAR_MIN, YEAR_MAX]).df()
    con.unregister("_tmp_ids")
    print(f"  {len(df):,} works with year {YEAR_MIN}-{YEAR_MAX}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_centroid(embeddings: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return mean embedding (float32) for the given row indices."""
    subset = embeddings[indices].astype(np.float32)
    return subset.mean(axis=0)


def build_centroids(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    work_id_to_idx: dict,
) -> dict:
    """Compute per-organism per-window centroids.

    Parameters
    ----------
    df : DataFrame with columns [work_id, organism, year]

    Returns
    -------
    centroids : dict[(organism, window_start, window_end), np.ndarray shape (768,)]
    """
    print("\nComputing centroids (one organism at a time) ...", flush=True)

    organisms = sorted(df["organism"].unique())
    centroids = {}

    for org in organisms:
        org_df = df[df["organism"] == org]
        print(f"  Organism '{org}': {len(org_df):,} papers", flush=True)

        for (ws, we) in WINDOWS:
            window_df = org_df[(org_df["year"] >= ws) & (org_df["year"] <= we)]
            if window_df.empty:
                continue

            # Resolve embedding indices (only rows present in embeddings array)
            idx_list = [
                work_id_to_idx[wid]
                for wid in window_df["work_id"]
                if wid in work_id_to_idx
            ]
            if not idx_list:
                continue

            indices = np.array(idx_list, dtype=np.int64)
            centroid = compute_centroid(embeddings, indices)
            centroids[(org, ws, we)] = centroid

        print(
            f"    -> {sum(1 for k in centroids if k[0] == org)} windows with data",
            flush=True,
        )

    print(f"  Total centroids computed: {len(centroids)}", flush=True)
    return centroids


# ---------------------------------------------------------------------------
# Drift and distance computations
# ---------------------------------------------------------------------------

def compute_drift_rows(centroids: dict) -> list:
    """Compute per-organism drift across consecutive windows.

    Returns list of dicts with keys:
        organism, window_start, window_end, drift_from_previous, distance_to_arabidopsis
    """
    print("\nComputing drift from previous window ...", flush=True)

    organisms = sorted({k[0] for k in centroids})
    rows = []

    for org in organisms:
        # Collect windows for this organism in chronological order
        org_windows = sorted(
            [(ws, we) for (o, ws, we) in centroids if o == org],
            key=lambda x: x[0],
        )

        prev_centroid = None
        for (ws, we) in org_windows:
            centroid = centroids[(org, ws, we)]

            if prev_centroid is None:
                drift = np.nan
            else:
                # cosine_distances expects 2-D arrays → shape (1, 768)
                drift = float(
                    cosine_distances(
                        centroid.reshape(1, -1),
                        prev_centroid.reshape(1, -1),
                    )[0, 0]
                )

            # distance_to_arabidopsis in same window
            ara_key = (ARABIDOPSIS_LABEL, ws, we)
            if org == ARABIDOPSIS_LABEL or ara_key not in centroids:
                dist_ara = np.nan
            else:
                dist_ara = float(
                    cosine_distances(
                        centroid.reshape(1, -1),
                        centroids[ara_key].reshape(1, -1),
                    )[0, 0]
                )

            rows.append(
                {
                    "organism": org,
                    "window_start": ws,
                    "window_end": we,
                    "drift_from_previous": drift,
                    "distance_to_arabidopsis": dist_ara,
                }
            )

            prev_centroid = centroid

    return rows


def compute_similarity_matrix_rows(centroids: dict) -> list:
    """Compute pairwise cosine similarity between organism centroids per decade.

    Uses decade-level centroids: for each organism, average all window
    centroids that fall within the decade.

    Returns list of dicts with keys: organism_a, organism_b, decade, cosine_similarity
    """
    print("\nComputing organism similarity matrix per decade ...", flush=True)

    organisms = sorted({k[0] for k in centroids})
    rows = []

    for decade_label, (d_start, d_end) in DECADES.items():
        # Build decade centroid for each organism: average windows that overlap decade
        decade_centroids = {}
        for org in organisms:
            window_vecs = [
                centroids[(org, ws, we)]
                for (o, ws, we) in centroids
                if o == org and ws >= d_start and we <= d_end
            ]
            if not window_vecs:
                continue
            decade_centroids[org] = np.mean(window_vecs, axis=0).astype(np.float32)

        present_orgs = sorted(decade_centroids.keys())
        if len(present_orgs) < 2:
            continue

        # Stack and compute full pairwise matrix
        mat = np.stack([decade_centroids[o] for o in present_orgs])  # (n_org, 768)
        sim_matrix = cosine_similarity(mat)  # (n_org, n_org)

        for i, org_a in enumerate(present_orgs):
            for j, org_b in enumerate(present_orgs):
                if j <= i:
                    continue  # upper triangle only, skip diagonal
                rows.append(
                    {
                        "organism_a": org_a,
                        "organism_b": org_b,
                        "decade": decade_label,
                        "cosine_similarity": float(sim_matrix[i, j]),
                    }
                )

    return rows


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_key_findings(drift_df: pd.DataFrame, sim_df: pd.DataFrame):
    """Print human-readable highlights."""
    print("\n" + "=" * 60, flush=True)
    print("KEY FINDINGS", flush=True)
    print("=" * 60, flush=True)

    # Most drifting organisms (mean drift across windows, ignoring NaN first window)
    valid_drift = drift_df.dropna(subset=["drift_from_previous"])
    if not valid_drift.empty:
        mean_drift = (
            valid_drift.groupby("organism")["drift_from_previous"]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nOrganisms ranked by mean semantic drift (cosine distance):", flush=True)
        print(f"  {'Organism':<30} {'Mean drift':>12}", flush=True)
        print("  " + "-" * 44, flush=True)
        for org, d in mean_drift.items():
            print(f"  {org:<30} {d:>12.4f}", flush=True)

    # Convergence / divergence: distance_to_arabidopsis over time
    ara_dist = drift_df.dropna(subset=["distance_to_arabidopsis"])
    if not ara_dist.empty:
        print("\nDistance to Arabidopsis (most recent window vs earliest):", flush=True)
        print(f"  {'Organism':<30} {'Earliest window':>16} {'Latest window':>14} {'Change':>8}", flush=True)
        print("  " + "-" * 72, flush=True)
        for org in sorted(ara_dist["organism"].unique()):
            org_rows = ara_dist[ara_dist["organism"] == org].sort_values("window_start")
            if len(org_rows) < 2:
                continue
            first_dist = org_rows.iloc[0]["distance_to_arabidopsis"]
            last_dist = org_rows.iloc[-1]["distance_to_arabidopsis"]
            change = last_dist - first_dist
            direction = "converging" if change < 0 else "diverging"
            print(
                f"  {org:<30} {first_dist:>16.4f} {last_dist:>14.4f} "
                f"{change:>+8.4f}  ({direction})",
                flush=True,
            )

    # Most similar organism pairs (latest decade)
    if not sim_df.empty:
        latest_decade = sorted(sim_df["decade"].unique())[-1]
        top_pairs = (
            sim_df[sim_df["decade"] == latest_decade]
            .sort_values("cosine_similarity", ascending=False)
            .head(5)
        )
        print(f"\nTop 5 most similar organism pairs in {latest_decade}:", flush=True)
        print(f"  {'Organism A':<25} {'Organism B':<25} {'Similarity':>10}", flush=True)
        print("  " + "-" * 62, flush=True)
        for _, row in top_pairs.iterrows():
            print(
                f"  {row['organism_a']:<25} {row['organism_b']:<25} "
                f"{row['cosine_similarity']:>10.4f}",
                flush=True,
            )

    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Semantic drift analysis using SPECTER2 embeddings"
    )
    ap.add_argument(
        "--db-path",
        default="data/processed/plant_science.duckdb",
        help="Path to the DuckDB database",
    )
    ap.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings/specter2_embeddings.npy",
        help="Path to SPECTER2 embeddings .npy file",
    )
    ap.add_argument(
        "--work-ids-path",
        default="data/processed/embeddings/specter2_work_ids.npy",
        help="Path to work IDs .npy file (string object array)",
    )
    ap.add_argument(
        "--classifications-dir",
        default="data/processed/classifications",
        help="Directory containing paper_a_organism.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory root (default: results/paper_a/{main,supplementary})",
    )
    args = ap.parse_args()

    global OUT_MAIN, OUT_SUPP
    if args.out_dir:
        OUT_MAIN = os.path.join(args.out_dir, "main")
        OUT_SUPP = os.path.join(args.out_dir, "supplementary")

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    # --- Load organism classifications ---------------------------------------
    organism_df = load_organism_classifications(args.classifications_dir)

    # --- Load embeddings & build index --------------------------------------
    embeddings, work_id_to_idx = load_embeddings_and_index(
        args.embeddings_path, args.work_ids_path
    )

    # --- Fetch years from DuckDB --------------------------------------------
    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='30GB'")
    con.execute("SET threads=8")
    year_df = fetch_work_years(con, organism_df["work_id"].tolist())
    con.close()

    # --- Build master dataframe (organism + year) ----------------------------
    df = organism_df.merge(year_df, on="work_id", how="inner")
    print(f"\nMaster dataframe: {len(df):,} rows (organism + year)", flush=True)

    # --- Compute centroids ---------------------------------------------------
    centroids = build_centroids(df, embeddings, work_id_to_idx)

    # Free embedding memory hint (Python GC may or may not act immediately)
    del embeddings

    # --- Compute drift -------------------------------------------------------
    drift_rows = compute_drift_rows(centroids)
    drift_df = pd.DataFrame(drift_rows).sort_values(
        ["organism", "window_start"]
    ).reset_index(drop=True)

    # --- Compute similarity matrix -------------------------------------------
    sim_rows = compute_similarity_matrix_rows(centroids)
    sim_df = pd.DataFrame(sim_rows).sort_values(
        ["decade", "organism_a", "organism_b"]
    ).reset_index(drop=True)

    # --- Save outputs --------------------------------------------------------
    centroids_path = os.path.join(OUT_MAIN, "semantic_drift_centroids.csv")
    sim_matrix_path = os.path.join(OUT_SUPP, "organism_similarity_matrix.csv")

    drift_df.to_csv(centroids_path, index=False)
    sim_df.to_csv(sim_matrix_path, index=False)

    print(f"\nSaved: {centroids_path}")
    print(f"Saved: {sim_matrix_path}")

    # --- Console highlights --------------------------------------------------
    print_key_findings(drift_df, sim_df)

    elapsed = int(time.time() - t0)
    print(f"\nSemantic drift analysis complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
