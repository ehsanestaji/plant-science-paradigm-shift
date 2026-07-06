"""
Hierarchical merging of BERTopic topics into macro-themes.

Clusters 1,632 BERTopic topic centroids (computed from SPECTER2 embeddings)
into ~30 interpretable macro-themes using agglomerative clustering on cosine
distance.

Inputs
------
results/topics/topic_assignments.csv  — work_id, topic_id, topic_prob, year
results/topics/topic_labels.csv       — topic_id, top_words, n_docs
data/processed/embeddings/specter2_embeddings.npy    — (2,770,088, 768) float16
data/processed/embeddings/specter2_work_ids.npy      — (2,770,088,) object

Outputs
-------
results/paper_a/main/macro_themes.csv
results/paper_a/supplementary/topic_hierarchy_full.csv
results/paper_a/supplementary/paper_macro_themes.csv
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge BERTopic topics into macro-themes via hierarchical clustering."
    )
    p.add_argument(
        "--topics-dir",
        default="results/topics",
        help="Directory containing topic_assignments.csv and topic_labels.csv",
    )
    p.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings/specter2_embeddings.npy",
        help="Path to SPECTER2 embeddings .npy file (shape N×768, float16)",
    )
    p.add_argument(
        "--work-ids-path",
        default="data/processed/embeddings/specter2_work_ids.npy",
        help="Path to work_id array .npy file (shape N, object/str)",
    )
    p.add_argument(
        "--out-dir",
        default="results/paper_a",
        help="Root output directory (main/ and supplementary/ created inside)",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=30,
        help="Number of macro-theme clusters (default: 30)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_topic_data(topics_dir: Path):
    """Load topic assignments and labels; return two DataFrames."""
    log("Loading topic_assignments.csv ...")
    assignments = pd.read_csv(
        topics_dir / "topic_assignments.csv",
        dtype={"work_id": str, "topic_id": int, "topic_prob": float},
    )
    log(f"  {len(assignments):,} rows loaded.")

    log("Loading topic_labels.csv ...")
    labels = pd.read_csv(
        topics_dir / "topic_labels.csv",
        dtype={"topic_id": int, "top_words": str, "n_docs": int},
    )
    log(f"  {len(labels):,} topics loaded (including outlier topic -1).")

    return assignments, labels


def compute_topic_centroids(
    assignments: pd.DataFrame,
    embeddings_path: Path,
    work_ids_path: Path,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute the mean SPECTER2 embedding for each non-outlier topic.

    Returns
    -------
    centroids : np.ndarray, shape (n_topics, 768), float32
    topic_ids : list[int] — topic ids in the same row order as centroids
    """
    log("Loading work_id index from embeddings file ...")
    work_ids_arr = np.load(str(work_ids_path), allow_pickle=True)
    work_id_to_idx: dict[str, int] = {wid: i for i, wid in enumerate(work_ids_arr)}
    del work_ids_arr
    log(f"  Index built for {len(work_id_to_idx):,} work_ids.")

    # Keep only non-outlier assignments
    non_outliers = assignments[assignments["topic_id"] != -1].copy()
    unique_topic_ids: list[int] = sorted(non_outliers["topic_id"].unique().tolist())
    n_topics = len(unique_topic_ids)
    log(f"  {n_topics} non-outlier topics to process.")

    # Build per-topic lists of embedding row indices
    log("Mapping papers to embedding row indices ...")
    non_outliers["emb_idx"] = non_outliers["work_id"].map(work_id_to_idx)
    missing = non_outliers["emb_idx"].isna().sum()
    if missing > 0:
        log(f"  WARNING: {missing:,} papers not found in embeddings — skipping.")
    non_outliers = non_outliers.dropna(subset=["emb_idx"])
    non_outliers["emb_idx"] = non_outliers["emb_idx"].astype(int)

    topic_to_indices: dict[int, np.ndarray] = {
        tid: grp["emb_idx"].values
        for tid, grp in non_outliers.groupby("topic_id")
    }
    del non_outliers

    # Load embeddings as memory-mapped array (4 GB, float16)
    log("Memory-mapping embeddings ...")
    embeddings = np.load(str(embeddings_path), mmap_mode="r")
    log(f"  Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Compute centroids topic-by-topic
    dim = embeddings.shape[1]
    centroids = np.zeros((n_topics, dim), dtype=np.float32)

    log(f"Computing centroids for {n_topics} topics ...")
    t0 = time.time()
    for i, tid in enumerate(unique_topic_ids):
        idx = topic_to_indices[tid]
        # Extract rows and cast to float32 before averaging
        centroids[i] = embeddings[idx].astype(np.float32).mean(axis=0)
        if (i + 1) % 100 == 0 or (i + 1) == n_topics:
            elapsed = time.time() - t0
            log(f"  {i + 1}/{n_topics} topics done ({elapsed:.1f}s elapsed)")

    del embeddings
    log("Centroids computed.")
    return centroids, unique_topic_ids


def cluster_topics(
    centroids: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Agglomerative clustering on cosine distance matrix.

    Returns cluster_labels array of shape (n_topics,).
    """
    log(f"Computing cosine distance matrix ({centroids.shape[0]} × {centroids.shape[0]}) ...")
    dist_matrix = 1.0 - cosine_similarity(centroids.astype(np.float64))
    # Clip to [0, 2] to avoid floating-point negatives
    np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)

    log(f"Running AgglomerativeClustering with n_clusters={n_clusters} ...")
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(dist_matrix)
    log("Clustering done.")
    return labels


def name_macro_theme(top_words_list: list[str]) -> str:
    """
    Given top_words strings from constituent topics, return a name made of the
    5 most frequent individual words across all topics.
    """
    word_counter: Counter = Counter()
    for tw in top_words_list:
        # top_words is comma-separated, e.g. "wheat, grain, drought, ..."
        words = [w.strip() for w in tw.split(",") if w.strip()]
        word_counter.update(words)
    top5 = [w for w, _ in word_counter.most_common(5)]
    return ", ".join(top5)


def build_outputs(
    assignments: pd.DataFrame,
    labels_df: pd.DataFrame,
    topic_ids: list[int],
    cluster_labels: np.ndarray,
    n_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the three output DataFrames.

    Returns (macro_themes_df, topic_hierarchy_df, paper_macro_themes_df).
    """
    # Map topic_id → macro_theme_id (0-based cluster label)
    topic_to_macro: dict[int, int] = {
        tid: int(cluster_labels[i]) for i, tid in enumerate(topic_ids)
    }
    # Outliers always map to macro_theme_id = -1
    topic_to_macro[-1] = -1

    # -----------------------------------------------------------------------
    # topic_hierarchy_full.csv
    # -----------------------------------------------------------------------
    topic_hierarchy = labels_df.copy()
    topic_hierarchy["macro_theme_id"] = topic_hierarchy["topic_id"].map(topic_to_macro)
    # topic_id=-1 row already covered by the mapping above
    topic_hierarchy = topic_hierarchy[["topic_id", "macro_theme_id", "top_words", "n_docs"]]

    # -----------------------------------------------------------------------
    # macro_themes.csv
    # -----------------------------------------------------------------------
    macro_rows = []

    # Non-outlier clusters
    labels_indexed = labels_df.set_index("topic_id")
    for cluster_id in range(n_clusters):
        member_tids = [tid for tid, cid in topic_to_macro.items() if cid == cluster_id]
        member_top_words = [
            labels_indexed.loc[tid, "top_words"]
            for tid in member_tids
            if tid in labels_indexed.index
        ]
        member_ndocs = [
            labels_indexed.loc[tid, "n_docs"]
            for tid in member_tids
            if tid in labels_indexed.index
        ]
        name = name_macro_theme(member_top_words)
        n_docs = int(sum(member_ndocs))
        constituent_ids_str = ";".join(str(t) for t in sorted(member_tids))
        all_words_str = " | ".join(member_top_words)
        macro_rows.append(
            {
                "macro_theme_id": cluster_id,
                "name": name,
                "n_docs": n_docs,
                "constituent_topic_ids": constituent_ids_str,
                "top_words": all_words_str,
            }
        )

    # Outlier row
    outlier_row = labels_indexed.loc[-1] if -1 in labels_indexed.index else None
    outlier_ndocs = int(outlier_row["n_docs"]) if outlier_row is not None else 0
    macro_rows.append(
        {
            "macro_theme_id": -1,
            "name": "outliers",
            "n_docs": outlier_ndocs,
            "constituent_topic_ids": "-1",
            "top_words": "outlier",
        }
    )

    macro_themes_df = pd.DataFrame(macro_rows).sort_values("macro_theme_id").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # paper_macro_themes.csv
    # -----------------------------------------------------------------------
    paper_macro = assignments[["work_id", "topic_id"]].copy()
    paper_macro["macro_theme_id"] = paper_macro["topic_id"].map(topic_to_macro)
    # Any topic_id not in the mapping (shouldn't happen) defaults to -1
    paper_macro["macro_theme_id"] = paper_macro["macro_theme_id"].fillna(-1).astype(int)
    paper_macro = paper_macro[["work_id", "macro_theme_id", "topic_id"]]

    return macro_themes_df, topic_hierarchy, paper_macro


def print_macro_theme_table(macro_themes_df: pd.DataFrame) -> None:
    """Print a summary table of macro-themes."""
    print()
    print("=" * 80)
    print(f"{'ID':>4}  {'n_docs':>9}  {'name'}")
    print("=" * 80)
    for _, row in macro_themes_df.sort_values("n_docs", ascending=False).iterrows():
        print(f"{row['macro_theme_id']:>4}  {row['n_docs']:>9,}  {row['name']}")
    print("=" * 80)
    print(f"Total macro-themes (excl. outliers): {(macro_themes_df['macro_theme_id'] >= 0).sum()}")
    print()


def save_outputs(
    macro_themes_df: pd.DataFrame,
    topic_hierarchy_df: pd.DataFrame,
    paper_macro_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    main_dir = out_dir / "main"
    supp_dir = out_dir / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    p1 = main_dir / "macro_themes.csv"
    p2 = supp_dir / "topic_hierarchy_full.csv"
    p3 = supp_dir / "paper_macro_themes.csv"

    log(f"Writing {p1} ...")
    macro_themes_df.to_csv(p1, index=False)

    log(f"Writing {p2} ...")
    topic_hierarchy_df.to_csv(p2, index=False)

    log(f"Writing {p3} ({len(paper_macro_df):,} rows) ...")
    paper_macro_df.to_csv(p3, index=False)

    log("All outputs saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    root = Path(os.getcwd())
    topics_dir = root / args.topics_dir
    embeddings_path = root / args.embeddings_path
    work_ids_path = root / args.work_ids_path
    out_dir = root / args.out_dir

    log("=== topic_hierarchy.py ===")
    log(f"topics_dir:       {topics_dir}")
    log(f"embeddings_path:  {embeddings_path}")
    log(f"work_ids_path:    {work_ids_path}")
    log(f"out_dir:          {out_dir}")
    log(f"n_clusters:       {args.n_clusters}")
    log("")

    # 1. Load topic data
    assignments, labels_df = load_topic_data(topics_dir)

    # 2. Compute topic centroids from embeddings
    centroids, topic_ids = compute_topic_centroids(
        assignments, embeddings_path, work_ids_path
    )

    # 3. Cluster centroids
    cluster_labels = cluster_topics(centroids, args.n_clusters)

    # 4. Build output DataFrames
    log("Building output DataFrames ...")
    macro_themes_df, topic_hierarchy_df, paper_macro_df = build_outputs(
        assignments, labels_df, topic_ids, cluster_labels, args.n_clusters
    )

    # 5. Print summary table
    print_macro_theme_table(macro_themes_df)

    # Verification stats
    log(f"macro_themes.csv rows:       {len(macro_themes_df)}")
    log(f"topic_hierarchy_full.csv rows: {len(topic_hierarchy_df)}")
    log(f"paper_macro_themes.csv rows:  {len(paper_macro_df):,}")
    assert len(paper_macro_df) == len(assignments), (
        f"Row count mismatch: {len(paper_macro_df)} vs {len(assignments)}"
    )

    # 6. Save outputs
    save_outputs(macro_themes_df, topic_hierarchy_df, paper_macro_df, out_dir)

    log("Done.")


if __name__ == "__main__":
    main()
