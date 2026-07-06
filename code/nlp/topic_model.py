"""
BERTopic topic modeling on SPECTER2 embeddings (resumable).

Runs UMAP dimensionality reduction + HDBSCAN clustering + BERTopic
on pre-computed embeddings. Each stage checkpoints its output so
the pipeline can be resumed after interruption (e.g., closing laptop lid).

Stages and their checkpoint files:
  Stage 1-3: BERTopic fit → topic_assignments.csv + topic_labels.csv
  Stage 4:   2D UMAP      → umap_2d.npy + umap_2d_work_ids.npy

Usage:
    python -m src.nlp.topic_model \
        --embeddings-path data/processed/embeddings/specter2_embeddings.npy \
        --work-ids-path data/processed/embeddings/specter2_work_ids.npy \
        --abstracts-path data/abstracts_for_embedding.parquet \
        --out-dir results/topics

Estimated time: 4-6h on M2 Max (2.77M embeddings).
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _stage_done(out_dir: str, stage: str) -> bool:
    """Check if a stage's output files already exist."""
    checks = {
        "bertopic": [
            os.path.join(out_dir, "topic_assignments.csv"),
            os.path.join(out_dir, "topic_labels.csv"),
        ],
        "umap2d": [
            os.path.join(out_dir, "umap_2d.npy"),
            os.path.join(out_dir, "umap_2d_work_ids.npy"),
        ],
    }
    return all(os.path.exists(f) for f in checks.get(stage, []))


def _get_umap_hdbscan(use_gpu: bool = False):
    """Return UMAP and HDBSCAN classes, preferring cuML GPU versions if available."""
    if use_gpu:
        try:
            from cuml.manifold import UMAP
            from cuml.cluster import HDBSCAN
            print("  Using cuML GPU-accelerated UMAP + HDBSCAN", flush=True)
            return UMAP, HDBSCAN
        except ImportError:
            print("  cuML not available, falling back to CPU", flush=True)

    from umap import UMAP
    from hdbscan import HDBSCAN
    print("  Using CPU UMAP + HDBSCAN", flush=True)
    return UMAP, HDBSCAN


def run_topic_model(
    embeddings_path: str,
    work_ids_path: str,
    abstracts_path: str,
    out_dir: str = "results/topics",
    use_gpu: bool = False,
):
    """Run BERTopic pipeline on pre-computed SPECTER2 embeddings."""

    print("=== BERTopic Topic Modeling Pipeline (resumable) ===", flush=True)
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────
    print("  Loading embeddings...", flush=True)
    embeddings = np.load(embeddings_path).astype(np.float32)
    work_ids = np.load(work_ids_path, allow_pickle=True)
    n_total = len(embeddings)
    print(f"  {n_total:,} embeddings ({embeddings.shape[1]}D)", flush=True)

    print("  Loading abstracts for tokenization...", flush=True)
    df = pd.read_parquet(abstracts_path, columns=["work_id", "abstract", "year"])
    # Align abstracts with work_ids
    df = df.set_index("work_id").loc[work_ids].reset_index()
    docs = df["abstract"].fillna("").tolist()
    years = df["year"].values
    print(f"  {len(docs):,} documents aligned", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # Stage 1-3: BERTopic (UMAP 5D + HDBSCAN + topic extraction)
    # ══════════════════════════════════════════════════════════════════
    if _stage_done(out_dir, "bertopic"):
        print("\n  SKIP Stages 1-3: BERTopic results already exist", flush=True)
        assigns = pd.read_csv(os.path.join(out_dir, "topic_assignments.csv"))
        n_topics = assigns[assigns["topic_id"] != -1]["topic_id"].nunique()
        n_outliers = (assigns["topic_id"] == -1).sum()
        print(f"  Loaded: {n_topics} topics, {n_outliers:,} outliers", flush=True)
    else:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        UMAP, HDBSCAN = _get_umap_hdbscan(use_gpu)

        # ── Step 1: UMAP 5D ──────────────────────────────────────────
        print("\n  Step 1: UMAP dimensionality reduction (5D)...", flush=True)
        t1 = time.time()
        umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,
            verbose=True,
        )

        # ── Step 2: HDBSCAN ──────────────────────────────────────────
        print("\n  Step 2: HDBSCAN clustering...", flush=True)
        hdbscan_model = HDBSCAN(
            min_cluster_size=100,
            min_samples=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # ── Step 3: BERTopic ─────────────────────────────────────────
        print("\n  Step 3: BERTopic model fitting...", flush=True)
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=50,
            max_df=0.95,
            ngram_range=(1, 2),
            max_features=50_000,
        )

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            calculate_probabilities=False,
            verbose=True,
        )

        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        elapsed_fit = time.time() - t1
        print(f"  BERTopic fit complete in {elapsed_fit / 3600:.1f}h", flush=True)

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"  {n_topics} topics found, {n_outliers:,} outliers "
              f"({100*n_outliers/n_total:.1f}%)", flush=True)

        # ── Save topic assignments ────────────────────────────────────
        print("\n  Saving topic assignments...", flush=True)
        assignments = pd.DataFrame({
            "work_id": work_ids,
            "topic_id": topics,
            "topic_prob": probs if probs is not None else np.nan,
            "year": years,
        })
        assignments.to_csv(
            os.path.join(out_dir, "topic_assignments.csv"),
            index=False,
        )
        print(f"  Saved topic_assignments.csv ({len(assignments):,} rows)",
              flush=True)

        # ── Save topic labels ─────────────────────────────────────────
        print("  Saving topic labels...", flush=True)
        topic_info = topic_model.get_topic_info()
        topic_labels = []
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                top_words = "outlier"
            else:
                words = topic_model.get_topic(tid)
                top_words = ", ".join([w for w, _ in words[:10]])
            topic_labels.append({
                "topic_id": tid,
                "top_words": top_words,
                "n_docs": row["Count"],
            })

        labels_df = pd.DataFrame(topic_labels)
        labels_df.to_csv(
            os.path.join(out_dir, "topic_labels.csv"),
            index=False,
        )
        print(f"  Saved topic_labels.csv ({len(labels_df)} topics)", flush=True)

        # Free memory
        del topic_model, topics, probs
        gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Stage 4: 2D UMAP for visualization
    # ══════════════════════════════════════════════════════════════════
    if _stage_done(out_dir, "umap2d"):
        print("\n  SKIP Stage 4: 2D UMAP already exists", flush=True)
    else:
        UMAP, _ = _get_umap_hdbscan(use_gpu)

        # Free docs from memory if still around
        try:
            del docs
        except NameError:
            pass
        gc.collect()

        print("\n  Step 4: 2D UMAP for visualization...", flush=True)
        t2 = time.time()
        umap_2d = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            low_memory=True,
            verbose=True,
        )
        coords_2d = umap_2d.fit_transform(embeddings)
        elapsed_2d = time.time() - t2
        print(f"  2D UMAP done in {elapsed_2d / 3600:.1f}h", flush=True)

        np.save(os.path.join(out_dir, "umap_2d.npy"),
                coords_2d.astype(np.float32))
        np.save(os.path.join(out_dir, "umap_2d_work_ids.npy"), work_ids)
        print("  Saved umap_2d.npy and umap_2d_work_ids.npy", flush=True)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print(f"\n=== Done ===", flush=True)
    print(f"  {n_topics} topics from {n_total:,} documents", flush=True)
    print(f"  Total time: {elapsed_total / 3600:.1f}h ({elapsed_total:.0f}s)",
          flush=True)
    print(f"  Outputs in: {out_dir}/", flush=True)


def main():
    ap = argparse.ArgumentParser(description="BERTopic topic modeling")
    ap.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings/specter2_embeddings.npy",
        help="Path to SPECTER2 embeddings",
    )
    ap.add_argument(
        "--work-ids-path",
        default="data/processed/embeddings/specter2_work_ids.npy",
        help="Path to work ID array",
    )
    ap.add_argument(
        "--abstracts-path",
        default="data/abstracts_for_embedding.parquet",
        help="Path to abstracts parquet (for text tokenization)",
    )
    ap.add_argument(
        "--out-dir",
        default="results/topics",
        help="Output directory for topic results",
    )
    ap.add_argument("--use-gpu", action="store_true",
                    help="Use cuML GPU-accelerated UMAP/HDBSCAN if available")
    args = ap.parse_args()

    run_topic_model(
        embeddings_path=args.embeddings_path,
        work_ids_path=args.work_ids_path,
        abstracts_path=args.abstracts_path,
        out_dir=args.out_dir,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
