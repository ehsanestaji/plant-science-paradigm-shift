"""
Compute SPECTER2 embeddings for plant science abstracts.

Loads abstracts_for_embedding.parquet, encodes with allenai/specter2_base
via sentence-transformers, saves as float16 .npy arrays with checkpointing.

Usage:
    python -m src.nlp.embeddings \
        --abstracts-path data/abstracts_for_embedding.parquet \
        --out-dir data/processed/embeddings \
        --batch-size 512 \
        --checkpoint-every 50000 \
        --model-cache /proj/nobackup/hpc2n2025-278/models/huggingface
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Constants ──────────────────────────────────────────────────────────
MODEL_NAME = "allenai/specter2_base"
EMBEDDING_DIM = 768
MAX_TOKENS = 512  # SPECTER2 limit


def _get_device():
    """Select best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        print("  Device: Apple MPS (Metal Performance Shaders)", flush=True)
        return "mps"
    elif torch.cuda.is_available():
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})", flush=True)
        return "cuda"
    else:
        print("  Device: CPU (this will be very slow!)", flush=True)
        return "cpu"


def _load_checkpoint(out_dir: str):
    """Load partial embeddings if checkpoint exists."""
    emb_path = os.path.join(out_dir, "specter2_embeddings_partial.npy")
    ids_path = os.path.join(out_dir, "specter2_work_ids_partial.npy")
    meta_path = os.path.join(out_dir, "checkpoint_meta.npy")

    if all(os.path.exists(p) for p in [emb_path, ids_path, meta_path]):
        emb = np.load(emb_path)
        ids = np.load(ids_path, allow_pickle=True)
        meta = np.load(meta_path, allow_pickle=True).item()
        n_done = meta["n_done"]
        print(f"  Resuming from checkpoint: {n_done:,} already embedded", flush=True)
        return emb, ids, n_done
    return None, None, 0


def _save_checkpoint(out_dir: str, embeddings: np.ndarray,
                     work_ids: np.ndarray, n_done: int):
    """Save partial checkpoint."""
    np.save(os.path.join(out_dir, "specter2_embeddings_partial.npy"),
            embeddings[:n_done])
    np.save(os.path.join(out_dir, "specter2_work_ids_partial.npy"),
            work_ids[:n_done])
    np.save(os.path.join(out_dir, "checkpoint_meta.npy"),
            {"n_done": n_done})
    print(f"  Checkpoint saved: {n_done:,} embeddings", flush=True)


def _cleanup_checkpoints(out_dir: str):
    """Remove checkpoint files after successful completion."""
    for suffix in ["_partial.npy", "_partial.npy", "_meta.npy"]:
        for prefix in ["specter2_embeddings", "specter2_work_ids", "checkpoint"]:
            path = os.path.join(out_dir, f"{prefix}{suffix}")
            if os.path.exists(path):
                os.remove(path)


def compute_embeddings(
    abstracts_path: str,
    out_dir: str = "data/processed/embeddings",
    batch_size: int = 512,
    checkpoint_every: int = 50_000,
    model_cache: str | None = None,
):
    """Compute SPECTER2 embeddings for all abstracts."""
    from sentence_transformers import SentenceTransformer

    print("=== SPECTER2 Embedding Pipeline ===", flush=True)
    t0 = time.time()

    # ── Load abstracts ─────────────────────────────────────────────────
    print(f"  Loading abstracts from {abstracts_path}...", flush=True)
    df = pd.read_parquet(abstracts_path)
    n_total = len(df)
    print(f"  {n_total:,} abstracts loaded", flush=True)

    # ── Prepare texts ──────────────────────────────────────────────────
    # SPECTER2 expects title + abstract; we only have abstract.
    # Truncate to ~512 tokens (~2500 chars as rough estimate).
    texts = df["abstract"].fillna("").str[:2500].tolist()
    work_ids = df["work_id"].values

    # ── Load model ─────────────────────────────────────────────────────
    device = _get_device()
    print(f"  Loading model: {MODEL_NAME}...", flush=True)
    load_kwargs = {"device": device}
    if model_cache:
        os.environ["HF_HOME"] = model_cache
        os.environ["TRANSFORMERS_CACHE"] = model_cache
        load_kwargs["cache_folder"] = model_cache
        print(f"  Using local cache: {model_cache}", flush=True)
    model = SentenceTransformer(MODEL_NAME, **load_kwargs)
    print(f"  Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}", flush=True)

    # ── Check for checkpoint ───────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    prev_emb, prev_ids, n_done = _load_checkpoint(out_dir)

    # ── Allocate output array ──────────────────────────────────────────
    embeddings = np.zeros((n_total, EMBEDDING_DIM), dtype=np.float16)
    all_work_ids = np.array(work_ids, dtype=object)

    if prev_emb is not None and n_done > 0:
        embeddings[:n_done] = prev_emb[:n_done]
        texts = texts[n_done:]
        print(f"  Remaining to embed: {len(texts):,}", flush=True)

    # ── Encode in batches ──────────────────────────────────────────────
    print(f"  Encoding with batch_size={batch_size}...", flush=True)
    n_batches = (len(texts) + batch_size - 1) // batch_size
    last_checkpoint = n_done

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_idx = i // batch_size + 1

        try:
            emb = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings[n_done:n_done + len(batch)] = emb.astype(np.float16)
            n_done += len(batch)

        except Exception as e:
            print(f"\n  ERROR at batch {batch_idx}: {e}", flush=True)
            print("  Saving checkpoint and exiting...", flush=True)
            _save_checkpoint(out_dir, embeddings, all_work_ids, n_done)
            sys.exit(1)

        # Progress
        if batch_idx % 50 == 0 or batch_idx == n_batches:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(
                f"  [{batch_idx}/{n_batches}] "
                f"{n_done:,}/{n_total:,} "
                f"({100 * n_done / n_total:.1f}%) "
                f"| {rate:.0f} docs/s "
                f"| ETA {eta / 3600:.1f}h",
                flush=True,
            )

        # Checkpoint
        if n_done - last_checkpoint >= checkpoint_every:
            _save_checkpoint(out_dir, embeddings, all_work_ids, n_done)
            last_checkpoint = n_done

    # ── Save final outputs ─────────────────────────────────────────────
    emb_path = os.path.join(out_dir, "specter2_embeddings.npy")
    ids_path = os.path.join(out_dir, "specter2_work_ids.npy")

    np.save(emb_path, embeddings)
    np.save(ids_path, all_work_ids)

    # Cleanup checkpoint files
    _cleanup_checkpoints(out_dir)

    emb_size = os.path.getsize(emb_path) / (1024 ** 3)
    elapsed = time.time() - t0
    print(f"\n=== Done ===", flush=True)
    print(f"  {n_total:,} embeddings saved to {emb_path} ({emb_size:.2f} GB)", flush=True)
    print(f"  Work IDs saved to {ids_path}", flush=True)
    print(f"  Total time: {elapsed / 3600:.1f}h ({elapsed:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Compute SPECTER2 embeddings")
    ap.add_argument(
        "--abstracts-path",
        default="data/abstracts_for_embedding.parquet",
        help="Path to abstracts parquet",
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed/embeddings",
        help="Output directory for embeddings",
    )
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Encoding batch size (default 512 for A100)")
    ap.add_argument("--checkpoint-every", type=int, default=50_000,
                    help="Save checkpoint every N papers")
    ap.add_argument("--model-cache", default=None,
                    help="Local HuggingFace cache dir (for HPC nodes without internet)")
    args = ap.parse_args()

    compute_embeddings(
        abstracts_path=args.abstracts_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        model_cache=args.model_cache,
    )


if __name__ == "__main__":
    main()
