"""
Classification for Paper A: organism type and research paradigm.

Strategy:
  1. Organism: keyword/regex matching on title+abstract (fast, accurate for named species)
  2. Paradigm: zero-shot on 50k sample → logistic regression on SPECTER2 embeddings → predict all

Usage:
    python -m src.nlp.classify_paper_a \
        --abstracts-path data/abstracts_for_embedding.parquet \
        --embeddings-path data/processed/embeddings/specter2_embeddings.npy \
        --work-ids-path data/processed/embeddings/specter2_work_ids.npy \
        --out-dir data/processed/classifications \
        --model-cache /proj/nobackup/hpc2n2025-278/models/huggingface
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Organism keyword patterns ────────────────────────────────────────
# Priority order: more specific first. First match wins for single-match papers.
# Multi-match papers get the first (most specific) match.
ORGANISM_PATTERNS = [
    ("arabidopsis", re.compile(r"arabidopsis", re.IGNORECASE)),
    ("rice", re.compile(r"\b(?:rice|oryza\s*sativa)\b", re.IGNORECASE)),
    ("wheat", re.compile(r"\b(?:wheat|triticum)\b", re.IGNORECASE)),
    ("maize", re.compile(r"\b(?:maize|corn|zea\s*mays)\b", re.IGNORECASE)),
    ("soybean", re.compile(r"\b(?:soybean|soy\s*bean|glycine\s*max)\b", re.IGNORECASE)),
    ("tomato", re.compile(r"\b(?:tomato|solanum\s*lycopersicum|lycopersicon)\b", re.IGNORECASE)),
    ("barley", re.compile(r"\b(?:barley|hordeum\s*vulgare)\b", re.IGNORECASE)),
    ("cotton", re.compile(r"\b(?:cotton|gossypium)\b", re.IGNORECASE)),
    ("potato", re.compile(r"\b(?:potato|solanum\s*tuberosum)\b", re.IGNORECASE)),
    ("tobacco", re.compile(r"\b(?:tobacco|nicotiana)\b", re.IGNORECASE)),
    ("other_crop", re.compile(
        r"\b(?:canola|rapeseed|brassica\s*napus|sunflower|helianthus|"
        r"sorghum|millet|pearl\s*millet|finger\s*millet|"
        r"cassava|manihot|yam|dioscorea|"
        r"sugarcane|saccharum|sugar\s*beet|"
        r"peanut|groundnut|arachis|chickpea|cicer|lentil|lens\s*culinaris|"
        r"pigeon\s*pea|cajanus|cowpea|vigna|common\s*bean|phaseolus|"
        r"oil\s*palm|elaeis|coconut|cocos\s*nucifera|"
        r"rubber|hevea|tea\s*plant|camellia\s*sinensis|"
        r"coffee|coffea|cacao|theobroma|"
        r"banana|musa\s+\w|plantain|"
        r"grape|vitis\s*vinifera|apple|malus\s*domestica|citrus|"
        r"strawberry|fragaria|peach|prunus\s*persica|"
        r"pepper|capsicum|cucumber|cucumis\s*sativus|melon|"
        r"lettuce|lactuca|carrot|daucus\s*carota|onion|allium\s*cepa|"
        r"garlic|allium\s*sativum)\b", re.IGNORECASE)),
    ("other_model_organism", re.compile(
        r"\b(?:marchantia|physcomitrella|physcomitrium|"
        r"chlamydomonas|synechocystis|"
        r"brachypodium|medicago\s*truncatula|lotus\s*japonicus|"
        r"populus|poplar|eucalyptus|"
        r"setaria\s*viridis|selaginella)\b", re.IGNORECASE)),
]

PARADIGM_LABELS = [
    "fundamental basic science research",
    "applied translational agricultural research",
]

PARADIGM_SAMPLE_SIZE = 50_000


def classify_organisms(df: pd.DataFrame) -> pd.DataFrame:
    """Classify organisms using keyword/regex matching."""
    print("  Classifying organisms (keyword/regex)...", flush=True)
    t0 = time.time()

    text = (df["title"].fillna("") + " " + df["abstract"].fillna("")).values
    n = len(text)
    labels = np.full(n, "non_specific", dtype=object)
    confidences = np.ones(n, dtype=np.float32)

    # Count matches per paper for confidence
    match_counts = np.zeros(n, dtype=np.int32)
    first_match = np.full(n, "", dtype=object)

    for organism, pattern in ORGANISM_PATTERNS:
        for i in range(n):
            if pattern.search(text[i]):
                match_counts[i] += 1
                if first_match[i] == "":
                    first_match[i] = organism

    # Assign labels
    for i in range(n):
        if match_counts[i] == 1:
            labels[i] = first_match[i]
            confidences[i] = 1.0
        elif match_counts[i] > 1:
            labels[i] = first_match[i]
            confidences[i] = 0.7  # lower confidence for multi-match

    elapsed = time.time() - t0
    result = pd.DataFrame({
        "work_id": df["work_id"].values,
        "predicted_label": labels,
        "confidence": confidences,
    })

    # Print summary
    counts = result["predicted_label"].value_counts()
    print(f"  Done in {elapsed:.1f}s ({n/elapsed:.0f} docs/s)", flush=True)
    print(f"  Distribution:", flush=True)
    for label, count in counts.items():
        print(f"    {label}: {count:,} ({100*count/n:.1f}%)", flush=True)

    return result


def classify_paradigms(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    emb_work_ids: np.ndarray,
    model_cache: str | None = None,
    device: int = 0,
) -> pd.DataFrame:
    """Classify paradigm using zero-shot on sample + logistic regression on embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n_total = len(embeddings)
    print(f"\n  Classifying paradigms ({n_total:,} papers)...", flush=True)
    print(f"  Step 1: Zero-shot classify {PARADIGM_SAMPLE_SIZE:,} sample...", flush=True)

    # ── Step 1: Zero-shot on sample ───────────────────────────────────
    from transformers import pipeline
    import torch

    if model_cache:
        os.environ["HF_HOME"] = model_cache
        os.environ["TRANSFORMERS_CACHE"] = model_cache

    # Check if GPU available
    use_gpu = torch.cuda.is_available()
    dev = device if use_gpu else -1
    print(f"  Device: {'GPU' if use_gpu else 'CPU'}", flush=True)

    classifier = pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-large",
        device=dev,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        model_kwargs={"cache_dir": model_cache} if model_cache else {},
    )

    # Build text lookup from parquet
    text_lookup = {}
    for _, row in df.iterrows():
        text_lookup[row["work_id"]] = (
            (row["title"] or "") + ". " + (row["abstract"] or "")
        )[:1000]  # truncate for speed

    # Random sample
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, size=min(PARADIGM_SAMPLE_SIZE, n_total), replace=False)
    sample_ids = emb_work_ids[sample_idx]
    sample_texts = [text_lookup.get(wid, "") for wid in sample_ids]

    # Classify sample in batches
    t0 = time.time()
    sample_labels = []
    sample_scores = []
    batch_size = 32
    for i in range(0, len(sample_texts), batch_size):
        batch = sample_texts[i:i + batch_size]
        outputs = classifier(batch, candidate_labels=PARADIGM_LABELS, batch_size=batch_size)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            sample_labels.append(out["labels"][0])
            sample_scores.append(out["scores"][0])
        done = min(i + batch_size, len(sample_texts))
        if done % 5000 < batch_size:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(sample_texts) - done) / rate if rate > 0 else 0
            print(f"    {done:,}/{len(sample_texts):,} ({100*done/len(sample_texts):.0f}%) "
                  f"| {rate:.1f} docs/s | ETA {eta/60:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"  Sample classified in {elapsed/60:.1f}min ({len(sample_texts)/elapsed:.1f} docs/s)",
          flush=True)

    # ── Step 2: Train logistic regression on embeddings ───────────────
    print(f"  Step 2: Training logistic regression on SPECTER2 embeddings...", flush=True)
    t1 = time.time()

    # Encode labels as 0/1
    label_to_int = {PARADIGM_LABELS[0]: 0, PARADIGM_LABELS[1]: 1}
    y_train = np.array([label_to_int[l] for l in sample_labels])
    X_train = embeddings[sample_idx].astype(np.float32)

    # Train with cross-validation
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"    5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})", flush=True)

    # Train on full sample
    clf.fit(X_train, y_train)
    elapsed2 = time.time() - t1
    print(f"    Trained in {elapsed2:.1f}s", flush=True)

    # ── Step 3: Predict all papers ────────────────────────────────────
    print(f"  Step 3: Predicting all {n_total:,} papers...", flush=True)
    t2 = time.time()

    # Predict in chunks to manage memory
    chunk_size = 100_000
    all_labels = []
    all_probs = []
    for i in range(0, n_total, chunk_size):
        X_chunk = embeddings[i:i + chunk_size].astype(np.float32)
        probs = clf.predict_proba(X_chunk)
        preds = clf.predict(X_chunk)
        all_labels.extend(preds)
        all_probs.extend(probs.max(axis=1))
        if (i + chunk_size) % 500_000 < chunk_size:
            print(f"    {min(i+chunk_size, n_total):,}/{n_total:,}", flush=True)

    elapsed3 = time.time() - t2
    print(f"    Predicted in {elapsed3:.1f}s ({n_total/elapsed3:.0f} docs/s)", flush=True)

    # Map back to string labels
    int_to_label = {0: PARADIGM_LABELS[0], 1: PARADIGM_LABELS[1]}
    result = pd.DataFrame({
        "work_id": emb_work_ids,
        "predicted_label": [int_to_label[l] for l in all_labels],
        "confidence": np.round(all_probs, 4),
    })

    counts = result["predicted_label"].value_counts()
    print(f"  Distribution:", flush=True)
    for label, count in counts.items():
        print(f"    {label}: {count:,} ({100*count/n_total:.1f}%)", flush=True)

    return result


def main():
    ap = argparse.ArgumentParser(description="Paper A: organism + paradigm classification")
    ap.add_argument("--abstracts-path", default="data/abstracts_for_embedding.parquet")
    ap.add_argument("--embeddings-path", default="data/processed/embeddings/specter2_embeddings.npy")
    ap.add_argument("--work-ids-path", default="data/processed/embeddings/specter2_work_ids.npy")
    ap.add_argument("--out-dir", default="data/processed/classifications")
    ap.add_argument("--model-cache", default=None)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    print("=== Paper A Classification Pipeline ===", flush=True)
    t0 = time.time()

    # Load data
    print(f"Loading abstracts from {args.abstracts_path}...", flush=True)
    df = pd.read_parquet(args.abstracts_path, columns=["work_id", "title", "abstract"])
    print(f"  {len(df):,} abstracts loaded", flush=True)

    print(f"Loading embeddings from {args.embeddings_path}...", flush=True)
    embeddings = np.load(args.embeddings_path)
    emb_work_ids = np.load(args.work_ids_path, allow_pickle=True)
    print(f"  {len(embeddings):,} embeddings loaded ({embeddings.shape[1]}D)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Task 1: Organisms (keyword) ──────────────────────────────────
    organism_path = os.path.join(args.out_dir, "paper_a_organism.csv")
    if os.path.exists(organism_path):
        print(f"\nSKIP: {organism_path} already exists", flush=True)
    else:
        result = classify_organisms(df)
        result.to_csv(organism_path, index=False)
        print(f"  Saved {organism_path}", flush=True)

    # ── Task 2: Paradigm (zero-shot sample + logreg on embeddings) ───
    paradigm_path = os.path.join(args.out_dir, "paper_a_paradigm.csv")
    if os.path.exists(paradigm_path):
        print(f"\nSKIP: {paradigm_path} already exists", flush=True)
    else:
        result = classify_paradigms(
            df, embeddings, emb_work_ids,
            model_cache=args.model_cache,
            device=args.device,
        )
        result.to_csv(paradigm_path, index=False)
        print(f"  Saved {paradigm_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== Done ({elapsed/60:.1f}min) ===", flush=True)


if __name__ == "__main__":
    main()
