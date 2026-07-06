"""
Analysis A: The Disruption Index (CD Index).

For each focal paper f, and for all papers p published AFTER f:
  D = papers p that cite f but do NOT cite any of f's references
  C = papers p that cite f AND cite at least one of f's references
  N = papers p that cite f's references but NOT f itself
  CD = (D - C) / (D + C + N)

CD → +1 : highly disruptive (replaced prior knowledge)
CD → -1 : highly consolidating (extended prior knowledge)
CD =  0 : neutral

Computed for the top-50K most-cited papers published ≤ 2018.
Processed in batches of 5,000 with checkpointing.

Output → results/novel/
  cd_index.csv             (per paper: work_id, year, D, C, N, cd_index)
  cd_by_year.csv           (mean CD per year)
  cd_top_disruptors.csv    (top 100 most disruptive)
  cd_top_consolidators.csv (top 100 most consolidating)

Usage:
    python -m src.novel.disruption_index --db-path data/processed/plant_science.duckdb
    python -m src.novel.disruption_index --db-path ... --batch-size 2000 --n-focal 20000
"""

import argparse
import sys
import time
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pickle

from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/novel"
CHECKPOINT_FILE = "results/novel/cd_checkpoint.pkl"


def _cp_save(completed_dfs, next_batch):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({"completed": completed_dfs, "next_batch": next_batch}, f)


def _cp_load():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    return None


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = "/pfs/proj/nobackup/fs/projnb10/hpc2n2025-278/tmp"
    os.makedirs(tmp, exist_ok=True)
    return tmp


# ── Focal sample ──────────────────────────────────────────────────────────────

def load_focal_sample(con, n_focal: int) -> pd.DataFrame:
    print(f"Loading focal sample (top {n_focal:,} cited papers ≤ 2018)…", flush=True)
    df = con.execute(f"""
        SELECT w.work_id, w.year, w.title, w.cited_by_count,
               w.journal_name, w.oa_status
        FROM works_clean w
        WHERE w.cited_by_count IS NOT NULL
          AND w.year IS NOT NULL
          AND w.year <= 2018
        ORDER BY w.cited_by_count DESC
        LIMIT {n_focal}
    """).df()
    print(f"  {len(df):,} focal papers, "
          f"citations range: {df['cited_by_count'].min()}–{df['cited_by_count'].max()}",
          flush=True)
    return df


# ── Batch CD computation ──────────────────────────────────────────────────────

CD_SQL = """
WITH
-- Reference set: papers that focal papers cite
focal_refs AS (
    SELECT c.citing_work_id AS focal_id,
           c.cited_work_id  AS ref_id
    FROM citations c
    WHERE c.citing_work_id IN (SELECT work_id FROM batch_focal)
),
-- Citing set: papers that cite each focal paper (published AFTER focal)
focal_citers AS (
    SELECT c.cited_work_id  AS focal_id,
           c.citing_work_id AS citer_id,
           w.year           AS citer_year
    FROM citations c
    JOIN works_clean w ON c.citing_work_id = w.work_id
    JOIN batch_focal bf ON c.cited_work_id = bf.work_id
    WHERE w.year > bf.year
),
-- For each (focal, citer): does citer overlap with focal's refs?
citer_ref_overlap AS (
    SELECT DISTINCT fc.focal_id, fc.citer_id
    FROM focal_citers fc
    WHERE EXISTS (
        SELECT 1
        FROM focal_refs fr
        JOIN citations c2 ON c2.citing_work_id = fc.citer_id
                          AND c2.cited_work_id  = fr.ref_id
        WHERE fr.focal_id = fc.focal_id
    )
),
-- Papers that cite at least one ref of focal (but not necessarily focal itself)
ref_citers AS (
    SELECT DISTINCT fr.focal_id,
                    c3.citing_work_id AS citer_id
    FROM focal_refs fr
    JOIN citations c3 ON c3.cited_work_id = fr.ref_id
    JOIN works_clean w3 ON c3.citing_work_id = w3.work_id
    JOIN batch_focal bf ON fr.focal_id = bf.work_id
    WHERE w3.year > bf.year
),
D_tbl AS (
    SELECT fc.focal_id, COUNT(*) AS D
    FROM focal_citers fc
    WHERE NOT EXISTS (
        SELECT 1 FROM citer_ref_overlap cro
        WHERE cro.focal_id = fc.focal_id AND cro.citer_id = fc.citer_id
    )
    GROUP BY fc.focal_id
),
C_tbl AS (
    SELECT focal_id, COUNT(*) AS C
    FROM citer_ref_overlap
    GROUP BY focal_id
),
N_tbl AS (
    SELECT rc.focal_id, COUNT(*) AS N
    FROM ref_citers rc
    WHERE NOT EXISTS (
        SELECT 1 FROM focal_citers fc
        WHERE fc.focal_id = rc.focal_id AND fc.citer_id = rc.citer_id
    )
    GROUP BY rc.focal_id
)
SELECT
    bf.work_id,
    bf.year,
    COALESCE(d.D, 0) AS D,
    COALESCE(c.C, 0) AS C,
    COALESCE(n.N, 0) AS N,
    CASE WHEN (COALESCE(d.D,0)+COALESCE(c.C,0)+COALESCE(n.N,0)) = 0 THEN NULL
         ELSE (COALESCE(d.D,0) - COALESCE(c.C,0)) * 1.0
              / (COALESCE(d.D,0) + COALESCE(c.C,0) + COALESCE(n.N,0))
    END AS cd_index
FROM batch_focal bf
LEFT JOIN D_tbl d ON d.focal_id = bf.work_id
LEFT JOIN C_tbl c ON c.focal_id = bf.work_id
LEFT JOIN N_tbl n ON n.focal_id = bf.work_id
"""


def compute_cd_batch(con, batch_df: pd.DataFrame) -> pd.DataFrame:
    con.register("batch_focal", batch_df[["work_id", "year"]])
    result = con.execute(CD_SQL).df()
    con.unregister("batch_focal")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--n-focal", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint if available")
    args = ap.parse_args()

    check_storage()
    tmp = _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='120GB'")
    con.execute("SET threads=8")
    con.execute(f"SET temp_directory='{tmp}'")

    focal = load_focal_sample(con, args.n_focal)

    # Resume from checkpoint
    completed_batches = []
    start_batch = 0
    if args.resume:
        cp = _cp_load()
        if cp is not None:
            completed_batches = cp.get("completed", [])
            start_batch = cp.get("next_batch", 0)
            print(f"Resuming from batch {start_batch} "
                  f"({len(completed_batches)} batches already done)", flush=True)

    # Process batches
    batch_size = args.batch_size
    batches = [focal.iloc[i:i+batch_size]
               for i in range(0, len(focal), batch_size)]
    total_batches = len(batches)

    all_results = completed_batches.copy()

    for idx, batch_df in enumerate(batches[start_batch:], start=start_batch):
        t_batch = time.time()
        print(f"\nBatch {idx+1}/{total_batches} "
              f"({len(batch_df)} papers)…", flush=True)
        try:
            batch_result = compute_cd_batch(con, batch_df)
            all_results.append(batch_result)
            elapsed_b = int(time.time() - t_batch)
            print(f"  Done in {elapsed_b}s. "
                  f"Mean CD={batch_result['cd_index'].mean():.3f}", flush=True)
            _cp_save(all_results, idx + 1)
        except Exception as e:
            print(f"  ERROR in batch {idx+1}: {e}", flush=True)
            print("  Saving partial results and continuing…", flush=True)
            continue

    con.close()

    if not all_results:
        print("No results produced.", flush=True)
        return

    # Combine and annotate
    cd_df = pd.concat(all_results, ignore_index=True)
    cd_df = cd_df.merge(
        focal[["work_id", "title", "cited_by_count", "journal_name", "oa_status"]],
        on="work_id", how="left"
    )

    # Save full CD index
    cd_df.to_csv(f"{OUT_DIR}/cd_index.csv", index=False)

    # By-year aggregate
    by_year = (cd_df.dropna(subset=["cd_index"])
                    .groupby("year")
                    .agg(mean_cd=("cd_index", "mean"),
                         median_cd=("cd_index", "median"),
                         n_papers=("work_id", "count"),
                         pct_disruptive=("cd_index", lambda x: (x > 0.5).mean() * 100),
                         pct_consolidating=("cd_index", lambda x: (x < -0.5).mean() * 100))
                    .reset_index())
    by_year.to_csv(f"{OUT_DIR}/cd_by_year.csv", index=False)

    # Top disruptors / consolidators
    (cd_df.dropna(subset=["cd_index"])
          .nlargest(100, "cd_index")
          .to_csv(f"{OUT_DIR}/cd_top_disruptors.csv", index=False))
    (cd_df.dropna(subset=["cd_index"])
          .nsmallest(100, "cd_index")
          .to_csv(f"{OUT_DIR}/cd_top_consolidators.csv", index=False))

    print("\n=== CD Index Summary ===")
    valid = cd_df.dropna(subset=["cd_index"])
    print(f"  Papers with CD score: {len(valid):,}")
    print(f"  Mean CD: {valid['cd_index'].mean():.4f}")
    print(f"  Disruptive (>0.5): {(valid['cd_index']>0.5).sum():,} "
          f"({(valid['cd_index']>0.5).mean()*100:.1f}%)")
    print(f"  Consolidating (<-0.5): {(valid['cd_index']<-0.5).sum():,} "
          f"({(valid['cd_index']<-0.5).mean()*100:.1f}%)")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis A complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
