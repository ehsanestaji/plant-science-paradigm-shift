"""
Analysis D: Sleeping Beauty Crops.

Apply the Ke et al. (2015) sleeping beauty metric to papers about
underutilised/minor crops (teff, amaranth, quinoa, fonio, millet, sorghum,
cassava, moringa).  For the top sleeping beauties, identify the "prince"
papers (citing papers in the awakening year) and what triggered the awakening.

Beauty score formula (Ke 2015):
    B = max over t in [0, t_awaken] of:
        [c(t) - c(t_awaken)*t/t_awaken]^2 / c(t_awaken)
where c(t) = cumulative citations at age t, t_awaken = year of peak annual rate.

Output → results/novel/
  sleeping_beauty_scores.csv
  sleeping_beauty_timeseries.csv
  awakening_triggers.csv

Usage:
    python -m src.novel.sleeping_beauty_crops --db-path data/processed/plant_science.duckdb
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

CROPS = {
    "teff":     ["%teff%"],
    "amaranth": ["%amaranth%"],
    "quinoa":   ["%quinoa%"],
    "fonio":    ["%fonio%"],
    "moringa":  ["%moringa%"],
    "millet":   ["%millet%"],
    "sorghum":  ["%sorghum%"],
    "cassava":  ["%cassava%"],
}

MIN_TOTAL_CITATIONS = 10    # ignore papers with < 10 citations total
MIN_PUB_YEAR = 1970         # publication must be at least 1970
MAX_PUB_YEAR = 2015         # need ≥ 9 post-pub years for meaningful curve
TOP_PER_CROP = 10           # top sleeping beauties to report per crop
TOP_TRIGGERS = 10           # trigger papers to find per sleeping beauty


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


# ── SQL ──────────────────────────────────────────────────────────────────────

def fetch_crop_papers(con) -> pd.DataFrame:
    print("Finding crop papers by title keyword search…", flush=True)
    rows = []
    for crop, patterns in CROPS.items():
        where = " OR ".join(f"lower(w.title) LIKE '{p}'" for p in patterns)
        df = con.execute(f"""
            SELECT w.work_id, w.year AS pub_year, w.title, w.cited_by_count,
                   '{crop}' AS crop
            FROM works_clean w
            WHERE ({where})
              AND w.year BETWEEN {MIN_PUB_YEAR} AND {MAX_PUB_YEAR}
              AND w.cited_by_count >= {MIN_TOTAL_CITATIONS}
        """).df()
        print(f"  {crop}: {len(df):,} papers", flush=True)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True).drop_duplicates("work_id")
    print(f"  Total crop papers: {len(all_df):,}", flush=True)
    return all_df


def fetch_citation_timeseries(con, work_ids: list) -> pd.DataFrame:
    """For each focal paper, get annual citation counts (by citing paper year)."""
    print(f"Fetching citation time series for {len(work_ids):,} crop papers…", flush=True)
    # Register work_ids as temp table for efficient join
    ids_df = pd.DataFrame({"work_id": work_ids})
    con.register("focal_ids", ids_df)
    df = con.execute("""
        SELECT
            cp.work_id  AS focal_id,
            w_c.year    AS citation_year,
            COUNT(*)    AS n_new_citations
        FROM focal_ids cp
        JOIN citations cit   ON cit.cited_work_id  = cp.work_id
        JOIN works_clean w_c ON cit.citing_work_id = w_c.work_id
        GROUP BY cp.work_id, w_c.year
        ORDER BY cp.work_id, w_c.year
    """).df()
    con.unregister("focal_ids")
    print(f"  {len(df):,} time-series rows", flush=True)
    return df


# ── Beauty score ─────────────────────────────────────────────────────────────

def beauty_score(pub_year: int, ts: pd.DataFrame) -> dict:
    """
    ts: DataFrame with columns [citation_year, n_new_citations] for ONE focal paper.
    Returns dict with B, awakening_year, sleep_years, c_at_awakening.
    """
    if len(ts) < 3:
        return {"B": 0.0, "awakening_year": None, "sleep_years": 0, "c_at_awakening": 0}

    max_year = int(ts["citation_year"].max())
    years = np.arange(pub_year, max_year + 1)
    annual = (ts.set_index("citation_year")["n_new_citations"]
                .reindex(years, fill_value=0).values.astype(float))
    cumulative = np.cumsum(annual)

    # t_awaken = index of max annual citations
    awaken_idx = int(np.argmax(annual))
    awaken_year = int(years[awaken_idx])
    c_awaken = float(cumulative[awaken_idx])

    if c_awaken < 1:
        return {"B": 0.0, "awakening_year": awaken_year,
                "sleep_years": awaken_year - pub_year, "c_at_awakening": 0}

    # Ke formula over [0, t_awaken]
    t_max = awaken_idx
    if t_max == 0:
        return {"B": 0.0, "awakening_year": awaken_year,
                "sleep_years": 0, "c_at_awakening": int(c_awaken)}

    t_vals = np.arange(0, t_max + 1, dtype=float)
    c_t = cumulative[:t_max + 1]
    linear = c_awaken * t_vals / t_max
    deviations = (c_t - linear) ** 2 / c_awaken
    B = float(deviations.max())

    return {
        "B": round(B, 2),
        "awakening_year": awaken_year,
        "sleep_years": awaken_year - pub_year,
        "c_at_awakening": int(c_awaken),
    }


def compute_all_scores(crop_papers: pd.DataFrame,
                       ts_df: pd.DataFrame) -> pd.DataFrame:
    print("Computing beauty scores…", flush=True)
    # Build a dict: work_id → timeseries DataFrame
    ts_map = {wid: grp for wid, grp in ts_df.groupby("focal_id")}

    records = []
    for _, row in crop_papers.iterrows():
        wid = row["work_id"]
        ts = ts_map.get(wid, pd.DataFrame(columns=["citation_year", "n_new_citations"]))
        score = beauty_score(int(row["pub_year"]), ts)
        records.append({
            "work_id": wid,
            "pub_year": int(row["pub_year"]),
            "crop": row["crop"],
            "title": str(row["title"])[:150],
            "total_citations": int(row["cited_by_count"]),
            **score,
        })
    df = pd.DataFrame(records).sort_values("B", ascending=False)
    return df


# ── Awakening triggers ────────────────────────────────────────────────────────

def find_triggers(con, top_beauties: pd.DataFrame) -> pd.DataFrame:
    if len(top_beauties) == 0:
        return pd.DataFrame()
    print(f"Finding awakening triggers for {len(top_beauties)} sleeping beauties…",
          flush=True)
    records = []
    for _, sb in top_beauties.iterrows():
        if sb["awakening_year"] is None:
            continue
        df = con.execute(f"""
            SELECT
                cit.citing_work_id AS trigger_id,
                w.title            AS trigger_title,
                w.year             AS trigger_year,
                w.cited_by_count   AS trigger_impact,
                STRING_AGG(DISTINCT c.concept_name, ' | ')  AS trigger_concepts
            FROM citations cit
            JOIN works_clean w ON cit.citing_work_id = w.work_id
            LEFT JOIN concepts c ON c.work_id = w.work_id AND c.level = 0
            WHERE cit.cited_work_id = '{sb["work_id"]}'
              AND w.year BETWEEN {sb["awakening_year"] - 1} AND {sb["awakening_year"] + 1}
            GROUP BY cit.citing_work_id, w.title, w.year, w.cited_by_count
            ORDER BY w.cited_by_count DESC
            LIMIT {TOP_TRIGGERS}
        """).df()
        df["sleeping_beauty_id"] = sb["work_id"]
        df["sb_crop"] = sb["crop"]
        df["sb_title"] = sb["title"][:80]
        records.append(df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    crop_papers = fetch_crop_papers(con)
    if len(crop_papers) == 0:
        print("No crop papers found — check title keywords.", flush=True)
        con.close()
        return

    ts_df = fetch_citation_timeseries(con, crop_papers["work_id"].tolist())

    # Scores
    scores_df = compute_all_scores(crop_papers, ts_df)
    scores_df.to_csv(f"{OUT_DIR}/sleeping_beauty_scores.csv", index=False)

    # Timeseries for top beauties (for figure)
    top_ids = scores_df.head(TOP_PER_CROP * len(CROPS))["work_id"].tolist()
    ts_top = ts_df[ts_df["focal_id"].isin(top_ids)].copy()
    # Merge pub_year for age calculation
    ts_top = ts_top.merge(scores_df[["work_id", "pub_year", "crop"]],
                          left_on="focal_id", right_on="work_id", how="left")
    ts_top["paper_age"] = ts_top["citation_year"] - ts_top["pub_year"]
    ts_top.to_csv(f"{OUT_DIR}/sleeping_beauty_timeseries.csv", index=False)

    # Top per crop
    print("\n=== Top Sleeping Beauties per Crop ===")
    for crop in CROPS:
        top = scores_df[scores_df["crop"] == crop].head(3)
        for _, r in top.iterrows():
            print(f"  [{crop}] B={r['B']:.0f}  "
                  f"slept={r['sleep_years']}yr  "
                  f"awoke={r['awakening_year']}  "
                  f"\"{r['title'][:70]}\"")

    # Triggers
    top_beauties = scores_df.groupby("crop").head(5).reset_index(drop=True)
    triggers_df = find_triggers(con, top_beauties)
    if len(triggers_df):
        triggers_df.to_csv(f"{OUT_DIR}/awakening_triggers.csv", index=False)

    con.close()
    elapsed = int(time.time() - t0)
    print(f"\nAnalysis D complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
