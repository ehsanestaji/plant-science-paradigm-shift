"""
G3 — 150-paper BERTopic label review template.

Selects 5 representative papers per macro-theme (30 themes × 5 = 150 papers)
for a domain expert to assess whether the auto-generated macro-theme label is
coherent and accurate.

Usage
-----
python3 -u -m src.analysis.paper_a.hardening.bertopic_review_template \
    --db-path data/processed/plant_science.duckdb
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOPIC_ASSIGNMENTS = PROJECT_ROOT / "results/topics/topic_assignments.csv"
TOPIC_LABELS = PROJECT_ROOT / "results/topics/topic_labels.csv"
MACRO_THEMES = PROJECT_ROOT / "results/paper_a/main/macro_themes.csv"
OUT_CSV = (
    PROJECT_ROOT / "results/paper_a/supplementary/hardening/bertopic_review_template.csv"
)

PAPERS_PER_THEME = 5
SEED = 42


def parse_constituent_ids(val):
    """Parse semicolon-separated topic IDs from macro_themes.csv."""
    if pd.isna(val):
        return []
    return [int(x) for x in str(val).split(";") if x.strip().lstrip("-").isdigit()]


def select_papers_for_theme(
    theme_topics: list,
    assignments: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Select n papers with highest topic_prob for any constituent topic,
    with year diversity (max 2 from any 5-year window).
    """
    pool = assignments[assignments["topic_id"].isin(theme_topics)].copy()
    if pool.empty:
        return pd.DataFrame()

    pool = pool.sort_values("topic_prob", ascending=False)

    # Year diversity: bin into 5-year windows
    pool["year_bin"] = (pool["year"] // 5) * 5
    selected = []
    bin_counts: dict = {}
    for _, row in pool.iterrows():
        yb = row["year_bin"]
        if bin_counts.get(yb, 0) >= 2:
            continue
        selected.append(row)
        bin_counts[yb] = bin_counts.get(yb, 0) + 1
        if len(selected) >= n:
            break

    # If diversity rule yielded fewer than n, top-up without constraint
    if len(selected) < n:
        picked_ids = {r["work_id"] for r in selected}
        for _, row in pool.iterrows():
            if row["work_id"] not in picked_ids:
                selected.append(row)
                picked_ids.add(row["work_id"])
            if len(selected) >= n:
                break

    return pd.DataFrame(selected[:n])


def query_meta(work_ids: list, db_path: str) -> pd.DataFrame:
    import duckdb

    ids_str = ", ".join(f"'{w}'" for w in work_ids)
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        f"""
        SELECT work_id, title, abstract
        FROM works_clean
        WHERE work_id IN ({ids_str})
        """
    ).df()
    con.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate BERTopic review template")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print("Loading topic assignments …")
    assignments = pd.read_csv(
        TOPIC_ASSIGNMENTS,
        dtype={"work_id": str, "topic_id": int},
    )
    print(f"  {len(assignments):,} rows")

    print("Loading macro themes …")
    themes = pd.read_csv(MACRO_THEMES)
    # Drop outlier row (macro_theme_id == -1)
    themes = themes[themes["macro_theme_id"] >= 0].copy()
    print(f"  {len(themes)} macro-themes")

    rows = []
    summary_rows = []

    for _, theme in themes.iterrows():
        theme_id = int(theme["macro_theme_id"])
        theme_name = str(theme["name"])
        theme_docs = int(theme["n_docs"])
        theme_words = str(theme.get("top_words", ""))
        constituent_ids = parse_constituent_ids(theme.get("constituent_topic_ids"))

        selected = select_papers_for_theme(constituent_ids, assignments, PAPERS_PER_THEME, rng)

        summary_rows.append(
            {
                "macro_theme_id": theme_id,
                "name": theme_name,
                "n_docs": theme_docs,
                "n_selected": len(selected),
            }
        )

        if selected.empty:
            continue

        for _, paper in selected.iterrows():
            rows.append(
                {
                    "macro_theme_id": theme_id,
                    "macro_theme_name": theme_name,
                    "macro_theme_top_words": theme_words,
                    "work_id": paper["work_id"],
                    "year": paper.get("year", pd.NA),
                    "title": "",
                    "abstract_snippet": "",
                    "topic_prob": round(float(paper["topic_prob"]), 4),
                    "label_accurate": "",
                    "suggested_label": "",
                    "coherence_score": "",
                    "notes": "",
                }
            )

    # Fetch metadata from DuckDB
    all_work_ids = [r["work_id"] for r in rows]
    print(f"\nQuerying DuckDB for {len(all_work_ids)} papers …")
    meta = query_meta(all_work_ids, args.db_path)
    meta_map = meta.set_index("work_id")

    for r in rows:
        wid = r["work_id"]
        if wid in meta_map.index:
            r["title"] = str(meta_map.loc[wid, "title"] or "")[:150]
            abstract = str(meta_map.loc[wid, "abstract"] or "")
            r["abstract_snippet"] = abstract[:400]

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out):,} rows → {OUT_CSV}")

    print("\n30-theme summary:")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
