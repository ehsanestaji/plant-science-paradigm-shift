"""
Analysis G: Dark Matter / Orphan Organisms.

Which organisms are massively under-researched relative to their agricultural
importance? Compares research attention (paper counts) to real-world importance
(FAO production tonnage, caloric contribution) to quantify the "research
attention gap" for neglected crops and model organisms.

Output → results/novel/
  organism_paper_counts.csv
  research_attention_gap.csv

Usage:
    python -m src.novel.orphan_organisms --db-path data/processed/plant_science.duckdb
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

# ── Organism definitions ─────────────────────────────────────────────────────
# Keywords for title + abstract matching (LIKE patterns)
ORGANISMS = {
    # Major staples
    "rice":        ["%oryza sativa%", "% rice %", "%rice %"],
    "wheat":       ["%triticum%", "% wheat %", "%wheat %"],
    "maize":       ["%zea mays%", "% maize %", "% corn %"],
    "soybean":     ["%glycine max%", "% soybean%"],
    "potato":      ["%solanum tuberosum%", "% potato%"],
    "tomato":      ["%solanum lycopersicum%", "% tomato%"],
    # Model organisms
    "arabidopsis": ["%arabidopsis%"],
    "tobacco":     ["%nicotiana%", "% tobacco %"],
    # Medium-importance crops
    "cassava":     ["%manihot%", "% cassava%"],
    "sorghum":     ["%sorghum%"],
    "millet":      ["%millet%", "%pennisetum%", "%eleusine%"],
    "chickpea":    ["%cicer arietinum%", "% chickpea%"],
    "lentil":      ["%lens culinaris%", "% lentil%"],
    "cowpea":      ["%vigna unguiculata%", "% cowpea%"],
    "yam":         ["%dioscorea%", "% yam %"],
    "plantain":    ["%plantain%", "% banana%", "%musa %"],
    "barley":      ["%hordeum%", "% barley%"],
    "sugarcane":   ["%saccharum%", "% sugarcane%"],
    # Minor / underutilised
    "teff":        ["%teff%", "%eragrostis tef%"],
    "quinoa":      ["%quinoa%", "%chenopodium quinoa%"],
    "fonio":       ["%fonio%", "%digitaria exilis%"],
    "moringa":     ["%moringa%"],
    # Perennials
    "oil palm":    ["%oil palm%", "%elaeis%"],
    "cocoa":       ["%theobroma%", "% cocoa%", "% cacao%"],
    "coffee":      ["%coffea%", "% coffee %"],
    "tea":         ["%camellia sinensis%"],
    "rubber":      ["%hevea%", "%rubber tree%"],
}

# FAO STAT approximate 2022 values (million tonnes)
FAO_PRODUCTION_MT = {
    "rice": 520, "wheat": 780, "maize": 1160, "soybean": 350,
    "potato": 375, "tomato": 187, "cassava": 310, "sorghum": 60,
    "millet": 30, "yam": 75, "plantain": 120, "teff": 6,
    "chickpea": 15, "lentil": 6.3, "cowpea": 9, "quinoa": 0.18,
    "fonio": 0.7, "moringa": 1.5, "oil palm": 400, "cocoa": 5.9,
    "coffee": 10.8, "tea": 7, "rubber": 14, "barley": 155,
    "sugarcane": 1900, "tobacco": 6.3,
    # arabidopsis: no production (model only)
}

# Approximate kcal per kg (dry-weight basis where applicable)
FAO_KCAL_PER_KG = {
    "rice": 1300, "wheat": 3400, "maize": 3650, "soybean": 4460,
    "potato": 770, "tomato": 180, "cassava": 1600, "sorghum": 3390,
    "millet": 3780, "yam": 1180, "plantain": 890, "teff": 3540,
    "chickpea": 3640, "lentil": 3530, "cowpea": 3360, "quinoa": 3680,
    "fonio": 3560, "barley": 3540, "sugarcane": 400,
}


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def count_organism_papers(con) -> pd.DataFrame:
    """Count papers per organism via title + abstract keyword matching."""
    print("Counting papers per organism…", flush=True)
    records = []
    for org, patterns in ORGANISMS.items():
        title_where = " OR ".join(f"lower(title) LIKE '{p}'" for p in patterns)
        abstract_where = " OR ".join(f"lower(abstract) LIKE '{p}'" for p in patterns)
        row = con.execute(f"""
            SELECT COUNT(*) AS n_papers,
                   MIN(year) AS first_year,
                   MAX(year) AS last_year
            FROM works_clean
            WHERE ({title_where})
               OR (abstract IS NOT NULL AND ({abstract_where}))
        """).fetchone()
        n_papers, first_year, last_year = row
        print(f"  {org}: {n_papers:,} papers ({first_year}–{last_year})", flush=True)
        records.append({
            "organism": org,
            "n_papers": n_papers,
            "first_year": first_year,
            "last_year": last_year,
        })
    return pd.DataFrame(records)


def compute_attention_gap(paper_counts: pd.DataFrame) -> pd.DataFrame:
    """Compute research attention gap relative to production importance."""
    df = paper_counts.copy()

    # Add FAO data
    df["production_mt"] = df["organism"].map(FAO_PRODUCTION_MT)
    df["kcal_per_kg"] = df["organism"].map(FAO_KCAL_PER_KG)

    # Papers per million tonnes
    mask = df["production_mt"].notna() & (df["production_mt"] > 0)
    df.loc[mask, "papers_per_mt"] = df.loc[mask, "n_papers"] / df.loc[mask, "production_mt"]

    # Caloric contribution (trillion kcal = MT * 1e6 * kcal/kg / 1e12)
    mask2 = mask & df["kcal_per_kg"].notna()
    df.loc[mask2, "caloric_trillion_kcal"] = (
        df.loc[mask2, "production_mt"] * 1e6 * df.loc[mask2, "kcal_per_kg"] / 1e12)
    cal_mask = df["caloric_trillion_kcal"].notna() & (df["caloric_trillion_kcal"] > 0)
    df.loc[cal_mask, "papers_per_trillion_kcal"] = (
        df.loc[cal_mask, "n_papers"] / df.loc[cal_mask, "caloric_trillion_kcal"])

    # Attention gap: log2 ratio vs median papers_per_mt
    valid = df[df["papers_per_mt"].notna()]
    if len(valid) > 0:
        median_ppm = valid["papers_per_mt"].median()
        df.loc[mask, "attention_gap_log2"] = np.log2(
            df.loc[mask, "papers_per_mt"] / median_ppm)

    return df.sort_values("n_papers", ascending=False)


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

    paper_counts = count_organism_papers(con)
    paper_counts.to_csv(f"{OUT_DIR}/organism_paper_counts.csv", index=False)

    gap_df = compute_attention_gap(paper_counts)
    gap_df.to_csv(f"{OUT_DIR}/research_attention_gap.csv", index=False)

    con.close()

    # Summary
    print("\n=== Research Attention Gap ===")
    print(f"{'Organism':<15} {'Papers':>10} {'Prod(MT)':>10} {'Pap/MT':>10} {'Gap(log2)':>10}")
    print("-" * 60)
    for _, r in gap_df.iterrows():
        prod = f"{r['production_mt']:.0f}" if pd.notna(r.get('production_mt')) else "N/A"
        ppm = f"{r['papers_per_mt']:.1f}" if pd.notna(r.get('papers_per_mt')) else "N/A"
        gap = f"{r['attention_gap_log2']:+.2f}" if pd.notna(r.get('attention_gap_log2')) else "N/A"
        print(f"  {r['organism']:<13} {r['n_papers']:>10,} {prod:>10} {ppm:>10} {gap:>10}")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis G complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
