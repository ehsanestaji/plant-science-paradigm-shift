"""
Cross-field & model-organism comparison via the OpenAlex API.

Produces CSVs in results/temporal/ for:
  - cross_field_growth.csv       : publication counts per year (1980-2024)
                                   for 5 scientific fields
  - cross_field_model_organisms.csv : publication counts per year (2000-2024)
                                      for 5 model organisms in plant / life science

No local database is required — data come directly from the OpenAlex API.

Usage:
    python -m src.biblio.cross_field_comparison
"""

import argparse
import sys
import os
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = "results/temporal"

OPENALEX_BASE = "https://api.openalex.org/works"

# OpenAlex concept IDs for scientific fields
FIELDS = {
    "plant_science":      "C59822182",
    "biology":            "C86803240",
    "medicine":           "C71924100",
    "chemistry":          "C185592680",
    "agricultural_science": "C118552586",
}

# OpenAlex concept IDs for model organisms
MODEL_ORGANISMS = {
    "mouse":       "C70994564",
    "zebrafish":   "C14723769",
    "drosophila":  "C507984",
    "c_elegans":   "C2777823",
    "arabidopsis": "C184235292",
}

FIELD_YEAR_RANGE    = range(1980, 2025)   # 1980–2024 inclusive
ORGANISM_YEAR_RANGE = range(2000, 2025)   # 2000–2024 inclusive

RATE_LIMIT_SLEEP = 0.1   # seconds between API requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def fetch_count(concept_id: str, year: int) -> int:
    """Return the number of works in OpenAlex matching concept + year."""
    params = {
        "filter": f"concepts.id:{concept_id},publication_year:{year}",
        "per_page": 1,
    }
    resp = requests.get(OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["meta"]["count"]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def cross_field_growth() -> pd.DataFrame:
    """
    Query publication counts per year (1980-2024) for each scientific field.

    Returns a long-format DataFrame with columns: field, year, n_papers.
    """
    print("Cross-field growth (1980-2024)...", flush=True)
    rows = []
    total_requests = len(FIELDS) * len(FIELD_YEAR_RANGE)
    done = 0

    for field_name, concept_id in FIELDS.items():
        print(f"  Field: {field_name}", flush=True)
        for year in FIELD_YEAR_RANGE:
            count = fetch_count(concept_id, year)
            rows.append({"field": field_name, "year": year, "n_papers": count})
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total_requests} requests done", flush=True)
            time.sleep(RATE_LIMIT_SLEEP)

    df = pd.DataFrame(rows)
    out_path = f"{OUT_DIR}/cross_field_growth.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}", flush=True)
    return df


def model_organism_comparison() -> pd.DataFrame:
    """
    Query publication counts per year (2000-2024) for each model organism.

    Returns a long-format DataFrame with columns: organism, year, n_papers.
    """
    print("Model organism comparison (2000-2024)...", flush=True)
    rows = []
    total_requests = len(MODEL_ORGANISMS) * len(ORGANISM_YEAR_RANGE)
    done = 0

    for organism_name, concept_id in MODEL_ORGANISMS.items():
        print(f"  Organism: {organism_name}", flush=True)
        for year in ORGANISM_YEAR_RANGE:
            count = fetch_count(concept_id, year)
            rows.append({"organism": organism_name, "year": year, "n_papers": count})
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total_requests} requests done", flush=True)
            time.sleep(RATE_LIMIT_SLEEP)

    df = pd.DataFrame(rows)
    out_path = f"{OUT_DIR}/cross_field_model_organisms.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fetch cross-field and model-organism publication counts from OpenAlex."
    )
    ap.add_argument(
        "--skip-fields",
        action="store_true",
        help="Skip the cross-field growth query (useful for re-running only organisms).",
    )
    ap.add_argument(
        "--skip-organisms",
        action="store_true",
        help="Skip the model-organism query (useful for re-running only fields).",
    )
    args = ap.parse_args()

    _ensure_dirs()
    t0 = time.time()

    if not args.skip_fields:
        cross_field_growth()

    if not args.skip_organisms:
        model_organism_comparison()

    elapsed = int(time.time() - t0)
    print(f"\nCross-field comparison complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
