"""
Theme 7: Gender Dynamics in Authorship.

Uses the gender_guesser library (no API needed) to infer gender from
first names. Tracks female/male ratio for first and last authors over time.

Limitations:
  - ~40% unclassifiable (East Asian names, initials-only)
  - Binary classification only (limitation of name-based methods)
  - Documented in paper as limitation

Usage:
    python -m src.dynamics.gender_diversity --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/dynamics"


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def _get_gender_detector():
    """Get gender detector, with fallback."""
    try:
        import gender_guesser.detector as gd
        return gd.Detector()
    except ImportError:
        print("  WARNING: gender_guesser not installed. Install with: "
              "pip install gender-guesser", flush=True)
        return None


def _extract_first_name(display_name: str) -> str | None:
    """Extract first name from 'First Last' or 'Last, First' format."""
    if not display_name:
        return None
    name = display_name.strip()
    if "," in name:
        parts = name.split(",", 1)
        after_comma = parts[1].strip().split() if len(parts) > 1 else []
        first = after_comma[0] if after_comma else ""
    else:
        parts = name.split()
        first = parts[0] if parts else ""
    # Skip initials
    if len(first) <= 1 or (len(first) == 2 and first.endswith(".")):
        return None
    return first.capitalize()


def _classify_gender(first_name: str, detector) -> str:
    """Classify gender: 'male', 'female', or 'unknown'."""
    if not first_name or not detector:
        return "unknown"
    result = detector.get_gender(first_name)
    if result in ("male", "mostly_male"):
        return "male"
    elif result in ("female", "mostly_female"):
        return "female"
    return "unknown"


def analyze_gender(con):
    """Gender analysis for first and last authors."""
    print("Fetching first and last authors...", flush=True)

    detector = _get_gender_detector()
    if detector is None:
        return

    # Get first authors (position=1) and last authors (max position per paper)
    print("  Querying first authors...", flush=True)
    first_authors = con.execute("""
        SELECT wa.work_id, a.display_name, w.year
        FROM work_authors wa
        JOIN authors a ON wa.author_id = a.author_id
        JOIN works_clean w ON wa.work_id = w.work_id
        WHERE wa.author_position = 1
          AND w.year >= 1980
          AND a.display_name IS NOT NULL
          AND wa.author_id != 'A9999999999'
    """).df()
    print(f"  {len(first_authors):,} first authors", flush=True)

    print("  Querying last authors...", flush=True)
    last_authors = con.execute("""
        WITH max_pos AS (
            SELECT work_id, MAX(author_position) AS last_pos
            FROM work_authors GROUP BY work_id
        )
        SELECT wa.work_id, a.display_name, w.year
        FROM work_authors wa
        JOIN authors a ON wa.author_id = a.author_id
        JOIN works_clean w ON wa.work_id = w.work_id
        JOIN max_pos mp ON wa.work_id = mp.work_id AND wa.author_position = mp.last_pos
        WHERE w.year >= 1980
          AND a.display_name IS NOT NULL
          AND wa.author_id != 'A9999999999'
          AND mp.last_pos > 1
    """).df()
    print(f"  {len(last_authors):,} last authors", flush=True)

    # Classify genders
    print("  Classifying first author genders...", flush=True)
    first_authors["first_name"] = first_authors["display_name"].apply(_extract_first_name)
    first_authors["gender"] = first_authors["first_name"].apply(
        lambda x: _classify_gender(x, detector))

    print("  Classifying last author genders...", flush=True)
    last_authors["first_name"] = last_authors["display_name"].apply(_extract_first_name)
    last_authors["gender"] = last_authors["first_name"].apply(
        lambda x: _classify_gender(x, detector))

    # Summarize by year
    def summarize(df, label):
        grouped = df.groupby(["year", "gender"]).size().reset_index(name="count")
        pivoted = grouped.pivot(index="year", columns="gender", values="count").fillna(0)
        for col in ["male", "female", "unknown"]:
            if col not in pivoted.columns:
                pivoted[col] = 0
        pivoted["total"] = pivoted["male"] + pivoted["female"] + pivoted["unknown"]
        pivoted["classified"] = pivoted["male"] + pivoted["female"]
        pivoted["pct_female"] = 100.0 * pivoted["female"] / pivoted["classified"].replace(0, 1)
        pivoted["pct_unknown"] = 100.0 * pivoted["unknown"] / pivoted["total"].replace(0, 1)
        pivoted = pivoted.reset_index()
        pivoted.to_csv(f"{OUT_DIR}/gender_{label}.csv", index=False)
        return pivoted

    first_summary = summarize(first_authors, "first_author")
    last_summary = summarize(last_authors, "last_author")

    # Print summary
    latest_first = first_summary.iloc[-1]
    latest_last = last_summary.iloc[-1]
    print(f"  First authors ({int(latest_first['year'])}): "
          f"{latest_first['pct_female']:.1f}% female "
          f"({latest_first['pct_unknown']:.0f}% unknown)", flush=True)
    print(f"  Last authors ({int(latest_last['year'])}): "
          f"{latest_last['pct_female']:.1f}% female "
          f"({latest_last['pct_unknown']:.0f}% unknown)", flush=True)

    # Overall classification rate
    total_classified = first_authors[first_authors["gender"] != "unknown"].shape[0]
    total_all = len(first_authors)
    print(f"  Overall classification rate: "
          f"{100*total_classified/total_all:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    con = create_database(args.db_path)

    t0 = time.time()
    analyze_gender(con)

    elapsed = int(time.time() - t0)
    print(f"\nGender analysis complete ({elapsed}s)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
