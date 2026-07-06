"""
Phase A: Data cleaning for the plant science metascience database.

Steps:
  1. Remove junk author A9999999999
  2. Cap years at 2024, remove pre-1900
  3. Normalize language codes (eng → en, etc.)
  4. Create summary stats materialized view

Usage:
    python -m src.db.clean --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage


# ISO 639-3 → ISO 639-1 mapping for common cases
LANG_NORM = {
    "eng": "en", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de",
    "spa": "es", "por": "pt", "ita": "it", "nld": "nl", "dut": "nl",
    "jpn": "ja", "zho": "zh", "chi": "zh", "kor": "ko", "rus": "ru",
    "pol": "pl", "tur": "tr", "ara": "ar", "hin": "hi", "tha": "th",
    "vie": "vi", "ces": "cs", "cze": "cs", "swe": "sv", "nor": "no",
    "dan": "da", "fin": "fi", "hun": "hu", "ron": "ro", "rum": "ro",
    "ukr": "uk", "bul": "bg", "hrv": "hr", "slk": "sk", "slo": "sk",
    "slv": "sl", "srp": "sr", "cat": "ca", "ell": "el", "gre": "el",
    "heb": "he", "ind": "id", "msa": "ms", "may": "ms", "fas": "fa",
    "per": "fa", "lit": "lt", "lav": "lv", "est": "et",
}

JUNK_AUTHOR_ID = "A9999999999"


def clean_database(con):
    t0 = time.time()

    # --- Step 1: Remove junk author ---
    print("Step 1: Removing junk author A9999999999...", flush=True)
    n_wa = con.execute(f"""
        DELETE FROM work_authors WHERE author_id = '{JUNK_AUTHOR_ID}'
    """).fetchone()
    n_a = con.execute(f"""
        DELETE FROM authors WHERE author_id = '{JUNK_AUTHOR_ID}'
    """).fetchone()
    print(f"  Removed {n_wa[0] if n_wa else 0} work_author links, "
          f"{n_a[0] if n_a else 0} author records", flush=True)

    # --- Step 2: Fix years ---
    print("Step 2: Fixing year outliers...", flush=True)
    n_future = con.execute("""
        UPDATE works SET year = NULL WHERE year > 2024
    """).fetchone()
    n_ancient = con.execute("""
        UPDATE works SET year = NULL WHERE year < 1900
    """).fetchone()
    print(f"  Nulled {n_future[0] if n_future else 0} future years, "
          f"{n_ancient[0] if n_ancient else 0} pre-1900 years", flush=True)

    # --- Step 3: Normalize language codes ---
    print("Step 3: Normalizing language codes...", flush=True)
    total_lang = 0
    for old, new in LANG_NORM.items():
        r = con.execute(f"""
            UPDATE works SET language = '{new}' WHERE language = '{old}'
        """).fetchone()
        if r and r[0] > 0:
            total_lang += r[0]
    print(f"  Normalized {total_lang} language codes", flush=True)

    # --- Step 4: Create useful views ---
    print("Step 4: Creating analysis views...", flush=True)

    con.execute("""
        CREATE OR REPLACE VIEW works_clean AS
        SELECT * FROM works
        WHERE year IS NOT NULL
          AND year BETWEEN 1900 AND 2024
          AND title IS NOT NULL
          AND length(title) > 5
    """)

    con.execute("""
        CREATE OR REPLACE VIEW corpus_stats AS
        SELECT
            COUNT(*)                                    AS total_works,
            COUNT(CASE WHEN abstract IS NOT NULL THEN 1 END) AS with_abstract,
            COUNT(CASE WHEN doi IS NOT NULL THEN 1 END)      AS with_doi,
            COUNT(CASE WHEN pmid IS NOT NULL THEN 1 END)     AS with_pmid,
            MIN(year)                                   AS min_year,
            MAX(year)                                   AS max_year,
            COUNT(DISTINCT journal_name)                AS n_journals,
            COUNT(DISTINCT language)                    AS n_languages
        FROM works_clean
    """)

    elapsed = int(time.time() - t0)
    print(f"\nCleaning complete ({elapsed}s)", flush=True)

    # Print summary
    stats = con.execute("SELECT * FROM corpus_stats").df()
    print("\n=== Corpus Summary ===")
    for col in stats.columns:
        print(f"  {col}: {stats[col].iloc[0]:,}" if isinstance(stats[col].iloc[0], (int, float))
              else f"  {col}: {stats[col].iloc[0]}")

    # Year distribution sample
    print("\n=== Year Distribution (last 10 years) ===")
    yr = con.execute("""
        SELECT year, COUNT(*) AS n
        FROM works_clean
        WHERE year >= 2015
        GROUP BY year ORDER BY year
    """).df()
    for _, row in yr.iterrows():
        print(f"  {int(row['year'])}: {int(row['n']):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    con = create_database(args.db_path)
    clean_database(con)
    con.close()


if __name__ == "__main__":
    main()
