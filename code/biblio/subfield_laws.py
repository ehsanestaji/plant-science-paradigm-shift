"""
Subfield-level bibliometric law testing.

Tests whether Price's law (citation power-law), Lotka's law (author
productivity), and Bradford's law (journal concentration) hold uniformly
across plant science subfields, or whether the exponents differ
meaningfully between disciplines.

Subfields tested (OpenAlex level-1 concepts):
    Botany, Agronomy, Horticulture, Ecology, Genetics,
    Biochemistry, Molecular biology, Food science

Outputs (results/bibliometrics/):
    subfield_price_law.csv    — per-subfield Price / citation power-law fit
    subfield_lotka_law.csv    — per-subfield Lotka author productivity fit
    subfield_bradford_law.csv — per-subfield Bradford journal concentration

Usage:
    python -m src.biblio.subfield_laws --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/bibliometrics"

# OpenAlex level-1 concept names to test (case-insensitive match stored lower)
SUBFIELDS = [
    "Botany",
    "Agronomy",
    "Horticulture",
    "Ecology",
    "Genetics",
    "Biochemistry",
    "Molecular biology",
    "Food science",
]

# Minimum concept-assignment confidence score and maximum level to accept
CONCEPT_SCORE_MIN = 0.3
CONCEPT_LEVEL_MAX = 1

# Minimum tail size to attempt a power-law fit
MIN_TAIL = 50


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Power-law fitting helpers
# ---------------------------------------------------------------------------

def _fit_powerlaw(data: np.ndarray, discrete: bool = True) -> dict:
    """
    Fit a power-law distribution using the `powerlaw` package (Alstott et al.)
    with automatic xmin estimation.

    Returns a dict with keys:
        alpha, xmin, sigma, n_tail, n_total, ks_D, ks_p,
        loglik_ratio_exp, p_vs_exp  (log-likelihood ratio vs. exponential)
    Falls back to NaN-filled dict if fitting fails or data is too small.
    """
    result = {
        "alpha": np.nan, "xmin": np.nan, "sigma": np.nan,
        "n_tail": 0, "n_total": len(data),
        "ks_D": np.nan, "ks_p": np.nan,
        "loglik_ratio_exp": np.nan, "p_vs_exp": np.nan,
    }

    if len(data) < MIN_TAIL:
        return result

    try:
        import powerlaw  # noqa: PLC0415 — imported lazily to keep startup light
    except ImportError:
        warnings.warn(
            "The `powerlaw` package is not installed. "
            "Run: pip install powerlaw\n"
            "Falling back to OLS log-log estimate.",
            stacklevel=3,
        )
        return _fit_powerlaw_ols(data, result)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(data, discrete=discrete, verbose=False)

        n_tail = int(np.sum(data >= fit.xmin))
        result.update({
            "alpha": float(fit.alpha),
            "xmin": float(fit.xmin),
            "sigma": float(fit.sigma),
            "n_tail": n_tail,
            "ks_D": float(fit.D),
        })

        # Compare power-law vs. exponential
        try:
            R, p = fit.distribution_compare("power_law", "exponential")
            result["loglik_ratio_exp"] = float(R)
            result["p_vs_exp"] = float(p)
        except Exception:
            pass

    except Exception as exc:
        warnings.warn(f"powerlaw.Fit failed: {exc}; using OLS fallback", stacklevel=2)
        return _fit_powerlaw_ols(data, result)

    return result


def _fit_powerlaw_ols(data: np.ndarray, base: dict) -> dict:
    """
    OLS log-log regression fallback.  Returns the same keys as _fit_powerlaw;
    fields that OLS cannot provide are left as NaN.
    """
    # Build frequency table: value -> count
    values, counts = np.unique(data, return_counts=True)
    mask = values > 0
    values, counts = values[mask].astype(float), counts[mask].astype(float)
    if len(values) < 5:
        return base

    x = np.log10(values)
    y = np.log10(counts)
    try:
        coeffs = np.polyfit(x, y, 1)
        base["alpha"] = float(-coeffs[0])
        base["xmin"] = float(values[0])
        base["n_tail"] = int(len(data))
    except Exception:
        pass
    return base


# ---------------------------------------------------------------------------
# 1. Price's law — citation power-law per subfield
# ---------------------------------------------------------------------------

def price_law_for_subfield(con, subfield: str) -> dict:
    """Fetch cited_by_count for works tagged with `subfield` and fit power-law."""
    sf_lower = subfield.lower()
    df = con.execute("""
        SELECT w.cited_by_count
        FROM works_clean w
        JOIN concepts c ON c.work_id = w.work_id
        WHERE LOWER(c.concept_name) = ?
          AND c.level <= ?
          AND c.score >= ?
          AND w.cited_by_count > 0
          AND w.cited_by_count IS NOT NULL
    """, [sf_lower, CONCEPT_LEVEL_MAX, CONCEPT_SCORE_MIN]).df()

    n_papers = len(df)
    print(f"    {subfield}: {n_papers:,} cited papers", flush=True)

    if n_papers < MIN_TAIL:
        return {
            "subfield": subfield, "n_papers": n_papers, "n_tail": 0,
            "alpha": np.nan, "xmin": np.nan, "sigma": np.nan,
            "ks_D": np.nan, "loglik_ratio_exp": np.nan, "p_vs_exp": np.nan,
        }

    data = df["cited_by_count"].values.astype(int)
    fit = _fit_powerlaw(data, discrete=True)

    return {
        "subfield": subfield,
        "n_papers": n_papers,
        "n_tail": fit["n_tail"],
        "alpha": fit["alpha"],
        "xmin": fit["xmin"],
        "sigma": fit["sigma"],
        "ks_D": fit["ks_D"],
        "loglik_ratio_exp": fit["loglik_ratio_exp"],
        "p_vs_exp": fit["p_vs_exp"],
    }


def run_price_law(con) -> pd.DataFrame:
    """Run Price's law analysis for all subfields."""
    print("\nPrice's law (citation power-law) by subfield...", flush=True)
    rows = []
    for sf in SUBFIELDS:
        rows.append(price_law_for_subfield(con, sf))
    df = pd.DataFrame(rows)
    out = f"{OUT_DIR}/subfield_price_law.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}", flush=True)
    return df


# ---------------------------------------------------------------------------
# 2. Lotka's law — author productivity per subfield
# ---------------------------------------------------------------------------

def lotka_law_for_subfield(con, subfield: str) -> dict:
    """
    Fetch per-author paper counts within a subfield and fit power-law.

    An author is counted once per work they co-authored that is tagged with
    the subfield concept.  The distribution of these per-author counts
    (how many subfield papers each author contributed to) is the input.
    """
    sf_lower = subfield.lower()
    df = con.execute("""
        SELECT wa.author_id, COUNT(DISTINCT wa.work_id) AS n_papers
        FROM work_authors wa
        JOIN concepts c ON c.work_id = wa.work_id
        WHERE LOWER(c.concept_name) = ?
          AND c.level <= ?
          AND c.score >= ?
          AND wa.author_id != 'A9999999999'
        GROUP BY wa.author_id
    """, [sf_lower, CONCEPT_LEVEL_MAX, CONCEPT_SCORE_MIN]).df()

    n_authors = len(df)
    print(f"    {subfield}: {n_authors:,} authors", flush=True)

    if n_authors < MIN_TAIL:
        return {
            "subfield": subfield, "n_authors": n_authors, "n_tail": 0,
            "alpha": np.nan, "xmin": np.nan, "sigma": np.nan,
            "ks_D": np.nan, "loglik_ratio_exp": np.nan, "p_vs_exp": np.nan,
            "pct_one_paper": np.nan,
        }

    data = df["n_papers"].values.astype(int)
    pct_one = 100.0 * np.sum(data == 1) / len(data)
    fit = _fit_powerlaw(data, discrete=True)

    return {
        "subfield": subfield,
        "n_authors": n_authors,
        "n_tail": fit["n_tail"],
        "alpha": fit["alpha"],
        "xmin": fit["xmin"],
        "sigma": fit["sigma"],
        "ks_D": fit["ks_D"],
        "loglik_ratio_exp": fit["loglik_ratio_exp"],
        "p_vs_exp": fit["p_vs_exp"],
        "pct_one_paper": round(pct_one, 2),
    }


def run_lotka_law(con) -> pd.DataFrame:
    """Run Lotka's law analysis for all subfields."""
    print("\nLotka's law (author productivity) by subfield...", flush=True)
    rows = []
    for sf in SUBFIELDS:
        rows.append(lotka_law_for_subfield(con, sf))
    df = pd.DataFrame(rows)
    out = f"{OUT_DIR}/subfield_lotka_law.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}", flush=True)
    return df


# ---------------------------------------------------------------------------
# 3. Bradford's law — journal concentration per subfield
# ---------------------------------------------------------------------------

def bradford_law_for_subfield(con, subfield: str) -> dict:
    """
    Compute Bradford zone metrics for a given subfield.

    Zone 1: fewest journals collectively producing the first 1/3 of all
            subfield papers.
    Zone 2: journals producing the second 1/3.
    Zone 3: the remainder.

    Bradford's multiplier k = |Zone2| / |Zone1| ≈ |Zone3| / |Zone2|
    (a constant k is the ideal; real data approximates it).
    """
    sf_lower = subfield.lower()
    df = con.execute("""
        SELECT w.journal_name, COUNT(*) AS n_papers
        FROM works_clean w
        JOIN concepts c ON c.work_id = w.work_id
        WHERE LOWER(c.concept_name) = ?
          AND c.level <= ?
          AND c.score >= ?
          AND w.journal_name IS NOT NULL
        GROUP BY w.journal_name
        ORDER BY n_papers DESC
    """, [sf_lower, CONCEPT_LEVEL_MAX, CONCEPT_SCORE_MIN]).df()

    total_journals = len(df)
    total_papers = int(df["n_papers"].sum())
    print(f"    {subfield}: {total_journals:,} journals, {total_papers:,} papers", flush=True)

    if total_journals < 3 or total_papers == 0:
        return {
            "subfield": subfield,
            "total_journals": total_journals,
            "total_papers": total_papers,
            "zone1_journals": np.nan,
            "zone2_journals": np.nan,
            "zone3_journals": np.nan,
            "zone1_paper_share": np.nan,
            "zone2_paper_share": np.nan,
            "zone3_paper_share": np.nan,
            "bradford_multiplier_k12": np.nan,
            "bradford_multiplier_k23": np.nan,
            "top1_journal": None,
            "top1_n_papers": np.nan,
        }

    df["cumulative"] = df["n_papers"].cumsum()
    third = total_papers / 3.0

    # Zone boundaries (by cumulative paper count)
    zone1_mask = df["cumulative"] <= third
    zone2_mask = (df["cumulative"] > third) & (df["cumulative"] <= 2 * third)
    zone3_mask = df["cumulative"] > 2 * third

    # Edge case: at least 1 journal per zone
    z1 = df[zone1_mask]
    if len(z1) == 0:
        z1 = df.iloc[:1]

    z2 = df[zone2_mask]
    z3 = df[zone3_mask]

    z1_j = len(z1)
    z2_j = len(z2)
    z3_j = len(z3)

    z1_share = round(100.0 * z1["n_papers"].sum() / total_papers, 2)
    z2_share = round(100.0 * z2["n_papers"].sum() / total_papers, 2) if len(z2) > 0 else np.nan
    z3_share = round(100.0 * z3["n_papers"].sum() / total_papers, 2) if len(z3) > 0 else np.nan

    # Bradford multiplier k (ideal ≈ constant)
    k12 = round(z2_j / z1_j, 3) if z1_j > 0 else np.nan
    k23 = round(z3_j / z2_j, 3) if (z2_j is not None and z2_j > 0) else np.nan

    return {
        "subfield": subfield,
        "total_journals": total_journals,
        "total_papers": total_papers,
        "zone1_journals": z1_j,
        "zone2_journals": z2_j if len(z2) > 0 else np.nan,
        "zone3_journals": z3_j if len(z3) > 0 else np.nan,
        "zone1_paper_share": z1_share,
        "zone2_paper_share": z2_share,
        "zone3_paper_share": z3_share,
        "bradford_multiplier_k12": k12,
        "bradford_multiplier_k23": k23,
        "top1_journal": str(df.iloc[0]["journal_name"]),
        "top1_n_papers": int(df.iloc[0]["n_papers"]),
    }


def run_bradford_law(con) -> pd.DataFrame:
    """Run Bradford's law analysis for all subfields."""
    print("\nBradford's law (journal concentration) by subfield...", flush=True)
    rows = []
    for sf in SUBFIELDS:
        rows.append(bradford_law_for_subfield(con, sf))
    df = pd.DataFrame(rows)
    out = f"{OUT_DIR}/subfield_bradford_law.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".3f"):
    """Format a value or return 'N/A' if NaN/None."""
    if val is None:
        return "N/A"
    try:
        if np.isnan(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return format(val, fmt)


def print_price_summary(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("PRICE'S LAW — Citation power-law exponents by subfield")
    print("=" * 72)
    header = f"{'Subfield':<20} {'N papers':>10} {'N tail':>8} "
    header += f"{'alpha':>7} {'xmin':>7} {'sigma':>7} {'KS-D':>7} {'p(vs exp)':>10}"
    print(header)
    print("-" * 72)
    for _, r in df.iterrows():
        print(
            f"{r['subfield']:<20} {r['n_papers']:>10,} {_fmt(r['n_tail'], 'd') if not np.isnan(r['n_tail']) else 'N/A':>8} "
            f"{_fmt(r['alpha']):>7} {_fmt(r['xmin']):>7} {_fmt(r['sigma']):>7} "
            f"{_fmt(r['ks_D']):>7} {_fmt(r['p_vs_exp']):>10}"
        )
    print()


def print_lotka_summary(df: pd.DataFrame):
    print("=" * 72)
    print("LOTKA'S LAW — Author productivity exponents by subfield")
    print("=" * 72)
    header = f"{'Subfield':<20} {'N authors':>10} {'N tail':>8} "
    header += f"{'alpha':>7} {'xmin':>7} {'sigma':>7} {'1-paper%':>9}"
    print(header)
    print("-" * 72)
    for _, r in df.iterrows():
        print(
            f"{r['subfield']:<20} {r['n_authors']:>10,} {_fmt(r['n_tail'], 'd') if not np.isnan(r['n_tail']) else 'N/A':>8} "
            f"{_fmt(r['alpha']):>7} {_fmt(r['xmin']):>7} {_fmt(r['sigma']):>7} "
            f"{_fmt(r['pct_one_paper'], '.1f'):>8}%"
        )
    print()


def print_bradford_summary(df: pd.DataFrame):
    print("=" * 72)
    print("BRADFORD'S LAW — Journal concentration zones by subfield")
    print("=" * 72)
    header = (
        f"{'Subfield':<20} {'Journals':>8} {'Z1 jrnls':>9} "
        f"{'Z1 share%':>10} {'k12':>6} {'k23':>6} {'Top journal':<30}"
    )
    print(header)
    print("-" * 72)
    for _, r in df.iterrows():
        top = str(r["top1_journal"])[:28] if r["top1_journal"] else "N/A"
        print(
            f"{r['subfield']:<20} {r['total_journals']:>8,} "
            f"{_fmt(r['zone1_journals'], 'd') if not (isinstance(r['zone1_journals'], float) and np.isnan(r['zone1_journals'])) else 'N/A':>9} "
            f"{_fmt(r['zone1_paper_share'], '.1f'):>9}% "
            f"{_fmt(r['bradford_multiplier_k12'], '.1f'):>6} "
            f"{_fmt(r['bradford_multiplier_k23'], '.1f'):>6} "
            f"{top:<30}"
        )
    print()


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Test bibliometric laws across plant science subfields."
    )
    ap.add_argument(
        "--db-path",
        default="data/processed/plant_science.duckdb",
        help="Path to the DuckDB database (default: %(default)s)",
    )
    ap.add_argument(
        "--read-only",
        action="store_true",
        default=True,
        help="Open database in read-only mode (default: True)",
    )
    ap.add_argument(
        "--memory-limit",
        default="40GB",
        help="DuckDB memory limit (default: %(default)s)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="DuckDB thread count (default: %(default)s)",
    )
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()

    con = create_database(args.db_path, read_only=args.read_only)
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")

    t0 = time.time()

    price_df = run_price_law(con)
    lotka_df = run_lotka_law(con)
    bradford_df = run_bradford_law(con)

    elapsed = int(time.time() - t0)
    print(f"\nSubfield bibliometric law analysis complete ({elapsed}s)\n", flush=True)

    # Print summary tables
    print_price_summary(price_df)
    print_lotka_summary(lotka_df)
    print_bradford_summary(bradford_df)

    con.close()


if __name__ == "__main__":
    main()
