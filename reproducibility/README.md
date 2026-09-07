# Reproducibility Package

Code for "Plant Science is Reorganizing Around Climate-Relevant Crops:
Evidence from 4.9 Million Publications (1900-2024)."

## Quick Start

```bash
pip install -e .
pip install -r requirements.txt
```

Analysis modules live under `code/` and are imported by their package names
(`db`, `biblio`, `novel`, …), not under `src/`.

```bash
python -m biblio.powerlaw_fitting --db-path <path-to-duckdb>
python -m biblio.subfield_laws --db-path <path-to-duckdb>
python -m biblio.productivity --db-path <path-to-duckdb>
python -m novel.disruption_index --db-path <path-to-duckdb>
```

## Full Database

The complete database (~24 GB), when deposited, is available at [ZENODO DOI].
A 1% sample DuckDB is not shipped in git (`*.duckdb` is gitignored). If a
local full database exists at `data/processed/plant_science.duckdb`, generate a
sample for the Zenodo record with:

```bash
python reproducibility/create_sample_db.py \
  --source data/processed/plant_science.duckdb \
  --output reproducibility/plant_science_sample.duckdb \
  --fraction 0.01
```

If that full database is absent, do not mention a sample DuckDB in the
manuscript; reviewers re-derive from the public APIs plus `code/`.
