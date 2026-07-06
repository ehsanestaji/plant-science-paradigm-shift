# Reproducibility Package

Code and sample data for "Bibliometric Laws and Structural Dynamics Across 4.9 Million Plant Science Publications (1900-2024)."

## Quick Start

```bash
pip install -r requirements.txt
python -m src.biblio.powerlaw_fitting --db-path reproducibility/plant_science_sample.duckdb
python -m src.biblio.subfield_laws --db-path reproducibility/plant_science_sample.duckdb
```

## Full Database

The complete database (24.1 GB) is available at [ZENODO DOI]. The sample database
(~250 MB, 1% stratified sample) included here reproduces the analysis pipeline but
will produce different parameter estimates due to sampling.

## Analysis Modules

See `src/` for all analysis code. Each module can be run independently:

```bash
python -m src.biblio.productivity --db-path <path>
python -m src.biblio.powerlaw_fitting --db-path <path>
python -m src.biblio.subfield_laws --db-path <path>
python -m src.novel.disruption_index --db-path <path>
```
