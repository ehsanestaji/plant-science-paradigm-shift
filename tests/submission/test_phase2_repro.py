import csv
import json
import subprocess
import sys
from pathlib import Path

from tests.submission.tex_util import REPO, read


def test_no_src_imports_in_code_or_repro_readme():
    hits = []
    for path in list((REPO / "code").rglob("*.py")) + [REPO / "reproducibility" / "README.md"]:
        text = path.read_text(encoding="utf-8")
        if "from src." in text or "import src." in text or "python -m src." in text:
            hits.append(str(path))
    assert hits == [], hits


def test_create_database_importable():
    from db.schema import create_database
    assert callable(create_database)


def test_query_string_table_exists_and_is_referenced():
    csv = (REPO / "config" / "method_query_strings.csv").read_text(encoding="utf-8")
    assert csv.splitlines()[0] == "method,boolean_query,source"
    assert csv.count("\n") >= 5
    assert "CRISPR" in csv
    supp = read("manuscript/sections/supplementary.tex")
    assert r"\label{tab:query-strings}" in supp
    assert r"tab:S3" not in supp
    methods = read("manuscript/sections/methods.tex")
    assert r"\ref{tab:query-strings}" in methods


def test_lexicon_exists_and_paper_cites_its_row_count():
    path = REPO / "config" / "taxon_lexicon_847.csv"
    assert path.exists()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) >= 1
    meta = json.loads(
        (REPO / "config" / "taxon_lexicon_847.meta.json").read_text()
    )
    assert meta["row_count"] == len(rows)
    methods = read("manuscript/sections/methods.tex")
    assert str(len(rows)) in methods
    if len(rows) != 847:
        assert "847" not in methods


def test_code_availability_has_no_slurm():
    assert "SLURM" not in read("manuscript/main.tex")
    assert not list(REPO.rglob("*.sbatch"))
    assert not list(REPO.rglob("*.slurm"))


def test_sample_duckdb_not_promised_in_git():
    readme = read("reproducibility/README.md")
    sample = REPO / "reproducibility" / "plant_science_sample.duckdb"
    if not sample.exists():
        assert "included here" not in readme
        assert "sample DuckDB is not shipped in git" in readme
    assert "[ZENODO DOI]" in readme
    assert "10.5281/zenodo.21199139" not in readme


def test_figure_source_manifest_complete():
    path = REPO / "results/paper_a/figure_source_manifest.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert {"display_item", "tex_label", "source_csv", "status"} <= set(rows[0].keys()) if rows else False
    assert len(rows) == 38
    fig6 = next(r for r in rows if r["tex_label"] == "fig:FA6")
    assert fig6["status"] in {"ok", "needs_duckdb"}
    for r in rows:
        if r["status"] == "ok" and not r["source_csv"].endswith(".tex"):
            assert (REPO / r["source_csv"]).exists(), r["source_csv"]
        if r["tex_label"] != "fig:FA6":
            assert r["status"] == "ok", r


def test_provenance_and_no_gbif_in_validation_sample():
    assert (REPO / "config" / "PROVENANCE.md").exists()
    sample = read("results/paper_a/supplementary/hardening/classifier_validation_sample.csv")
    assert "Occurrence Download" not in sample
    assert "GBIF" not in sample
