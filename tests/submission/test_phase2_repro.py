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
