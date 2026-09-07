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
