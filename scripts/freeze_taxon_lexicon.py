"""Freeze the taxon lexicon from committed classification dictionaries."""
from __future__ import annotations

import ast
import csv
import json
import re
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "code/nlp/classify_paper_a.py",
    "code/novel/orphan_organisms.py",
    "code/novel/sleeping_beauty_crops.py",
]


def assignment_value(rel: str, name: str) -> ast.expr:
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ):
            return node.value
    raise ValueError(f"{name} not found in {rel}")


def clean_regex_name(raw: str) -> str | None:
    if r"\w" in raw:
        return None
    name = re.sub(r"\\b|\(\?:|\)|\?", "", raw)
    name = re.sub(r"\\s[+*]", " ", name)
    name = re.sub(r"[^A-Za-z -]", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower() or None


def tokens_from_classify() -> list[tuple[str, str]]:
    value = assignment_value(SOURCE_FILES[0], "ORGANISM_PATTERNS")
    rows: list[tuple[str, str]] = []
    if not isinstance(value, ast.List):
        raise ValueError("ORGANISM_PATTERNS must be a list")
    for item in value.elts:
        if not isinstance(item, ast.Tuple) or len(item.elts) != 2:
            continue
        organism = ast.literal_eval(item.elts[0])
        compile_call = item.elts[1]
        if not isinstance(compile_call, ast.Call) or not compile_call.args:
            continue
        pattern = ast.literal_eval(compile_call.args[0])
        for raw in pattern.split("|"):
            name = clean_regex_name(raw)
            if name:
                rows.append((name, organism))
    return rows


def tokens_from_dict(rel: str, assignment: str) -> list[tuple[str, str]]:
    value = ast.literal_eval(assignment_value(rel, assignment))
    rows: list[tuple[str, str]] = []
    for organism, patterns in value.items():
        rows.append((organism.lower(), organism.lower()))
        for pattern in patterns:
            name = pattern.replace("%", "").strip().lower()
            if name:
                rows.append((name, organism.lower()))
    return rows


def main() -> None:
    seen: dict[str, str] = {}
    sources = (
        tokens_from_classify()
        + tokens_from_dict(SOURCE_FILES[1], "ORGANISMS")
        + tokens_from_dict(SOURCE_FILES[2], "CROPS")
    )
    for name, organism_bin in sources:
        seen.setdefault(name, organism_bin)

    output = REPO / "config" / "taxon_lexicon_847.csv"
    rows = sorted(seen.items())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "taxon_id",
                "common_or_scientific_name",
                "organism_bin",
                "source",
            ],
        )
        writer.writeheader()
        for index, (name, organism_bin) in enumerate(rows, start=1):
            writer.writerow(
                {
                    "taxon_id": f"T{index:04d}",
                    "common_or_scientific_name": name,
                    "organism_bin": organism_bin,
                    "source": "committed classification dictionaries",
                }
            )

    metadata = {
        "freeze_date": date.today().isoformat(),
        "row_count": len(rows),
        "source_files": SOURCE_FILES,
        "note": "Do not cite 847 if row_count differs; cite row_count.",
    }
    (REPO / "config" / "taxon_lexicon_847.meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
