from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MS = REPO / "manuscript"


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def tex_files() -> list[Path]:
    return sorted((MS / "sections").glob("*.tex")) + [MS / "main.tex"]


def all_tex() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in tex_files())


def labels(text: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", text)


def refs(text: str) -> list[str]:
    return re.findall(r"\\(?:page)?ref\{([^}]+)\}", text)


def cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        keys.update(k.strip() for k in group.split(",") if k.strip())
    return keys


def bib_keys(text: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.M))
