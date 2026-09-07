from collections import Counter

from tests.submission.tex_util import MS, all_tex, labels, read


def test_each_label_defined_exactly_once():
    counts = Counter(labels(all_tex()))
    dupes = {k: n for k, n in counts.items() if n != 1}
    assert dupes == {}, f"duplicate or missing labels: {dupes}"


def test_retired_and_old_colliding_names_absent():
    text = all_tex()
    for forbidden in ("fig:S8", "tab:S6", "tab:S1", "tab:S3"):
        assert forbidden not in text, f"{forbidden} must be retired"


def test_one_off_reorder_scripts_deleted():
    leftovers = list(MS.glob("rename_*.py")) + list(MS.glob("reorder_*.py"))
    assert leftovers == [], leftovers


FORBIDDEN_CITES = {"bhatt2020", "rhodes2021", "nair2010", "singh2022specter2"}
LIVE_CITES = {
    "ipcc2022", "jackson2011", "somerville1986", "meyerowitz1989",
    "feuillet2011", "voss2015", "funk2017disruption", "ke2015sleeping",
    "kuhn1962", "hegarty2008", "varshney2012", "price1963",
    "nestler2010", "frese2007", "sunagawa2015", "priem2022openalex",
    "caracciolo2013", "singh2023specter2", "grootendorst2022bertopic",
    "fao2023",
}


def test_forbidden_cite_keys_absent():
    text = all_tex() + read("manuscript/references.bib")
    for key in FORBIDDEN_CITES:
        assert key not in text, key


def test_live_cite_set_matches_bib():
    from tests.submission.tex_util import bib_keys, cite_keys

    cited = cite_keys(all_tex())
    bib = bib_keys(read("manuscript/references.bib"))
    assert cited == LIVE_CITES
    assert bib == LIVE_CITES


def test_every_remaining_entry_has_doi_or_misc_url():
    import re

    raw = read("manuscript/references.bib")
    entries = re.split(r"\n(?=@)", raw)
    for entry in entries:
        if not entry.strip():
            continue
        kind = re.match(r"@(\w+)", entry).group(1).lower()
        if kind == "misc":
            assert "url =" in entry or "doi =" in entry, entry[:80]
        else:
            assert "doi =" in entry, entry[:80]
