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


def test_collab_caption_is_23_point_1():
    supp = read("manuscript/sections/supplementary.tex")
    start = supp.index("\\subsection*{International collaboration by organism}")
    block = supp[start:supp.index("\\end{figure}", start)]
    assert "18.9" not in block
    assert "23.1" in block


def test_attention_gap_table_uses_z_and_includes_oil_palm():
    supp = read("manuscript/sections/supplementary.tex")
    start = supp.index("\\label{tab:si-attention-gap}")
    block = supp[start:start + 2500]
    assert "oil palm" in block.lower() or "Oil palm" in block
    for z_value in ("-5.21", "-4.03", "-3.87", "-3.12", "+8.94", "+4.71"):
        assert z_value in block
    assert "Papers/Mt" not in block


def test_sleeping_beauty_prose_matches_table_s2_names():
    main = read("manuscript/sections/main_text.tex")
    start = main.index("\\subsection*{Sleeping beauties on minor crops are awakening}")
    block = main[start:main.index("\\subsection*", start + 1)]
    for excluded_species in ("teff", "quinoa", "cowpea", "breadfruit"):
        assert excluded_species not in block.lower()
    assert "Moringa" in block
    assert "millet" in block.lower()
    assert "amaranth" in block.lower()


def test_most_convergent_organism_is_tobacco():
    main = read("manuscript/sections/main_text.tex")
    start = main.index("\\subsection*{Semantic drift")
    block = main[start:main.index("\\subsection*", start + 1)]
    assert "Cotton and rice converge most strongly" not in block
    assert "Tobacco" in block or "tobacco" in block
    assert "$d = 0.016$" in block
    assert "$d = 0.047$" in block


MANUSCRIPT_COUNT_FILES = (
    "manuscript/sections/main_text.tex",
    "manuscript/sections/methods.tex",
    "manuscript/sections/supplementary.tex",
    "manuscript/sections/item_lists.tex",
)


def test_corpus_states_harvested_and_classified():
    methods = read("manuscript/sections/methods.tex")
    assert "4,912,447" in methods
    assert "2,466,390" in methods
    assert "each with at least one author affiliation and abstract text" not in methods

    combined = "\n".join(read(path) for path in MANUSCRIPT_COUNT_FILES)
    for stale in ("2,770,088", "2.77 million", "56\\%"):
        assert stale not in combined, stale

    main = read("manuscript/sections/main_text.tex")
    assert "46.1\\%" not in main
    assert "46.1%" not in main


def test_methods_does_not_cite_query_table_yet():
    methods = read("manuscript/sections/methods.tex")
    assert "tab:query-strings" not in methods
    assert "tab:S3" not in methods
    assert "code/novel/method" in methods or "method\\_diffusion.py" in methods


def test_no_introduction_or_policy_headings():
    main = read("manuscript/sections/main_text.tex")
    assert r"\section*{Introduction}" not in main
    assert r"\subsection*{Policy implications}" not in main


def test_paper_c_is_not_the_only_methods():
    blob = read("manuscript/sections/methods.tex") + read("manuscript/main.tex")
    assert "Paper C, in preparation" not in blob
    assert "in preparation" not in blob
    for needle in (
        "PubMed 1990--2024",
        "OpenAlex 1900--2024",
        "SPECTER2",
        "sleeping-beauty index",
        "CD disruption index",
        "studentised residual",
        "code/collect/",
        "code/db/clean.py",
    ):
        assert needle in blob, needle


def test_methods_follows_references_in_main_tex():
    tex = read("manuscript/main.tex")
    i_bib = tex.index("\\bibliography{references}")
    i_methods = tex.index("\\input{sections/methods}")
    i_da = tex.index("Data Availability")
    i_ed = tex.rindex("\\input{sections/extended_data}")
    assert i_bib < i_methods < i_da < i_ed


def test_data_availability_uses_placeholder_not_404():
    tex = read("manuscript/main.tex")
    assert "[ZENODO DOI]" in tex
    assert "10.5281/zenodo.21199139" not in tex
    assert "https://doi.org/10.5281/zenodo." not in tex
