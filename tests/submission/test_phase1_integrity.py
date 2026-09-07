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
