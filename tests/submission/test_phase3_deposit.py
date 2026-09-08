import json

from tests.submission.tex_util import read


def test_zenodo_json_is_dataset_ccby():
    meta = json.loads(read(".zenodo.json"))
    assert meta["upload_type"] == "dataset"
    assert meta["license"] == "CC-BY-4.0"
    assert "MIT" in meta["description"]
    assert "CC BY 4.0" in meta["description"]
