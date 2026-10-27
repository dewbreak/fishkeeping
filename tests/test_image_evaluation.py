import importlib.util
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "image_eval", Path(__file__).parents[1] / "scripts/evaluate_images.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_frozen_labels():
    data = evaluation.load_dataset()
    assert (
        evaluation.fingerprint(data)
        == "f1a41e44fa9aeb4ba149c6bf02f4695db516fd100beaa7fbf18691100c5fd7c1"
    )
    assert len(data["cases"]) == 8
    assert {c["id"] for c in data["cases"] if c.get("live")} == {
        "long_screenshot",
        "water_table",
        "scene_shapes",
    }


def test_synthetic_generation_and_metric(tmp_path):
    case = evaluation.load_dataset()["cases"][0]
    path, metadata = evaluation.generate_case(case, tmp_path)
    with Image.open(path) as picture:
        assert picture.size == (1400, 1600)
    assert "25 C" in "\n".join(metadata["truth"])
    assert evaluation.edit_distance("filter", "filler") == 1
