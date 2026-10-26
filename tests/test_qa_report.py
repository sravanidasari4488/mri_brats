"""QA overlays, review flags, and the HTML report order."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.postprocess import PostConfig
from scripts.qa_report import evaluate_case, review_reasons, tumor_center, write_report


def _ball(shape, center, radius, value, target):
    coords = np.indices(shape)
    delta = coords - np.asarray(center).reshape(3, 1, 1, 1)
    target[np.sum(delta * delta, axis=0) <= radius * radius] = value


def test_low_precision_is_listed_before_the_gallery(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    shape = (28, 28, 28)
    label = np.zeros(shape, dtype=np.uint8)
    _ball(shape, (14, 14, 14), 6, 2, label)
    probs = np.zeros((3,) + shape, dtype=np.float32)
    _ball(shape, (14, 14, 14), 10, 0.95, probs[2])
    image = np.zeros((4,) + shape, dtype=np.float32)
    image[0] = 0.2
    _ball(shape, (14, 14, 14), 10, 1.0, image[0])
    config = PostConfig(0.5, 0.5, 0.5, 0, 0, 0, 0, 1)
    record = evaluate_case(
        "low_precision",
        image,
        label,
        probs,
        (1.0, 1.0, 1.0),
        np.eye(4),
        root,
        config,
    )
    assert record["regions"]["WT"]["precision"] < 0.9
    assert record["center"] == (14, 14, 14)
    assert any(reason.startswith("WT precision") for reason in record["reasons"])
    for name in ("axial", "coronal", "sagittal", "render"):
        assert (root / record["images"][name]).is_file()

    clean = {
        "case_id": "clean",
        "regions": {
            name: {"dice": 1.0, "hd95_mm": 0.0, "precision": 0.97, "lesion_fp": 0}
            for name in ("WT", "TC", "ET")
        },
        "volume_error_pct": {"WT": 1.0, "TC": -12.0, "ET": None},
        "watertight": {"WT": True, "TC": True, "ET": None},
        "reasons": review_reasons(
            {name: {"precision": 0.97} for name in ("WT", "TC", "ET")},
            {"WT": True, "TC": True, "ET": None},
        ),
        "images": record["images"],
    }
    assert clean["reasons"] == []
    open_shell = {
        "case_id": "open_shell",
        "regions": {
            name: {"dice": 0.8, "hd95_mm": 2.0, "precision": 0.95, "lesion_fp": 1}
            for name in ("WT", "TC", "ET")
        },
        "volume_error_pct": {"WT": 0.5, "TC": None, "ET": None},
        "watertight": {"WT": True, "TC": False, "ET": None},
        "reasons": [],
        "images": record["images"],
    }
    open_shell["reasons"] = review_reasons(open_shell["regions"], open_shell["watertight"])
    assert open_shell["reasons"] == ["TC mesh not watertight"]

    report = write_report([clean, record, open_shell], root / "index.html")
    text = report.read_text(encoding="utf-8")
    assert text.index("Manual review") < text.index("All cases")
    review = text.split("All cases")[0]
    assert "low_precision" in review
    assert "open_shell" in review
    assert "clean" not in review.split("<ul>")[-1]
    assert "Dice" in text and "HD95" in text and "FP components" in text
    assert "WT +1.0%, TC -12.0%, ET —" in text
    assert "TC not watertight" in text


def test_cutoffs_and_undefined_distance():
    ground_truth = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction = np.zeros((5, 5, 5), dtype=np.uint8)
    prediction[1, 1, 1] = 2
    assert tumor_center(ground_truth, prediction) == (1, 1, 1)
    ground_truth[3, 3, 3] = 2
    assert tumor_center(ground_truth, prediction) == (3, 3, 3)

    exact = {name: {"precision": 0.9} for name in ("WT", "TC", "ET")}
    assert review_reasons(exact, {"WT": True, "TC": True, "ET": None}) == []
    assert review_reasons(exact) == []
    low = dict(exact)
    low["TC"] = {"precision": 0.899}
    assert review_reasons(low, {"WT": None, "TC": None, "ET": None}) == ["TC precision 0.899"]

    import tempfile

    root = Path(tempfile.mkdtemp())
    blank_images = {"axial": "a.png", "coronal": "c.png", "sagittal": "s.png", "render": "r.png"}
    record = {
        "case_id": "empty_hd",
        "regions": {
            name: {"dice": 0.0, "hd95_mm": float("nan"), "precision": 0.95, "lesion_fp": 0}
            for name in ("WT", "TC", "ET")
        },
        "volume_error_pct": {"WT": None, "TC": None, "ET": None},
        "watertight": {"WT": None, "TC": None, "ET": None},
        "reasons": [],
        "images": blank_images,
    }
    text = write_report([record], root / "index.html").read_text(encoding="utf-8")
    assert "— / — / —" in text
    assert "No cases flagged." in text
    assert "WT —, TC —, ET —" in text


def test_thin_shell_shrinkage_does_not_flag_a_case():
    import tempfile

    shape = (40, 40, 40)
    label = np.zeros(shape, dtype=np.uint8)
    _ball(shape, (20, 20, 20), 12, 2, label)
    _ball(shape, (20, 20, 20), 9, 3, label)
    _ball(shape, (20, 20, 20), 7, 1, label)
    probs = np.zeros((3,) + shape, dtype=np.float32)
    probs[0][label == 3] = 0.95
    probs[1][(label == 1) | (label == 3)] = 0.95
    probs[2][label > 0] = 0.95
    image = np.ones((4,) + shape, dtype=np.float32)
    record = evaluate_case(
        "thin_rim",
        image,
        label,
        probs,
        (1.0, 1.0, 1.0),
        np.eye(4),
        Path(tempfile.mkdtemp()),
        PostConfig(0.5, 0.5, 0.5, 0, 0, 0, 0, 1),
    )
    assert set(record["volume_error_pct"]) == {"WT", "TC", "ET"}
    assert all(record["watertight"][name] for name in ("WT", "TC", "ET"))
    assert all(record["volume_error_pct"][name] < -3.0 for name in ("WT", "TC", "ET"))
    assert record["reasons"] == []


if __name__ == "__main__":
    test_cutoffs_and_undefined_distance()
    test_low_precision_is_listed_before_the_gallery()
    test_thin_shell_shrinkage_does_not_flag_a_case()
    print("qa ok")
