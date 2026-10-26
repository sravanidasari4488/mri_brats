"""abhinav_v3 case lists, and refusing metrics on its training cases."""

import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.evaluate import TrainingCaseError, aggregate_scores, evaluation_case_ids, score_case
from recon.splits import (
    ABHINAV_HELDOUT,
    ABHINAV_TRAIN,
    ABHINAV_VAL,
    check_reportable,
    read_case_list,
)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refused(call):
    try:
        call()
    except TrainingCaseError:
        return True
    return False


def test_lists_are_disjoint_and_cover_brats2020():
    heldout = read_case_list(ABHINAV_HELDOUT)
    train = read_case_list(ABHINAV_TRAIN)
    val = read_case_list(ABHINAV_VAL)
    assert (len(heldout), len(train), len(val)) == (60, 289, 20)
    assert not set(heldout) & set(train)
    assert not set(heldout) & set(val)
    assert not set(train) & set(val)
    assert len(set(heldout) | set(train) | set(val)) == 369
    assert evaluation_case_ids("abhinav_v3") == heldout
    assert evaluation_case_ids("ours") is None


def test_metrics_on_training_cases_are_refused():
    trained = read_case_list(ABHINAV_TRAIN)[0]
    heldout = read_case_list(ABHINAV_HELDOUT)[0]
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1, 1, 1] = 2

    assert _refused(lambda: check_reportable([heldout, trained], "abhinav_v3"))
    assert _refused(lambda: score_case(label, label, case_id=trained, model="abhinav_v3"))
    assert _refused(lambda: score_case(label, label, model="abhinav_v3"))
    check_reportable([trained], "ours")

    scored = score_case(label, label, case_id=heldout, model="abhinav_v3")
    assert scored["case_id"] == heldout
    assert aggregate_scores([scored], model="abhinav_v3")["n_cases"] == 1
    leaked = dict(scored, case_id=trained)
    assert _refused(lambda: aggregate_scores([scored, leaked], model="abhinav_v3"))

    selection = read_case_list(ABHINAV_VAL)[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_reportable([selection], "abhinav_v3")
    assert any("select" in str(item.message) for item in caught)


def test_tuner_and_cache_refuse_training_cases():
    tune = _script("tune_postprocess")
    cache = _script("cache_probs")
    trained = read_case_list(ABHINAV_TRAIN)[0]
    heldout = read_case_list(ABHINAV_HELDOUT)[:2]
    root = Path(tempfile.mkdtemp())

    probs_dir = root / "probs"
    probs_dir.mkdir()
    for case_id in (heldout[0], trained):
        np.savez_compressed(
            probs_dir / f"{case_id}.npz",
            probs=np.zeros((3, 4, 4, 4), dtype=np.float16),
            label=np.zeros((4, 4, 4), dtype=np.uint8),
            channel_order=np.array(["WT", "ET", "TC"]),
            case_id=np.array(case_id),
            model=np.array("abhinav_v3"),
        )
    assert _refused(lambda: tune.load_cached_cases(probs_dir, "abhinav_v3"))
    (probs_dir / f"{trained}.npz").unlink()
    assert [case.case_id for case in tune.load_cached_cases(probs_dir, "abhinav_v3")] == [heldout[0]]
    try:
        tune.load_cached_cases(probs_dir, "ours")
        raise AssertionError("expected a model mismatch")
    except ValueError:
        pass
    assert tune.default_cache_dir("abhinav_v3").name == "val_probs_abhinav_v3"

    data = root / "data"
    for case_id in heldout + [trained]:
        folder = data / case_id
        folder.mkdir(parents=True)
        for name in ("flair", "t1", "t1c", "t2", "mask"):
            (folder / f"{name}.nii.gz").write_bytes(b"")
    listed = root / "list.txt"
    listed.write_text("\n".join(heldout) + "\n", encoding="utf-8")
    selected, _ = cache.select_cases(data, "abhinav_v3", case_list=listed)
    assert [path.name for path in selected] == heldout
    listed.write_text(f"{heldout[0]}\n{trained}\n", encoding="utf-8")
    assert _refused(lambda: cache.select_cases(data, "abhinav_v3", case_list=listed))
    assert cache.default_out_dir("abhinav_v3").name == "val_probs_abhinav_v3"


if __name__ == "__main__":
    test_lists_are_disjoint_and_cover_brats2020()
    test_metrics_on_training_cases_are_refused()
    test_tuner_and_cache_refuse_training_cases()
    print("splits ok")
