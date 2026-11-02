"""Quality gates for preprocessed real MRI studies."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.qc_real import abc2_largest_stats, abc2_volume_cm3, qc_study


def _volume(path: Path, data, affine=None):
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), affine), str(path))


def _ready(tmp_path, *, failed=(), flair_mm=5.0, extra_meta=None):
    study = "study_test"
    ready = tmp_path / study / "04_ready"
    ready.mkdir(parents=True)
    affine = np.eye(4)
    shape = (70, 70, 70)
    mask = np.zeros(shape, np.uint8)
    mask[5:65, 5:65, 5:65] = 1
    rng = np.random.default_rng(0)
    for name in ("flair", "t1", "t1c", "t2"):
        vol = np.zeros(shape, np.float32)
        vol[mask > 0] = rng.normal(size=int(mask.sum())).astype(np.float32)
        _volume(ready / f"{name}.nii.gz", vol, affine)
    _volume(ready / "brain_mask.nii.gz", mask, affine)
    meta = {
        "study_id": study,
        "tier": "FULL",
        "reference": {"role": "t1", "acquisition": "3D"},
        "original_thickness_mm": {"flair": flair_mm, "t1": 1.1, "t1c": 5.0, "t2": 5.0},
        "registration": {
            "values": {"t1": None, "flair": -0.3, "t1c": -0.4, "t2": -0.35},
            "failed": list(failed),
        },
        "registration_failed": list(failed),
    }
    if extra_meta:
        meta.update(extra_meta)
    (ready / "preprocess.json").write_text(json.dumps(meta), encoding="utf-8")
    (ready / "registration_check.png").write_bytes(b"png")
    return study, tmp_path


def test_registration_failed_is_fail(tmp_path):
    study, root = _ready(tmp_path, failed=("t1c", "t2"))
    result = qc_study(study, out_root=root)
    assert result.status == "FAIL"
    assert any("registration_failed" in reason for reason in result.reasons)


def test_thick_flair_is_warn(tmp_path):
    study, root = _ready(tmp_path, failed=(), flair_mm=5.0)
    result = qc_study(study, out_root=root)
    assert result.status == "WARN"
    assert "thick FLAIR: tumor boundary accuracy reduced" in result.reasons
    assert any(reason.startswith("thick slices: t1c") for reason in result.reasons)


def test_thin_slices_pass_without_thickness_warn(tmp_path):
    study, root = _ready(tmp_path, flair_mm=1.2)
    ready = root / study / "04_ready"
    meta = json.loads((ready / "preprocess.json").read_text(encoding="utf-8"))
    meta["original_thickness_mm"] = {"flair": 1.2, "t1": 1.1, "t1c": 1.0, "t2": 1.4}
    (ready / "preprocess.json").write_text(json.dumps(meta), encoding="utf-8")
    result = qc_study(study, out_root=root)
    assert result.status == "PASS"
    assert result.reasons == []


def test_abc2_volume_of_box():
    empty = np.zeros((12, 12, 12), dtype=bool)
    assert abc2_volume_cm3(empty, np.eye(4)) == 0.0
    mask = np.zeros((30, 30, 20), dtype=bool)
    mask[10:20, 10:16, 5:15] = True
    volume = abc2_volume_cm3(mask, np.eye(4))
    assert 0.15 < volume < 0.5


def test_fail_study_never_has_pred_seg(tmp_path):
    study, root = _ready(tmp_path, failed=("t1c", "t2"))
    out = root / study
    affine = np.eye(4)
    pred = np.ones((70, 70, 70), dtype=np.uint8)
    nib.save(nib.Nifti1Image(pred, affine), str(out / "pred_seg.nii.gz"))
    nib.save(nib.Nifti1Image(np.ones((70, 70, 70), np.float32), affine), str(out / "wt_prob.nii.gz"))
    assert (out / "pred_seg.nii.gz").is_file()
    result = qc_study(study, out_root=root)
    assert result.status == "FAIL"
    assert not (out / "pred_seg.nii.gz").is_file()
    assert not (out / "wt_prob.nii.gz").is_file()
    result_again = qc_study(study, out_root=root)
    assert result_again.status == "FAIL"
    assert not (out / "pred_seg.nii.gz").is_file()


def test_abc2_uses_largest_component_only():
    mask = np.zeros((40, 40, 24), dtype=bool)
    mask[4:8, 4:8, 4:8] = True
    mask[12:28, 10:22, 6:18] = True
    stats = abc2_largest_stats(mask, np.eye(4))
    assert stats["wt_components"] == 2
    assert stats["largest_component_cm3"] > stats["component_volumes_cm3"][1]
    only_large = np.zeros_like(mask)
    only_large[12:28, 10:22, 6:18] = True
    assert abs(stats["abc2_largest_cm3"] - abc2_volume_cm3(only_large, np.eye(4))) < 1e-6
    assert stats["ratio"] == stats["abc2_largest_cm3"] / stats["largest_component_cm3"]


if __name__ == "__main__":
    import tempfile

    test_registration_failed_is_fail(Path(tempfile.mkdtemp()))
    test_thick_flair_is_warn(Path(tempfile.mkdtemp()))
    test_thin_slices_pass_without_thickness_warn(Path(tempfile.mkdtemp()))
    test_abc2_volume_of_box()
    test_abc2_uses_largest_component_only()
    test_fail_study_never_has_pred_seg(Path(tempfile.mkdtemp()))
    print("qc_real ok")
