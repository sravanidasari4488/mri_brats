"""Geometry of load_case: seg centre of mass stays inside the brain, in mm."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import NormalizeIntensity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.io import MODALITY_SUFFIXES, load_case  # noqa: E402


def _write_case(case_dir):
    """LPS-style affine so RAS reorientation actually permutes the array."""
    shape = (16, 20, 12)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = -1.5
    affine[1, 1] = -1.25
    affine[2, 2] = 2.0
    affine[:3, 3] = (40.0, -15.0, 8.0)

    brain = np.zeros(shape, dtype=np.float32)
    brain[3:13, 4:16, 2:10] = 1.0
    ramp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    case_dir.mkdir()
    for index, suffix in enumerate(MODALITY_SUFFIXES):
        volume = brain * (ramp * 0.05 + (index + 1) * 20.0)
        nib.save(
            nib.Nifti1Image(volume, affine),
            case_dir / f"{case_dir.name}_{suffix}.nii.gz",
        )

    seg = np.zeros(shape, dtype=np.uint8)
    seg[6:10, 7:12, 4:8] = 1
    seg[6:10, 7:12, 4:6] = 2
    seg[7:9, 8:11, 5:7] = 4
    seg_path = case_dir / f"{case_dir.name}_seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, affine), seg_path)
    return seg_path


def _world(affine, ijk):
    hom = np.array([ijk[0], ijk[1], ijk[2], 1.0], dtype=np.float64)
    return affine @ hom


def test_seg_com_inside_brain_bbox_and_shapes_match(tmp_path):
    case_dir = tmp_path / "BraTS2021_00042"
    seg_path = _write_case(case_dir)
    loaded = load_case(case_dir)

    image = loaded["image"]
    seg = loaded["seg"]
    affine = loaded["affine"]

    assert loaded["case_id"] == "BraTS2021_00042"
    assert image.ndim == 4 and image.shape[0] == 4
    assert image.shape[1:] == seg.shape
    assert seg.dtype == np.uint8
    assert set(np.unique(seg)).issubset({0, 1, 2, 3})

    raw_seg = np.rint(np.asanyarray(nib.load(seg_path).dataobj)).astype(np.int16)
    assert int((seg == 3).sum()) == int((raw_seg == 4).sum())
    assert int((seg == 1).sum()) == int((raw_seg == 1).sum())
    assert int((seg == 2).sum()) == int((raw_seg == 2).sum())

    com = np.argwhere(seg > 0).astype(np.float64).mean(axis=0)
    world = _world(affine, com)

    brain = np.any(image != 0, axis=0)
    brain_ijk = np.argwhere(brain).astype(np.float64)
    brain_hom = np.concatenate([brain_ijk, np.ones((len(brain_ijk), 1))], axis=1)
    brain_mm = (affine @ brain_hom.T).T[:, :3]
    lo = brain_mm.min(axis=0)
    hi = brain_mm.max(axis=0)
    assert np.all(world[:3] >= lo - 1e-3)
    assert np.all(world[:3] <= hi + 1e-3)

    raw = nib.load(seg_path)
    raw_com = np.argwhere(raw_seg > 0).astype(np.float64).mean(axis=0)
    assert np.allclose(world, _world(raw.affine, raw_com), atol=1e-4)

    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
    for index, suffix in enumerate(MODALITY_SUFFIXES):
        canonical = nib.as_closest_canonical(
            nib.load(case_dir / f"{case_dir.name}_{suffix}.nii.gz")
        )
        raw_vol = np.asarray(canonical.get_fdata(dtype=np.float32))
        expected = np.asarray(normalize(raw_vol[None]), dtype=np.float32)[0]
        assert np.allclose(image[index], expected, atol=1e-5)
        nonzero = image[index][image[index] != 0]
        assert abs(float(nonzero.mean())) < 1e-4
        assert abs(float(nonzero.std()) - 1.0) < 1e-4


def test_brats2020_and_processed_names_load_in_the_same_order(tmp_path):
    shape = (6, 6, 6)
    layouts = {
        "processed": ("flair", "t1", "t1c", "t2", "mask"),
        "BraTS20_Training_002": tuple(
            f"BraTS20_Training_002_{name}" for name in ("flair", "t1", "t1ce", "t2", "seg")
        ),
    }
    for folder, names in layouts.items():
        case_dir = tmp_path / folder
        case_dir.mkdir()
        for index, name in enumerate(names[:4]):
            nib.save(nib.Nifti1Image(np.full(shape, index + 1.0, np.float32), np.eye(4)), case_dir / f"{name}.nii.gz")
        seg = np.zeros(shape, dtype=np.uint8)
        seg[2:4, 2:4, 2:4] = 4
        nib.save(nib.Nifti1Image(seg, np.eye(4)), case_dir / f"{names[4]}.nii.gz")
        loaded = load_case(case_dir, normalize=False)
        assert [float(loaded["image"][i].mean()) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]
        assert int((loaded["seg"] == 3).sum()) == 8


def test_case_insensitive_names_and_grid_mismatch(tmp_path):
    from recon.io import input_files_hash, load_case, validate_case_grid

    shape = (6, 6, 6)
    case_dir = tmp_path / "plain_names"
    case_dir.mkdir()
    names = ("FLAIR.nii.gz", "T1.nii.gz", "T1c.nii.gz", "T2.nii.gz")
    for index, name in enumerate(names):
        nib.save(nib.Nifti1Image(np.full(shape, index + 1.0, np.float32), np.eye(4)), case_dir / name)
    loaded = load_case(case_dir, normalize=False)
    assert [float(loaded["image"][i].mean()) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]
    first = input_files_hash(case_dir)
    assert first == input_files_hash(case_dir)
    nib.save(nib.Nifti1Image(np.full(shape, 9.0, np.float32), np.eye(4)), case_dir / "T2.nii.gz")
    assert input_files_hash(case_dir) != first

    other = tmp_path / "mismatch"
    other.mkdir()
    for name in names:
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)), other / name)
    nib.save(nib.Nifti1Image(np.ones((5, 6, 6), np.float32), np.eye(4)), other / "T2.nii.gz")
    try:
        validate_case_grid(other)
        raise AssertionError("expected a grid mismatch")
    except ValueError as exc:
        message = str(exc)
        assert "T2" in message
        assert "FLAIR" in message or "grid" in message.lower()

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 2.0
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)), other / "T2.nii.gz")
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), affine), other / "T1c.nii.gz")
    try:
        validate_case_grid(other)
        raise AssertionError("expected an affine mismatch")
    except ValueError as exc:
        message = str(exc)
        assert "T1 with contrast" in message
        assert "affine" in message.lower() or "orientation" in message.lower()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_seg_com_inside_brain_bbox_and_shapes_match(Path(tempfile.mkdtemp()))
    test_brats2020_and_processed_names_load_in_the_same_order(Path(tempfile.mkdtemp()))
    test_case_insensitive_names_and_grid_mismatch(Path(tempfile.mkdtemp()))
    print("io ok")
