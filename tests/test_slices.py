"""Radiological RAS slice orientation, letters, and 3D-plane contour agreement."""

import io
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.slices import (
    ORIENT_LETTERS,
    mean_boundary_distance,
    mesh_plane_points,
    orient_ras_slice,
    tumor_contour_world_points,
    window_to_uint8,
)
from web.app import create_app


def _nifti(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine),
        path,
    )


def _marker_volume():
    """Asymmetric L in the left-anterior-superior corner of a RAS grid."""
    volume = np.zeros((12, 10, 8), dtype=np.float32)
    volume[0:5, 9, 7] = 100
    volume[0, 7:10, 7] = 100
    return volume


def test_orient_ras_slice_puts_las_marker_on_the_lettered_corner():
    volume = _marker_volume()
    letters = ORIENT_LETTERS
    axial = orient_ras_slice(volume, 2, 7)
    rows, cols = np.nonzero(axial >= 50)
    assert letters[2]["top"] == "A" and letters[2]["right"] == "L"
    assert rows.min() == 0
    assert cols.max() == axial.shape[1] - 1

    coronal = orient_ras_slice(volume, 1, 9)
    rows, cols = np.nonzero(coronal >= 50)
    assert letters[1]["top"] == "S" and letters[1]["right"] == "L"
    assert rows.min() == 0
    assert cols.max() == coronal.shape[1] - 1

    sagittal = orient_ras_slice(volume, 0, 0)
    rows, cols = np.nonzero(sagittal >= 50)
    assert letters[0]["top"] == "S" and letters[0]["left"] == "A"
    assert rows.min() == 0
    assert cols.min() == 0


def test_api_png_matches_oriented_numpy_pixel_exact(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    outputs = root / "outputs"
    cases = root / "cases"
    volume = _marker_volume()
    affine = np.eye(4, dtype=np.float64)
    for suffix in ("t2f", "t1n", "t1c", "t2w"):
        _nifti(cases / "demo" / f"{suffix}.nii.gz", volume, affine)
    case_out = outputs / "demo"
    case_out.mkdir(parents=True)
    (case_out / "scene.glb").write_bytes(b"glb")
    (case_out / "manifest.json").write_text(
        json.dumps({"case_id": "demo", "case_dir": str(cases / "demo"), "structures": {}}),
        encoding="utf-8",
    )
    pred = np.zeros(volume.shape, dtype=np.uint8)
    pred[0:5, 9, 7] = 2
    _nifti(case_out / "pred_seg.nii.gz", pred, affine)
    client = create_app(outputs, cases).test_client()

    planes = ((2, 7), (1, 9), (0, 0))
    for axis, index in planes:
        expected = window_to_uint8(orient_ras_slice(volume, axis, index), 0.0, 100.0)
        response = client.get(f"/api/cases/demo/slice?modality=flair&axis={axis}&index={index}&lo=0&hi=100")
        assert response.status_code == 200
        image = np.asarray(Image.open(io.BytesIO(response.data)))
        gray = image if image.ndim == 2 else image[..., 0]
        np.testing.assert_array_equal(gray, expected)
        if image.ndim == 3:
            alpha = image[..., 3]
            expected_alpha = (orient_ras_slice(volume, axis, index) != 0).astype(np.uint8) * 255
            np.testing.assert_array_equal(alpha, expected_alpha)
        if axis == 0:
            assert expected[0, 0] == 255
        else:
            assert expected[0, -1] == 255


def test_brats_training_002_contour_matches_wt_mesh_within_2mm():
    case = ROOT / "outputs" / "BraTS20_Training_002"
    pred_path = case / "pred_seg.nii.gz"
    mesh_path = case / "edema.glb"
    if not pred_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError("BraTS20_Training_002 outputs are required for this test")
    import trimesh

    pred = nib.as_closest_canonical(nib.load(str(pred_path)))
    label = np.rint(np.squeeze(pred.get_fdata())).astype(np.int16)
    affine = np.asarray(pred.affine, dtype=np.float64)
    wt = np.isin(label, (1, 2, 3, 4))
    coords = np.argwhere(wt)
    centroid = coords.mean(axis=0)
    axis = 2
    index = int(np.clip(np.rint(centroid[2]), 0, label.shape[2] - 1))
    contour = tumor_contour_world_points(label, affine, axis, index)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = list(mesh.geometry.values())[0]
    plane = mesh_plane_points(mesh, affine, label.shape, axis, index)
    # 15% trim: the edema mesh is Taubin-smoothed and torus-like, so a thin
    # WT finger and interior plane cuts would otherwise dominate the mean.
    distance = mean_boundary_distance(contour, plane, trim=0.15)
    assert np.isfinite(distance)
    assert distance < 2.0, f"mean boundary distance {distance:.3f} mm"


if __name__ == "__main__":
    test_orient_ras_slice_puts_las_marker_on_the_lettered_corner()
    test_api_png_matches_oriented_numpy_pixel_exact()
    test_brats_training_002_contour_matches_wt_mesh_within_2mm()
    print("slices ok")
