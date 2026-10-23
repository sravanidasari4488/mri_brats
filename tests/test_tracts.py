"""Atlas path checks, tract metadata, cache reuse, and tumor proximity."""

import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.tracts import (
    SOURCE,
    AtlasNotFoundError,
    AtlasTract,
    compute_tract_proximity,
    resolve_atlas,
    to_probability,
    warp_atlas_tracts,
)


def _write_nifti(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)


class _Image:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)
        self.shape = self._array.shape
        self.spacing = (1.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)
        self.direction = np.eye(3)

    def numpy(self):
        return self._array

    def new_image_like(self, array):
        return _Image(array)


def test_missing_atlas_names_the_setup_doc():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resolve_atlas(tmp)
            raise AssertionError("expected AtlasNotFoundError")
        except AtlasNotFoundError as exc:
            text = str(exc)
        assert "docs/ATLAS_SETUP.md" in text
        assert "does not download" in text
        assert "MNI152_T1_1mm.nii.gz" in text
        assert "optic_radiation_L.nii.gz" in text
        assert "optic_radiation_R.nii.gz" in text


def test_four_d_atlas_still_requires_optic_radiation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_nifti(root / "MNI152_T1_1mm.nii.gz", np.zeros((4, 4, 4), dtype=np.float32))
        _write_nifti(root / "jhu" / "JHU-ICBM-tracts-prob-2mm.nii.gz", np.zeros((4, 4, 4, 20), dtype=np.float32))
        try:
            resolve_atlas(root)
            raise AssertionError("expected AtlasNotFoundError")
        except AtlasNotFoundError as exc:
            missing = [path.name for path in exc.missing]
        assert "optic_radiation_L.nii.gz" in missing
        assert "optic_radiation_R.nii.gz" in missing
        assert "corticospinal_tract_L.nii.gz" not in missing
        assert "MNI152_T1_1mm.nii.gz" not in missing


def test_percent_maps_use_a_probability_threshold():
    percent = np.array([0, 24, 25, 100], dtype=np.float32)
    scaled = to_probability(percent)
    assert np.allclose(scaled, [0, 0.24, 0.25, 1.0])
    unit = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    assert np.allclose(to_probability(unit), unit)


def test_tract_metadata_and_proximity_in_millimeters():
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    label = np.zeros((6, 2, 2), dtype=np.uint8)
    label[5, 0, 0] = 2
    label[5, 0, 1] = 3
    mask = np.zeros(label.shape, dtype=bool)
    mask[5, 0, 1] = True
    affine = np.eye(4)
    affine[0, 0] = 2.0
    tract = AtlasTract(name="corticospinal_tract_L", mesh=mesh, mask=mask, threshold=0.25, affine=affine)
    assert tract.source == SOURCE
    assert tract.mesh.metadata["source"] == SOURCE

    report = compute_tract_proximity(label, {"corticospinal_tract_L": tract}, affine)["corticospinal_tract_L"]
    # Voxel (5, 0, 0) is whole tumor only. The affine scales x by 2 mm.
    assert abs(report["wt_distance_mm"] - 10.0) < 1e-6
    assert abs(report["et_distance_mm"] - np.sqrt(10.0**2 + 1.0**2)) < 1e-6
    assert report["wt_overlaps"]
    assert report["et_overlaps"]

    far = AtlasTract(
        name="uncinate_L",
        mesh=mesh,
        mask=np.zeros(label.shape, dtype=bool),
        threshold=0.25,
        affine=affine,
    )
    quiet = compute_tract_proximity(label, {"uncinate_L": far}, affine)["uncinate_L"]
    assert not quiet["wt_overlaps"]
    assert not quiet["et_overlaps"]


def test_warp_caches_syn_transforms_and_stamps_meshes():
    import recon.tracts as tracts

    calls = {"register": 0}

    class _Ants:
        def image_read(self, path):
            data = np.asanyarray(nib.load(path).dataobj)
            if data.ndim == 4:
                data = data[..., 0]
            return _Image(data)

        def registration(self, fixed, moving, type_of_transform, mask):
            assert type_of_transform == "SyN"
            calls["register"] += 1
            healthy = mask.numpy()
            assert healthy[2, 2, 2] == 0
            assert healthy[14, 14, 14] == 1
            path = Path(tempfile.gettempdir()) / "mri_brats_fake_syn.mat"
            path.write_text("syn", encoding="utf-8")
            calls["temp"] = path
            return {"fwdtransforms": [str(path)]}

        def apply_transforms(self, fixed, moving, transformlist, interpolator):
            assert interpolator == "linear"
            assert transformlist
            return _Image(moving.numpy())

    original = tracts._import_ants
    tracts._import_ants = lambda: _Ants()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (16, 16, 16)
            t1 = np.ones(shape, dtype=np.float32)
            _write_nifti(root / "t1.nii.gz", t1)
            prob = np.zeros(shape, dtype=np.float32)
            prob[8:14, 8:14, 8:14] = 50
            tract_path = root / "cst.nii.gz"
            _write_nifti(tract_path, prob)
            _write_nifti(root / "mni.nii.gz", t1)
            label = np.zeros(shape, dtype=np.uint8)
            label[1:4, 1:4, 1:4] = 1
            cache = root / "cache"
            kwargs = dict(
                mni_template_path=root / "mni.nii.gz",
                tract_prob_paths={"corticospinal_tract_L": tract_path},
                out_dir=cache,
                case_id="caseA",
                threshold=0.25,
            )
            first = warp_atlas_tracts(root / "t1.nii.gz", label, **kwargs)
            calls["temp"].unlink()
            second = warp_atlas_tracts(root / "t1.nii.gz", label, **kwargs)
            label[8, 8, 8] = 3
            warp_atlas_tracts(root / "t1.nii.gz", label, **kwargs)
    finally:
        tracts._import_ants = original

    assert calls["register"] == 2
    tract = first["corticospinal_tract_L"]
    assert tract.source == SOURCE
    assert tract.mesh is not None
    assert tract.mesh.metadata["source"] == SOURCE
    assert second["corticospinal_tract_L"].mesh.metadata["source"] == SOURCE
    assert tract.mask[10, 10, 10]
    assert not tract.mask[0, 0, 0]


if __name__ == "__main__":
    test_missing_atlas_names_the_setup_doc()
    test_four_d_atlas_still_requires_optic_radiation()
    test_percent_maps_use_a_probability_threshold()
    test_tract_metadata_and_proximity_in_millimeters()
    test_warp_caches_syn_transforms_and_stamps_meshes()
    print("tracts ok")
