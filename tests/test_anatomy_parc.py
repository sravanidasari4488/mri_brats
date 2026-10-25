"""Full parcellation: device order, nearest resample, lobes, tumor cutout, cache."""

import logging
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.anatomy import (
    LOBE_LABELS,
    cortical_parcellation_supported,
    excise_tumor,
    paint_lobes,
    parcellate_case,
    resample_parcellation,
    resolve_structural_atlas,
    select_device,
)
from recon.tracts import AtlasNotFoundError, _cache_paths, _store_transforms


def _write(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine),
        path,
    )


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


def test_device_prefers_cuda_then_mps_then_cpu():
    assert select_device(cuda_available=True, mps_available=True).type == "cuda"
    assert select_device(cuda_available=False, mps_available=True).type == "mps"
    assert select_device(cuda_available=False, mps_available=False).type == "cpu"
    assert select_device("cpu").type == "cpu"


def test_cortical_flag_is_absent_on_the_robust_port():
    class _Robust:
        def predict(self, T1, affine, *, batch_size=None, return_prob=False):
            return T1

        labels_segmentation = np.array([0, 2, 3, 42])

    class _Parc:
        def predict(self, T1, affine, *, parc=False):
            return T1

    assert not cortical_parcellation_supported(_Robust())
    assert cortical_parcellation_supported(_Parc())


def test_resample_is_nearest_and_uses_the_tumor_affine():
    source = np.zeros((2, 1, 1), dtype=np.int16)
    source[0, 0, 0] = 5
    source[1, 0, 0] = 9
    src_affine = np.eye(4)
    src_affine[0, 0] = 2.0
    target_affine = np.eye(4)
    out = resample_parcellation(source, src_affine, (4, 1, 1), target_affine)
    assert out.shape == (4, 1, 1)
    assert set(np.unique(out)).issubset({0, 5, 9})
    assert 5 in out and 9 in out
    assert not np.any((out != 0) & (out != 5) & (out != 9))
    same = resample_parcellation(out, target_affine, out.shape, target_affine)
    assert np.array_equal(same, out)


def test_lobes_replace_cortex_and_tumor_is_logged():
    shape = (12, 12, 12)
    synth = np.zeros(shape, dtype=np.int16)
    synth[1, 1, 1] = 3
    synth[6, 6, 4] = 3
    synth[10, 10, 10] = 10
    mni = np.zeros(shape, dtype=np.int16)
    mni[1, 1, 1] = 3
    mni[6, 6, 4] = 3
    mni[10, 10, 10] = 9
    painted = paint_lobes(synth, mni)
    assert painted[1, 1, 1] == LOBE_LABELS["frontal"]
    assert painted[6, 6, 4] == LOBE_LABELS["frontal"]
    assert painted[10, 10, 10] == 10

    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[6, 6, 6] = 1
    messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    log = logging.getLogger("recon.anatomy")
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        cleaned, removed = excise_tumor(painted, tumor, {10: "left thalamus"})
    finally:
        log.removeHandler(handler)

    assert cleaned[6, 6, 4] == 0
    assert cleaned[1, 1, 1] == LOBE_LABELS["frontal"]
    assert cleaned[10, 10, 10] == 10
    assert removed["frontal"] == 1
    assert removed["left thalamus"] == 0
    logged = " ".join(messages)
    assert "frontal: removed 1 voxels inside the tumor dilated by 2 voxels" in logged
    assert "left thalamus: removed 0 voxels" in logged


def test_missing_structural_atlas_names_the_setup_doc():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resolve_structural_atlas(tmp)
            raise AssertionError("expected AtlasNotFoundError")
        except AtlasNotFoundError as exc:
            text = str(exc)
    assert "docs/ATLAS_SETUP.md" in text
    assert "does not download" in text
    assert "MNI-maxprob-thr25-1mm.nii.gz" in text


def test_parcellate_reuses_cached_transforms_and_writes_anatomy_seg():
    import recon.tracts as tracts

    shape = (12, 12, 12)
    synth = np.zeros(shape, dtype=np.int16)
    synth[1, 1, 1] = 3
    synth[6, 6, 4] = 42
    synth[10, 10, 10] = 10
    lobes = np.zeros(shape, dtype=np.int16)
    lobes[1, 1, 1] = 3
    lobes[6, 6, 4] = 8
    lobes[10, 10, 10] = 9
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[6, 6, 6] = 1
    calls = {"register": 0, "nearest": 0}

    def runner(t1, affine, device):
        assert device.type == "cpu"
        assert t1.shape == shape
        return synth, {10: "left thalamus"}, False

    class _Ants:
        def image_read(self, path):
            data = np.asanyarray(nib.load(path).dataobj)
            if data.ndim == 4:
                data = data[..., 0]
            return _Image(data)

        def registration(self, fixed, moving, type_of_transform, mask):
            calls["register"] += 1
            path = Path(tempfile.gettempdir()) / "mri_brats_lobe_syn.mat"
            path.write_text("syn", encoding="utf-8")
            return {"fwdtransforms": [str(path)]}

        def apply_transforms(self, fixed, moving, transformlist, interpolator):
            assert interpolator == "nearestNeighbor"
            assert transformlist
            calls["nearest"] += 1
            return _Image(lobes)

    original = tracts._import_ants
    tracts._import_ants = lambda: _Ants()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.eye(4)
            _write(root / "t1.nii.gz", np.ones(shape, dtype=np.float32), affine)
            template = root / "MNI152_T1_1mm.nii.gz"
            _write(template, np.ones(shape, dtype=np.float32), affine)
            _write(root / "MNI" / "MNI-maxprob-thr25-1mm.nii.gz", lobes, affine)
            cache = root / "cache"
            cache_case, _expected = _cache_paths(cache, "caseA", template, tumor)
            stored = root / "syn.mat"
            stored.write_text("syn", encoding="utf-8")
            _store_transforms(cache_case, _expected, [str(stored)])
            out = root / "anatomy_seg.nii.gz"
            raw, cleaned, removed = parcellate_case(
                root / "t1.nii.gz",
                tumor,
                affine,
                out_path=out,
                case_id="caseA",
                atlas_dir=root,
                cache_dir=cache,
                device="cpu",
                synthseg_runner=runner,
            )
            saved = nib.load(out)
            assert calls["register"] == 0
            assert calls["nearest"] == 1
            assert np.allclose(saved.affine, affine)
            assert saved.shape[:3] == shape
            assert raw[10, 10, 10] == 10
            assert raw[1, 1, 1] == LOBE_LABELS["frontal"]
            assert raw[6, 6, 4] == LOBE_LABELS["temporal"]
            assert cleaned[6, 6, 4] == 0
            assert cleaned[1, 1, 1] == LOBE_LABELS["frontal"]
            assert int(np.rint(saved.get_fdata())[1, 1, 1]) == LOBE_LABELS["frontal"]
            assert removed["temporal"] == 1
            assert removed["left thalamus"] == 0

            parcellate_case(
                root / "t1.nii.gz",
                tumor,
                affine,
                case_id="caseB",
                atlas_dir=root,
                cache_dir=cache,
                device="cpu",
                synthseg_runner=runner,
            )
            assert calls["register"] == 1
    finally:
        tracts._import_ants = original


def test_desikan_labels_skip_the_lobe_warp():
    shape = (4, 4, 4)
    synth = np.zeros(shape, dtype=np.int16)
    synth[1, 1, 1] = 1003
    tumor = np.zeros(shape, dtype=np.uint8)

    def runner(t1, affine, device):
        return synth, {1003: "ctx-lh-superiorfrontal"}, True

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "t1.nii.gz", np.ones(shape, dtype=np.float32))
        _raw, cleaned, removed = parcellate_case(
            root / "t1.nii.gz",
            tumor,
            np.eye(4),
            case_id="dk",
            atlas_dir=root,
            device="cpu",
            synthseg_runner=runner,
        )
    assert cleaned[1, 1, 1] == 1003
    assert removed["ctx-lh-superiorfrontal"] == 0
