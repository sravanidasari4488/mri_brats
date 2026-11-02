"""Brain-shell settings, SynthSeg tumor cutout, and the default-off toggle."""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.anatomy import (
    BRAIN_FACE_BUDGET,
    BRAIN_SIGMA,
    BRAIN_TAUBIN_ITERS,
    CORTEX_SIGMA,
    DEEP_SIGMA,
    DEFAULT_STRUCTURE_SIGMA,
    SYNTHSEG_STRUCTURES,
    WM_SIGMA,
    cleaned_structure_masks,
    deep_grey_matter_volume_mask,
    grey_matter_total_volume_mask,
    grey_matter_volume_mask,
    hemisphere_tissue_mask,
    iter_glass_shells,
    mesh_brain_surface,
    mesh_parcellation_labels,
    mesh_white_matter,
    paint_vertex_lobe_ids,
    possible_blood_products_mask,
    structure_sigma,
)
from recon.scene import reconstruct_tumor


def test_brain_surface_uses_stronger_smoothing_and_face_budget():
    import recon.anatomy as anatomy

    seen = {}
    original = anatomy.mesh_from_mask

    def fake(mask, affine, sigma=0.7, taubin_iters=15, decimate_to=None):
        seen["sigma"] = sigma
        seen["taubin_iters"] = taubin_iters
        seen["decimate_to"] = decimate_to
        seen["mask"] = mask
        return None

    anatomy.mesh_from_mask = fake
    try:
        mesh_brain_surface(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    finally:
        anatomy.mesh_from_mask = original
    assert seen["sigma"] == BRAIN_SIGMA == 0.5
    assert seen["taubin_iters"] == BRAIN_TAUBIN_ITERS == 10
    assert seen["decimate_to"] == BRAIN_FACE_BUDGET == 300_000
    assert seen["mask"].dtype == bool


def test_synthseg_subtracts_tumor_dilated_by_2_and_logs():
    shape = (12, 12, 12)
    parc = np.zeros(shape, dtype=np.int16)
    parc[5, 5, 3] = 4    # lateral ventricle, 2 voxels from the tumor (Chebyshev)
    parc[5, 5, 2] = 5    # inferior lateral ventricle, 3 voxels away
    parc[9, 9, 9] = 16   # brainstem, far away
    parc[1, 1, 1] = 14   # third ventricle: not a requested structure
    parc[2, 2, 2] = 26   # accumbens, part of basal ganglia
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[5, 5, 5] = 1

    messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    log = logging.getLogger("recon.anatomy")
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        masks, removed = cleaned_structure_masks(parc, tumor)
    finally:
        log.removeHandler(handler)

    assert removed["lateral_ventricles"] == 1
    assert not masks["lateral_ventricles"][5, 5, 3]
    assert masks["lateral_ventricles"][5, 5, 2]
    assert removed["brainstem"] == 0
    assert masks["brainstem"][9, 9, 9]
    assert masks["basal_ganglia"][2, 2, 2]
    assert not any(mask[1, 1, 1] for mask in masks.values())
    assert set(masks) == set(SYNTHSEG_STRUCTURES)
    logged = " ".join(messages)
    assert "lateral_ventricles: removed 1 voxels inside the tumor dilated by 2 voxels" in logged
    assert "brainstem: removed 0 voxels" in logged


def test_include_anatomy_is_off_unless_requested():
    shape = (16, 16, 16)
    label = np.zeros(shape, dtype=np.uint8)
    label[6:11, 6:11, 6:11] = 2
    brain = np.zeros(shape, dtype=bool)
    brain[2:14, 2:14, 2:14] = True
    parc = np.zeros(shape, dtype=np.int16)
    parc[0:4, 0:4, 0:4] = 16  # 64 voxels, above the 50-voxel mesh floor
    parc[15, 15, 15] = 10  # one thalamus voxel, logged and skipped
    affine = np.eye(4)
    with tempfile.TemporaryDirectory() as tmp:
        off = reconstruct_tumor(
            label,
            affine,
            str(Path(tmp) / "off.glb"),
            brain_mask=brain,
            synthseg=parc,
        )
        on = reconstruct_tumor(
            label,
            affine,
            str(Path(tmp) / "on.glb"),
            include_anatomy=True,
            brain_mask=brain,
            synthseg=parc,
        )
    assert "brain" not in off.geometry
    assert "Brainstem" not in off.geometry
    assert "edema" in off.geometry
    assert "brain" not in on.geometry
    assert "Brainstem" in on.geometry
    assert "Left Thalamus" not in on.geometry
    assert "Frontal Lobe" not in on.geometry
    assert on.geometry["Brainstem"].faces.shape[0] <= BRAIN_FACE_BUDGET


def test_parcellation_sigma_and_small_labels():
    from recon.labels import anatomy as label_of

    assert structure_sigma(label_of(3)) == CORTEX_SIGMA == 1.0
    assert structure_sigma(label_of(1028)) == CORTEX_SIGMA
    assert structure_sigma(label_of(10)) == DEEP_SIGMA == 0.6
    assert structure_sigma(label_of(49)) == DEEP_SIGMA
    assert structure_sigma(label_of(12)) == DEEP_SIGMA
    assert structure_sigma(label_of(17)) == DEEP_SIGMA
    assert structure_sigma(label_of(18)) == DEEP_SIGMA
    assert structure_sigma(label_of(16)) == DEFAULT_STRUCTURE_SIGMA == 0.7

    parc = np.zeros((8, 8, 8), dtype=np.int16)
    parc[:5, :4, :4] = 101  # lobe paint: not a solid mesh
    parc[:5, :4, 4:7] = 10  # thalamus, 60 voxels
    parc[7, 7, 7] = 11  # caudate, 1 voxel
    seen = []

    def fake(mask, affine, sigma=0.7, taubin_iters=15, decimate_to=None):
        seen.append((int(np.count_nonzero(mask)), sigma))
        return None

    import recon.anatomy as anatomy

    original = anatomy.mesh_from_mask
    anatomy.mesh_from_mask = fake
    messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    log = logging.getLogger("recon.anatomy")
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        assert list(mesh_parcellation_labels(parc, np.eye(4))) == []
    finally:
        anatomy.mesh_from_mask = original
        log.removeHandler(handler)

    assert (80, CORTEX_SIGMA) not in seen
    assert (60, DEEP_SIGMA) in seen
    assert all(count >= 50 for count, _sigma in seen)
    logged = " ".join(messages)
    assert "Skipping Left Caudate" in logged
    assert "1 voxels is under 50" in logged
    assert all("Frontal Lobe" not in message for message in messages)


def test_glass_hemisphere_fills_tumor_and_skips_solid_lobes():
    shape = (16, 16, 16)
    parc = np.zeros(shape, dtype=np.int16)
    parc[1:7, 2:8, 2:8] = 3
    parc[1:7, 2:8, 8:10] = 2
    parc[1:6, 2:7, 2:6] = 101
    parc[10:15, 2:8, 2:8] = 42
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[4:10, 4:10, 4:10] = 1
    left = hemisphere_tissue_mask(parc, "left", tumor)
    right = hemisphere_tissue_mask(parc, "right", tumor)
    assert left[5, 5, 5]
    assert not right[5, 5, 5]
    assert not left[1, 2, 9]
    assert not np.any(np.isin(parc, (4, 5, 14, 15, 24)) & left)

    names = [structure.mesh_name for structure, _mask, _mesh in mesh_parcellation_labels(parc, np.eye(4))]
    assert "Frontal Lobe" not in names
    assert "Left Cerebral Cortex" not in names

    mesh = __import__("trimesh").Trimesh(
        vertices=[[3.0, 4.0, 3.0], [3.0, 5.0, 3.0], [4.0, 4.0, 3.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    paint_vertex_lobe_ids(mesh, parc, np.eye(4))
    assert mesh.vertex_attributes["lobe_id"][0] == 101
    assert tuple(mesh.visual.vertex_colors[0, :3]) == (236, 118, 142)

    shells = [structure.mesh_name for structure, _mask, _mesh in iter_glass_shells(parc, tumor, np.eye(4))]
    assert "Left Cortex" in shells
    assert "Right Cortex" in shells
    assert "Frontal Lobe" not in shells
    parc[2:4, 2:4, 2:4] = 8
    parc[12:14, 2:4, 2:4] = 47
    parc[8:10, 8:10, 8:10] = 16
    names = [structure.mesh_name for structure, _mask, _mesh in iter_glass_shells(parc, tumor, np.eye(4))]
    assert "Left Cerebellar Cortex" in names
    assert "Right Cerebellar Cortex" in names
    assert "Brainstem" in names


def test_white_matter_uses_sigma_0_6_and_is_carved():
    import recon.anatomy as anatomy

    shape = (12, 12, 12)
    parc = np.zeros(shape, dtype=np.int16)
    parc[1:11, 1:11, 1:11] = 2
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[4:8, 4:8, 4:8] = 1
    seen = {}
    original = anatomy.mesh_from_mask

    def fake(mask, affine, sigma=0.7, taubin_iters=15, decimate_to=None):
        seen["sigma"] = sigma
        seen["voxels"] = int(np.count_nonzero(mask))
        return None

    anatomy.mesh_from_mask = fake
    try:
        mesh_white_matter(parc, tumor, np.eye(4))
    finally:
        anatomy.mesh_from_mask = original
    assert seen["sigma"] == WM_SIGMA == 0.6
    assert seen["voxels"] < int(np.count_nonzero(parc == 2))


def test_grey_matter_total_is_cortical_plus_deep():
    shape = (12, 12, 12)
    parc = np.zeros(shape, dtype=np.int16)
    parc[1:4, 1:4, 1:4] = 3
    parc[8:11, 1:4, 1:4] = 42
    parc[1:3, 8:11, 1:3] = 10
    parc[8:10, 8:11, 1:3] = 49
    parc[5:7, 5:7, 8:10] = 11
    tumor = np.zeros(shape, dtype=np.uint8)
    left = int(np.count_nonzero(grey_matter_volume_mask(parc, tumor, "left")))
    right = int(np.count_nonzero(grey_matter_volume_mask(parc, tumor, "right")))
    deep = int(np.count_nonzero(deep_grey_matter_volume_mask(parc, tumor)))
    total = int(np.count_nonzero(grey_matter_total_volume_mask(parc, tumor)))
    assert left == 27
    assert right == 27
    assert deep == 32
    assert total == left + right + deep


def test_possible_blood_products_are_t1_bright_and_not_enhancing():
    shape = (10, 10, 10)
    t1 = np.ones(shape, dtype=np.float32)
    t1c = np.ones(shape, dtype=np.float32)
    parc = np.zeros(shape, dtype=np.int16)
    parc[1:9, 1:9, 1:9] = 2
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[3:8, 3:8, 3:8] = 1
    t1[3:8, 3:8, 3:8] = 20
    t1c[3:8, 3:8, 3:8] = 20
    t1c[4, 4, 4] = 80
    mask = possible_blood_products_mask(t1, t1c, tumor, parc)
    assert int(np.count_nonzero(mask)) >= 50
    assert not mask[4, 4, 4]
    assert mask[5, 5, 5]
    blob = " ".join(
        [
            "Possible blood products (experimental, T1-hyperintense)",
            "Not validated — BraTS has no hemorrhage labels; confirmation needs SWI/GRE.",
        ]
    )
    assert "hemorrhage" in blob
    from recon.labels import BLOOD_MESH, BLOOD_NOTE

    assert BLOOD_MESH == "Possible blood products (experimental, T1-hyperintense)"
    assert "hemorrhage" not in BLOOD_MESH.lower()
    assert BLOOD_NOTE == "Not validated — BraTS has no hemorrhage labels; confirmation needs SWI/GRE."


if __name__ == "__main__":
    test_brain_surface_uses_stronger_smoothing_and_face_budget()
    test_synthseg_subtracts_tumor_dilated_by_2_and_logs()
    test_include_anatomy_is_off_unless_requested()
    test_parcellation_sigma_and_small_labels()
    test_glass_hemisphere_fills_tumor_and_skips_solid_lobes()
    test_white_matter_uses_sigma_0_6_and_is_carved()
    test_grey_matter_total_is_cortical_plus_deep()
    test_possible_blood_products_are_t1_bright_and_not_enhancing()
    print("anatomy ok")
