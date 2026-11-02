"""Optional brain surface, SynthSeg structure meshes, and a full parcellation.

These shells stay out of the scene unless ``reconstruct_tumor`` is called with
``include_anatomy=True`` (the UI toggle; ``build_scene`` defaults on).

SynthSeg labels follow the FreeSurfer lookup the robust SynthSeg model writes.
Lateral ventricles include the inferior lateral ventricle. Basal ganglia are
caudate, putamen, pallidum, and accumbens. Third and fourth ventricles are
not part of the lateral-ventricle shell.

``parcellate_case`` runs the DIPY PyTorch SynthSeg port. That port has no
Desikan-Killiany switch: cortex is one label per hemisphere. When the loaded
model cannot emit cortical parcels, subcortical SynthSeg labels are kept and
the frontal, parietal, temporal, occipital, and insular lobes are painted from
the FSL MNI structural atlas warped with the ANTs transforms cached by
``recon.tracts``. The result is resampled with nearest neighbours onto the
tumor segmentation grid, the dilated whole-tumor mask is cleared from every
label, and ``anatomy_seg.nii.gz`` is written for the case.
"""

from __future__ import annotations

import inspect
import logging
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nibabel.processing import resample_from_to
from scipy import ndimage

from recon.mesh import mesh_from_mask
from recon.postprocess import CONN

logger = logging.getLogger(__name__)

# Glass cortex: light smooth so gyri stay. Heavy smoothing lived here before.
BRAIN_SIGMA = 0.5
BRAIN_TAUBIN_ITERS = 10
BRAIN_FACE_BUDGET = 300_000
TUMOR_DILATION_VOXELS = 2
GLASS_SIGMA = 0.5
GLASS_TAUBIN_ITERS = 10
HEMISPHERE_FACE_BUDGET = 300_000

# Cortex is smoothed enough to look anatomical. A higher sigma would round
# neighbouring parcels into each other. Deep nuclei use a smaller sigma so the
# surface does not shrink off the structure.
CORTEX_SIGMA = 1.0
DEEP_SIGMA = 0.6
DEFAULT_STRUCTURE_SIGMA = 0.7
MIN_STRUCTURE_VOXELS = 50
_DEEP_LEGENDS = frozenset(
    {"Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala"}
)

# name -> SynthSeg / FreeSurfer label ids (left and right combined)
SYNTHSEG_STRUCTURES = {
    "lateral_ventricles": (4, 5, 43, 44),
    "brainstem": (16,),
    "cerebellum": (7, 8, 46, 47),
    "thalamus": (10, 49),
    "basal_ganglia": (11, 12, 13, 26, 50, 51, 52, 58),
}



def mesh_brain_surface(brain_mask, affine):
    """Legacy wrapper. Glass hemispheres use :func:`mesh_hemisphere_shell`."""
    mask = np.squeeze(np.asarray(brain_mask))
    if mask.ndim != 3:
        raise ValueError(f"Brain mask must be 3D, got {mask.shape}")
    return mesh_from_mask(
        mask.astype(bool),
        affine,
        sigma=GLASS_SIGMA,
        taubin_iters=GLASS_TAUBIN_ITERS,
        decimate_to=HEMISPHERE_FACE_BUDGET,
    )


# Grey-matter glass shell: cerebral cortex only. White matter is a second
# inner glass layer. Ventricles, CSF, cerebellum, and brainstem are excluded.
_LEFT_HEMISPHERE_IDS = (3,)
_RIGHT_HEMISPHERE_IDS = (42,)
_WM_IDS = (2, 41, 7, 46)
_VENTRICLE_AND_CSF = (4, 5, 14, 15, 24, 43, 44)
_LOBE_IDS = (101, 102, 103, 104, 105)
# Outer GM glass, WM glass, cerebellum, brainstem, CSF, and lobe/DK paints
# are not meshed as solid interiors.
_GLASS_SKIP_IDS = frozenset((2, 3, 7, 8, 16, 24, 41, 42, 46, 47) + _LOBE_IDS)
WM_SIGMA = 0.6
WM_TAUBIN_ITERS = 10
WM_FACE_BUDGET = 200_000
BLOOD_MIN_VOXELS = 50


def _dilated_tumor(tumor_label, dilation_voxels=TUMOR_DILATION_VOXELS) -> np.ndarray:
    tumor = _whole_tumor(tumor_label)
    if dilation_voxels:
        return ndimage.binary_dilation(tumor, structure=CONN, iterations=int(dilation_voxels))
    return tumor.astype(bool)


def _hemisphere_core(parc, side) -> np.ndarray:
    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    ids = _LEFT_HEMISPHERE_IDS if side == "left" else _RIGHT_HEMISPHERE_IDS
    tissue = np.isin(labels, ids)
    if side == "left":
        tissue |= (labels >= 1000) & (labels < 2000)
    else:
        tissue |= (labels >= 2000) & (labels < 3000)
    lobe = np.isin(labels, _LOBE_IDS)
    if not np.any(lobe):
        return tissue
    left_core = np.isin(labels, _LEFT_HEMISPHERE_IDS)
    right_core = np.isin(labels, _RIGHT_HEMISPHERE_IDS)
    if np.any(left_core) or np.any(right_core):
        left_d = ndimage.distance_transform_edt(~left_core) if np.any(left_core) else np.full(labels.shape, np.inf)
        right_d = ndimage.distance_transform_edt(~right_core) if np.any(right_core) else np.full(labels.shape, np.inf)
        nearer_left = left_d <= right_d
        tissue |= lobe & (nearer_left if side == "left" else ~nearer_left)
        return tissue
    mid = labels.shape[0] / 2.0
    xs = np.indices(labels.shape)[0]
    tissue |= lobe & ((xs < mid) if side == "left" else (xs >= mid))
    return tissue


def hemisphere_tissue_mask(parc, side, tumor_label) -> np.ndarray:
    """Grey-matter cortex for one hemisphere, with the tumor filled so the glass has no hole."""
    tissue = _hemisphere_core(parc, side)
    other = _hemisphere_core(parc, "right" if side == "left" else "left")
    tumor = _dilated_tumor(tumor_label)
    if not np.any(tumor):
        return tissue
    if np.any(tissue) or np.any(other):
        this_d = ndimage.distance_transform_edt(~tissue) if np.any(tissue) else np.full(tissue.shape, np.inf)
        other_d = ndimage.distance_transform_edt(~other) if np.any(other) else np.full(tissue.shape, np.inf)
        return tissue | (tumor & (this_d <= other_d))
    return tissue | tumor


def mesh_hemisphere_shell(parc, tumor_label, affine, side):
    """Folded grey-matter glass: sigma 0.5, Taubin 10, at most ~300k faces."""
    mask = hemisphere_tissue_mask(parc, side, tumor_label)
    logger.info("meshing %s hemisphere (%d voxels)", side, int(np.count_nonzero(mask)))
    mesh = mesh_from_mask(
        mask,
        affine,
        sigma=GLASS_SIGMA,
        taubin_iters=GLASS_TAUBIN_ITERS,
        decimate_to=HEMISPHERE_FACE_BUDGET,
    )
    if mesh is None:
        return None, mask
    paint_vertex_lobe_ids(mesh, parc, affine)
    return mesh, mask


def mesh_glass_part(parc, tumor_label, affine, label_ids):
    """Cerebellum or brainstem glass mesh. Tumor voxels that touch it are filled."""
    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    raw = np.isin(labels, label_ids)
    if not np.any(raw):
        return None, raw
    margin = _dilated_tumor(tumor_label)
    mask = raw | (margin & ndimage.binary_dilation(raw, structure=CONN, iterations=max(TUMOR_DILATION_VOXELS, 1)))
    mesh = mesh_from_mask(
        mask,
        affine,
        sigma=GLASS_SIGMA,
        taubin_iters=GLASS_TAUBIN_ITERS,
        decimate_to=HEMISPHERE_FACE_BUDGET,
    )
    return mesh, mask


def white_matter_mask(parc, tumor_label) -> np.ndarray:
    """Cerebral and cerebellar white matter with the dilated tumor carved out."""
    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    raw = np.isin(labels, _WM_IDS)
    return raw & ~_dilated_tumor(tumor_label)


def mesh_white_matter(parc, tumor_label, affine):
    """Warm inner glass layer: sigma 0.6, tumor carved with a 2-voxel margin."""
    mask = white_matter_mask(parc, tumor_label)
    if int(np.count_nonzero(mask)) < MIN_STRUCTURE_VOXELS:
        return None, mask
    logger.info("meshing white matter (%d voxels)", int(np.count_nonzero(mask)))
    mesh = mesh_from_mask(
        mask,
        affine,
        sigma=WM_SIGMA,
        taubin_iters=WM_TAUBIN_ITERS,
        decimate_to=WM_FACE_BUDGET,
    )
    return mesh, mask


# Thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens.
_DEEP_GM_IDS = (10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58)


def grey_matter_volume_mask(parc, tumor_label, side) -> np.ndarray:
    """Carved cortical grey-matter voxels, not the filled glass shell."""
    return _hemisphere_core(parc, side) & ~_dilated_tumor(tumor_label)


def deep_grey_matter_volume_mask(parc, tumor_label) -> np.ndarray:
    """Deep nuclei included in Grey matter total, with the dilated tumor carved out."""
    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    return np.isin(labels, _DEEP_GM_IDS) & ~_dilated_tumor(tumor_label)


def grey_matter_total_volume_mask(parc, tumor_label) -> np.ndarray:
    """Cortical GM (both hemispheres) plus deep GM."""
    cortical = grey_matter_volume_mask(parc, tumor_label, "left") | grey_matter_volume_mask(
        parc, tumor_label, "right"
    )
    return cortical | deep_grey_matter_volume_mask(parc, tumor_label)


def possible_blood_products_mask(t1, t1c, tumor_label, parc) -> np.ndarray:
    """T1-hyperintense non-enhancing voxels inside the predicted tumor.

    Bright on native T1 (above the 99th percentile of normal-appearing white
    matter) and not enhancing on T1c relative to T1. Components under 50
    voxels are dropped. This is an experimental heuristic, not a hemorrhage
    label.
    """
    from recon.evaluate import regions_from_label
    from recon.postprocess import remove_small_components

    native = np.asarray(t1, dtype=np.float32)
    contrast = np.asarray(t1c, dtype=np.float32)
    if native.shape != contrast.shape:
        raise ValueError(f"T1 shape {native.shape} does not match T1c {contrast.shape}")
    wt = regions_from_label(tumor_label)["WT"]
    if wt.shape != native.shape:
        raise ValueError(f"Tumor shape {wt.shape} does not match T1 {native.shape}")
    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16) if parc is not None else None
    finite = np.isfinite(native) & np.isfinite(contrast)
    nawm = np.zeros(native.shape, dtype=bool)
    if labels is not None and labels.shape == native.shape:
        nawm = np.isin(labels, (2, 41)) & ~wt & finite
    if int(np.count_nonzero(nawm)) < 500:
        positive = finite & (native > 0) & ~wt
        if np.any(positive):
            nawm = positive
        else:
            nawm = finite & ~wt
    if not np.any(nawm) or not np.any(wt):
        return np.zeros(native.shape, dtype=bool)
    t1_thr = float(np.percentile(native[nawm], 99))
    delta = contrast - native
    delta_thr = float(np.percentile(delta[nawm], 95))
    candidate = wt & finite & (native > t1_thr) & (delta <= delta_thr)
    return remove_small_components(candidate, BLOOD_MIN_VOXELS)


def paint_vertex_lobe_ids(mesh, parc, affine):
    """Nearest lobe (or DK) label at each vertex, stored as vertex colors."""
    from recon.labels import anatomy as anatomy_by_id, rgba

    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    homo = np.column_stack([verts, np.ones(len(verts))])
    vox = (inv @ homo.T).T[:, :3]
    ijk = np.rint(vox).astype(np.int32)
    for axis in range(3):
        ijk[:, axis] = np.clip(ijk[:, axis], 0, labels.shape[axis] - 1)

    lobe_present = np.isin(labels, _LOBE_IDS) | ((labels >= 1000) & (labels < 3000))
    if np.any(lobe_present):
        _dist, inds = ndimage.distance_transform_edt(~lobe_present, return_indices=True)
        nearest = labels[tuple(inds)]
        ids = nearest[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    else:
        ids = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    colors[:, 3] = 255
    for index, label_id in enumerate(ids):
        spec = anatomy_by_id(int(label_id))
        if spec is None:
            colors[index] = (159, 184, 214, 255)
            continue
        colors[index, :3] = spec.rgb
    mesh.visual.vertex_colors = colors
    mesh.vertex_attributes["lobe_id"] = ids.astype(np.int16)
    return mesh


def _whole_tumor(tumor_label) -> np.ndarray:
    from recon.evaluate import regions_from_label

    return regions_from_label(tumor_label)["WT"]


def cleaned_structure_masks(synthseg, tumor_label, dilation_voxels=TUMOR_DILATION_VOXELS):
    """SynthSeg structures with the dilated whole-tumor mask removed.

    Returns ``(masks, removed_voxels)``. ``removed_voxels`` counts how many
    structure voxels fell inside the tumor after a 26-connected dilation, and
    the same count is written to the ``recon.anatomy`` log.
    """
    parc = np.rint(np.squeeze(np.asarray(synthseg))).astype(np.int16)
    if parc.ndim != 3:
        raise ValueError(f"SynthSeg parcellation must be 3D, got {parc.shape}")
    tumor = _whole_tumor(tumor_label)
    if tumor.shape != parc.shape:
        raise ValueError(f"SynthSeg shape {parc.shape} does not match tumor shape {tumor.shape}")
    if dilation_voxels < 0:
        raise ValueError(f"dilation_voxels must be >= 0, got {dilation_voxels}")
    margin = (
        ndimage.binary_dilation(tumor, structure=CONN, iterations=int(dilation_voxels))
        if dilation_voxels
        else np.zeros(tumor.shape, dtype=bool)
    )
    masks = {}
    removed = {}
    for name, ids in SYNTHSEG_STRUCTURES.items():
        raw = np.isin(parc, ids)
        removed[name] = int(np.count_nonzero(raw & margin))
        masks[name] = raw & ~margin
        logger.info(
            "%s: removed %d voxels inside the tumor dilated by %d voxels",
            name,
            removed[name],
            int(dilation_voxels),
        )
    return masks, removed


def structure_sigma(structure) -> float:
    """Smoothing width for one parcellation label."""
    if structure.group == "Cortex":
        return CORTEX_SIGMA
    if structure.legend_name in _DEEP_LEGENDS:
        return DEEP_SIGMA
    return DEFAULT_STRUCTURE_SIGMA


def centroid_mm(mask, affine) -> list[float]:
    """Mean voxel of ``mask``, mapped through ``affine`` into millimeters."""
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if len(coords) == 0:
        return [0.0, 0.0, 0.0]
    center = coords.mean(axis=0)
    hom = np.array([center[0], center[1], center[2], 1.0], dtype=np.float64)
    world = np.asarray(affine, dtype=np.float64) @ hom
    return [float(world[0]), float(world[1]), float(world[2])]


def ensure_scene_group(scene, group):
    """Create an empty parent node so glTF export can group meshes under it."""
    if group is None:
        return
    if group not in scene.graph.transforms.parents:
        scene.graph.update(frame_to=group, matrix=np.eye(4))


def mesh_parcellation_labels(parc, affine, *, min_voxels=MIN_STRUCTURE_VOXELS):
    """Mesh every known parcellation label with ``mesh_from_mask``.

    Cortex uses sigma 1.0. Thalamus, caudate, putamen, pallidum, hippocampus,
    and amygdala use sigma 0.6. Other labels use sigma 0.7. Labels under
    ``min_voxels`` are logged and skipped. Each kept mesh is checked against
    its voxel volume.

    Yields ``(structure, mask, mesh)``.
    """
    from recon.evaluate import check_mesh_volume
    from recon.labels import rgba, structures_in_volume

    for structure, mask in structures_in_volume(parc):
        label_id = structure.label_id
        if label_id is not None and (
            int(label_id) in _GLASS_SKIP_IDS or 1000 <= int(label_id) < 3000 or structure.group == "Cortex"
        ):
            continue
        voxels = int(np.count_nonzero(mask))
        if voxels < int(min_voxels):
            logger.info(
                "Skipping %s (label %s, %s): %d voxels is under %d",
                structure.mesh_name,
                structure.label_id,
                structure.group,
                voxels,
                int(min_voxels),
            )
            continue
        mesh = mesh_from_mask(mask, affine, sigma=structure_sigma(structure))
        if mesh is None:
            logger.info("Skipping %s: mesh_from_mask returned no surface", structure.mesh_name)
            continue
        mesh.visual.face_colors = rgba(structure)
        check_mesh_volume(mesh, mask, affine, structure.mesh_name)
        yield structure, mask, mesh


def mesh_synthseg_structures(synthseg, tumor_label, affine):
    """Mesh each cleaned structure. Empty results are ``None``.

    Returns ``(meshes, removed_voxels)``.
    """
    masks, removed = cleaned_structure_masks(synthseg, tumor_label)
    meshes = {name: mesh_from_mask(mask, affine) for name, mask in masks.items()}
    return meshes, removed


# DIPY SynthSeg robust labels. Cortex is one region per hemisphere (3, 42),
# not a Desikan-Killiany parcel. Names match ``dipy.nn.torch.synthseg``.
SYNTHSEG_LABEL_NAMES = {
    0: "background",
    2: "left cerebral white matter",
    3: "left cerebral cortex",
    4: "left lateral ventricle",
    5: "left inferior lateral ventricle",
    7: "left cerebellum white matter",
    8: "left cerebellum cortex",
    10: "left thalamus",
    11: "left caudate",
    12: "left putamen",
    13: "left pallidum",
    14: "3rd ventricle",
    15: "4th ventricle",
    16: "brain-stem",
    17: "left hippocampus",
    18: "left amygdala",
    24: "CSF",
    26: "left accumbens area",
    28: "left ventral DC",
    41: "right cerebral white matter",
    42: "right cerebral cortex",
    43: "right lateral ventricle",
    44: "right inferior lateral ventricle",
    46: "right cerebellum white matter",
    47: "right cerebellum cortex",
    49: "right thalamus",
    50: "right caudate",
    51: "right putamen",
    52: "right pallidum",
    53: "right hippocampus",
    54: "right amygdala",
    58: "right accumbens area",
    60: "right ventral DC",
}

# Replaced by lobe labels when a cortical parcellation is available.
CORTEX_LABELS = (3, 42)

# FSL MNI structural max-prob atlas. XML indices are 0-based; the summary
# image (MNI-maxprob-thr25-1mm) stores index + 1.
MNI_LOBE_INDEX = {
    3: "frontal",
    4: "insula",
    5: "occipital",
    6: "parietal",
    8: "temporal",
}

# FreeSurfer subcortical ids stop near 60, so these do not collide.
LOBE_LABELS = {
    "frontal": 101,
    "parietal": 102,
    "temporal": 103,
    "occipital": 104,
    "insula": 105,
}

# Bilateral MNI structures and their FreeSurfer (left, right) ids. The atlas
# has one label for both sides, so the side comes from the MNI x coordinate.
MNI_PAIRED_INDEX = {
    1: (11, 50),  # caudate
    2: (8, 47),  # cerebellum
    7: (12, 51),  # putamen
    9: (10, 49),  # thalamus
}
_RIGHT_OFFSET = 10

_DK_MIN = 1000
_AFFINE_ATOL = 1e-4
_CORTICAL_FLAGS = ("parc", "parcellate", "cortical_parcellation")
_STRUCTURAL_NAMES = (
    "MNI/MNI-maxprob-thr25-1mm.nii.gz",
    "MNI/MNI-maxprob-thr25-1mm.nii",
    "MNI/MNI-maxprob-thr0-1mm.nii.gz",
    "MNI/MNI-maxprob-thr0-1mm.nii",
    "MNI/MNI-maxprob-thr50-1mm.nii.gz",
    "MNI/MNI-maxprob-thr50-1mm.nii",
    "MNI-maxprob-thr25-1mm.nii.gz",
    "MNI-maxprob-thr0-1mm.nii.gz",
    "MNI-maxprob-thr50-1mm.nii.gz",
    "MNI/MNI-prob-1mm.nii.gz",
    "MNI/MNI-prob-1mm.nii",
)


def select_device(device=None, *, cuda_available=None, mps_available=None) -> torch.device:
    """``cuda``, then ``mps``, then ``cpu``."""
    if device is not None:
        return torch.device(device)
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if mps_available is None:
        mps = getattr(torch.backends, "mps", None)
        mps_available = mps is not None and mps.is_available()
    if cuda_available:
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def cortical_parcellation_supported(model) -> bool:
    """True when this SynthSeg object can emit Desikan-Killiany cortex."""
    predict = getattr(model, "predict", None)
    if predict is not None:
        try:
            names = inspect.signature(predict).parameters
        except (TypeError, ValueError):
            names = {}
        if any(name in names for name in _CORTICAL_FLAGS):
            return True
    labels = getattr(model, "labels_segmentation", None)
    if labels is not None and np.any(np.asarray(labels) >= _DK_MIN):
        return True
    return False


def _has_dk(labels) -> bool:
    return bool(np.any(np.asarray(labels) >= _DK_MIN))


def _labels(data) -> np.ndarray:
    """3D integer labels. A singleton channel axis is dropped; spatial axes are not."""
    array = np.asarray(data)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Parcellation must be 3D, got {array.shape}")
    return np.rint(array).astype(np.int16)


def _label_names(extra=None) -> dict[int, str]:
    names = dict(SYNTHSEG_LABEL_NAMES)
    names.update({label_id: name for name, label_id in LOBE_LABELS.items()})
    if extra:
        for key, value in extra.items():
            names[int(key)] = str(value)
    return names


def resample_parcellation(data, src_affine, target_shape, target_affine) -> np.ndarray:
    """Nearest-neighbour resample onto the tumor segmentation grid.

    The returned volume's affine is the tumor affine. A mismatch raises.
    """
    labels = _labels(data)
    src_affine = np.asarray(src_affine, dtype=np.float64)
    target_affine = np.asarray(target_affine, dtype=np.float64)
    target_shape = tuple(int(v) for v in target_shape)
    if labels.shape == target_shape and np.allclose(src_affine, target_affine, atol=_AFFINE_ATOL):
        return labels
    image = nib.Nifti1Image(labels, src_affine)
    resampled = resample_from_to(image, (target_shape, target_affine), order=0, cval=0.0)
    out = np.rint(np.asanyarray(resampled.dataobj)).astype(np.int16)
    if out.shape != target_shape or not np.allclose(resampled.affine, target_affine, atol=_AFFINE_ATOL):
        raise ValueError(
            f"Parcellation affine does not match the tumor segmentation "
            f"(shape {out.shape} vs {target_shape})"
        )
    return out


def paint_lobes(synthseg, mni_labels) -> np.ndarray:
    """Keep SynthSeg labels and replace cortex with MNI structural lobes."""
    out = _labels(synthseg)
    lobes = _labels(mni_labels)
    if out.shape != lobes.shape:
        raise ValueError(f"MNI lobe map shape {lobes.shape} does not match SynthSeg shape {out.shape}")
    cortex = np.isin(out, CORTEX_LABELS)
    for index, name in MNI_LOBE_INDEX.items():
        out[cortex & (lobes == index)] = np.int16(LOBE_LABELS[name])
    return out


def excise_tumor(parc, tumor_label, names=None, dilation_voxels=TUMOR_DILATION_VOXELS):
    """Zero every anatomical label inside the dilated whole-tumor mask.

    Returns ``(cleaned, removed_voxels)`` and logs one line per structure.
    """
    labels = _labels(parc)
    tumor = _whole_tumor(tumor_label)
    if tumor.shape != labels.shape:
        raise ValueError(f"Parcellation shape {labels.shape} does not match tumor shape {tumor.shape}")
    if dilation_voxels < 0:
        raise ValueError(f"dilation_voxels must be >= 0, got {dilation_voxels}")
    margin = (
        ndimage.binary_dilation(tumor, structure=CONN, iterations=int(dilation_voxels))
        if dilation_voxels
        else np.zeros(tumor.shape, dtype=bool)
    )
    cleaned = labels.copy()
    cleaned[margin] = 0
    lookup = _label_names(names)
    removed = {}
    for label_id in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        count = int(np.count_nonzero((labels == label_id) & margin))
        structure = lookup.get(label_id, f"label_{label_id}")
        removed[structure] = count
        logger.info(
            "%s: removed %d voxels inside the tumor dilated by %d voxels",
            structure,
            count,
            int(dilation_voxels),
        )
    return cleaned, removed


def resolve_structural_atlas(atlas_dir=None) -> tuple[Path, Path]:
    """Return ``(MNI152 T1, MNI structural atlas)`` or raise ``AtlasNotFoundError``."""
    from recon.tracts import AtlasNotFoundError, default_atlas_dir

    root = Path(atlas_dir) if atlas_dir is not None else default_atlas_dir()
    template = None
    for name in ("MNI152_T1_1mm.nii.gz", "MNI152_T1_1mm.nii"):
        path = root / name
        if path.is_file():
            template = path
            break
    structural = None
    for name in _STRUCTURAL_NAMES:
        path = root / name
        if path.is_file():
            structural = path
            break
    missing = []
    if template is None:
        missing.append(root / "MNI152_T1_1mm.nii.gz")
    if structural is None:
        missing.append(root / "MNI" / "MNI-maxprob-thr25-1mm.nii.gz")
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise AtlasNotFoundError(
            "The FSL MNI structural atlas is missing. This project does not download it.\n"
            "Place MNI152_T1_1mm.nii.gz and MNI/MNI-maxprob-thr25-1mm.nii.gz as described in docs/ATLAS_SETUP.md.\n"
            f"Missing:\n{lines}",
            missing,
        )
    return template, structural


def split_hemispheres(labels, affine) -> np.ndarray:
    """Add ``_RIGHT_OFFSET`` to paired MNI structures right of the MNI midline."""
    out = _labels(labels)
    affine = np.asarray(affine, dtype=np.float64)
    i, j, k = np.indices(out.shape, sparse=True)
    world_x = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + affine[0, 3]
    right = np.isin(out, tuple(MNI_PAIRED_INDEX)) & (world_x > 0)
    out[right] += np.int16(_RIGHT_OFFSET)
    return out


def atlas_parcellation(warped) -> np.ndarray:
    """FreeSurfer-style labels from a hemisphere-split MNI atlas alone."""
    mni = _labels(warped)
    out = np.zeros(mni.shape, dtype=np.int16)
    for index, name in MNI_LOBE_INDEX.items():
        out[mni == index] = np.int16(LOBE_LABELS[name])
    for index, (left, right) in MNI_PAIRED_INDEX.items():
        out[mni == index] = np.int16(left)
        out[mni == index + _RIGHT_OFFSET] = np.int16(right)
    return out


def _as_label_nifti(path: Path, *, split=False) -> tuple[Path, Path | None]:
    """3D max-prob labels. A 4D probability atlas becomes argmax labels (index + 1)."""
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 3:
        if not split:
            return path, None
        winner = _labels(data)
    elif data.ndim == 4:
        probs = np.asarray(data, dtype=np.float32)
        winner = np.argmax(probs, axis=-1).astype(np.int16) + 1
        winner[np.max(probs, axis=-1) <= 0] = 0
    else:
        raise ValueError(f"MNI structural atlas must be 3D or 4D, got {data.shape} at {path}")
    if split:
        winner = split_hemispheres(winner, image.affine)
    temporary = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temporary.close()
    dest = Path(temporary.name)
    nib.save(nib.Nifti1Image(winner, image.affine), str(dest))
    return dest, dest


def warp_mni_lobes(
    t1_path, tumor_label, *, atlas_dir=None, cache_dir=None, case_id=None, split=False
):
    """Warp the MNI structural atlas with the cached tract transforms.

    Interpolation is nearest neighbour. Returns ``(labels, patient_affine)``.
    With ``split``, paired structures right of the midline carry index + 10.
    Aborts after 20 minutes so a stuck ANTs SyN cannot block a batch.
    """
    from recon.timing import STAGE_TIMEOUT_S, run_with_timeout

    def _run():
        return _warp_mni_lobes_impl(
            t1_path,
            tumor_label,
            atlas_dir=atlas_dir,
            cache_dir=cache_dir,
            case_id=case_id,
            split=split,
        )

    return run_with_timeout(STAGE_TIMEOUT_S, _run)


def _warp_mni_lobes_impl(
    t1_path, tumor_label, *, atlas_dir=None, cache_dir=None, case_id=None, split=False
):
    from recon.tracts import _import_ants, _tumor_wt, patient_mni_transforms

    template, structural = resolve_structural_atlas(atlas_dir)
    ants = _import_ants()
    fixed = ants.image_read(str(t1_path))
    _tumor_wt(tumor_label, fixed.shape)
    transforms = patient_mni_transforms(
        t1_path,
        tumor_label,
        template,
        cache_dir=cache_dir,
        case_id=case_id,
        ants=ants,
        fixed=fixed,
    )
    label_path, temporary = _as_label_nifti(structural, split=split)
    try:
        moving = ants.image_read(str(label_path))
        warped = ants.apply_transforms(fixed, moving, transforms, interpolator="nearestNeighbor")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    labels = np.rint(np.asarray(warped.numpy())).astype(np.int16)
    # The array is on the T1 voxel grid. ANTs reports LPS; meshing uses the
    # nibabel RAS affine of that T1, so resampling with the ANTs affine
    # lands the lobes in empty space.
    t1_affine = np.asarray(nib.load(str(t1_path)).affine, dtype=np.float64)
    return labels, t1_affine


def _import_synthseg():
    try:
        from dipy.nn.torch.synthseg import SynthSeg
    except ImportError as exc:
        raise ImportError(
            "DIPY's PyTorch SynthSeg is required (dipy.nn.torch.synthseg). Install dipy."
        ) from exc
    return SynthSeg


def _bind_cpu_numpy(model, device: torch.device) -> None:
    """Run the port on ``device`` and copy probabilities to CPU before NumPy.

    DIPY's ``__predict`` calls ``.numpy()`` on the model output. That fails
    for CUDA and MPS tensors.
    """
    model.device = torch.device(device)
    model.model = model.model.to(model.device)

    def _predict(x_test):
        return model.model(x_test).detach().to("cpu").numpy()

    model._SynthSeg__predict = _predict


def _unpack_synthseg(result, model) -> tuple[np.ndarray, dict]:
    names = {}
    if isinstance(result, tuple):
        labels = result[0]
        if len(result) > 1 and isinstance(result[1], dict):
            names = result[1]
    else:
        labels = result
    if not names:
        names = getattr(model, "label_dict", None) or {}
    parsed = {int(key): str(value) for key, value in names.items()}
    return np.rint(np.squeeze(np.asarray(labels))).astype(np.int16), parsed


def run_synthseg(t1, affine, device):
    """Segment one T1 with DIPY SynthSeg, enabling cortex if the port has it.

    Returns ``(labels, names, has_cortical_parcellation)``.
    """
    SynthSeg = _import_synthseg()
    model = SynthSeg(verbose=False, use_cuda=device.type == "cuda")
    _bind_cpu_numpy(model, device)
    cortical = cortical_parcellation_supported(model)
    kwargs = {}
    if cortical:
        params = inspect.signature(model.predict).parameters
        for name in _CORTICAL_FLAGS:
            if name in params:
                kwargs[name] = True
                break
    result = model.predict(np.asarray(t1, dtype=np.float32), np.asarray(affine, dtype=np.float64), **kwargs)
    labels, names = _unpack_synthseg(result, model)
    return labels, names, cortical and _has_dk(labels)


def save_anatomy_seg(label, reference_path, out_path, tumor_seg_path=None) -> None:
    """Write ``label`` into the reference NIfTI grid and check the tumor affine.

    Labels stay int16 so lobe ids and Desikan-Killiany ids are preserved.
    When ``tumor_seg_path`` is set, the written affine and shape must match
    that tumor segmentation.
    """
    orig = nib.load(str(reference_path))
    canon = nib.as_closest_canonical(orig)
    transform = ornt_transform(io_orientation(canon.affine), io_orientation(orig.affine))
    native = apply_orientation(np.asarray(label), transform)
    if native.shape != orig.shape[:3]:
        raise ValueError(
            f"Reoriented parcellation shape {native.shape} does not match {reference_path} shape {orig.shape[:3]}"
        )
    nib.save(nib.Nifti1Image(np.asarray(native, dtype=np.int16), orig.affine), str(out_path))
    if tumor_seg_path is None:
        return
    saved = nib.load(str(out_path))
    tumor = nib.load(str(tumor_seg_path))
    if saved.shape[:3] != tumor.shape[:3] or not np.allclose(saved.affine, tumor.affine, atol=_AFFINE_ATOL):
        raise ValueError("anatomy_seg affine does not match the tumor segmentation")


def _write_ras(label, affine, out_path) -> None:
    nib.save(
        nib.Nifti1Image(np.asarray(label, dtype=np.int16), np.asarray(affine, dtype=np.float64)),
        str(out_path),
    )
    saved = nib.load(str(out_path))
    if saved.shape[:3] != tuple(np.asarray(label).shape) or not np.allclose(saved.affine, affine, atol=_AFFINE_ATOL):
        raise ValueError("anatomy_seg affine does not match the tumor segmentation")


def parcellate_case(
    t1_path,
    tumor_label,
    tumor_affine,
    *,
    out_path=None,
    reference_path=None,
    tumor_seg_path=None,
    case_id=None,
    atlas_dir=None,
    cache_dir=None,
    device=None,
    synthseg_runner=None,
    on_lobes=None,
):
    """Build one case's anatomical parcellation and optionally save it.

    ``tumor_label`` and ``tumor_affine`` are the BraTS segmentation grid.
    The return value is ``(raw, cleaned, removed_voxels)``. ``raw`` still
    contains tumor overlap. ``cleaned`` is what ``anatomy_seg.nii.gz`` stores.

    ``synthseg_runner`` replaces DIPY in tests. It is
    ``(t1, affine, device) -> (labels, names, has_cortical)``.
    """
    t1_path = Path(t1_path)
    if not t1_path.is_file():
        raise FileNotFoundError(f"Patient T1 was not found: {t1_path}")
    chosen = select_device(device)
    logger.info("anatomy device=%s", chosen.type)
    t1_image = nib.load(str(t1_path))
    t1 = np.squeeze(np.asarray(t1_image.get_fdata(dtype=np.float32)))
    t1_affine = np.asarray(t1_image.affine, dtype=np.float64)
    tumor_shape = np.squeeze(np.asarray(tumor_label)).shape
    if synthseg_runner is None and chosen.type == "cpu":
        logger.info("CPU: warping the MNI structural atlas for lobes (SynthSeg is reserved for a GPU)")
        return _parcellate_from_atlas(
            t1_path, t1, tumor_label, tumor_shape, tumor_affine,
            atlas_dir, cache_dir, case_id, out_path, reference_path, tumor_seg_path,
            on_lobes=on_lobes,
        )
    runner = synthseg_runner if synthseg_runner is not None else run_synthseg
    try:
        synthseg, names, reported_cortical = runner(t1, t1_affine, chosen)
    except Exception as exc:
        if synthseg_runner is not None:
            raise
        logger.warning(
            "SynthSeg is unavailable (%s); using the warped MNI structural atlas alone", exc
        )
        return _parcellate_from_atlas(
            t1_path, t1, tumor_label, tumor_shape, tumor_affine,
            atlas_dir, cache_dir, case_id, out_path, reference_path, tumor_seg_path,
            on_lobes=on_lobes,
        )
    synthseg = _labels(synthseg)
    if synthseg.shape != t1.shape:
        raise ValueError(f"SynthSeg shape {synthseg.shape} does not match the T1 grid {t1.shape}")
    on_grid = resample_parcellation(synthseg, t1_affine, tumor_shape, tumor_affine)
    if on_lobes is not None:
        on_lobes()
    if _has_dk(on_grid):
        logger.info("SynthSeg cortical parcellation enabled")
        raw = on_grid
    else:
        if reported_cortical:
            logger.info("SynthSeg reported cortical parcellation but returned no Desikan-Killiany labels")
        logger.info(
            "SynthSeg has no Desikan-Killiany cortical parcellation; "
            "warping the FSL MNI structural atlas for lobe labels"
        )
        warped, warped_affine = warp_mni_lobes(
            t1_path,
            tumor_label,
            atlas_dir=atlas_dir,
            cache_dir=cache_dir,
            case_id=case_id,
        )
        lobes = resample_parcellation(warped, warped_affine, on_grid.shape, tumor_affine)
        raw = paint_lobes(on_grid, lobes)
    return _finish_parcellation(
        raw, tumor_label, names, tumor_affine, out_path, reference_path, tumor_seg_path
    )


def _parcellate_from_atlas(
    t1_path, t1, tumor_label, tumor_shape, tumor_affine,
    atlas_dir, cache_dir, case_id, out_path, reference_path, tumor_seg_path,
    on_lobes=None,
):
    """Lobes and deep nuclei from the FSL MNI structural atlas, no SynthSeg."""
    if on_lobes is not None:
        on_lobes()
    warped, warped_affine = warp_mni_lobes(
        t1_path,
        tumor_label,
        atlas_dir=atlas_dir,
        cache_dir=cache_dir,
        case_id=case_id,
        split=True,
    )
    atlas = atlas_parcellation(warped)
    if atlas.shape == np.asarray(t1).shape:
        atlas[np.asarray(t1) <= 0] = 0
    raw = resample_parcellation(atlas, warped_affine, tumor_shape, tumor_affine)
    return _finish_parcellation(
        raw, tumor_label, None, tumor_affine, out_path, reference_path, tumor_seg_path
    )


def _finish_parcellation(raw, tumor_label, names, tumor_affine, out_path, reference_path, tumor_seg_path):
    cleaned, removed = excise_tumor(raw, tumor_label, names)
    if out_path is not None and reference_path is not None:
        save_anatomy_seg(cleaned, reference_path, out_path, tumor_seg_path)
    elif out_path is not None:
        _write_ras(cleaned, tumor_affine, out_path)
    return raw, cleaned, removed


def iter_glass_shells(parc, tumor_label, affine):
    """Yield ``(structure, mask, mesh)`` for GM hemispheres, WM, cerebellum, and brainstem."""
    from recon.labels import LEFT_CORTEX, RIGHT_CORTEX, WHITE_MATTER, get

    for side, spec in (("left", LEFT_CORTEX), ("right", RIGHT_CORTEX)):
        mesh, mask = mesh_hemisphere_shell(parc, tumor_label, affine, side)
        if mesh is not None:
            yield spec, mask, mesh
    wm_mesh, wm_mask = mesh_white_matter(parc, tumor_label, affine)
    if wm_mesh is not None:
        yield WHITE_MATTER, wm_mask, wm_mesh
    for label_ids, mesh_name in (
        ((8,), "Left Cerebellar Cortex"),
        ((47,), "Right Cerebellar Cortex"),
        ((16,), "Brainstem"),
    ):
        mesh, mask = mesh_glass_part(parc, tumor_label, affine, label_ids)
        if mesh is None:
            continue
        yield get(mesh_name), mask, mesh


def add_anatomy_meshes(scene, affine, tumor_label, brain_mask=None, synthseg=None):
    """Add glass hemispheres and deep-structure meshes.

    Names and colors come from ``recon.labels``. Left and right stay separate.
    Returns the per-label removed-voxel counts (empty if no parcellation).
    """
    from recon.labels import rgba

    if synthseg is None:
        logger.info("include_anatomy: no SynthSeg parcellation, skipping structures")
        if brain_mask is not None:
            brain = mesh_brain_surface(brain_mask, affine)
            if brain is not None:
                from recon.labels import BRAIN

                brain.visual.face_colors = rgba(BRAIN)
                ensure_scene_group(scene, BRAIN.group)
                scene.add_geometry(
                    brain,
                    geom_name=BRAIN.mesh_name,
                    node_name=BRAIN.mesh_name,
                    parent_node_name=BRAIN.group,
                )
        return {}

    for structure, _mask, mesh in iter_glass_shells(synthseg, tumor_label, affine):
        colors = getattr(mesh.visual, "vertex_colors", None)
        if colors is None or len(np.asarray(colors)) == 0:
            mesh.visual.face_colors = rgba(structure)
        ensure_scene_group(scene, structure.group)
        scene.add_geometry(
            mesh,
            geom_name=structure.mesh_name,
            node_name=structure.mesh_name,
            parent_node_name=structure.group,
        )

    cleaned, removed = excise_tumor(synthseg, tumor_label)
    for structure, _mask, mesh in mesh_parcellation_labels(cleaned, affine):
        ensure_scene_group(scene, structure.group)
        scene.add_geometry(
            mesh,
            geom_name=structure.mesh_name,
            node_name=structure.mesh_name,
            parent_node_name=structure.group,
        )
    return removed
