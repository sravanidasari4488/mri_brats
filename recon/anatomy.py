"""Optional brain surface, SynthSeg structure meshes, and a full parcellation.

These shells stay out of the scene unless ``reconstruct_tumor`` is called with
``include_anatomy=True`` (the UI toggle, default off).

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

# Stronger than the tumor shells (sigma 0.7, 15 Taubin iterations).
BRAIN_SIGMA = 1.5
BRAIN_TAUBIN_ITERS = 30
BRAIN_FACE_BUDGET = 150_000
TUMOR_DILATION_VOXELS = 2

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
    """Transparent outer shell: sigma 1.5, extra Taubin, about 150k faces."""
    mask = np.squeeze(np.asarray(brain_mask))
    if mask.ndim != 3:
        raise ValueError(f"Brain mask must be 3D, got {mask.shape}")
    return mesh_from_mask(
        mask.astype(bool),
        affine,
        sigma=BRAIN_SIGMA,
        taubin_iters=BRAIN_TAUBIN_ITERS,
        decimate_to=BRAIN_FACE_BUDGET,
    )


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


def _as_label_nifti(path: Path) -> tuple[Path, Path | None]:
    """3D max-prob labels. A 4D probability atlas becomes argmax labels (index + 1)."""
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 3:
        return path, None
    if data.ndim != 4:
        raise ValueError(f"MNI structural atlas must be 3D or 4D, got {data.shape} at {path}")
    probs = np.asarray(data, dtype=np.float32)
    winner = np.argmax(probs, axis=-1).astype(np.int16) + 1
    winner[np.max(probs, axis=-1) <= 0] = 0
    temporary = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    temporary.close()
    dest = Path(temporary.name)
    nib.save(nib.Nifti1Image(winner, image.affine), str(dest))
    return dest, dest


def warp_mni_lobes(t1_path, tumor_label, *, atlas_dir=None, cache_dir=None, case_id=None):
    """Warp the MNI structural atlas with the cached tract transforms.

    Interpolation is nearest neighbour. Returns ``(labels, patient_affine)``.
    """
    from recon.tracts import _import_ants, _tumor_wt, affine_from_ants, patient_mni_transforms

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
    label_path, temporary = _as_label_nifti(structural)
    try:
        moving = ants.image_read(str(label_path))
        warped = ants.apply_transforms(fixed, moving, transforms, interpolator="nearestNeighbor")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    labels = np.rint(np.asarray(warped.numpy())).astype(np.int16)
    return labels, affine_from_ants(fixed)


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
    runner = synthseg_runner if synthseg_runner is not None else run_synthseg
    synthseg, names, reported_cortical = runner(t1, t1_affine, chosen)
    synthseg = _labels(synthseg)
    if synthseg.shape != t1.shape:
        raise ValueError(f"SynthSeg shape {synthseg.shape} does not match the T1 grid {t1.shape}")
    tumor_shape = np.squeeze(np.asarray(tumor_label)).shape
    on_grid = resample_parcellation(synthseg, t1_affine, tumor_shape, tumor_affine)
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
    cleaned, removed = excise_tumor(raw, tumor_label, names)
    if out_path is not None and reference_path is not None:
        save_anatomy_seg(cleaned, reference_path, out_path, tumor_seg_path)
    elif out_path is not None:
        _write_ras(cleaned, tumor_affine, out_path)
    return raw, cleaned, removed


def add_anatomy_meshes(scene, affine, tumor_label, brain_mask=None, synthseg=None):
    """Add the brain shell and one mesh per anatomical label.

    Names and colors come from ``recon.labels``. Left and right stay separate.
    Returns the per-label removed-voxel counts (empty if no parcellation).
    """
    from recon.labels import BRAIN, rgba

    if brain_mask is not None:
        brain = mesh_brain_surface(brain_mask, affine)
        if brain is not None:
            brain.visual.face_colors = rgba(BRAIN)
            ensure_scene_group(scene, BRAIN.group)
            scene.add_geometry(
                brain,
                geom_name=BRAIN.mesh_name,
                node_name=BRAIN.mesh_name,
                parent_node_name=BRAIN.group,
            )
    else:
        logger.info("include_anatomy: no brain mask, skipping the outer shell")

    if synthseg is None:
        logger.info("include_anatomy: no SynthSeg parcellation, skipping structures")
        return {}

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
