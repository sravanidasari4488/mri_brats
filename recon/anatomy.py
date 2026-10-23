"""Optional brain surface and SynthSeg structure meshes.

These shells stay out of the scene unless ``reconstruct_tumor`` is called with
``include_anatomy=True`` (the UI toggle, default off).

SynthSeg labels follow the FreeSurfer lookup the robust SynthSeg model writes.
Lateral ventricles include the inferior lateral ventricle. Basal ganglia are
caudate, putamen, pallidum, and accumbens. Third and fourth ventricles are
not part of the lateral-ventricle shell.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from recon.mesh import mesh_from_mask
from recon.postprocess import CONN

logger = logging.getLogger(__name__)

# Stronger than the tumor shells (sigma 0.7, 15 Taubin iterations).
BRAIN_SIGMA = 1.5
BRAIN_TAUBIN_ITERS = 30
BRAIN_FACE_BUDGET = 150_000
TUMOR_DILATION_VOXELS = 2

# name -> SynthSeg / FreeSurfer label ids (left and right combined)
SYNTHSEG_STRUCTURES = {
    "lateral_ventricles": (4, 5, 43, 44),
    "brainstem": (16,),
    "cerebellum": (7, 8, 46, 47),
    "thalamus": (10, 49),
    "basal_ganglia": (11, 12, 13, 26, 50, 51, 52, 58),
}

ANATOMY_COLORS = {
    "brain": (180, 170, 160, 40),
    "lateral_ventricles": (80, 170, 220, 200),
    "brainstem": (200, 160, 140, 220),
    "cerebellum": (190, 140, 120, 220),
    "thalamus": (160, 90, 160, 220),
    "basal_ganglia": (90, 160, 120, 220),
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


def mesh_synthseg_structures(synthseg, tumor_label, affine):
    """Mesh each cleaned structure. Empty results are ``None``.

    Returns ``(meshes, removed_voxels)``.
    """
    masks, removed = cleaned_structure_masks(synthseg, tumor_label)
    meshes = {name: mesh_from_mask(mask, affine) for name, mask in masks.items()}
    return meshes, removed


def add_anatomy_meshes(scene, affine, tumor_label, brain_mask=None, synthseg=None):
    """Add the brain shell and SynthSeg meshes onto an existing scene.

    Returns the per-structure removed-voxel counts (empty if no parcellation).
    """
    if brain_mask is not None:
        brain = mesh_brain_surface(brain_mask, affine)
        if brain is not None:
            brain.visual.face_colors = ANATOMY_COLORS["brain"]
            scene.add_geometry(brain, geom_name="brain", node_name="brain")
    else:
        logger.info("include_anatomy: no brain mask, skipping the outer shell")

    if synthseg is None:
        logger.info("include_anatomy: no SynthSeg parcellation, skipping structures")
        return {}

    meshes, removed = mesh_synthseg_structures(synthseg, tumor_label, affine)
    for name, mesh in meshes.items():
        if mesh is None:
            continue
        mesh.visual.face_colors = ANATOMY_COLORS[name]
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    return removed
