"""Smooth mask -> marching cubes -> Taubin (no Laplacian) -> world mm.

Nested shells are the whole tumor, the tumor core, and enhancing tumor:
edema = WT, core = TC, enhancing = ET.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from recon.labels import CORE, EDEMA, ENHANCING, rgba

SHELL_NAMES = (EDEMA, CORE, ENHANCING)


class MeshCheck(NamedTuple):
    watertight: bool
    volume_error_pct: float


def mesh_from_mask(mask, affine, sigma=0.7, taubin_iters=15, decimate_to=None):
    """Binary mask to a mesh in scanner millimeters.

    Gaussian smooth (sigma 0.7), pad, marching cubes at 0.5, Taubin smooth,
    then the RAS affine. Optional quadric decimation when ``decimate_to`` is a
    face count. No Laplacian smoothing.
    """
    if np.asarray(mask).sum() == 0:
        return None
    field = ndimage.gaussian_filter(np.asarray(mask, dtype=np.float32), sigma)
    field = np.pad(field, 1)  # close meshes that meet the volume border
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5)
    verts -= 1.0  # undo the pad; vertices are still in voxel units
    mesh = trimesh.Trimesh(verts, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=taubin_iters)
    mesh.apply_transform(np.asarray(affine, dtype=np.float64))
    if decimate_to and len(mesh.faces) > decimate_to:
        mesh = mesh.simplify_quadric_decimation(face_count=decimate_to)
    trimesh.repair.fix_normals(mesh)
    return mesh


def check_mesh(mesh, mask, affine) -> MeshCheck:
    """Watertight flag and signed volume error (%) against the voxel volume.

    Voxel volume is ``mask.sum() * |det(affine[:3, :3])|`` in mm³. The error
    is undefined when the mesh is open or the mask is empty.
    """
    vox_vol = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    v_vox = float(np.asarray(mask).sum()) * vox_vol
    watertight = bool(mesh.is_watertight)
    if not watertight or v_vox == 0.0:
        return MeshCheck(watertight, float("nan"))
    err = 100.0 * (abs(float(mesh.volume)) - v_vox) / v_vox
    return MeshCheck(watertight, float(err))


def region_shells(label) -> dict[str, np.ndarray]:
    """Nested shells. Edema is WT, core is TC, enhancing is ET."""
    from recon.evaluate import regions_from_label

    regions = regions_from_label(label)
    return {
        EDEMA: regions["WT"],
        CORE: regions["TC"],
        ENHANCING: regions["ET"],
    }


def _paint(mesh, name):
    if mesh is not None:
        mesh.visual.face_colors = rgba(name)
    return mesh


def mesh_shells(label, affine, **kwargs) -> dict[str, trimesh.Trimesh | None]:
    """One closed shell per nested region. Empty regions are ``None``."""
    return {
        name: _paint(mesh_from_mask(mask, affine, **kwargs), name)
        for name, mask in region_shells(label).items()
    }


def iter_parcellation_meshes(parc, affine, **kwargs):
    """Mesh each known anatomical label.

    Sigma and the 50-voxel cutoff live in ``recon.anatomy.mesh_parcellation_labels``.
    """
    from recon.anatomy import mesh_parcellation_labels

    yield from mesh_parcellation_labels(parc, affine)
