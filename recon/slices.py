"""MRI slice PNGs with radiological RAS orientation and tumor-contour overlays.

Volumes are RAS: +i right, +j anterior, +k superior. One function,
:func:`orient_ras_slice`, rotates every plane for both the 2D panels and
the 3D plane textures.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SEQUENCES = {
    "flair": "t2f",
    "t2f": "t2f",
    "t1": "t1n",
    "t1n": "t1n",
    "t1c": "t1c",
    "t2": "t2w",
    "t2w": "t2w",
}

SEQUENCE_ORDER = ("flair", "t1", "t1c", "t2")

# Screen-edge letters after :func:`orient_ras_slice`.
ORIENT_LETTERS = {
    0: {"left": "A", "right": "P", "top": "S", "bottom": "I"},
    1: {"left": "R", "right": "L", "top": "S", "bottom": "I"},
    2: {"left": "R", "right": "L", "top": "A", "bottom": "P"},
}

_OVERLAY = {
    2: (245, 197, 66, 255),
    1: (192, 57, 43, 255),
    3: (59, 130, 246, 255),
    4: (59, 130, 246, 255),
}


def resolve_sequence(name: str) -> str:
    key = (name or "flair").strip().lower()
    if key not in SEQUENCES:
        raise ValueError("modality must be flair, t1, t1c, or t2")
    return SEQUENCES[key]


def brain_window(volume, brain_mask=None) -> tuple[float, float]:
    """1st–99th percentile of brain voxels, or of foreground if no mask."""
    data = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(data)
    if brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool) & finite
    else:
        mask = finite & (data > np.nanmin(data))
    if int(np.count_nonzero(mask)) < 16:
        mask = finite
    sample = data[mask]
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, (1, 99))
    if hi <= lo:
        hi = float(lo) + 1.0
    return float(lo), float(hi)


def orient_ras_slice(volume, axis: int, index: int) -> np.ndarray:
    """Radiological RAS view of one plane, with explicit ``rot90`` / flip.

    Axial (fixed z): ``array[:, :, z]``, anterior up, patient left on screen right.
    Coronal (fixed y): ``array[:, y, :]``, superior up, radiological left/right.
    Sagittal (fixed x): ``array[x, :, :]``, superior up, anterior on screen left.
    """
    data = np.asarray(volume)
    axis = int(axis)
    index = int(index)
    if axis == 2:
        slab = data[:, :, index]
        return np.fliplr(np.rot90(slab, 1))
    if axis == 1:
        slab = data[:, index, :]
        return np.fliplr(np.rot90(slab, 1))
    if axis == 0:
        slab = data[index, :, :]
        return np.fliplr(np.rot90(slab, 1))
    raise ValueError("axis must be 0, 1, or 2")


def take_slice(volume, axis: int, index: int) -> np.ndarray:
    """Oriented RAS slab. Same as :func:`orient_ras_slice`."""
    return orient_ras_slice(volume, axis, index)


def oriented_pixel_to_ijk(axis: int, index: int, row: int, col: int, shape) -> tuple[int, int, int]:
    """Map a PNG pixel (row, col) back to RAS voxel indices."""
    nx, ny, nz = (int(n) for n in shape[:3])
    axis = int(axis)
    if axis == 2:
        return nx - 1 - int(col), ny - 1 - int(row), int(index)
    if axis == 1:
        return nx - 1 - int(col), int(index), nz - 1 - int(row)
    if axis == 0:
        return int(index), ny - 1 - int(col), nz - 1 - int(row)
    raise ValueError("axis must be 0, 1, or 2")


def voxel_to_world(affine, ijk) -> np.ndarray:
    hom = np.array([float(ijk[0]), float(ijk[1]), float(ijk[2]), 1.0], dtype=np.float64)
    return (np.asarray(affine, dtype=np.float64) @ hom)[:3]


def slice_texture_corners(affine, shape, axis: int, index: int) -> list[list[float]]:
    """World xyz of the oriented PNG's four UV corners (0,0), (1,0), (1,1), (0,1).

    ``flipY=false`` in three.js, so UV (0, 0) is the top-left PNG pixel.
    """
    rows, cols = orient_ras_slice(np.zeros(shape[:3], dtype=np.uint8), axis, index).shape
    pixels = ((0, 0), (0, cols - 1), (rows - 1, cols - 1), (rows - 1, 0))
    corners = []
    for row, col in pixels:
        ijk = oriented_pixel_to_ijk(axis, index, row, col, shape)
        corners.append(voxel_to_world(affine, ijk).tolist())
    return corners


def slice_extent_mm(affine, shape, axis: int) -> tuple[float, float]:
    """Width and height in millimetres of the oriented PNG."""
    from nibabel.affines import voxel_sizes

    spacing = [float(abs(v)) for v in voxel_sizes(np.asarray(affine, dtype=np.float64))]
    nx, ny, nz = (int(n) for n in shape[:3])
    if int(axis) == 2:
        return nx * spacing[0], ny * spacing[1]
    if int(axis) == 1:
        return nx * spacing[0], nz * spacing[2]
    return ny * spacing[1], nz * spacing[2]


def window_to_uint8(image, lo: float, hi: float) -> np.ndarray:
    finite = np.isfinite(image)
    scaled = np.zeros(image.shape, dtype=np.uint8)
    if not np.any(finite):
        return scaled
    if hi <= lo:
        scaled[finite & (image >= lo)] = 255
        return scaled
    values = np.clip((np.asarray(image, dtype=np.float32) - lo) / (hi - lo), 0, 1)
    scaled[finite] = (values[finite] * 255).astype(np.uint8)
    return scaled


def render_slice_png(volume, axis: int, index: int, lo: float, hi: float, alpha=None) -> bytes:
    image = orient_ras_slice(volume, axis, index)
    gray = window_to_uint8(image, lo, hi)
    if alpha is None:
        buf = io.BytesIO()
        Image.fromarray(gray, mode="L").save(buf, format="PNG")
        return buf.getvalue()
    mask = orient_ras_slice(np.asarray(alpha, dtype=np.float32), axis, index)
    rgba = np.zeros(gray.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def render_overlay_png(label, axis: int, index: int) -> bytes:
    """RGBA tumor contours. Transparent where there is no edge."""
    slab = orient_ras_slice(np.rint(np.asarray(label)).astype(np.int16), axis, index)
    rgba = np.zeros(slab.shape + (4,), dtype=np.uint8)
    for value in (2, 3, 1):
        mask = slab == value
        if value == 3:
            mask = (slab == 3) | (slab == 4)
        if not np.any(mask):
            continue
        eroded = ndimage.binary_erosion(mask)
        edge = mask & ~eroded
        if np.any(edge):
            edge = ndimage.binary_dilation(edge)
        rgba[edge] = _OVERLAY[value]
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def tumor_contour_world_points(label, affine, axis: int, index: int) -> np.ndarray:
    """World-mm coordinates of WT contour pixels on the oriented plane."""
    labels = np.rint(np.asarray(label)).astype(np.int16)
    slab = orient_ras_slice(labels, axis, index)
    whole = np.isin(slab, (1, 2, 3, 4))
    if not np.any(whole):
        return np.zeros((0, 3), dtype=np.float64)
    eroded = ndimage.binary_erosion(whole)
    edge = whole & ~eroded
    rows, cols = np.nonzero(edge)
    points = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        ijk = oriented_pixel_to_ijk(axis, index, row, col, labels.shape)
        points.append(voxel_to_world(affine, ijk))
    return np.asarray(points, dtype=np.float64)


def mesh_plane_points(mesh, affine, shape, axis: int, index: int) -> np.ndarray:
    """Points where ``mesh`` meets the RAS slice plane."""
    import trimesh

    origin = voxel_to_world(affine, [index if axis == 0 else 0, index if axis == 1 else 0, index if axis == 2 else 0])
    axis_vec = np.asarray(affine, dtype=np.float64)[:3, int(axis)]
    normal = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
    lines = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    if lines is None or len(lines) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(lines, dtype=np.float64).reshape(-1, 3)


def mean_boundary_distance(points_a, points_b, trim: float = 0.0) -> float:
    """Symmetric mean nearest-neighbour distance in millimetres.

    ``trim`` drops that fraction of the farthest matches on each side so a
    thin label finger that smoothing removed from the mesh does not dominate.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    from scipy.spatial import cKDTree

    def _mean(distances) -> float:
        values = np.sort(np.asarray(distances, dtype=np.float64))
        keep = max(1, int(round(len(values) * (1.0 - float(trim)))))
        return float(values[:keep].mean())

    ab = _mean(cKDTree(b).query(a, k=1)[0])
    ba = _mean(cKDTree(a).query(b, k=1)[0])
    return float(0.5 * (ab + ba))


def cache_path(cache_dir: Path, kind: str, axis: int, index: int, extra: str = "") -> Path:
    name = f"ras1_{kind}_a{int(axis)}_i{int(index)}"
    if extra:
        name += f"_{extra}"
    return cache_dir / f"{name}.png"


def window_token(lo: float, hi: float) -> str:
    return f"w{lo:.4g}_{hi:.4g}".replace(".", "p")
