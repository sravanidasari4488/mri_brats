"""BraTS 2021+ NIfTI case loader.

Channel order matches the four-modality stack used to build the training
cache (FLAIR, T1, T1ce, T2), which is BraTS 2021+ ``t2f``, ``t1n``, ``t1c``,
``t2w``. Intensity normalization is the training transform
``NormalizeIntensityd(nonzero=True, channel_wise=True)``: per-channel z-score
of nonzero voxels, population standard deviation, zeros left at zero.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import NormalizeIntensity
from nibabel.affines import voxel_sizes

# FLAIR, T1, T1ce, T2 — training stack order, BraTS 2021+ filenames.
MODALITY_SUFFIXES: tuple[str, ...] = ("t2f", "t1n", "t1c", "t2w")
SEG_SUFFIX = "seg"

# Same arguments as get_train_transforms / get_val_transforms.
_normalize_like_training = NormalizeIntensity(nonzero=True, channel_wise=True)


def load_case(case_dir: str | Path, *, normalize: bool = True) -> dict:
    """Load one BraTS 2021+ case folder.

    Reads ``t2f``, ``t1n``, ``t1c``, ``t2w``, and ``seg``, reorients each
    volume to RAS with ``nib.as_closest_canonical``, and returns

    ``image`` ``(4, H, W, D)`` float32, ``seg`` ``(H, W, D)`` uint8,
    the RAS ``affine``, voxel ``spacing``, and ``case_id``.

    Labels stay 1 = NCR, 2 = ED, 3 = ET. Legacy ET label 4 is rewritten to 3.
    ``normalize=False`` leaves raw intensities for a model whose own
    preprocessing applies the z-score.
    """
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise NotADirectoryError(f"BraTS case folder not found: {case_dir}")

    channels: list[np.ndarray] = []
    affine: np.ndarray | None = None
    spatial_shape: tuple[int, ...] | None = None
    for suffix in MODALITY_SUFFIXES:
        data, vol_affine = _read_canonical(_find_volume(case_dir, suffix), label=False)
        _check_grid(suffix, data.shape, vol_affine, spatial_shape, affine)
        spatial_shape = data.shape
        affine = vol_affine
        channels.append(data.astype(np.float32, copy=False))

    seg, seg_affine = _read_canonical(_find_volume(case_dir, SEG_SUFFIX), label=True)
    _check_grid(SEG_SUFFIX, seg.shape, seg_affine, spatial_shape, affine)
    assert affine is not None

    stacked = np.stack(channels, axis=0)
    if normalize:
        image = np.asarray(_normalize_like_training(stacked), dtype=np.float32)
    else:
        image = np.asarray(stacked, dtype=np.float32)
    return {
        "image": np.ascontiguousarray(image),
        "seg": np.ascontiguousarray(seg),
        "affine": affine,
        "spacing": tuple(float(v) for v in voxel_sizes(affine)),
        "case_id": case_dir.name,
    }


def _find_volume(case_dir: Path, suffix: str) -> Path:
    suffix = suffix.lower()
    matches: list[Path] = []
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".nii.gz"):
            base = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            base = name[: -len(".nii")]
        else:
            continue
        if base == suffix or base.endswith("_" + suffix):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No {suffix} NIfTI in {case_dir} (expected *_{suffix}.nii or *_{suffix}.nii.gz)"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise FileNotFoundError(f"Multiple {suffix} files in {case_dir}: {names}")
    return matches[0]


def _read_canonical(path: Path, *, label: bool) -> tuple[np.ndarray, np.ndarray]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.squeeze(np.asarray(img.get_fdata(dtype=np.float32)))
    if data.ndim != 3:
        raise ValueError(f"{path.name} has shape {data.shape}, expected a 3D volume")
    affine = np.array(img.affine, dtype=np.float64, copy=True)
    if label:
        labels = np.rint(data).astype(np.int16, copy=False)
        labels[labels == 4] = 3
        return labels.astype(np.uint8), affine
    return data, affine


def _check_grid(name: str, shape, affine, ref_shape, ref_affine) -> None:
    if ref_shape is not None and shape != ref_shape:
        raise ValueError(f"{name} shape {shape} does not match {ref_shape}")
    if ref_affine is not None and not np.allclose(affine, ref_affine, atol=1e-4):
        raise ValueError(f"{name} RAS affine does not match the other volumes")
