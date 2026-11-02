"""BraTS 2021+ NIfTI case loader.

Channel order matches the four-modality stack used to build the training
cache (FLAIR, T1, T1ce, T2), which is BraTS 2021+ ``t2f``, ``t1n``, ``t1c``,
``t2w``. Intensity normalization is the training transform
``NormalizeIntensityd(nonzero=True, channel_wise=True)``: per-channel z-score
of nonzero voxels, population standard deviation, zeros left at zero.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import NormalizeIntensity
from nibabel.affines import voxel_sizes

# FLAIR, T1, T1ce, T2 — training stack order, BraTS 2021+ filenames.
MODALITY_SUFFIXES: tuple[str, ...] = ("t2f", "t1n", "t1c", "t2w")
SEG_SUFFIX = "seg"

# BraTS 2020 names (``*_flair``, ``*_t1ce``) and the processed cache layout
# (``flair``, ``t1``, ``t1c``, ``t2``, ``mask``), tried when the 2021+ name is absent.
SUFFIX_ALIASES: dict[str, tuple[str, ...]] = {
    "t2f": ("flair",),
    "t1n": ("t1",),
    "t1c": ("t1ce",),
    "t2w": ("t2",),
    "seg": ("mask",),
}

UPLOAD_ROLES: tuple[str, ...] = (*MODALITY_SUFFIXES, SEG_SUFFIX, "synthseg")
ROLE_LABELS: dict[str, str] = {
    "t2f": "FLAIR",
    "t1n": "T1",
    "t1c": "T1 with contrast",
    "t2w": "T2",
    "seg": "tumor segmentation",
    "synthseg": "brain structure map",
}

_CONTRAST_WORDS = {"c", "ce", "gd", "gad", "gado", "post", "contrast", "enhanced", "plus"}
_ROLE_TOKENS = {
    "synthseg": {"synthseg", "aseg"},
    SEG_SUFFIX: {"seg", "segmentation", "mask", "label", "labels", "gt", "truth"},
    "t2f": {"flair", "t2f", "t2flair"},
    "t1c": {"t1c", "t1ce", "t1gd", "t1post", "t1contrast", "t1wce", "t1gad"},
    "t2w": {"t2", "t2w"},
    "t1n": {"t1", "t1n", "t1w", "t1native", "mprage"},
}


def _name_tokens(filename: str) -> list[str]:
    name = Path(str(filename).replace("\\", "/")).name.lower()
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return [token for token in re.split(r"[^a-z0-9]+", name) if token]


def identify_modality(filename: str) -> str | None:
    """Upload role for a NIfTI file name, or None when the name does not say.

    Roles are ``t2f`` (FLAIR), ``t1n`` (T1), ``t1c`` (T1 with contrast),
    ``t2w`` (T2), ``seg`` (tumor labels), and ``synthseg``. Covers BraTS
    2020 and 2021+ names, the processed ``flair/t1/t1c/t2/mask`` layout, and
    scanner-style names such as ``T1_post_contrast`` or ``sub-01_T2w``.
    """
    tokens = _name_tokens(filename)
    present = set(tokens)
    for role in ("synthseg", SEG_SUFFIX, "t2f", "t1c"):
        if present & _ROLE_TOKENS[role]:
            return role
    if present & {"t1", "t1w"} and present & _CONTRAST_WORDS:
        return "t1c"
    for role in ("t2w", "t1n"):
        if present & _ROLE_TOKENS[role]:
            return role
    return None


def guess_case_id(names, folder: str | None = None) -> str:
    """Case id from the uploaded folder name, else from the file names."""
    if folder:
        candidate = Path(str(folder).replace("\\", "/")).name
    else:
        roles = {identify_modality(name): name for name in names}
        source = roles.get("t1n") or roles.get("t2f") or next(iter(names), "")
        role_words = set().union(*_ROLE_TOKENS.values()) | _CONTRAST_WORDS
        kept = [token for token in re.split(r"[^A-Za-z0-9]+", Path(str(source).replace("\\", "/")).name)
                if token and token.lower() not in role_words and token.lower() not in {"nii", "gz"}]
        candidate = "_".join(kept)
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._-")[:128]
    return candidate or "scan"


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
    paths = case_sequence_paths(case_dir)
    for suffix in MODALITY_SUFFIXES:
        data, vol_affine = _read_canonical(paths[suffix], label=False)
        _check_grid(suffix, data.shape, vol_affine, spatial_shape, affine, ref_name=MODALITY_SUFFIXES[0] if affine is not None else None)
        spatial_shape = data.shape
        affine = vol_affine
        channels.append(data.astype(np.float32, copy=False))

    try:
        seg_path = _find_volume(case_dir, SEG_SUFFIX)
        seg, seg_affine = _read_canonical(seg_path, label=True)
        _check_grid(SEG_SUFFIX, seg.shape, seg_affine, spatial_shape, affine, ref_name="t1n")
    except FileNotFoundError:
        assert spatial_shape is not None
        seg = np.zeros(spatial_shape, dtype=np.uint8)
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


def _nifti_bases(case_dir: Path) -> list[tuple[str, Path]]:
    bases = []
    for path in sorted(Path(case_dir).iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".nii.gz"):
            bases.append((name[: -len(".nii.gz")], path))
        elif name.endswith(".nii"):
            bases.append((name[: -len(".nii")], path))
    return bases


def _find_volume(case_dir: Path, suffix: str) -> Path:
    suffix = suffix.lower()
    bases = _nifti_bases(case_dir)
    names = (suffix,) + SUFFIX_ALIASES.get(suffix, ())
    for name in names:
        matches = [path for base, path in bases if base == name or base.endswith("_" + name)]
        if len(matches) > 1:
            listed = ", ".join(path.name for path in matches)
            raise FileNotFoundError(f"Multiple {ROLE_LABELS.get(suffix, name)} files in {case_dir}: {listed}")
        if matches:
            return matches[0]
    identified = [path for _base, path in bases if identify_modality(path.name) == suffix]
    if len(identified) > 1:
        listed = ", ".join(path.name for path in identified)
        raise FileNotFoundError(f"Multiple {ROLE_LABELS.get(suffix, suffix)} files in {case_dir}: {listed}")
    if identified:
        return identified[0]
    label = ROLE_LABELS.get(suffix, suffix)
    expected = " or ".join(f"*_{name}.nii[.gz]" for name in names)
    raise FileNotFoundError(f"Missing scans: {label} (no {expected} in {case_dir})")


def case_sequence_paths(case_dir: Path) -> dict[str, Path]:
    """Paths of the four required sequences, or a missing-scan error."""
    case_dir = Path(case_dir)
    missing = []
    paths: dict[str, Path] = {}
    for suffix in MODALITY_SUFFIXES:
        try:
            paths[suffix] = _find_volume(case_dir, suffix)
        except FileNotFoundError:
            missing.append(ROLE_LABELS[suffix])
    if missing:
        raise FileNotFoundError(f"Missing scans: {', '.join(missing)}")
    return paths


def validate_case_grid(case_dir: Path) -> dict[str, Path]:
    """Require four sequences on one RAS grid. Raises with the sequence name."""
    paths = case_sequence_paths(case_dir)
    ref_shape = None
    ref_affine = None
    ref_name = None
    for suffix, path in paths.items():
        data, affine = _read_canonical(path, label=False)
        _check_grid(suffix, data.shape, affine, ref_shape, ref_affine, ref_name=ref_name)
        ref_shape, ref_affine, ref_name = data.shape, affine, suffix
    return paths


def input_files_hash(case_dir: Path) -> str:
    """Stable hash of the four sequence files, so anatomy caches can be reused."""
    digest = hashlib.sha256()
    for suffix, path in case_sequence_paths(case_dir).items():
        digest.update(suffix.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:20]


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


def _check_grid(name: str, shape, affine, ref_shape, ref_affine, ref_name=None) -> None:
    label = ROLE_LABELS.get(name, name)
    other = ROLE_LABELS.get(ref_name, ref_name) if ref_name else "the other volumes"
    if ref_shape is not None and shape != ref_shape:
        raise ValueError(
            f"{label} shape {tuple(int(n) for n in shape)} does not match {other} "
            f"{tuple(int(n) for n in ref_shape)}. All four sequences must share the same grid."
        )
    if ref_affine is not None and not np.allclose(affine, ref_affine, atol=1e-4):
        raise ValueError(
            f"{label} RAS affine does not match {other}. "
            "All four sequences must share the same orientation and spacing."
        )
