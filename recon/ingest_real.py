"""Convert and preprocess real hospital MRI studies into BraTS-like NIfTIs.

Walks the same DICOM tree as the inventory, converts the selected FLAIR/T1/T1c/T2
series, N4-corrects, rigid-registers to a 1 mm RAS reference, skull-strips once,
and writes nonzero z-scored volumes. Output folders and NIfTI headers carry the
hashed study id only — no names, raw IDs, dates, or source folder names.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inventory_real import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_OUT,
    collect_studies,
)

logger = logging.getLogger(__name__)

ABHINAV_SRC = ROOT / "third_party" / "abhinav_mri" / "src"
DEFAULT_OUTPUT = ROOT / "outputs" / "real"
INVENTORY_TIERS = ("FULL", "NO_CONTRAST", "FLAIR_T2_ONLY")
CLI_TIERS = ("FULL", "NO_CONTRAST", "FLAIR_T2_ONLY", "ALL")
MODALITY_ROLES = ("FLAIR", "T1", "T1c", "T2")
ROLE_FILE = {"FLAIR": "flair", "T1": "t1", "T1c": "t1c", "T2": "t2"}
FILE_ROLE = {value: key for key, value in ROLE_FILE.items()}
ISO_MM = (1.0, 1.0, 1.0)
MI_ACCEPT = -0.15
MOVING_MASK_DILATE_MM = 10.0
FOV_MIN_MM = 180.0
SI_COVERAGE = 0.80
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|\\\\|/)[^\s\"']+")


@dataclass
class PreprocessSummary:
    tier: str
    candidates: int = 0
    skipped: int = 0
    processed: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    processed_ids: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)


def _ensure_abhinav() -> None:
    src = str(ABHINAV_SRC)
    if ABHINAV_SRC.is_dir() and src not in sys.path:
        sys.path.insert(0, src)


def _scrub(text: str) -> str:
    return _PATH_RE.sub("<path>", str(text or ""))


def _user_scripts_dirs() -> list[Path]:
    dirs: list[Path] = []
    exe = Path(sys.executable).resolve().parent
    dirs.append(exe)
    dirs.append(exe / "Scripts")
    version = f"Python{sys.version_info.major}{sys.version_info.minor}"
    dirs.append(Path.home() / "AppData" / "Roaming" / "Python" / version / "Scripts")
    dirs.append(ROOT / "caches" / "tools")
    return dirs


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found)
    extra = os.pathsep.join(str(path) for path in _user_scripts_dirs() if path.is_dir())
    search = extra + os.pathsep + os.environ.get("PATH", "")
    found = shutil.which(name, path=search)
    if found:
        return Path(found)
    windows = f"{name}.exe"
    for folder in _user_scripts_dirs():
        candidate = folder / windows
        if candidate.is_file():
            return candidate
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def find_dcm2niix() -> Path | None:
    return _which("dcm2niix")


def find_hd_bet() -> Path | None:
    return _which("hd-bet")


def _series_rank(item: dict, *, prefer_3d: bool, prefer_axial: bool) -> tuple:
    thick = item.get("slice_thickness_mm")
    try:
        thickness = float(thick)
    except (TypeError, ValueError):
        thickness = 99.0
    is_3d = (item.get("acquisition") or "").upper() == "3D"
    axial = "axial" in (item.get("orientation") or "").lower()
    n_slices = int(item.get("n_slices") or 0)
    rank = []
    if prefer_3d:
        rank.append(0 if is_3d else 1)
    if prefer_axial:
        rank.append(0 if axial else 1)
    rank.extend((thickness, -n_slices))
    return tuple(rank)


def _parse_pair(text: str) -> tuple[float, float] | None:
    if not text or "x" not in str(text).lower():
        return None
    try:
        left, right = str(text).lower().split("x", 1)
        return float(left), float(right)
    except (TypeError, ValueError):
        return None


def series_extent(item: dict) -> dict:
    """Physical FOV (mm) and superior-inferior coverage from inventory geometry."""
    thick = item.get("slice_thickness_mm")
    try:
        thickness = float(thick) if thick is not None else None
    except (TypeError, ValueError):
        thickness = None
    n_slices = int(item.get("n_slices") or 0)
    stack = None if thickness is None else thickness * n_slices
    spacing = _parse_pair(item.get("in_plane_spacing_mm") or "")
    matrix = _parse_pair(item.get("matrix") or "")
    inplane = None
    if spacing is not None and matrix is not None:
        inplane = (abs(matrix[0] * spacing[0]), abs(matrix[1] * spacing[1]))
    orientation = (item.get("orientation") or "").lower()
    if "axial" in orientation:
        si = stack
    elif inplane is not None:
        si = max(inplane)
    else:
        si = stack
    return {
        "si_mm": si,
        "stack_mm": stack,
        "inplane_fov_mm": None if inplane is None else min(inplane),
        "inplane_fov": inplane,
    }


def coverage_reject_reasons(item: dict, *, ref_si_mm: float | None) -> list[str]:
    """Reject small-FOV zooms and slabs that miss the reference brain SI."""
    extent = series_extent(item)
    reasons = []
    inplane = extent["inplane_fov_mm"]
    if inplane is not None and inplane < FOV_MIN_MM:
        reasons.append(f"FOV {inplane:.0f} mm < {FOV_MIN_MM:.0f} mm")
    si = extent["si_mm"]
    if ref_si_mm is not None and si is not None and si < SI_COVERAGE * float(ref_si_mm):
        reasons.append(
            f"SI {si:.0f} mm < {SI_COVERAGE:.0%} of reference brain {float(ref_si_mm):.0f} mm"
        )
    return reasons


def pick_channel_series(
    series: list[dict],
    modality: str,
    *,
    ref_si_mm: float | None = None,
    rejected: list | None = None,
) -> dict | None:
    candidates = [
        item
        for item in series
        if item.get("usable_as_input")
        and item.get("modality") == modality
        and item.get("source_path")
    ]
    kept = []
    for item in candidates:
        reasons = coverage_reject_reasons(item, ref_si_mm=ref_si_mm)
        if reasons:
            why = "; ".join(reasons)
            desc = item.get("series_description") or "(unnamed)"
            logger.info("reject %s %s: %s", modality, desc, why)
            print(f"  reject {ROLE_FILE.get(modality, modality)} {desc}: {why}", flush=True)
            if rejected is not None:
                rejected.append(
                    {
                        "modality": ROLE_FILE.get(modality, modality),
                        "series_description": desc,
                        "reason": why,
                        **{key: series_extent(item).get(key) for key in ("si_mm", "inplane_fov_mm", "stack_mm")},
                    }
                )
            continue
        kept.append(item)
    if not kept:
        return None
    prefer_3d = modality == "T1"
    return min(kept, key=lambda item: _series_rank(item, prefer_3d=prefer_3d, prefer_axial=True))


def choose_channels(
    series: list[dict],
    *,
    ref_si_mm: float | None = None,
    rejected: list | None = None,
    keep: dict[str, dict] | None = None,
) -> dict[str, dict]:
    chosen: dict[str, dict] = dict(keep or {})
    for modality in MODALITY_ROLES:
        if modality in chosen:
            continue
        item = pick_channel_series(series, modality, ref_si_mm=ref_si_mm, rejected=rejected)
        if item is not None:
            chosen[modality] = item
    return chosen


def mask_si_mm(mask_path: Path) -> float | None:
    """Superior-inferior extent of a RAS brain mask, millimetres."""
    image = nib.load(str(mask_path))
    mask = np.asanyarray(image.dataobj) > 0
    if not np.any(mask):
        return None
    zs = np.where(mask)[2]
    spacing = float(abs(image.header.get_zooms()[2]))
    return float((int(zs.max()) - int(zs.min()) + 1) * spacing)


def choose_reference(chosen: dict[str, dict]) -> tuple[str, str]:
    """Return (modality, reason) using 3D T1 (thinnest) > T1c > 2D T1 > FLAIR."""
    t1 = chosen.get("T1")
    if t1 is not None and (t1.get("acquisition") or "").upper() == "3D":
        return "T1", "3D T1 (thinnest)"
    if chosen.get("T1c") is not None:
        return "T1c", "T1c (no 3D T1)"
    if t1 is not None:
        return "T1", "2D T1 (no 3D T1, no T1c)"
    if chosen.get("FLAIR") is not None:
        return "FLAIR", "FLAIR (no T1/T1c)"
    raise RuntimeError("No reference sequence (need 3D T1, T1c, 2D T1, or FLAIR)")


def public_series(item: dict) -> dict:
    thick = item.get("slice_thickness_mm")
    try:
        thickness = float(thick) if thick is not None else None
    except (TypeError, ValueError):
        thickness = None
    return {
        "series_description": item.get("series_description") or "(unnamed)",
        "acquisition": item.get("acquisition") or "",
        "slice_thickness_mm": thickness,
        "n_slices": int(item.get("n_slices") or 0),
        "orientation": item.get("orientation") or "unknown",
    }


def study_is_done(ready_dir: Path) -> bool:
    meta_path = ready_dir / "preprocess.json"
    if not meta_path.is_file():
        return False
    if not (ready_dir / "brain_mask.nii.gz").is_file():
        return False
    if not (ready_dir / "registration_check.png").is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    chosen = meta.get("series_chosen") or {}
    if not chosen:
        return False
    return all((ready_dir / f"{role}.nii.gz").is_file() for role in chosen)


def strip_nifti_identity(path: Path) -> Path:
    """Rewrite a NIfTI so the header has no descrip/ids/aux text; keep geometry."""
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim > 3:
        data = np.take(data, 0, axis=3)
    data = np.ascontiguousarray(data)
    clean = nib.Nifti1Image(data, image.affine)
    hdr = clean.header
    for key in ("descrip", "aux_file", "intent_name", "db_name"):
        try:
            hdr[key] = b""
        except Exception:
            pass
    try:
        hdr.extensions.clear()
    except Exception:
        pass
    clean.set_qform(image.affine, code=1)
    clean.set_sform(image.affine, code=1)
    nib.save(clean, str(path))
    return path


def _save_nifti(data, affine, path: Path, *, dtype=None) -> Path:
    array = np.ascontiguousarray(data if dtype is None else np.asarray(data, dtype=dtype))
    image = nib.Nifti1Image(array, np.asarray(affine, dtype=np.float64))
    for key in ("descrip", "aux_file", "intent_name", "db_name"):
        try:
            image.header[key] = b""
        except Exception:
            pass
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    return path


def _reorient_ras(path: Path) -> Path:
    image = nib.load(str(path))
    canonical = nib.as_closest_canonical(image)
    data = np.ascontiguousarray(np.asanyarray(canonical.dataobj))
    if data.ndim > 3:
        data = np.take(data, 0, axis=3)
    _save_nifti(data, canonical.affine, path)
    return path


def _pick_dcm2niix_output(produced: list[Path], expected_slices: int) -> Path:
    scored = []
    for path in produced:
        if "mask" in path.name.lower():
            continue
        image = nib.load(str(path))
        shape = image.shape[:3]
        n_slices = int(shape[2]) if len(shape) >= 3 else 0
        scored.append((abs(n_slices - int(expected_slices or n_slices)), -int(np.prod(shape)), path))
    if not scored:
        raise FileNotFoundError("dcm2niix produced no usable NIfTI")
    scored.sort()
    return scored[0][2]


def convert_series(item: dict, dest: Path, *, role: str) -> str:
    """Convert one series to ``dest``. Returns the converter name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = Path(item["source_path"])
    kind = item.get("source_kind") or ("nifti" if source.is_file() else "dicom")
    if kind == "nifti" or source.is_file():
        shutil.copy2(source, dest)
        strip_nifti_identity(dest)
        return "copy_nifti"

    series_dir = source if source.is_dir() else source.parent
    dcm2niix = find_dcm2niix()
    if dcm2niix is not None:
        with tempfile.TemporaryDirectory(prefix="dcm2niix_") as tmp:
            tmp_dir = Path(tmp)
            cmd = [
                str(dcm2niix),
                "-z",
                "y",
                "-b",
                "n",
                "-ba",
                "y",
                "-w",
                "1",
                "-f",
                f"raw_{role}",
                "-o",
                str(tmp_dir),
                str(series_dir),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            produced = sorted(tmp_dir.glob("*.nii*"))
            if result.returncode == 0 and produced:
                chosen = _pick_dcm2niix_output(produced, int(item.get("n_slices") or 0))
                shutil.copy2(chosen, dest)
                strip_nifti_identity(dest)
                return "dcm2niix"
            logger.warning("dcm2niix failed for %s (exit %s); trying SimpleITK", role, result.returncode)

    _ensure_abhinav()
    from preprocessing.dicom_to_nifti import convert_dicom_series

    convert_dicom_series(series_dir, dest, reorient_ras=False)
    strip_nifti_identity(dest)
    return "simpleitk"


def n4_correct(src: Path, dest: Path) -> Path:
    _ensure_abhinav()
    from preprocessing.bias_correction import n4_bias_correct

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        n4_bias_correct(src, dest)
    except Exception as exc:
        logger.warning("N4 failed (%s); retrying with shrink_factor=4", _scrub(str(exc)))
        n4_bias_correct(src, dest, shrink_factor=4)
    strip_nifti_identity(dest)
    return dest


def _ants():
    import ants

    return ants


def _resample_isotropic_bspline(src: Path, dest: Path) -> Path:
    ants = _ants()
    image = ants.image_read(str(src))
    resampled = ants.resample_image(image, list(ISO_MM), use_voxels=False, interp_type=4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(resampled, str(dest))
    strip_nifti_identity(dest)
    _reorient_ras(dest)
    return dest


def _mutual_information(ants, fixed, moving) -> float | None:
    try:
        return float(ants.image_mutual_information(fixed, moving))
    except Exception:
        return None


def mi_accepted(value: float | None, threshold: float = MI_ACCEPT) -> bool:
    """Mattes MI from ANTs is a cost: more negative is better. Accept if ``value < threshold``."""
    return value is not None and float(value) < float(threshold)


def _brainish_ants_mask(ants, image, *, dilate_mm: float = 0.0):
    data = np.asarray(image.numpy(), dtype=np.float32)
    mask = _brainish_mask(data)
    if dilate_mm > 0:
        spacing = float(np.mean(np.asarray(image.spacing, dtype=np.float64)))
        iterations = max(1, int(round(dilate_mm / max(spacing, 0.5))))
        mask = ndimage.binary_dilation(mask, iterations=iterations)
    return image.new_image_like(mask.astype(np.float32))


def _rotation_init(ants, center, axis: int, degrees: float, path: Path) -> str:
    from scipy.spatial.transform import Rotation

    angles = [0.0, 0.0, 0.0]
    angles[axis] = float(np.deg2rad(degrees))
    matrix = Rotation.from_euler("xyz", angles).as_matrix().tolist()
    transform = ants.create_ants_transform(
        transform_type="AffineTransform",
        precision="float",
        dimension=3,
        matrix=matrix,
        offset=(0.0, 0.0, 0.0),
        center=tuple(float(v) for v in center),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.write_transform(transform, str(path))
    return str(path)


def _score_warped(ants, warped, fixed_masked, brain_mask):
    masked = warped * brain_mask
    return _mutual_information(ants, fixed_masked, masked)


def rigid_register_to_reference(
    moving_path: Path,
    fixed_path: Path,
    dest: Path,
    work: Path,
    role: str,
    *,
    fixed_mask_path: Path,
) -> dict:
    """Multi-start rigid register onto the brain-masked reference.

    Starts: identity (scanner/header coordinates), center-of-mass, and ±10°
    about each axis. The warped volume with the best (lowest) Mattes MI is kept.
    """
    ants = _ants()
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    brain_mask = ants.image_read(str(fixed_mask_path))
    brain_mask = brain_mask > 0
    fixed_masked = fixed * brain_mask
    moving_mask = _brainish_ants_mask(ants, moving, dilate_mm=MOVING_MASK_DILATE_MM)

    starts: list[tuple[str, object]] = [
        ("identity", "Identity"),
        ("com", f"[{fixed_path},{moving_path},1]"),
    ]
    try:
        center = ants.get_center_of_mass(brain_mask)
    except Exception:
        center = tuple(float(v) for v in fixed.get_center_of_mass())
    for axis, axis_name in enumerate("xyz"):
        for degrees in (-10.0, 10.0):
            label = f"rot_{axis_name}{degrees:+.0f}"
            init_path = work / f"init_{role}_{label}.mat"
            try:
                starts.append((label, _rotation_init(ants, center, axis, degrees, init_path)))
            except Exception as exc:
                logger.warning("skip init %s for %s (%s)", label, role, _scrub(str(exc)))

    best: dict | None = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    for name, initial in starts:
        prefix = str(work / f"xfm_{role}_{name}_")
        try:
            reg = ants.registration(
                fixed=fixed_masked,
                moving=moving,
                type_of_transform="Rigid",
                aff_metric="mattes",
                initial_transform=initial,
                mask=brain_mask,
                moving_mask=moving_mask,
                mask_all_stages=True,
                outprefix=prefix,
                verbose=False,
            )
            transforms = list(reg.get("fwdtransforms") or [])
            if not transforms:
                raise RuntimeError("no forward transforms")
            warped = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=transforms,
                interpolator="bSpline",
            )
            metric = _score_warped(ants, warped, fixed_masked, brain_mask)
        except Exception as exc:
            logger.warning("rigid start %s failed for %s (%s)", name, role, _scrub(str(exc)))
            continue
        if metric is None:
            continue
        print(f"    start {name}: MI={metric:.4f}", flush=True)
        if best is None or metric < float(best["value"]):
            best = {"value": float(metric), "init": name, "warped": warped, "transforms": transforms}

    if best is None:
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=["Identity"],
            interpolator="bSpline",
        )
        metric = _score_warped(ants, warped, fixed_masked, brain_mask)
        best = {
            "value": None if metric is None else float(metric),
            "init": "identity_resample",
            "warped": warped,
            "transforms": ["Identity"],
        }
        print(f"    fallback identity resample: MI={best['value']}", flush=True)

    ants.image_write(best["warped"], str(dest))
    strip_nifti_identity(dest)
    accepted = mi_accepted(best["value"])
    print(
        f"    kept {best['init']} MI={best['value']} ({'accept' if accepted else 'FAIL threshold'})",
        flush=True,
    )
    return {
        "value": best["value"],
        "init": best["init"],
        "accepted": accepted,
        "failed": not accepted,
    }


def _ensure_hd_bet_params() -> Path:
    """HD-BET 1.x writes weights to ~/hd-bet_params and fails if the folder is missing."""
    params = Path.home() / "hd-bet_params"
    params.mkdir(parents=True, exist_ok=True)
    model = params / "0.model"
    if not model.is_file():
        import urllib.request

        url = "https://zenodo.org/record/2540695/files/0.model?download=1"
        print("  downloading HD-BET weights (fold 0, fast mode)", flush=True)
        tmp = params / "0.model.part"
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(model)
    return params


def _run_hd_bet_cli(src: Path, stripped: Path, mask: Path) -> None:
    _ensure_hd_bet_params()
    cli = find_hd_bet()
    if cli is None:
        raise RuntimeError("hd-bet executable not found")
    stripped.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hdbet_") as tmp:
        tmp_out = Path(tmp) / "brain.nii.gz"
        cmd = [
            str(cli),
            "-i",
            str(src),
            "-o",
            str(tmp_out),
            "-device",
            "cpu",
            "-mode",
            "fast",
            "-tta",
            "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"hd-bet CLI exit {result.returncode}: {_scrub(result.stderr[-400:])}")
        produced = tmp_out if tmp_out.is_file() else None
        if produced is None:
            candidates = sorted(Path(tmp).glob("*.nii*"))
            non_mask = [path for path in candidates if "mask" not in path.name.lower()]
            produced = (non_mask or candidates)[0] if candidates else None
        if produced is None:
            raise FileNotFoundError("HD-BET produced no NIfTI")
        shutil.copy2(produced, stripped)
        mask_hits = sorted(Path(tmp).glob("*mask*.nii*"))
        if not mask_hits:
            sibling = produced.with_name(produced.name.replace(".nii.gz", "_mask.nii.gz"))
            if sibling.is_file():
                mask_hits = [sibling]
        if mask_hits:
            shutil.copy2(mask_hits[0], mask)
        else:
            image = nib.load(str(stripped))
            brain = np.asanyarray(image.dataobj) != 0
            _save_nifti(brain.astype(np.uint8), image.affine, mask, dtype=np.uint8)
        strip_nifti_identity(stripped)
        strip_nifti_identity(mask)


def skull_strip_reference(src: Path, stripped: Path, mask: Path) -> str:
    _ensure_hd_bet_params()
    try:
        _run_hd_bet_cli(src, stripped, mask)
        return "hd-bet"
    except Exception as exc:
        logger.warning("HD-BET CLI failed (%s); trying Python API", _scrub(str(exc)))
    _ensure_abhinav()
    from preprocessing.skull_strip import hd_bet_available, skull_strip

    if not hd_bet_available():
        raise RuntimeError("HD-BET is not installed")
    result = skull_strip(src, stripped, mask, prefer_hd_bet=True, device="cpu", mode="fast")
    if result.method != "hd-bet":
        raise RuntimeError("HD-BET unavailable; refusing the Otsu fallback")
    strip_nifti_identity(stripped)
    strip_nifti_identity(mask)
    return "hd-bet"


def apply_mask_and_zscore(src: Path, mask_path: Path, dest: Path) -> Path:
    from monai.transforms import NormalizeIntensity

    image = nib.load(str(src))
    mask_img = nib.load(str(mask_path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    mask = np.asarray(mask_img.dataobj) > 0
    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} != image shape {data.shape}")
    data = data.copy()
    data[~mask] = 0
    stacked = np.asarray(NormalizeIntensity(nonzero=True, channel_wise=True)(data[None]), dtype=np.float32)[0]
    stacked[~mask] = 0
    return _save_nifti(stacked, image.affine, dest, dtype=np.float32)


def _window(slice2d: np.ndarray) -> np.ndarray:
    finite = slice2d[np.isfinite(slice2d)]
    if finite.size == 0:
        return np.zeros_like(slice2d, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((slice2d - lo) / (hi - lo), 0, 1)
    return scaled.astype(np.float32)


def _brainish_mask(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data) & (data != 0)]
    if finite.size < 16:
        return data != 0
    thresh = float(np.percentile(finite, 20.0))
    mask = data > thresh
    mask = ndimage.binary_fill_holes(mask)
    labeled, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labeled, range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        mask = labeled == keep
    return mask


def _outline(mask2d: np.ndarray) -> np.ndarray:
    if not np.any(mask2d):
        return np.zeros_like(mask2d, dtype=bool)
    eroded = ndimage.binary_erosion(mask2d, iterations=1)
    return mask2d & ~eroded


def write_registration_check(reference: np.ndarray, others: dict[str, np.ndarray], dest: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_mask = _brainish_mask(reference)
    coords = np.argwhere(ref_mask)
    if coords.size == 0:
        coords = np.argwhere(np.isfinite(reference))
    center = coords.mean(axis=0).astype(int) if coords.size else np.array(reference.shape) // 2
    ix, iy, iz = (
        int(np.clip(center[0], 0, reference.shape[0] - 1)),
        int(np.clip(center[1], 0, reference.shape[1] - 1)),
        int(np.clip(center[2], 0, reference.shape[2] - 1)),
    )
    views = (
        ("axial", reference[:, :, iz], 2, iz),
        ("coronal", reference[:, iy, :], 1, iy),
        ("sagittal", reference[ix, :, :], 0, ix),
    )
    colors = {"flair": (0.2, 0.7, 1.0), "t1": (0.2, 1.0, 0.4), "t1c": (1.0, 0.35, 0.2), "t2": (1.0, 0.85, 0.2)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, sl, axis, index) in zip(axes, views):
        ax.imshow(_window(sl.T), cmap="gray", origin="lower")
        for role, volume in others.items():
            mask3d = _brainish_mask(volume)
            if axis == 0:
                mask2d = mask3d[index, :, :]
            elif axis == 1:
                mask2d = mask3d[:, index, :]
            else:
                mask2d = mask3d[:, :, index]
            edge = _outline(mask2d)
            overlay = np.zeros((*edge.shape, 4), dtype=np.float32)
            rgb = colors.get(role, (1.0, 0.0, 1.0))
            overlay[edge] = (*rgb, 1.0)
            ax.imshow(np.transpose(overlay, (1, 0, 2)), origin="lower")
        ax.set_title(title)
        ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], color=colors.get(role, "w"), lw=2, label=role)
        for role in others
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    return data, np.asarray(image.affine, dtype=np.float64)


def _convert_n4(item: dict, work: Path, role: str, conversion: dict[str, str], native: dict[str, Path]) -> None:
    raw = work / f"raw_{role}.nii.gz"
    print(f"  convert {role}", flush=True)
    conversion[role] = convert_series(item, raw, role=role)
    n4_path = work / f"n4_{role}.nii.gz"
    print(f"  N4 {role}", flush=True)
    n4_correct(raw, n4_path)
    _reorient_ras(n4_path)
    native[role] = n4_path


def preprocess_study(study_id: str, payload: dict, *, out_root: Path, tier: str) -> dict:
    from recon.timing import StageClock

    clock = StageClock()
    clock.start("preprocess")
    series = payload.get("series") or []
    rejected: list[dict] = []
    # Reference first (FOV filter only). Brain SI is measured after HD-BET.
    seed = choose_channels(series, ref_si_mm=None, rejected=None)
    if not seed:
        raise RuntimeError("No usable FLAIR/T1/T1c/T2 series")
    ref_modality, ref_reason = choose_reference(seed)
    logger.info("%s reference=%s (%s)", study_id, ROLE_FILE[ref_modality], ref_reason)
    print(f"  reference: {ROLE_FILE[ref_modality]} - {ref_reason}", flush=True)

    study_dir = out_root / study_id
    ready = study_dir / "04_ready"
    work = study_dir / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    ready.mkdir(parents=True, exist_ok=True)

    conversion: dict[str, str] = {}
    native: dict[str, Path] = {}
    ref_role = ROLE_FILE[ref_modality]
    _convert_n4(seed[ref_modality], work, ref_role, conversion, native)

    ref_1mm = work / "ref_1mm.nii.gz"
    print(f"  resample {ref_role} to 1 mm RAS", flush=True)
    _resample_isotropic_bspline(native[ref_role], ref_1mm)
    codes = "".join(aff2axcodes(nib.load(str(ref_1mm)).affine))
    if codes != "RAS":
        _reorient_ras(ref_1mm)

    stripped = work / "ref_stripped.nii.gz"
    mask_path = work / "brain_mask.nii.gz"
    print("  HD-BET skull-strip reference (-device cpu -mode fast -tta 0)", flush=True)
    skull_method = skull_strip_reference(ref_1mm, stripped, mask_path)
    brain_si = mask_si_mm(mask_path)
    print(f"  reference brain SI: {brain_si:.0f} mm" if brain_si else "  reference brain SI: unknown", flush=True)

    chosen = choose_channels(
        series,
        ref_si_mm=brain_si,
        rejected=rejected,
        keep={ref_modality: seed[ref_modality]},
    )
    for modality, item in chosen.items():
        role = ROLE_FILE[modality]
        if role in native:
            continue
        _convert_n4(item, work, role, conversion, native)

    registered: dict[str, Path] = {ref_role: ref_1mm}
    metrics: dict[str, float | None] = {ref_role: None}
    inits: dict[str, str | None] = {ref_role: None}
    failed_roles: list[str] = []
    for role, path in native.items():
        if role == ref_role:
            continue
        dest = work / f"reg_{role}.nii.gz"
        print(f"  rigid register {role} -> {ref_role} (header init, masked, multi-start)", flush=True)
        result = rigid_register_to_reference(
            path, ref_1mm, dest, work, role, fixed_mask_path=mask_path
        )
        metrics[role] = result["value"]
        inits[role] = result["init"]
        registered[role] = dest
        if result["failed"]:
            failed_roles.append(role)

    ref_data, _ref_aff = _load_volume(ref_1mm)
    other_data = {role: _load_volume(path)[0] for role, path in registered.items() if role != ref_role}
    png_path = ready / "registration_check.png"
    write_registration_check(ref_data, other_data, png_path)

    ready_mask = ready / "brain_mask.nii.gz"
    shutil.copy2(mask_path, ready_mask)
    strip_nifti_identity(ready_mask)

    registered_dir = study_dir / "02_registered"
    if registered_dir.exists():
        shutil.rmtree(registered_dir)
    registered_dir.mkdir(parents=True, exist_ok=True)
    for role, path in registered.items():
        dest = registered_dir / f"{role}.nii.gz"
        shutil.copy2(path, dest)
        strip_nifti_identity(dest)

    for role, path in registered.items():
        print(f"  mask + nonzero z-score {role}", flush=True)
        apply_mask_and_zscore(path, ready_mask, ready / f"{role}.nii.gz")

    meta = {
        "study_id": study_id,
        "tier": tier,
        "reference": {
            "role": ref_role,
            "reason": ref_reason,
            **public_series(chosen[ref_modality]),
        },
        "series_chosen": {ROLE_FILE[mod]: public_series(item) for mod, item in chosen.items()},
        "original_thickness_mm": {
            ROLE_FILE[mod]: public_series(item)["slice_thickness_mm"] for mod, item in chosen.items()
        },
        "registration": {
            "transform": "rigid",
            "metric": "mattes_mutual_information",
            "interpolator": "bSpline",
            "spacing_mm": list(ISO_MM),
            "orientation": "RAS",
            "initial_transform": "identity_header",
            "accept_threshold": MI_ACCEPT,
            "values": metrics,
            "init": inits,
            "failed": failed_roles,
        },
        "registration_failed": failed_roles,
        "series_rejected": rejected,
        "reference_brain_si_mm": None if brain_si is None else round(float(brain_si), 1),
        "conversion": conversion,
        "n4": True,
        "skull_strip": {
            "method": skull_method,
            "device": "cpu",
            "mode": "fast",
            "tta": 0,
        },
        "normalization": "nonzero_zscore_channelwise",
    }
    clock.stop()
    meta["stage_times_s"] = clock.as_dict()
    (ready / "preprocess.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return meta


def load_inventory_rows(path: Path, tier: str) -> list[dict]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if tier == "ALL":
        keep = set(INVENTORY_TIERS)
        return [row for row in rows if row.get("tier") in keep]
    return [row for row in rows if row.get("tier") == tier]


def print_summary(summary: PreprocessSummary) -> None:
    print("=== Preprocess summary ===", flush=True)
    print(f"tier: {summary.tier}", flush=True)
    print(f"candidates: {summary.candidates}", flush=True)
    print(f"skipped (already done): {summary.skipped}", flush=True)
    print(f"processed: {summary.processed}", flush=True)
    print(f"failed: {summary.failed}", flush=True)
    if summary.processed_ids:
        print("processed ids:", flush=True)
        for study_id in summary.processed_ids:
            print(f"  {study_id}", flush=True)
    if summary.failures:
        print("failures:", flush=True)
        for study_id, message in summary.failures:
            print(f"  {study_id}: {message}", flush=True)


def run(
    *,
    root: Path = DEFAULT_DATA,
    inventory: Path = DEFAULT_OUT,
    out_root: Path = DEFAULT_OUTPUT,
    tier: str = "FULL",
    limit: int | None = None,
    force: bool = False,
    studies: list[str] | None = None,
) -> PreprocessSummary:
    if tier not in CLI_TIERS:
        raise ValueError(f"tier must be one of {CLI_TIERS}")
    if not inventory.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory}")
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}")

    rows = load_inventory_rows(inventory, tier)
    rows.sort(key=lambda row: row.get("study_id") or "")
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    wanted = [row["study_id"] for row in rows if row.get("study_id")]
    if studies:
        keep = set(studies)
        missing = sorted(keep - set(wanted))
        wanted = [study_id for study_id in wanted if study_id in keep]
        if missing:
            print(f"not in {tier} inventory: {', '.join(missing)}", flush=True)
    tier_by_id = {row["study_id"]: row.get("tier") or tier for row in rows}

    print(f"Scanning {root} for {len(wanted)} {tier} stud{'y' if len(wanted)==1 else 'ies'}...", flush=True)
    disk_studies, _duplicates = collect_studies(root)

    summary = PreprocessSummary(tier=tier, candidates=len(wanted))
    out_root.mkdir(parents=True, exist_ok=True)

    for index, study_id in enumerate(wanted, start=1):
        ready = out_root / study_id / "04_ready"
        print(f"[{index}/{len(wanted)}] {study_id}", flush=True)
        if study_is_done(ready) and not force:
            print("  skip (already done)", flush=True)
            summary.skipped += 1
            summary.skipped_ids.append(study_id)
            continue
        if force and ready.exists():
            shutil.rmtree(ready, ignore_errors=True)
        payload = disk_studies.get(study_id)
        if payload is None:
            message = "study not found on disk"
            print(f"  FAIL {message}", flush=True)
            summary.failed += 1
            summary.failures.append((study_id, message))
            continue
        try:
            preprocess_study(study_id, payload, out_root=out_root, tier=tier_by_id.get(study_id, tier))
        except Exception as exc:
            message = _scrub(f"{exc.__class__.__name__}: {exc}")
            logger.exception("Preprocess failed for %s", study_id)
            print(f"  FAIL {message}", flush=True)
            summary.failed += 1
            summary.failures.append((study_id, message))
            continue
        print("  done", flush=True)
        summary.processed += 1
        summary.processed_ids.append(study_id)
    return summary
