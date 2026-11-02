"""Quality gates for preprocessed real MRI studies (Prompt C).

Status is PASS, WARN, or FAIL. FAIL blocks inference.

FAIL
    missing 04_ready files or preprocess.json
    registration_failed on any sequence
    sequences not on one RAS ~1 mm grid
    brain mask missing, empty, or volume outside 200–2500 cm3
    non-finite voxel values
    a sequence required by the study tier is missing

WARN
    FLAIR original thickness > 3 mm: "thick FLAIR: tumor boundary accuracy reduced"
    any other input thicker than 3 mm: "thick slices: {role} {mm} mm"
    2D-only reference (no 3D T1)
    missing T1c: "no contrast: enhancing tumor may be missed"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes
from nibabel.orientations import aff2axcodes

from recon.ingest_real import DEFAULT_OUTPUT, mi_accepted

READY_MAX_MM = 3.0
BRAIN_VOL_CM3 = (200.0, 2500.0)
ISO_TOL_MM = 0.15
TIER_REQUIRED = {
    "FULL": ("flair", "t1", "t1c", "t2"),
    "NO_CONTRAST": ("flair", "t1", "t2"),
    "FLAIR_T2_ONLY": ("flair", "t2"),
}


@dataclass
class QCResult:
    study_id: str
    status: str
    reasons: list[str] = field(default_factory=list)
    tier: str = ""
    ready_dir: Path | None = None

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    def as_dict(self) -> dict:
        return {
            "study_id": self.study_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "tier": self.tier,
        }


def ready_dir_for(study_id: str, out_root: Path = DEFAULT_OUTPUT) -> Path:
    return Path(out_root) / study_id / "04_ready"


def _load_meta(ready: Path) -> dict:
    path = ready / "preprocess.json"
    if not path.is_file():
        raise FileNotFoundError("missing preprocess.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _volume_cm3(mask, affine) -> float:
    voxel = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    return float(np.asarray(mask).sum()) * voxel / 1000.0


def _check_grid(ready: Path, roles: list[str], result: QCResult) -> None:
    ref_shape = None
    ref_affine = None
    ref_name = None
    for role in roles:
        path = ready / f"{role}.nii.gz"
        if not path.is_file():
            result.reasons.append(f"missing {role}.nii.gz")
            result.status = "FAIL"
            continue
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
        if data.ndim != 3:
            result.reasons.append(f"{role} is not 3D")
            result.status = "FAIL"
            continue
        if not np.isfinite(data).all():
            result.reasons.append(f"non-finite values in {role}")
            result.status = "FAIL"
        affine = np.asarray(image.affine, dtype=np.float64)
        codes = "".join(aff2axcodes(affine))
        if codes != "RAS":
            result.reasons.append(f"{role} orientation is {codes}, expected RAS")
            result.status = "FAIL"
        spacing = voxel_sizes(affine)
        if any(abs(float(v) - 1.0) > ISO_TOL_MM for v in spacing[:3]):
            pretty = "x".join(f"{float(v):.2f}" for v in spacing[:3])
            result.reasons.append(f"{role} spacing {pretty} mm, expected 1 mm isotropic")
            result.status = "FAIL"
        shape = tuple(int(n) for n in data.shape[:3])
        if ref_shape is None:
            ref_shape, ref_affine, ref_name = shape, affine, role
            continue
        if shape != ref_shape:
            result.reasons.append(f"{role} shape {shape} does not match {ref_name} {ref_shape}")
            result.status = "FAIL"
        elif not np.allclose(affine, ref_affine, atol=1e-4):
            result.reasons.append(f"{role} affine does not match {ref_name}")
            result.status = "FAIL"


def _check_mask(ready: Path, result: QCResult) -> None:
    path = ready / "brain_mask.nii.gz"
    if not path.is_file():
        result.reasons.append("missing brain_mask.nii.gz")
        result.status = "FAIL"
        return
    image = nib.load(str(path))
    mask = np.asanyarray(image.dataobj) > 0
    if not np.any(mask):
        result.reasons.append("empty brain mask")
        result.status = "FAIL"
        return
    volume = _volume_cm3(mask, image.affine)
    lo, hi = BRAIN_VOL_CM3
    if volume < lo or volume > hi:
        result.reasons.append(f"brain mask volume {volume:.1f} cm3 outside {lo:.0f}–{hi:.0f} cm3")
        result.status = "FAIL"


def _check_registration(meta: dict, result: QCResult) -> None:
    failed = list(meta.get("registration_failed") or (meta.get("registration") or {}).get("failed") or [])
    values = ((meta.get("registration") or {}).get("values") or {})
    if not failed:
        for role, value in values.items():
            if role == (meta.get("reference") or {}).get("role"):
                continue
            if value is None:
                continue
            if not mi_accepted(value):
                failed.append(role)
    if failed:
        result.reasons.append("registration_failed: " + ", ".join(failed))
        result.status = "FAIL"


def _check_tier_sequences(meta: dict, ready: Path, result: QCResult) -> list[str]:
    tier = str(meta.get("tier") or "FULL")
    required = TIER_REQUIRED.get(tier, TIER_REQUIRED["FULL"])
    present = []
    for role in ("flair", "t1", "t1c", "t2"):
        if (ready / f"{role}.nii.gz").is_file():
            present.append(role)
    missing = [role for role in required if role not in present]
    if missing:
        result.reasons.append("missing sequences: " + ", ".join(missing))
        result.status = "FAIL"
    if "t1c" not in present:
        result.reasons.append("no contrast: enhancing tumor may be missed")
        if result.status != "FAIL":
            result.status = "WARN"
    return present


def _check_thickness(meta: dict, result: QCResult) -> None:
    thickness = meta.get("original_thickness_mm") or {}
    for role, value in thickness.items():
        try:
            mm = float(value)
        except (TypeError, ValueError):
            continue
        if mm <= READY_MAX_MM:
            continue
        if role == "flair":
            result.reasons.append("thick FLAIR: tumor boundary accuracy reduced")
        else:
            result.reasons.append(f"thick slices: {role} {mm:g} mm")
        if result.status != "FAIL":
            result.status = "WARN"
    reference = meta.get("reference") or {}
    acquisition = str(reference.get("acquisition") or "").upper()
    if acquisition == "2D":
        result.reasons.append("2D-only reference (no 3D T1)")
        if result.status != "FAIL":
            result.status = "WARN"


PRED_FILE_NAMES = ("pred_seg.nii.gz", "wt_prob.nii.gz")


def clear_stale_predictions(study_id: str, *, out_root: Path = DEFAULT_OUTPUT) -> list[Path]:
    """Remove prediction files so a FAIL study cannot show leftover contours."""
    out_dir = Path(out_root) / study_id
    removed = []
    for folder in (out_dir, out_dir / "04_ready"):
        for name in PRED_FILE_NAMES:
            path = folder / name
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def qc_study(study_id: str, *, out_root: Path = DEFAULT_OUTPUT, clear_predictions: bool = True) -> QCResult:
    ready = ready_dir_for(study_id, out_root)
    result = QCResult(study_id=study_id, status="PASS", ready_dir=ready)
    if not ready.is_dir():
        result.status = "FAIL"
        result.reasons.append("missing 04_ready")
        if clear_predictions:
            clear_stale_predictions(study_id, out_root=out_root)
        return result
    try:
        meta = _load_meta(ready)
    except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
        if clear_predictions:
            clear_stale_predictions(study_id, out_root=out_root)
        return result
    result.tier = str(meta.get("tier") or "")
    present = _check_tier_sequences(meta, ready, result)
    _check_registration(meta, result)
    _check_grid(ready, present, result)
    _check_mask(ready, result)
    _check_thickness(meta, result)
    if result.status == "PASS" and result.reasons:
        result.status = "WARN"
    if result.failed and clear_predictions:
        clear_stale_predictions(study_id, out_root=out_root)
    return result


def qc_studies(study_ids, *, out_root: Path = DEFAULT_OUTPUT) -> list[QCResult]:
    return [qc_study(study_id, out_root=out_root) for study_id in study_ids]


def voxel_volume_cm3(mask, affine) -> float:
    voxel = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    return float(np.asarray(mask).sum()) * voxel / 1000.0


def abc2_volume_cm3(mask, affine) -> float:
    """ABC/2 ellipsoid volume in cm3 from a binary tumor mask.

    A and B are the longest in-plane diameter and its perpendicular on the
    axial slice with the most tumor voxels. C is the superior-inferior extent.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        return 0.0
    affine = np.asarray(affine, dtype=np.float64)
    spacing = np.asarray(voxel_sizes(affine)[:3], dtype=np.float64)
    areas = mask.sum(axis=(0, 1))
    k = int(np.argmax(areas))
    sl = mask[:, :, k]
    coords = np.argwhere(sl)
    if coords.size == 0:
        return 0.0
    # Physical 2D points on this slice (axis 0,1).
    pts = coords.astype(np.float64) * spacing[:2]
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((delta**2).sum(axis=-1))
    a_idx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    a_vec = pts[a_idx[1]] - pts[a_idx[0]]
    a_mm = float(dist[a_idx])
    if a_mm <= 0:
        return 0.0
    direction = a_vec / a_mm
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    proj = pts @ perp
    b_mm = float(proj.max() - proj.min())
    zs = np.argwhere(mask)[:, 2]
    c_mm = float((int(zs.max()) - int(zs.min()) + 1) * spacing[2])
    return (a_mm * b_mm * c_mm) / 2.0 / 1000.0


def labeled_wt_components(wt, affine):
    """Connected WT components, largest first. Each item has mask, volume_cm3, centroid_ijk."""
    from recon.postprocess import CONN
    from scipy import ndimage

    wt = np.asarray(wt, dtype=bool)
    if not np.any(wt):
        return []
    labeled, count = ndimage.label(wt, structure=CONN)
    parts = []
    for index in range(1, count + 1):
        mask = labeled == index
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0)
        parts.append(
            {
                "id": index,
                "mask": mask,
                "volume_cm3": voxel_volume_cm3(mask, affine),
                "centroid_ijk": tuple(float(v) for v in centroid),
                "n_voxels": int(mask.sum()),
            }
        )
    parts.sort(key=lambda item: item["volume_cm3"], reverse=True)
    return parts


def abc2_largest_stats(wt, affine) -> dict:
    """ABC/2 on the largest WT component, plus the voxel volume of that component."""
    parts = labeled_wt_components(wt, affine)
    if not parts:
        return {
            "wt_components": 0,
            "largest_component_cm3": 0.0,
            "abc2_cm3": 0.0,
            "abc2_largest_cm3": 0.0,
            "ratio": None,
            "component_volumes_cm3": [],
        }
    largest = parts[0]
    abc2 = abc2_volume_cm3(largest["mask"], affine)
    vol = float(largest["volume_cm3"])
    ratio = None if vol <= 0 else abc2 / vol
    return {
        "wt_components": len(parts),
        "largest_component_cm3": vol,
        "abc2_cm3": abc2,
        "abc2_largest_cm3": abc2,
        "ratio": ratio,
        "component_volumes_cm3": [float(p["volume_cm3"]) for p in parts],
    }


def nearest_lobe_name(parc, ijk) -> str:
    """Lobe name at or nearest to a voxel (MNI 101-105, else catalog name)."""
    from recon.labels import anatomy as anatomy_by_id
    from scipy import ndimage

    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    ijk = tuple(int(np.clip(round(float(v)), 0, labels.shape[axis] - 1)) for axis, v in enumerate(ijk))
    lobe_ids = (101, 102, 103, 104, 105)
    names = {
        101: "Frontal Lobe",
        102: "Parietal Lobe",
        103: "Temporal Lobe",
        104: "Occipital Lobe",
        105: "Insular Lobe",
    }
    present = np.isin(labels, lobe_ids)
    if np.any(present):
        if present[ijk]:
            return names.get(int(labels[ijk]), "unknown")
        _dist, inds = ndimage.distance_transform_edt(~present, return_indices=True)
        nearest = int(labels[inds[0][ijk], inds[1][ijk], inds[2][ijk]])
        return names.get(nearest, "unknown")
    spec = anatomy_by_id(int(labels[ijk]))
    return spec.full_name if spec is not None else "unknown"


def load_study_label(study_id: str, *, out_root: Path = DEFAULT_OUTPUT):
    pred = Path(out_root) / study_id / "pred_seg.nii.gz"
    if not pred.is_file():
        raise FileNotFoundError(f"no pred_seg for {study_id}")
    image = nib.load(str(pred))
    label = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
    return label, np.asarray(image.affine, dtype=np.float64)


def load_study_parc(study_id: str, *, out_root: Path = DEFAULT_OUTPUT):
    out_dir = Path(out_root) / study_id
    for name in ("synthseg_raw.nii.gz", "anatomy_seg.nii.gz"):
        path = out_dir / name
        if path.is_file():
            image = nib.load(str(path))
            parc = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
            return parc
    return None


def report_study_wt_components(study_id: str, *, out_root: Path = DEFAULT_OUTPUT, probs=None) -> list[dict]:
    label, affine = load_study_label(study_id, out_root=out_root)
    parc = load_study_parc(study_id, out_root=out_root)
    return describe_wt_components(label, affine, probs=probs, parc=parc)


def describe_wt_components(label, affine, *, probs=None, parc=None) -> list[dict]:
    """Per-component WT stats: volume, centroid, nearest lobe, mean pWT, contains TC."""
    from recon.evaluate import regions_from_label
    from recon.infer import CHANNEL_ORDER

    regions = regions_from_label(label)
    parts = labeled_wt_components(regions["WT"], affine)
    wt_index = CHANNEL_ORDER.index("WT")
    spacing = np.asarray(voxel_sizes(np.asarray(affine, dtype=np.float64))[:3], dtype=np.float64)
    origin = np.asarray(affine, dtype=np.float64)[:3, 3]
    direction = np.asarray(affine, dtype=np.float64)[:3, :3]
    rows = []
    for rank, part in enumerate(parts, start=1):
        ijk = part["centroid_ijk"]
        world = direction @ np.array(ijk, dtype=np.float64) + origin
        mean_p = None
        if probs is not None:
            wt_prob = np.asarray(probs[wt_index], dtype=np.float32)
            mean_p = float(wt_prob[part["mask"]].mean()) if np.any(part["mask"]) else None
        lobe = nearest_lobe_name(parc, ijk) if parc is not None else None
        rows.append(
            {
                "rank": rank,
                "id": part["id"],
                "volume_cm3": part["volume_cm3"],
                "centroid_ijk": ijk,
                "centroid_mm": tuple(float(v) for v in world),
                "nearest_lobe": lobe,
                "mean_wt_prob": mean_p,
                "contains_tc": bool(np.any(regions["TC"] & part["mask"])),
                "n_voxels": part["n_voxels"],
                "spacing_mm": tuple(float(v) for v in spacing),
            }
        )
    return rows
