"""Inventory real hospital MRI folders (DICOM and NIfTI), read-only.

Walks a data root, groups files by StudyInstanceUID / SeriesInstanceUID,
guesses FLAIR/T1/T1c/T2, and writes one CSV row per unique study. Folder
names, PatientID, and UIDs are hashed; they are not copied into the CSVs.

Example:
    python scripts/inventory_real.py
    python scripts/inventory_real.py --root "C:\\Users\\Sravani\\Downloads\\MRI DATA"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = Path(r"C:\Users\Sravani\Downloads\MRI DATA")
DEFAULT_OUT = ROOT / "reports" / "real_inventory.csv"
DEFAULT_DUPES = ROOT / "reports" / "duplicates.csv"
REQUIRED = ("FLAIR", "T1", "T1c", "T2")
TIERS = ("FULL", "NO_CONTRAST", "FLAIR_T2_ONLY", "UNUSABLE")
READY_MAX_MM = 3.0
_HASH_SALT = b"mri_brats_real_inventory_v1"
_SKIP_DIR_NAMES = {"__macosx", ".ds_store"}
_PROTOCOL_WORDS = {"mr", "study", "mri", "data", "nifti", "dicom", "scan", "series"}

# Contrast: token CE / Gd / post / +C. Philips CLEAR / SENSE / SPIR are not contrast.
_CONTRAST = re.compile(r"(?:^|[^a-z0-9])(?:ce|gd|gad|gado|gadolinium|post)(?:[^a-z0-9]|$)")
_T1 = re.compile(r"(?:^|[^a-z0-9])(?:[es]*t1w?|t1w|mprage|mp2rage|tfe)(?:[^a-z0-9]|$)")
_T2 = re.compile(r"(?:^|[^a-z0-9])(?:[es]*t2w?|t2w|tse|drive)(?:[^a-z0-9]|$)")
_FFE = re.compile(r"(?:^|[^a-z0-9])ffe(?:[^a-z0-9]|$)")
_FLAIR = re.compile(r"flair")
_DWI = re.compile(r"(?:^|[^a-z0-9])(?:dwi|adc|dti|trace|tracew|isodwi|ddwi)(?:[^a-z0-9]|$)")
_SWI = re.compile(r"(?:^|[^a-z0-9])(?:swi|swip|swan|sws|venobold|t2star|t2\*|hemo)(?:[^a-z0-9]|$)")
_EXCLUDED = re.compile(
    r"(?:^|[^a-z0-9])(?:survey|surv|mobiview|spine|btfe|stir|localizer|scout)(?:[^a-z0-9]|$)"
)
_MIP = re.compile(r"(?:^|[^a-z0-9])mip(?:[^a-z0-9]|$)")
_VEN = re.compile(r"(?:^|[^a-z0-9])(?:pca|ven|vven|mrv)(?:[^a-z0-9]|$)")
_MRA = re.compile(r"(?:^|[^a-z0-9])(?:s3di|tof|mra|angio)(?:[^a-z0-9]|$)")


def hashed_id(kind: str, text: str) -> str:
    payload = kind.encode("ascii") + b"\0" + str(text).encode("utf-8", errors="replace")
    digest = hashlib.sha256(_HASH_SALT + payload).hexdigest()
    return f"{kind}_{digest[:12]}"


def folder_has_identifying_text(name: str) -> bool:
    """True when a folder name looks like a person (name, age, sex), not an accession."""
    cleaned = re.sub(r"(?i)\b(?:mr|study|mri|data|nifti|dicom|scan|series)\b", " ", name)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    if re.search(r"\d+\s*y(?:ears?)?\b", cleaned, re.I):
        return True
    if re.search(r"\b(?:male|female)\b", cleaned, re.I):
        return True
    words = re.findall(r"[A-Za-z]{3,}", cleaned)
    kept = [word for word in words if word.lower() not in _PROTOCOL_WORDS]
    if any("." in token for token in re.split(r"\s+", name) if re.search(r"[A-Za-z]{3,}\.[A-Za-z]{3,}", token)):
        return True
    return sum(len(word) for word in kept) >= 5


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _has_contrast(blob: str, compact: str) -> bool:
    if _CONTRAST.search(compact):
        return True
    collapsed = re.sub(r"\s+", "", blob)
    return "+c" in collapsed or "c+" in collapsed


def guess_modality(*parts: str, image_type=()) -> tuple[str, str, bool]:
    """Return (label, rule, usable_as_input)."""
    blob = _norm(" ".join(str(p) for p in parts if p))
    compact = re.sub(r"[^a-z0-9*+]+", " ", blob)
    types = " ".join(str(v).lower() for v in (image_type or ()))
    collapsed = compact.replace(" ", "")

    if _EXCLUDED.search(compact):
        return "excluded", "not brain tumor protocol", False
    if "swi" in collapsed or _SWI.search(compact):
        return "SWI", "recorded, not used as input (SWI)", False
    if _VEN.search(compact):
        return "VEN", "recorded, not used as input (PCA/VEN)", False
    if _MRA.search(compact):
        return "MRA", "recorded, not used as input (s3DI/MRA)", False
    if _MIP.search(compact):
        return "MIP", "recorded, not used as input (MIP)", False
    if "adc" in types:
        return "DWI", "ImageType ADC", False
    if _DWI.search(compact):
        return "DWI", "description matches dwi/adc/dti", False
    if _FLAIR.search(compact) and _T1.search(compact) and not re.search(r"t2", compact):
        if _has_contrast(blob, compact):
            return "T1c", "T1-FLAIR + CE/Gd/post/+C", True
        return "T1", "T1-FLAIR (IR T1, not T2-FLAIR)", True
    if _FLAIR.search(compact):
        return "FLAIR", "description contains flair", True
    t1 = bool(_T1.search(compact))
    t2 = bool(_T2.search(compact))
    ffe = bool(_FFE.search(compact))
    contrast = _has_contrast(blob, compact)
    if t1 and contrast:
        return "T1c", "T1-like + CE/Gd/post/+C", True
    if ffe and not t1:
        return "T2*", "FFE without T1 (GRE / T2*)", False
    if t1:
        return "T1", "description matches t1/t1w/mprage/tfe", True
    if t2:
        return "T2", "description matches t2/t2w/tse", True
    return "other", "no FLAIR/T1/T1c/T2 tokens", False


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _is_junk(path: Path) -> bool:
    name = path.name
    if name.startswith("._") or name.startswith("."):
        return True
    return name.lower() in {".ds_store", "thumbs.db", "dicomdir"}


def _skip_dir(path: Path) -> bool:
    return path.name.lower() in _SKIP_DIR_NAMES or path.name.startswith("._")


def _dicom_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            preamble = handle.read(132)
    except OSError:
        return False
    return len(preamble) >= 132 and preamble[128:132] == b"DICM"


def orientation_from_iop(iop) -> str:
    try:
        values = [float(v) for v in iop]
    except (TypeError, ValueError):
        return "unknown"
    if len(values) < 6:
        return "unknown"
    row = np.asarray(values[:3], dtype=np.float64)
    col = np.asarray(values[3:6], dtype=np.float64)
    normal = np.cross(row, col)
    if not np.any(np.isfinite(normal)) or float(np.linalg.norm(normal)) < 1e-6:
        return "unknown"
    axis = int(np.argmax(np.abs(normal)))
    names = ("sagittal", "coronal", "axial")
    label = names[axis]
    if abs(float(normal[axis]) / float(np.linalg.norm(normal))) < 0.9:
        return f"oblique {label}"
    return label


def orientation_from_affine(affine) -> str:
    from nibabel.orientations import aff2axcodes

    try:
        codes = "".join(aff2axcodes(np.asarray(affine, dtype=np.float64)))
    except Exception:
        return "unknown"
    last = codes[2] if len(codes) > 2 else ""
    if last in {"S", "I"}:
        return "axial"
    if last in {"A", "P"}:
        return "coronal"
    if last in {"R", "L"}:
        return "sagittal"
    return codes or "unknown"


def _fmt_spacing(values) -> str:
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return ""
    if len(nums) >= 2:
        return f"{nums[0]:.3f}x{nums[1]:.3f}"
    if len(nums) == 1:
        return f"{nums[0]:.3f}"
    return ""


def _fmt_matrix(rows, cols) -> str:
    try:
        if rows is None or cols is None:
            return ""
        return f"{int(cols)}x{int(rows)}"
    except (TypeError, ValueError):
        return ""


def _acquisition(flag, description: str, n_slices: int, thickness) -> str:
    text = (flag or "").strip().upper()
    if text in {"2D", "3D"}:
        return text
    blob = _norm(description)
    if re.search(r"(?:^|[^a-z0-9])3d(?:[^a-z0-9]|$)", blob):
        return "3D"
    try:
        thick = float(thickness)
    except (TypeError, ValueError):
        thick = None
    if n_slices >= 80 and thick is not None and thick <= 1.5:
        return "3D"
    return "2D" if n_slices else ""


def _read_dicom_header(path: Path):
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def _public_series(record: dict) -> dict:
    """Series fields that may be written to CSV. No names, folders, or raw UIDs."""
    return {
        "series_description": record.get("series_description") or "(unnamed)",
        "modality": record.get("modality"),
        "rule": record.get("rule"),
        "usable_as_input": bool(record.get("usable_as_input")),
        "acquisition": record.get("acquisition") or "",
        "slice_thickness_mm": record.get("slice_thickness_mm"),
        "in_plane_spacing_mm": record.get("in_plane_spacing_mm") or "",
        "matrix": record.get("matrix") or "",
        "n_slices": record.get("n_slices") or 0,
        "orientation": record.get("orientation") or "unknown",
    }


def _classify_header(desc, protocol, bolus, image_type) -> tuple[str, str, bool]:
    modality, rule, usable = guess_modality(desc, protocol, bolus, image_type=image_type)
    if bolus and modality == "T1":
        return "T1c", "ContrastBolusAgent set on T1", True
    return modality, rule, usable


def _geometry(ds, n_slices: int, desc: str) -> dict:
    thickness = getattr(ds, "SliceThickness", None)
    try:
        thick = float(thickness) if thickness is not None else None
    except (TypeError, ValueError):
        thick = None
    return {
        "acquisition": _acquisition(getattr(ds, "MRAcquisitionType", None), desc, n_slices, thickness),
        "slice_thickness_mm": thick,
        "in_plane_spacing_mm": _fmt_spacing(getattr(ds, "PixelSpacing", None)),
        "matrix": _fmt_matrix(getattr(ds, "Rows", None), getattr(ds, "Columns", None)),
        "n_slices": int(n_slices),
        "orientation": orientation_from_iop(getattr(ds, "ImageOrientationPatient", None)),
    }


def _series_from_dicom(files: list[Path], study_dir: Path, root: Path) -> dict:
    files = sorted(files)
    ds = _read_dicom_header(files[0])
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    n_slices = max(len(files), n_frames)
    desc = str(getattr(ds, "SeriesDescription", "") or getattr(ds, "ProtocolName", "") or "")
    protocol = str(getattr(ds, "ProtocolName", "") or "")
    image_type = list(getattr(ds, "ImageType", []) or [])
    bolus = str(getattr(ds, "ContrastBolusAgent", "") or "")
    modality, rule, usable = _classify_header(desc, protocol, bolus, image_type)
    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
    patient_id = str(getattr(ds, "PatientID", "") or "")
    try:
        folder_key = str(study_dir.relative_to(root)).replace("\\", "/")
    except ValueError:
        folder_key = study_dir.name
    fallback_study = folder_key
    fallback_series = f"{folder_key}/{files[0].parent.name}"
    return {
        "series_description": desc or "(unnamed)",
        "modality": modality,
        "rule": rule,
        "usable_as_input": usable,
        **_geometry(ds, n_slices, desc),
        "study_uid_hash": hashed_id("study", study_uid or fallback_study),
        "series_uid_hash": hashed_id("series", series_uid or fallback_series),
        "patient_id_hash": hashed_id("patient", patient_id or study_uid or "unknown"),
        "folder_hash": hashed_id("folder", folder_key),
        "identifying_folder": folder_has_identifying_text(study_dir.name),
        "n_files": len(files),
        "source_kind": "dicom",
        "source_path": files[0].parent,
    }


def _series_from_nifti(path: Path, root: Path) -> dict:
    import nibabel as nib

    image = nib.load(str(path))
    shape = tuple(int(n) for n in image.shape[:3])
    zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
    n_slices = int(shape[2]) if len(shape) >= 3 else int(shape[-1] if shape else 0)
    thickness = zooms[2] if len(zooms) >= 3 else None
    desc = path.name
    modality, rule, usable = guess_modality(desc)
    acq = _acquisition(None, desc, n_slices, thickness)
    study_dir = path.parent
    try:
        folder_key = str(study_dir.relative_to(root)).replace("\\", "/")
    except ValueError:
        folder_key = study_dir.name
    file_key = str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else path.name
    return {
        "series_description": "(nifti)",
        "modality": modality,
        "rule": f"filename: {rule}",
        "usable_as_input": usable,
        "acquisition": acq,
        "slice_thickness_mm": thickness,
        "in_plane_spacing_mm": _fmt_spacing(zooms[:2] if len(zooms) >= 2 else zooms),
        "matrix": _fmt_matrix(shape[1] if len(shape) >= 2 else None, shape[0] if shape else None),
        "n_slices": n_slices,
        "orientation": orientation_from_affine(image.affine),
        "study_uid_hash": hashed_id("study", folder_key),
        "series_uid_hash": hashed_id("series", file_key),
        "patient_id_hash": hashed_id("patient", folder_key),
        "folder_hash": hashed_id("folder", folder_key),
        "identifying_folder": folder_has_identifying_text(study_dir.name),
        "n_files": 1,
        "source_kind": "nifti",
        "source_path": path,
    }


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or _is_junk(path):
            continue
        if any(_skip_dir(part) for part in path.parents):
            continue
        yield path


def _study_dir_for_dicom(path: Path, root: Path) -> Path:
    series_dir = path.parent
    study_dir = series_dir.parent
    if study_dir == root or _skip_dir(study_dir):
        return series_dir
    return study_dir


def _keep_series(copies: list[dict]) -> tuple[dict, list[dict]]:
    """Prefer a non-identifying folder, then the copy with more files."""
    ranked = sorted(
        copies,
        key=lambda item: (item["identifying_folder"], -int(item.get("n_files") or 0), item["folder_hash"]),
    )
    return ranked[0], ranked[1:]


def _dupe_reason(kept: dict, dropped: dict) -> str:
    if kept["identifying_folder"] != dropped["identifying_folder"]:
        return "named_vs_numeric"
    return "copy_folder"


def collect_studies(root: Path) -> tuple[dict[str, dict], list[dict]]:
    """Unique studies by StudyInstanceUID, unique series by SeriesInstanceUID."""
    dicoms_by_folder: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    niftis: list[Path] = []
    for path in _iter_files(root):
        if _is_nifti(path):
            niftis.append(path)
            continue
        suffix = path.suffix.lower()
        if suffix == ".dcm" or (suffix == "" and _dicom_magic(path)):
            study_dir = _study_dir_for_dicom(path, root)
            dicoms_by_folder[(study_dir, path.parent.name)].append(path)

    raw_series: list[dict] = []
    for (study_dir, _series_name), files in dicoms_by_folder.items():
        try:
            raw_series.append(_series_from_dicom(files, study_dir, root))
        except Exception as exc:
            folder_key = study_dir.name
            raw_series.append(
                {
                    "series_description": "(unreadable DICOM)",
                    "modality": "other",
                    "rule": f"read error: {exc.__class__.__name__}",
                    "usable_as_input": False,
                    "acquisition": "",
                    "slice_thickness_mm": None,
                    "in_plane_spacing_mm": "",
                    "matrix": "",
                    "n_slices": len(files),
                    "orientation": "unknown",
                    "study_uid_hash": hashed_id("study", folder_key),
                    "series_uid_hash": hashed_id("series", f"{folder_key}/{_series_name}"),
                    "patient_id_hash": hashed_id("patient", "unknown"),
                    "folder_hash": hashed_id("folder", folder_key),
                    "identifying_folder": folder_has_identifying_text(study_dir.name),
                    "n_files": len(files),
                }
            )
    for path in niftis:
        try:
            raw_series.append(_series_from_nifti(path, root))
        except Exception as exc:
            folder_key = path.parent.name
            raw_series.append(
                {
                    "series_description": "(unreadable NIfTI)",
                    "modality": "other",
                    "rule": f"read error: {exc.__class__.__name__}",
                    "usable_as_input": False,
                    "acquisition": "",
                    "slice_thickness_mm": None,
                    "in_plane_spacing_mm": "",
                    "matrix": "",
                    "n_slices": 0,
                    "orientation": "unknown",
                    "study_uid_hash": hashed_id("study", folder_key),
                    "series_uid_hash": hashed_id("series", path.name),
                    "patient_id_hash": hashed_id("patient", "unknown"),
                    "folder_hash": hashed_id("folder", folder_key),
                    "identifying_folder": folder_has_identifying_text(path.parent.name),
                    "n_files": 1,
                }
            )

    by_series: dict[str, list[dict]] = defaultdict(list)
    for record in raw_series:
        by_series[record["series_uid_hash"]].append(record)

    duplicates: list[dict] = []
    kept_series: list[dict] = []
    identifying_studies: set[str] = set()
    for series_hash, copies in by_series.items():
        kept, dropped = _keep_series(copies)
        kept_series.append(kept)
        if any(item["identifying_folder"] for item in copies):
            identifying_studies.add(kept["study_uid_hash"])
            identifying_studies.update(item["study_uid_hash"] for item in copies)
        for item in dropped:
            duplicates.append(
                {
                    "study_id_kept": kept["study_uid_hash"],
                    "study_id_dropped": item["study_uid_hash"],
                    "patient_id_kept": kept["patient_id_hash"],
                    "patient_id_dropped": item["patient_id_hash"],
                    "series_id": series_hash,
                    "folder_id_kept": kept["folder_hash"],
                    "folder_id_dropped": item["folder_hash"],
                    "kept_identifying_folder": "yes" if kept["identifying_folder"] else "no",
                    "dropped_identifying_folder": "yes" if item["identifying_folder"] else "no",
                    "n_files_kept": kept["n_files"],
                    "n_files_dropped": item["n_files"],
                    "reason": _dupe_reason(kept, item),
                }
            )

    studies: dict[str, dict] = {}
    for record in kept_series:
        sid = record["study_uid_hash"]
        if sid not in studies:
            studies[sid] = {
                "patient_ids": Counter(),
                "identifying_folder": False,
                "folder_hashes": set(),
                "series": [],
            }
        bucket = studies[sid]
        bucket["patient_ids"][record["patient_id_hash"]] += 1
        bucket["identifying_folder"] = (
            bucket["identifying_folder"]
            or record["identifying_folder"]
            or record["study_uid_hash"] in identifying_studies
        )
        bucket["folder_hashes"].add(record["folder_hash"])
        bucket["series"].append(record)
    return studies, duplicates


def study_tier(present: dict[str, bool]) -> str:
    if present["FLAIR"] and present["T1"] and present["T1c"] and present["T2"]:
        return "FULL"
    if present["FLAIR"] and present["T1"] and present["T2"]:
        return "NO_CONTRAST"
    if present["FLAIR"] and present["T2"]:
        return "FLAIR_T2_ONLY"
    return "UNUSABLE"


def summarize_study(series: list[dict]) -> dict:
    present = {name: False for name in REQUIRED}
    best_thickness = {name: None for name in REQUIRED}
    for item in series:
        if not item.get("usable_as_input"):
            continue
        modality = item.get("modality")
        if modality not in present:
            continue
        present[modality] = True
        thick = item.get("slice_thickness_mm")
        if thick is None:
            continue
        thick = float(thick)
        current = best_thickness[modality]
        if current is None or thick < current:
            best_thickness[modality] = thick
    required_thickness = [best_thickness[name] for name in REQUIRED if best_thickness[name] is not None]
    missing = [name for name in REQUIRED if not present[name]]
    worst = max(required_thickness) if required_thickness else None
    if missing:
        status = f"INCOMPLETE (missing {', '.join(missing)})"
    elif worst is not None and worst > READY_MAX_MM:
        status = "DEGRADED"
    else:
        status = "READY"
    return {
        "has_FLAIR": "yes" if present["FLAIR"] else "no",
        "has_T1": "yes" if present["T1"] else "no",
        "has_T1c": "yes" if present["T1c"] else "no",
        "has_T2": "yes" if present["T2"] else "no",
        "worst_slice_thickness_mm": worst,
        "status": status,
        "missing_sequences": ", ".join(missing),
        "tier": study_tier(present),
        "input_series_count": sum(1 for item in series if item.get("usable_as_input")),
    }


def _series_summary(series: list[dict]) -> str:
    chunks = []
    for index, item in enumerate(series, start=1):
        thick = item.get("slice_thickness_mm")
        thick_s = "" if thick is None else f"{float(thick):.2f}mm"
        usable = "input" if item.get("usable_as_input") else "not-input"
        chunks.append(
            " | ".join(
                [
                    f"[{index}]",
                    str(item.get("series_description") or ""),
                    f"{item.get('modality')} ({item.get('rule')})",
                    usable,
                    str(item.get("acquisition") or ""),
                    thick_s,
                    str(item.get("in_plane_spacing_mm") or ""),
                    str(item.get("matrix") or ""),
                    f"{item.get('n_slices')} slices",
                    str(item.get("orientation") or ""),
                ]
            )
        )
    return " || ".join(chunks)


def write_duplicates(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study_id_kept",
        "study_id_dropped",
        "patient_id_kept",
        "patient_id_dropped",
        "series_id",
        "folder_id_kept",
        "folder_id_dropped",
        "kept_identifying_folder",
        "dropped_identifying_folder",
        "n_files_kept",
        "n_files_dropped",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(studies: dict[str, dict], out_path: Path) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study_id",
        "patient_id",
        "identifying_folder",
        "series_count",
        "input_series_count",
        "series",
        "series_summary",
        "has_FLAIR",
        "has_T1",
        "has_T1c",
        "has_T2",
        "worst_slice_thickness_mm",
        "status",
        "missing_sequences",
        "tier",
    ]
    counts = {name: 0 for name in TIERS}
    rows = []
    for study_id, payload in sorted(studies.items()):
        series = payload["series"]
        summary = summarize_study(series)
        counts[summary["tier"]] += 1
        patient_id = payload["patient_ids"].most_common(1)[0][0] if payload["patient_ids"] else hashed_id("patient", "unknown")
        rows.append(
            {
                "study_id": study_id,
                "patient_id": patient_id,
                "identifying_folder": "yes" if payload["identifying_folder"] else "no",
                "series_count": len(series),
                "input_series_count": summary.pop("input_series_count"),
                "series": json.dumps([_public_series(item) for item in series], ensure_ascii=False),
                "series_summary": _series_summary(series),
                **summary,
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Read-only inventory of real MRI folders")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA, help="Folder to scan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Inventory CSV path")
    parser.add_argument("--duplicates", type=Path, default=DEFAULT_DUPES, help="Duplicate series CSV path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root
    if not root.is_dir():
        print(f"Data folder not found: {root}", file=sys.stderr)
        return 1
    print(f"Scanning {root} (read-only)...", flush=True)
    studies, duplicates = collect_studies(root)
    counts = write_csv(studies, args.out)
    write_duplicates(duplicates, args.duplicates)
    patients = {payload["patient_ids"].most_common(1)[0][0] for payload in studies.values() if payload["patient_ids"]}
    identifying = sum(1 for payload in studies.values() if payload["identifying_folder"])
    named_pairs = sum(1 for row in duplicates if row["reason"] == "named_vs_numeric")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.duplicates} ({len(duplicates)} duplicate series copies dropped)")
    print(f"Unique studies: {len(studies)}")
    print(f"Unique patients: {len(patients)}")
    print("Tier counts:")
    for name in TIERS:
        print(f"  {name}: {counts[name]}")
    print(f"Studies whose source folder name looks identifying: {identifying} (hashed IDs only)")
    print(f"Named-vs-numeric series pairs: {named_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
