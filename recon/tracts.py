"""Atlas-estimated tracts warped from MNI into the patient T1.

BraTS has no diffusion MRI. These meshes are expected locations from the
JHU white-matter tractography atlas, not patient tractography. Every tract
object is stamped ``source="atlas-estimated"``.

The MNI template and the tract maps are never downloaded. Missing files raise
``AtlasNotFoundError`` and point at ``docs/ATLAS_SETUP.md``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from recon.mesh import mesh_from_mask
from recon.postprocess import CONN

SOURCE = "atlas-estimated"
DEFAULT_THRESHOLD = 0.25
TUMOR_MARGIN_VOXELS = 5

# 0-based volumes in FSL's JHU-ICBM-tracts-prob-2mm (left is the even index).
# Arcuate/SLF is the superior longitudinal fasciculus plus its temporal part.
# Optic radiation is not one of the 20 probability volumes.
JHU_PROB_VOLUMES = {
    "corticospinal_tract_L": (2,),
    "corticospinal_tract_R": (3,),
    "arcuate_slf_L": (14, 18),
    "arcuate_slf_R": (15, 19),
    "ifof_L": (10,),
    "ifof_R": (11,),
    "optic_radiation_L": (),
    "optic_radiation_R": (),
    "uncinate_L": (16,),
    "uncinate_R": (17,),
}

_TEMPLATE_NAMES = ("MNI152_T1_1mm.nii.gz", "MNI152_T1_1mm.nii")
_PROB_4D_NAMES = ("JHU-ICBM-tracts-prob-2mm.nii.gz", "JHU-ICBM-tracts-prob-2mm.nii")


class AtlasNotFoundError(FileNotFoundError):
    """Raised when the local MNI template or a JHU tract map is absent."""

    def __init__(self, message: str, missing: list[Path]):
        self.missing = missing
        super().__init__(message)


@dataclass
class AtlasTract:
    """One warped tract. ``source`` is always ``atlas-estimated``."""

    name: str
    mesh: object
    mask: np.ndarray
    threshold: float
    affine: np.ndarray
    source: str = SOURCE

    def __post_init__(self):
        self.source = SOURCE
        if self.mesh is None:
            return
        meta = dict(getattr(self.mesh, "metadata", {}) or {})
        meta["source"] = SOURCE
        meta["name"] = self.name
        meta["threshold"] = float(self.threshold)
        self.mesh.metadata = meta


def default_atlas_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "atlases"


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "caches" / "tract_transforms"


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _tract_file(jhu_dir: Path, name: str) -> Path | None:
    return _first_existing(jhu_dir, (f"{name}.nii.gz", f"{name}.nii"))


def resolve_atlas(atlas_dir: Path | None = None) -> tuple[Path, dict]:
    """Return ``(template_path, {tract: source})`` or raise ``AtlasNotFoundError``.

    ``source`` is a 3D NIfTI path, or ``(path_to_4d, volume_indices)``.
    """
    root = Path(atlas_dir) if atlas_dir is not None else default_atlas_dir()
    jhu_dir = root / "jhu"
    template = _first_existing(root, _TEMPLATE_NAMES)
    prob_4d = _first_existing(jhu_dir, _PROB_4D_NAMES)
    sources = {}
    missing = []
    if template is None:
        missing.append(root / _TEMPLATE_NAMES[0])
    for name, volumes in JHU_PROB_VOLUMES.items():
        individual = _tract_file(jhu_dir, name)
        if individual is not None:
            sources[name] = individual
        elif volumes and prob_4d is not None:
            sources[name] = (prob_4d, volumes)
        else:
            missing.append(jhu_dir / f"{name}.nii.gz")
    if missing:
        raise AtlasNotFoundError(_missing_message(root, prob_4d, missing), missing)
    return template, sources


def _missing_message(root: Path, prob_4d: Path | None, missing: list[Path]) -> str:
    lines = "\n".join(f"  {path}" for path in missing)
    four_d = "found" if prob_4d is not None else "not found"
    return (
        "Atlas files are missing. This project does not download them.\n"
        "Place the MNI152 1mm T1 and the JHU tract probability maps as described in docs/ATLAS_SETUP.md.\n"
        f"Atlas directory: {root}\n"
        f"4D JHU tractography atlas: {four_d}.\n"
        "Optic radiation is not a volume in JHU-ICBM-tracts-prob-2mm, so optic_radiation_L/R.nii.gz "
        "are required even when that 4D file is present.\n"
        f"Missing:\n{lines}"
    )


def _case_id_from_path(t1_path, case_id) -> str:
    if case_id:
        return str(case_id)
    name = Path(t1_path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(t1_path).stem


def _tumor_sha256(label) -> str:
    lab = np.ascontiguousarray(np.rint(np.squeeze(np.asarray(label))).astype(np.int16))
    return hashlib.sha256(lab.tobytes() + str(lab.shape).encode("ascii")).hexdigest()


def _import_ants():
    try:
        import ants
    except ImportError as exc:
        raise ImportError(
            "ANTsPy is required to warp atlas tracts. Install antspyx. "
            "Atlas files are still read from disk and are not downloaded."
        ) from exc
    return ants


def affine_from_ants(image) -> np.ndarray:
    """Voxel-index to world millimeters, matching ``mesh_from_mask``."""
    spacing = np.asarray(image.spacing, dtype=np.float64).reshape(-1)[:3]
    direction = np.asarray(image.direction, dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.origin, dtype=np.float64).reshape(-1)[:3]
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def _tumor_wt(label, shape) -> np.ndarray:
    from recon.evaluate import regions_from_label

    tumor = regions_from_label(label)["WT"]
    if tumor.shape != tuple(shape):
        raise ValueError(f"Tumor label shape {tumor.shape} does not match the patient T1 shape {tuple(shape)}")
    return tumor


def _cache_paths(cache_dir: Path, case_id: str, template: Path, label) -> tuple[Path, dict]:
    meta = {
        "case_id": case_id,
        "template": str(Path(template).resolve()),
        "tumor_sha256": _tumor_sha256(label),
    }
    return Path(cache_dir) / case_id, meta


def _load_cached_transforms(cache_case: Path, expected: dict) -> list[str] | None:
    manifest_path = cache_case / "manifest.json"
    if not manifest_path.is_file():
        return None
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved.get("template") != expected["template"] or saved.get("tumor_sha256") != expected["tumor_sha256"]:
        return None
    paths = [cache_case / name for name in saved.get("transforms", [])]
    if not paths or any(not path.is_file() for path in paths):
        return None
    return [str(path) for path in paths]


def _store_transforms(cache_case: Path, expected: dict, transforms: list[str]) -> list[str]:
    cache_case.mkdir(parents=True, exist_ok=True)
    stored = []
    copied = []
    for index, src in enumerate(transforms):
        src_path = Path(src)
        name = f"{index:02d}_{src_path.name}"
        dest = cache_case / name
        shutil.copy2(src_path, dest)
        stored.append(name)
        copied.append(str(dest))
    manifest = dict(expected)
    manifest["transforms"] = stored
    (cache_case / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return copied


def _register(ants, fixed, template_path, label):
    tumor = _tumor_wt(label, fixed.shape)
    margin = ndimage.binary_dilation(tumor, structure=CONN, iterations=TUMOR_MARGIN_VOXELS)
    healthy = ((np.asarray(fixed.numpy()) > 0) & ~margin).astype(np.float32)
    reg = ants.registration(
        fixed=fixed,
        moving=ants.image_read(str(template_path)),
        type_of_transform="SyN",
        mask=fixed.new_image_like(healthy),
    )
    transforms = list(reg["fwdtransforms"])
    if not transforms:
        raise RuntimeError("ANTs SyN returned no forward transforms.")
    return transforms


def _moving_image(ants, source):
    """3D probability image in atlas space, and a temp file to delete after the warp."""
    if isinstance(source, tuple):
        path, volumes = source
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4:
            raise ValueError(f"Expected a 4D JHU atlas at {path}, got shape {data.shape}")
        parts = [np.asarray(data[..., int(index)], dtype=np.float32) for index in volumes]
        prob = parts[0] if len(parts) == 1 else np.maximum.reduce(parts)
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        tmp.close()
        nib.save(nib.Nifti1Image(np.asarray(prob, dtype=np.float32), img.affine), tmp.name)
        return ants.image_read(tmp.name), Path(tmp.name)
    return ants.image_read(str(source)), None


def to_probability(values) -> np.ndarray:
    """Scale a JHU percent map (0–100) into 0–1. Maps already in 0–1 stay put."""
    prob = np.asarray(values, dtype=np.float32)
    peak = float(np.nanmax(prob)) if prob.size else 0.0
    if peak > 1.5:
        prob = prob / 100.0
    return prob


def _require_file(path: Path, kind: str) -> None:
    if not Path(path).is_file():
        raise AtlasNotFoundError(
            f"{kind} was not found: {path}\n"
            "This project does not download atlas or patient files. See docs/ATLAS_SETUP.md.",
            [Path(path)],
        )


def warp_atlas_tracts(
    t1_path,
    label,
    mni_template_path=None,
    tract_prob_paths=None,
    out_dir=None,
    threshold=DEFAULT_THRESHOLD,
    atlas_dir=None,
    case_id=None,
):
    """Register MNI to the patient T1 and mesh JHU tract probabilities.

    ``t1_path`` is the patient T1. ``label`` is the BraTS segmentation on that
    grid (WT is dilated by 5 voxels and excluded from the SyN cost). Transforms
    are cached under ``out_dir/<case_id>`` and reused while the template path
    and the tumor label stay the same.

    ``tract_prob_paths`` overrides the standard JHU set with ``{name: nifti}``.
    Otherwise files are resolved from ``atlas_dir`` (default ``atlases/``).

    Returns ``{name: AtlasTract}``. Probabilities are thresholded at
    ``threshold`` (default 0.25) after percent maps are scaled to 0–1.
    """
    if mni_template_path is None or tract_prob_paths is None:
        template, sources = resolve_atlas(atlas_dir)
        if mni_template_path is not None:
            template = Path(mni_template_path)
        if tract_prob_paths is not None:
            sources = {name: Path(path) for name, path in tract_prob_paths.items()}
    else:
        template = Path(mni_template_path)
        sources = {name: Path(path) for name, path in tract_prob_paths.items()}

    _require_file(t1_path, "Patient T1")
    _require_file(template, "MNI152 1mm T1 template")
    missing = [Path(source[0] if isinstance(source, tuple) else source) for source in sources.values()]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        raise AtlasNotFoundError(_missing_message(default_atlas_dir(), None, missing), missing)

    ants = _import_ants()
    fixed = ants.image_read(str(t1_path))
    _tumor_wt(label, fixed.shape)
    cache_root = Path(out_dir) if out_dir is not None else default_cache_dir()
    case = _case_id_from_path(t1_path, case_id)
    cache_case, expected = _cache_paths(cache_root, case, template, label)
    transforms = _load_cached_transforms(cache_case, expected)
    if transforms is None:
        transforms = _store_transforms(cache_case, expected, _register(ants, fixed, template, label))

    affine = affine_from_ants(fixed)
    tracts = {}
    for name, source in sources.items():
        moving, temporary = _moving_image(ants, source)
        try:
            warped = ants.apply_transforms(fixed, moving, transforms, interpolator="linear")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        prob = to_probability(warped.numpy())
        mask = prob >= float(threshold)
        mesh = mesh_from_mask(mask, affine)
        tracts[name] = AtlasTract(
            name=name,
            mesh=mesh,
            mask=mask,
            threshold=float(threshold),
            affine=affine,
        )
    return tracts


def _points_mm(mask, affine) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homo = np.concatenate([coords.astype(np.float64), np.ones((len(coords), 1))], axis=1)
    return (np.asarray(affine, dtype=np.float64) @ homo.T).T[:, :3]


def _min_vertex_distance_mm(points, vertices) -> float:
    if len(points) == 0 or len(vertices) == 0:
        return float("nan")
    tree = cKDTree(np.asarray(vertices, dtype=np.float64))
    distances, _ = tree.query(points, k=1)
    return float(np.min(distances))


def compute_tract_proximity(tumor_label, tracts, affine) -> dict:
    """Minimum tumor-to-tract distance in mm, plus voxel overlap.

    Distances use a KD-tree on the tract mesh vertices (already in mm).
    Whole tumor and enhancing tumor are reported separately. Overlap is the
    thresholded tract mask against that region, not a mesh intersection.
    """
    from recon.evaluate import regions_from_label

    regions = regions_from_label(tumor_label)
    report = {}
    for name, tract in tracts.items():
        mask = np.asarray(tract.mask, dtype=bool)
        if mask.shape != regions["WT"].shape:
            raise ValueError(f"Tract {name} mask shape {mask.shape} does not match the tumor label {regions['WT'].shape}")
        vertices = np.zeros((0, 3)) if tract.mesh is None else np.asarray(tract.mesh.vertices)
        report[name] = {
            "wt_distance_mm": _min_vertex_distance_mm(_points_mm(regions["WT"], affine), vertices),
            "et_distance_mm": _min_vertex_distance_mm(_points_mm(regions["ET"], affine), vertices),
            "wt_overlaps": bool(np.any(mask & regions["WT"])),
            "et_overlaps": bool(np.any(mask & regions["ET"])),
        }
    return report
