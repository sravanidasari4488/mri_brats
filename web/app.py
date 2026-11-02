"""Local viewer and case API.

Upload saves a BraTS folder and runs ``build_scene``. The viewer then loads
``scene.glb``, ``manifest.json``, and PNG MRI slices cut on the same RAS grid
as the meshes.

Run:
    python web/app.py
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED = ("t2f", "t1n", "t1c", "t2w")
_ROLE_NAMES = {"t2f": "FLAIR", "t1n": "T1", "t1c": "T1 with contrast", "t2w": "T2"}
# Four flipped views at overlap 0.5: the setting the tuned thresholds were fit on, ~3x faster than eight.
UPLOAD_TTA = "fast"
UPLOAD_OVERLAP = 0.5
_BUILDING: set[str] = set()
_BUILDING_LOCK = threading.Lock()
_JOBS: dict[str, object] = {}
_JOBS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def create_app(outputs_dir: Path | None = None, cases_dir: Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
    app.config["OUTPUTS"] = Path(outputs_dir) if outputs_dir else ROOT / "outputs"
    app.config["CASES"] = Path(cases_dir) if cases_dir else ROOT / "data" / "cases"
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    app.config["VOLUMES"] = {}

    @app.get("/")
    @app.get("/viewer")
    def viewer_page():
        return send_file(Path(app.static_folder) / "index.html")

    @app.errorhandler(FileNotFoundError)
    def missing(exc):
        return jsonify({"error": str(exc)}), 404

    @app.post("/api/identify")
    def identify():
        """Role of each uploaded file name, so the page can show plain scan names."""
        from recon.io import guess_case_id, identify_modality

        payload = request.get_json(silent=True) or {}
        names = [str(name) for name in payload.get("names", [])][:200]
        folder = payload.get("folder") or None
        return jsonify(
            {
                "roles": [identify_modality(name) for name in names],
                "required": list(_REQUIRED),
                "case_id": guess_case_id(names, folder),
            }
        )

    @app.post("/api/cases")
    def create_case():
        try:
            case_id = _safe_case_id(request.form.get("case_id") or _guess_case_id())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _BUILDING_LOCK:
            if case_id in _BUILDING:
                return jsonify({"error": "This scan is already being built. Please wait for it to finish."}), 409
            _BUILDING.add(case_id)
        try:
            return _enqueue_case(app, case_id)
        except Exception:
            with _BUILDING_LOCK:
                _BUILDING.discard(case_id)
            raise

    @app.get("/api/cases/<case_id>/job")
    def case_job(case_id: str):
        try:
            case_id = _safe_case_id(case_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _JOBS_LOCK:
            progress = _JOBS.get(case_id)
        if progress is not None:
            payload = progress.snapshot()
            payload["case_id"] = case_id
            if payload["status"] == "done":
                payload["viewer"] = f"/viewer?case={case_id}"
            return jsonify(payload)
        out = app.config["OUTPUTS"] / case_id
        if (out / "scene.glb").is_file() and (out / "manifest.json").is_file():
            return jsonify(
                {
                    "case_id": case_id,
                    "status": "done",
                    "stage": "Done",
                    "viewer": f"/viewer?case={case_id}",
                    "stages": {},
                    "error": None,
                    "warning": None,
                    "elapsed": 0,
                }
            )
        return jsonify({"error": f"No job for case {case_id}"}), 404

    @app.get("/api/cases/<case_id>/manifest.json")
    def manifest(case_id: str):
        path = _output_file(app, case_id, "manifest.json")
        return send_file(path, mimetype="application/json", max_age=0)

    @app.get("/api/cases/<case_id>/scene.glb")
    def scene_glb(case_id: str):
        path = _output_file(app, case_id, "scene.glb")
        return send_file(path, mimetype="model/gltf-binary", max_age=0)

    @app.get("/api/cases/<case_id>/volume")
    def volume_meta(case_id: str):
        from recon.slices import SEQUENCE_ORDER, resolve_sequence

        modality = request.args.get("modality", "flair")
        try:
            suffix = resolve_sequence(modality)
            data, affine = _cached_volume(app, case_id, modality)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        lo, hi = _foreground_bbox(data)
        brain = _brain_mask(app, case_id, data)
        windows = {}
        for name in SEQUENCE_ORDER:
            try:
                volume, _ = _cached_volume(app, case_id, name)
            except (FileNotFoundError, ValueError):
                continue
            mask = brain if brain is not None and brain.shape == volume.shape else None
            wlo, whi = _window_for(volume, mask)
            windows[name] = {"lo": wlo, "hi": whi, "suffix": resolve_sequence(name)}
        key = modality if modality in windows else "flair"
        chosen = windows.get(key, next(iter(windows.values()), {"lo": 0.0, "hi": 1.0}))
        from nibabel.affines import voxel_sizes
        from recon.slices import ORIENT_LETTERS, slice_extent_mm

        spacing = [float(abs(v)) for v in voxel_sizes(np.asarray(affine, dtype=np.float64))]
        extents = {str(axis): list(slice_extent_mm(affine, data.shape, axis)) for axis in (0, 1, 2)}
        return jsonify(
            {
                "modality": suffix,
                "shape": [int(v) for v in data.shape],
                "bbox": {"lo": lo, "hi": hi},
                "affine": np.asarray(affine, dtype=np.float64).round(6).tolist(),
                "spacing": spacing,
                "extent_mm": extents,
                "letters": {str(axis): dict(letters) for axis, letters in ORIENT_LETTERS.items()},
                "window": {"lo": chosen["lo"], "hi": chosen["hi"]},
                "windows": windows,
            }
        )

    @app.get("/api/cases/<case_id>/slice")
    def slice_png(case_id: str):
        modality = request.args.get("modality", "flair")
        try:
            axis = int(request.args.get("axis", "2"))
            index = int(request.args.get("index", "0"))
        except (TypeError, ValueError):
            return jsonify({"error": "axis and index must be integers"}), 400
        try:
            data, _affine = _cached_volume(app, case_id, modality)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if axis not in (0, 1, 2):
            return jsonify({"error": "axis must be 0, 1, or 2"}), 400
        if index < 0 or index >= data.shape[axis]:
            return jsonify({"error": f"index {index} is outside axis {axis} (length {data.shape[axis]})"}), 400
        brain = _brain_mask(app, case_id, data)
        default_lo, default_hi = _window_for(data, brain if brain is not None and brain.shape == data.shape else None)
        lo = _optional_float(request.args.get("lo"), default_lo)
        hi = _optional_float(request.args.get("hi"), default_hi)
        alpha = _combined_foreground(app, case_id, data.shape)
        png = _cached_slice_png(app, case_id, modality, data, axis, index, lo, hi, alpha=alpha)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=120)

    @app.get("/api/cases/<case_id>/overlay")
    def overlay_png(case_id: str):
        try:
            axis = int(request.args.get("axis", "2"))
            index = int(request.args.get("index", "0"))
        except (TypeError, ValueError):
            return jsonify({"error": "axis and index must be integers"}), 400
        try:
            label = _cached_pred(app, case_id)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        if axis not in (0, 1, 2):
            return jsonify({"error": "axis must be 0, 1, or 2"}), 400
        if index < 0 or index >= label.shape[axis]:
            return jsonify({"error": f"index {index} is outside axis {axis} (length {label.shape[axis]})"}), 400
        png = _cached_overlay_png(app, case_id, label, axis, index)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=120)

    return app


def _enqueue_case(app: Flask, case_id: str):
    """Save the uploaded scans, validate the grid, and start a background build."""
    dest = app.config["CASES"] / case_id
    dest.mkdir(parents=True, exist_ok=True)
    saved = {}
    for suffix in (*_REQUIRED, "seg", "synthseg"):
        upload = request.files.get(suffix)
        if upload is None or not upload.filename:
            continue
        saved[suffix] = _save_nifti(upload, dest, suffix)
    missing = [suffix for suffix in _REQUIRED if suffix not in saved]
    if missing:
        names = ", ".join(_ROLE_NAMES[suffix] for suffix in missing)
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)
        return jsonify({"error": f"Missing scans: {names}"}), 400
    if "seg" not in saved:
        _write_empty_seg(saved["t1n"], dest / "seg.nii.gz")

    try:
        from recon.io import validate_case_grid

        validate_case_grid(dest)
    except (FileNotFoundError, ValueError) as exc:
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)
        return jsonify({"error": str(exc)}), 400

    from recon.progress import BuildProgress

    anatomy = _form_flag("anatomy", default=True)
    tracts = _form_flag("tracts", default=False)
    model, checkpoint = _upload_model()
    synthseg = saved.get("synthseg")
    outputs = app.config["OUTPUTS"]
    progress = BuildProgress()
    with _JOBS_LOCK:
        _JOBS[case_id] = progress
    thread = threading.Thread(
        target=_run_build_job,
        kwargs={
            "case_id": case_id,
            "dest": dest,
            "outputs": outputs,
            "anatomy": anatomy,
            "tracts": tracts,
            "checkpoint": checkpoint,
            "model": model,
            "synthseg": synthseg,
            "progress": progress,
        },
        daemon=True,
        name=f"build-{case_id}",
    )
    thread.start()
    snap = progress.snapshot()
    return jsonify(
        {
            "case_id": case_id,
            "job": f"/api/cases/{case_id}/job",
            "status": snap["status"],
            "stage": snap["stage"],
            "viewer": f"/viewer?case={case_id}",
        }
    ), 202


def _run_build_job(
    *,
    case_id,
    dest,
    outputs,
    anatomy,
    tracts,
    checkpoint,
    model,
    synthseg,
    progress,
):
    """Worker: never touches Flask ``request``."""
    try:
        from recon.scene import build_scene

        build_scene(
            dest,
            anatomy=anatomy,
            tracts=tracts,
            checkpoint=checkpoint,
            output_root=outputs,
            synthseg=synthseg,
            model=model,
            tta=UPLOAD_TTA,
            overlap=UPLOAD_OVERLAP,
            progress=progress,
        )
        if progress.status == "running":
            progress.finish()
    except Exception as exc:
        if progress.status == "running":
            progress.fail(str(exc))
        logger.exception("reconstruction failed for %s", case_id)
    finally:
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)


def _form_flag(name: str, *, default: bool = False) -> bool:
    """HTML checkbox. Anatomy defaults on so a 3D brain is built even if the box is omitted."""
    raw = request.form.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "on", "yes"}


def _safe_case_id(value: str) -> str:
    text = (value or "").strip()
    if not _CASE_ID.match(text):
        raise ValueError(f"Case id must be letters, numbers, dot, underscore, or hyphen: {value!r}")
    return text


def _upload_model() -> tuple[str, Path]:
    """``abhinav_v3`` when its checkpoint is present, else the local checkpoint.

    ``BRATS_UPLOAD_MODEL=ours`` forces the local checkpoint.
    """
    from recon.infer import ABHINAV_CHECKPOINT, ABHINAV_V3, OURS
    from scripts.reconstruct import default_checkpoint

    wanted = os.environ.get("BRATS_UPLOAD_MODEL", ABHINAV_V3)
    if wanted == ABHINAV_V3 and Path(ABHINAV_CHECKPOINT).is_file():
        return ABHINAV_V3, Path(ABHINAV_CHECKPOINT)
    return OURS, default_checkpoint()


def _guess_case_id() -> str:
    from recon.io import guess_case_id

    names = [upload.filename for upload in request.files.values() if upload and upload.filename]
    return guess_case_id(names)


def _save_nifti(storage, dest: Path, suffix: str) -> Path:
    filename = (storage.filename or "").lower()
    ext = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    path = dest / f"{suffix}{ext}"
    storage.save(path)
    return path


def _write_empty_seg(reference: Path, dest: Path) -> None:
    image = nib.load(str(reference))
    shape = image.shape[:3]
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), image.affine), str(dest))


def _output_file(app: Flask, case_id: str, name: str) -> Path:
    case_id = _safe_case_id(case_id)
    path = app.config["OUTPUTS"] / case_id / name
    if not path.is_file():
        raise FileNotFoundError(f"No {name} for case {case_id}")
    return path


def _case_dir(app: Flask, case_id: str) -> Path:
    case_id = _safe_case_id(case_id)
    manifest = app.config["OUTPUTS"] / case_id / "manifest.json"
    if manifest.is_file():
        import json

        meta = json.loads(manifest.read_text(encoding="utf-8"))
        recorded = meta.get("case_dir")
        if recorded and Path(recorded).is_dir():
            return Path(recorded)
    fallback = app.config["CASES"] / case_id
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"No images for case {case_id}")


def _find_modality(case_dir: Path, suffix: str) -> Path:
    from recon.io import _find_volume

    return _find_volume(case_dir, suffix)


def _foreground_bbox(volume) -> tuple[list[int], list[int]]:
    """Inclusive voxel box of values above the background minimum."""
    data = np.asarray(volume)
    mask = np.isfinite(data) & (data > np.nanmin(data))
    if not np.any(mask):
        return [0, 0, 0], [int(n) - 1 for n in data.shape]
    coords = np.argwhere(mask)
    return coords.min(axis=0).astype(int).tolist(), coords.max(axis=0).astype(int).tolist()


def _cached_volume(app: Flask, case_id: str, modality: str):
    from recon.slices import resolve_sequence

    suffix = resolve_sequence(modality)
    key = (case_id, suffix)
    cache = app.config["VOLUMES"]
    if key not in cache:
        path = _find_modality(_case_dir(app, case_id), suffix)
        image = nib.as_closest_canonical(nib.load(str(path)))
        data = np.squeeze(np.asarray(image.get_fdata(dtype=np.float32)))
        if data.ndim != 3:
            raise ValueError(f"{path.name} is not a 3D volume")
        cache[key] = (data, np.asarray(image.affine, dtype=np.float64))
    return cache[key]


def _cached_pred(app: Flask, case_id: str):
    cache = app.config["VOLUMES"]
    key = (case_id, "pred")
    if key not in cache:
        path = _output_file(app, case_id, "pred_seg.nii.gz")
        image = nib.as_closest_canonical(nib.load(str(path)))
        data = np.rint(np.squeeze(np.asarray(image.get_fdata(dtype=np.float32)))).astype(np.int16)
        if data.ndim != 3:
            raise ValueError("pred_seg is not a 3D volume")
        cache[key] = data
    return cache[key]


def _brain_mask(app: Flask, case_id: str, volume) -> np.ndarray:
    cache = app.config["VOLUMES"]
    key = (case_id, "brain")
    if key in cache:
        return cache[key]
    for name in ("anatomy_seg.nii.gz", "pred_seg.nii.gz"):
        path = app.config["OUTPUTS"] / _safe_case_id(case_id) / name
        if not path.is_file():
            continue
        image = nib.as_closest_canonical(nib.load(str(path)))
        labels = np.rint(np.squeeze(np.asarray(image.get_fdata(dtype=np.float32)))).astype(np.int16)
        mask = labels != 0
        if np.any(mask):
            cache[key] = mask
            return mask
    data = np.asarray(volume)
    mask = np.isfinite(data) & (data > np.nanmin(data))
    cache[key] = mask
    return mask


def _window_for(volume, brain_mask) -> tuple[float, float]:
    from recon.slices import brain_window

    return brain_window(volume, brain_mask)


def _optional_float(raw, default: float) -> float:
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _slice_cache_dir(app: Flask, case_id: str) -> Path:
    dest = app.config["OUTPUTS"] / _safe_case_id(case_id) / "slices"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _combined_foreground(app: Flask, case_id: str, shape) -> np.ndarray:
    """Opaque MRI voxels. Prefer anatomy ∪ tumor; else sequences that are not 0.

    Normalized BraTS volumes are not exactly 0 outside the brain, so using
    ``volume != 0`` alone leaves a black rectangle. Label masks follow the
    brain. Cases without parcellation still use exact zeros (synthetic tests).
    """
    from recon.slices import SEQUENCE_ORDER
    from scipy import ndimage

    cache = app.config["VOLUMES"]
    key = (case_id, "fg5")
    if key in cache:
        return cache[key]
    fg = np.zeros(tuple(int(n) for n in shape[:3]), dtype=bool)
    has_parc = False
    out_dir = app.config["OUTPUTS"] / _safe_case_id(case_id)
    for name in ("anatomy_seg.nii.gz", "synthseg_raw.nii.gz", "pred_seg.nii.gz"):
        path = out_dir / name
        if not path.is_file():
            continue
        image = nib.as_closest_canonical(nib.load(str(path)))
        labels = np.rint(np.squeeze(np.asarray(image.get_fdata(dtype=np.float32)))).astype(np.int16)
        if labels.shape != fg.shape:
            continue
        if name != "pred_seg.nii.gz":
            has_parc = has_parc or np.any(labels)
        fg |= labels != 0
    if has_parc:
        fg = ndimage.binary_dilation(fg)
        cache[key] = fg
        return fg
    for name in SEQUENCE_ORDER:
        try:
            volume, _ = _cached_volume(app, case_id, name)
        except (FileNotFoundError, ValueError):
            continue
        if volume.shape == fg.shape:
            fg |= np.isfinite(volume) & (volume != 0)
    if not np.any(fg):
        fg = np.ones(fg.shape, dtype=bool)
    cache[key] = fg
    return fg


def _cached_slice_png(app, case_id, modality, data, axis, index, lo, hi, alpha=None) -> bytes:
    from recon.slices import cache_path, render_slice_png, resolve_sequence, window_token

    suffix = resolve_sequence(modality)
    extra = window_token(lo, hi) + ("_rgba_fg5" if alpha is not None else "")
    path = cache_path(_slice_cache_dir(app, case_id), suffix, axis, index, extra)
    if path.is_file():
        return path.read_bytes()
    png = render_slice_png(data, axis, index, lo, hi, alpha=alpha)
    path.write_bytes(png)
    return png


def _cached_overlay_png(app, case_id, label, axis, index) -> bytes:
    from recon.slices import cache_path, render_overlay_png

    path = cache_path(_slice_cache_dir(app, case_id), "overlay", axis, index)
    if path.is_file():
        return path.read_bytes()
    png = render_overlay_png(label, axis, index)
    path.write_bytes(png)
    return png


def _slice_png(volume: np.ndarray, axis: int, index: int) -> bytes:
    """PNG of one radiological RAS slice (same orientation as the 2D panels)."""
    from recon.slices import brain_window, render_slice_png

    lo, hi = brain_window(volume)
    return render_slice_png(volume, axis, index, lo, hi)


app = create_app()


def _warm_imports() -> None:
    """Import torch/MONAI and the scene code once, so the first request is not the slow one."""
    import recon.io  # noqa: F401
    import recon.scene  # noqa: F401


if __name__ == "__main__":
    threading.Thread(target=_warm_imports, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
