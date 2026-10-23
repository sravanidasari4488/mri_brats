"""Local viewer and case API.

Upload saves a BraTS folder and runs ``build_scene``. The viewer then loads
``scene.glb``, ``manifest.json``, and PNG MRI slices cut on the same RAS grid
as the meshes.

Run:
    python web/app.py
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODALITY_SUFFIX = {"flair": "t2f", "t2f": "t2f", "t1c": "t1c"}
_REQUIRED = ("t2f", "t1n", "t1c", "t2w")


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

    @app.post("/api/cases")
    def create_case():
        try:
            case_id = _safe_case_id(request.form.get("case_id") or _guess_case_id())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
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
            return jsonify({"error": f"Missing NIfTI uploads: {', '.join(missing)}"}), 400
        if "seg" not in saved:
            _write_empty_seg(saved["t1n"], dest / "seg.nii.gz")

        try:
            from scripts.reconstruct import default_checkpoint
            from recon.scene import build_scene

            checkpoint = default_checkpoint()
            manifest = build_scene(
                dest,
                anatomy=request.form.get("anatomy") in {"1", "true", "on"},
                tracts=request.form.get("tracts") in {"1", "true", "on"},
                checkpoint=checkpoint,
                output_root=app.config["OUTPUTS"],
                synthseg=saved.get("synthseg"),
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        case_id = manifest["case_id"]
        return jsonify({"case_id": case_id, "viewer": f"/viewer?case={case_id}"})

    @app.get("/api/cases/<case_id>/manifest.json")
    def manifest(case_id: str):
        path = _output_file(app, case_id, "manifest.json")
        return send_file(path, mimetype="application/json")

    @app.get("/api/cases/<case_id>/scene.glb")
    def scene_glb(case_id: str):
        path = _output_file(app, case_id, "scene.glb")
        return send_file(path, mimetype="model/gltf-binary")

    @app.get("/api/cases/<case_id>/volume")
    def volume_meta(case_id: str):
        modality = request.args.get("modality", "flair")
        try:
            data, affine = _cached_volume(app, case_id, modality)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "modality": _MODALITY_SUFFIX[modality],
                "shape": [int(v) for v in data.shape],
                "affine": np.asarray(affine, dtype=np.float64).round(6).tolist(),
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
        png = _slice_png(data, axis, index)
        return send_file(io.BytesIO(png), mimetype="image/png")

    return app


def _safe_case_id(value: str) -> str:
    text = (value or "").strip()
    if not _CASE_ID.match(text):
        raise ValueError(f"Case id must be letters, numbers, dot, underscore, or hyphen: {value!r}")
    return text


def _guess_case_id() -> str:
    upload = request.files.get("t1n")
    name = Path(upload.filename).name if upload and upload.filename else "case"
    if name.lower().endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.lower().endswith(".nii"):
        name = name[: -len(".nii")]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "case"
    return name


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
    matches = []
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
        raise FileNotFoundError(f"No {suffix} NIfTI in {case_dir}")
    return matches[0]


def _cached_volume(app: Flask, case_id: str, modality: str):
    if modality not in _MODALITY_SUFFIX:
        raise ValueError("modality must be flair or t1c")
    suffix = _MODALITY_SUFFIX[modality]
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


def _slice_png(volume: np.ndarray, axis: int, index: int) -> bytes:
    """PNG of one RAS slice. Row 0 is the low index of the second free axis."""
    slab = np.take(volume, index, axis=axis)
    image = np.transpose(slab)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((image - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(scaled, mode="L").save(buf, format="PNG")
    return buf.getvalue()


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
