"""QA sheets for held-out validation cases.

For each case, write axial, coronal, and sagittal PNGs through the tumor
centre (green ground-truth contour, red prediction contour) and a 3D
screenshot of the tumor meshes. ``reports/qa/index.html`` lists Dice, HD95,
precision, false-positive components, and mesh volume error. Cases with any
region precision below 0.9, or any tumor-mesh volume error beyond 3% in
magnitude, are repeated at the top for manual review.

Example:
    python scripts/qa_report.py
    python scripts/qa_report.py --num-cases 10 --out-dir reports/qa
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.evaluate import REGION_NAMES, regions_from_label, score_case  # noqa: E402
from recon.mesh import check_mesh, mesh_from_mask  # noqa: E402
from recon.postprocess import PostConfig, brain_mask_from_image, postprocess  # noqa: E402

_DEFAULT_DATA = Path(r"C:\Users\Sravani\Desktop\MRI project\data\processed\brats_nifti")
_DEFAULT_CKPT = Path(
    r"C:\Users\Sravani\Desktop\MRI project\models\checkpoints\brats_pretrain_smoke\best_model.pt"
)
_DEFAULT_CACHE = ROOT / "caches" / "val_probs"
_DEFAULT_OUT = ROOT / "reports" / "qa"

PRECISION_MIN = 0.9
VOLUME_ERROR_MAX_PCT = 3.0
_PLANES = (("sagittal", 0), ("coronal", 1), ("axial", 2))
_COMPARTMENTS = ("edema", "necrosis", "enhancing")
_RENDER_COLORS = {
    "edema": (1.0, 0.86, 0.0, 0.35),
    "necrosis": (0.86, 0.16, 0.16, 1.0),
    "enhancing": (0.24, 0.47, 1.0, 1.0),
}


def tumor_center(ground_truth, prediction) -> tuple[int, int, int]:
    """Voxel centre of the ground-truth whole tumor, else the prediction."""
    for source in (ground_truth, prediction):
        mask = regions_from_label(source)["WT"]
        if mask.any():
            com = np.argwhere(mask).mean(axis=0)
            return tuple(int(np.clip(round(float(value)), 0, mask.shape[axis] - 1)) for axis, value in enumerate(com))
    shape = np.squeeze(np.asarray(ground_truth)).shape
    return tuple(int(size // 2) for size in shape)


def review_reasons(regions: dict, volume_error_pct: dict) -> list[str]:
    """Precision below 0.9, or a mesh volume error outside ±3%."""
    reasons = []
    for name in REGION_NAMES:
        precision = regions[name]["precision"]
        if precision < PRECISION_MIN:
            reasons.append(f"{name} precision {precision:.3f}")
    for name in _COMPARTMENTS:
        error = volume_error_pct.get(name)
        if error is None or not np.isfinite(error):
            continue
        if abs(error) > VOLUME_ERROR_MAX_PCT:
            reasons.append(f"{name} volume error {error:+.1f}%")
    return reasons


def _window(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0, 1)


def save_slice_panels(flair, ground_truth, prediction, center, dest: Path) -> dict[str, str]:
    """PNG overlays at the tumor centre. Contours are the whole-tumor outline."""
    dest.mkdir(parents=True, exist_ok=True)
    gt = regions_from_label(ground_truth)["WT"]
    pred = regions_from_label(prediction)["WT"]
    paths = {}
    for name, axis in _PLANES:
        index = center[axis]
        image = _window(np.take(flair, index, axis=axis))
        gt_slice = np.take(gt, index, axis=axis)
        pred_slice = np.take(pred, index, axis=axis)
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
        ax.imshow(image.T, cmap="gray", origin="lower")
        if np.any(gt_slice):
            ax.contour(gt_slice.T.astype(float), levels=[0.5], colors="#1f9d3a", linewidths=1.2)
        if np.any(pred_slice):
            ax.contour(pred_slice.T.astype(float), levels=[0.5], colors="#d11f1f", linewidths=1.2)
        ax.set_title(f"{name} {index}", color="white")
        ax.set_axis_off()
        fig.patch.set_facecolor("black")
        path = dest / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        paths[name] = path
    return paths


def compartment_masks(label) -> dict[str, np.ndarray]:
    lab = np.rint(np.squeeze(np.asarray(label))).astype(np.int16)
    return {
        "edema": lab == 2,
        "necrosis": lab == 1,
        "enhancing": (lab == 3) | (lab == 4),
    }


def mesh_volume_errors(label, affine) -> tuple[dict, dict, dict]:
    """Per-compartment mesh, signed volume error (%), and watertight flag."""
    meshes = {}
    errors = {}
    watertight = {}
    for name, mask in compartment_masks(label).items():
        if not np.any(mask):
            meshes[name] = None
            errors[name] = None
            watertight[name] = None
            continue
        mesh = mesh_from_mask(mask, affine)
        meshes[name] = mesh
        if mesh is None:
            errors[name] = None
            watertight[name] = None
            continue
        closed, error = check_mesh(mesh, mask, affine)
        watertight[name] = closed
        errors[name] = None if not np.isfinite(error) else float(error)
    return meshes, errors, watertight


def save_render(meshes: dict, dest: Path) -> Path:
    """Offscreen 3D view of the edema, necrosis, and enhancing meshes."""
    fig = plt.figure(figsize=(5.2, 5.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    points = []
    for name in _COMPARTMENTS:
        mesh = meshes.get(name)
        if mesh is None or len(mesh.faces) == 0:
            continue
        triangles = mesh.vertices[mesh.faces]
        if len(triangles) > 4000:
            triangles = triangles[:: int(np.ceil(len(triangles) / 4000))]
        color = _RENDER_COLORS[name]
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=color, edgecolors="none"))
        points.append(mesh.vertices)
    if points:
        stacked = np.concatenate(points, axis=0)
        center = (stacked.min(axis=0) + stacked.max(axis=0)) / 2
        radius = max(float((stacked.max(axis=0) - stacked.min(axis=0)).max()) / 2, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=25, azim=-55)
    ax.set_axis_off()
    fig.patch.set_facecolor("#111315")
    ax.set_facecolor("#111315")
    path = dest / "render.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _affine(image, spacing) -> np.ndarray:
    matrix = getattr(image, "affine", None)
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(-1)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing[:3]
    return affine


def _flair(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 4:
        return np.asarray(array[0], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def evaluate_case(case_id, image, label, probs, spacing, affine, out_dir: Path, config=None, model="ours") -> dict:
    """Score one case and write its four PNGs under ``out_dir/images/<case_id>``."""
    if config is None:
        config = PostConfig.for_model(model)
    brain = brain_mask_from_image(image, config)
    prediction = postprocess(np.asarray(probs, dtype=np.float32), brain, config)
    regions = score_case(prediction, label, spacing, case_id=case_id, model=model)
    meshes, errors, watertight = mesh_volume_errors(prediction, affine)
    folder = out_dir / "images" / case_id
    center = tumor_center(label, prediction)
    panels = save_slice_panels(_flair(image), label, prediction, center, folder)
    render = save_render(meshes, folder)
    reasons = review_reasons(regions, errors)
    return {
        "case_id": case_id,
        "center": center,
        "regions": {name: regions[name] for name in REGION_NAMES},
        "volume_error_pct": errors,
        "watertight": watertight,
        "reasons": reasons,
        "images": {
            "sagittal": panels["sagittal"].relative_to(out_dir).as_posix(),
            "coronal": panels["coronal"].relative_to(out_dir).as_posix(),
            "axial": panels["axial"].relative_to(out_dir).as_posix(),
            "render": render.relative_to(out_dir).as_posix(),
        },
    }


def _fmt(value, digits: int) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _metric_cell(record, key, digits) -> str:
    return " / ".join(_fmt(record["regions"][name][key], digits) for name in REGION_NAMES)


def _error_cell(record) -> str:
    parts = []
    for name in _COMPARTMENTS:
        error = record["volume_error_pct"].get(name)
        closed = record["watertight"].get(name)
        if error is None and closed is None:
            parts.append(f"{name} —")
        elif error is None:
            parts.append(f"{name} not watertight")
        else:
            parts.append(f"{name} {error:+.1f}%")
    return ", ".join(parts)


def write_report(cases: list[dict], path: Path) -> Path:
    """HTML report. Flagged cases are listed before the full gallery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged = [case for case in cases if case["reasons"]]
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>Validation QA</title>",
        "<style>",
        "body{font:15px/1.45 Segoe UI,sans-serif;margin:24px;background:#f6f5f2;color:#1c1c1c}",
        "table{border-collapse:collapse;width:100%;background:white}",
        "th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}",
        "th{font-size:13px;color:#555}",
        ".flag{background:#fde8e4}",
        ".card{background:white;padding:16px;margin:16px 0}",
        ".card img{width:220px;margin-right:8px;background:#111}",
        "h1,h2{font-weight:600}",
        "</style></head><body>",
        "<h1>Validation QA</h1>",
        "<p>Slices are through the whole-tumor centre. Green is ground truth, red is the prediction. "
        "Metrics are WT / TC / ET. A case is listed for manual review when any region precision is below "
        f"{PRECISION_MIN:.1f} or any mesh volume error exceeds {VOLUME_ERROR_MAX_PCT:.0f}% in magnitude.</p>",
        "<h2>Manual review</h2>",
    ]
    if not flagged:
        parts.append("<p>No cases flagged.</p>")
    else:
        parts.append("<ul>")
        for case in flagged:
            case_id = html.escape(case["case_id"])
            reasons = "; ".join(html.escape(reason) for reason in case["reasons"])
            parts.append(f"<li><a href='#case-{case_id}'>{case_id}</a>: {reasons}</li>")
        parts.append("</ul>")
    parts.append("<h2>All cases</h2>")
    parts.append(
        "<table><thead><tr><th>Case</th><th>Dice</th><th>HD95 (mm)</th><th>Precision</th>"
        "<th>FP components</th><th>Mesh volume error</th></tr></thead><tbody>"
    )
    for case in cases:
        case_id = html.escape(case["case_id"])
        klass = " class='flag'" if case["reasons"] else ""
        parts.append(
            f"<tr{klass}><td><a href='#case-{case_id}'>{case_id}</a></td>"
            f"<td>{_metric_cell(case, 'dice', 3)}</td>"
            f"<td>{_metric_cell(case, 'hd95_mm', 1)}</td>"
            f"<td>{_metric_cell(case, 'precision', 3)}</td>"
            f"<td>{_metric_cell(case, 'lesion_fp', 0)}</td>"
            f"<td>{html.escape(_error_cell(case))}</td></tr>"
        )
    parts.append("</tbody></table>")
    for case in cases:
        case_id = html.escape(case["case_id"])
        banner = ""
        if case["reasons"]:
            banner = "<p><strong>Review:</strong> " + "; ".join(html.escape(r) for r in case["reasons"]) + "</p>"
        images = "".join(
            f"<img src='{html.escape(case['images'][name])}' alt='{name}'>"
            for name in ("axial", "coronal", "sagittal", "render")
        )
        parts.append(f"<section class='card' id='case-{case_id}'><h2>{case_id}</h2>{banner}{images}</section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _load_cached(cache_dir: Path, case_id: str, model: str = "ours"):
    """Cached probabilities in ``CHANNEL_ORDER``, or None when this case was not cached."""
    path = cache_dir / f"{case_id}.npz"
    if not path.is_file():
        return None
    from recon.infer import CHANNEL_ORDER, reorder_probabilities

    with np.load(path, allow_pickle=False) as data:
        probs = np.asarray(data["probs"], dtype=np.float32)
        if "model" in data.files and str(data["model"]) != model:
            raise ValueError(f"{path.name} was cached with model {str(data['model'])!r}, not {model!r}")
        if "channel_order" in data.files:
            order = tuple(str(name) for name in np.asarray(data["channel_order"]).reshape(-1))
        else:
            order = CHANNEL_ORDER
    return reorder_probabilities(probs, order)


def default_cache_dir(model: str) -> Path:
    return ROOT / "caches" / "val_probs_abhinav_v3" if model == "abhinav_v3" else _DEFAULT_CACHE


def default_out_dir(model: str) -> Path:
    return ROOT / "reports" / "qa_abhinav_v3" if model == "abhinav_v3" else _DEFAULT_OUT


def run(
    data_dir: Path,
    out_dir: Path,
    num_cases: int,
    cache_dir: Path,
    checkpoint: Path,
    config=None,
    model_kind: str = "ours",
    case_list: Path | None = None,
) -> Path:
    from recon.infer import load_model, predict_probs, roi_for
    from scripts.cache_probs import _load_case, _training_transform, select_cases

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"BraTS directory not found: {data_dir}. Point --data-dir at the processed NIfTI cases."
        )
    selected, source = select_cases(data_dir, model_kind, num_cases=num_cases, case_list=case_list)
    print(f"model={model_kind} {len(selected)} cases, {', '.join(source)}", flush=True)
    if config is None:
        config = PostConfig.for_model(model_kind)
    transform = None
    model = None
    records = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, case_dir in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case_dir.name}", flush=True)
        if transform is None:
            transform = _training_transform(model_kind)
        image, label, spacing = _load_case(case_dir, transform)
        probs = _load_cached(cache_dir, case_dir.name, model_kind)
        if probs is None:
            if model is None:
                model = load_model(checkpoint, None, model_kind)
            print("  no cache, running inference", flush=True)
            probs = predict_probs(model, image, roi_for(model_kind), model_name=model_kind)
        if probs.shape[1:] != label.shape:
            raise RuntimeError(f"{case_dir.name}: probs {probs.shape} vs label {label.shape}")
        records.append(
            evaluate_case(
                case_dir.name,
                image,
                label,
                probs,
                spacing,
                _affine(image, spacing),
                out_dir,
                config,
                model_kind,
            )
        )
    report = write_report(records, out_dir / "index.html")
    print(f"wrote {report}", flush=True)
    flagged = sum(1 for record in records if record["reasons"])
    print(f"flagged {flagged} of {len(records)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="HTML QA report for held-out validation cases")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Default: caches/val_probs or caches/val_probs_abhinav_v3")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", choices=("ours", "abhinav_v3"), default="ours")
    parser.add_argument("--case-list", type=Path, default=None, help="Default for abhinav_v3: splits/abhinav_heldout.txt")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: reports/qa or reports/qa_abhinav_v3")
    parser.add_argument("--num-cases", type=int, default=10)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.model == "abhinav_v3":
            from recon.infer import resolve_checkpoint

            checkpoint = resolve_checkpoint(args.model)
        else:
            checkpoint = _DEFAULT_CKPT
    run(
        args.data_dir,
        args.out_dir or default_out_dir(args.model),
        args.num_cases,
        args.cache_dir or default_cache_dir(args.model),
        checkpoint,
        model_kind=args.model,
        case_list=args.case_list,
    )


if __name__ == "__main__":
    main()
