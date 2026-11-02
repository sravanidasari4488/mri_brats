"""QC and infer preprocessed real MRI studies.

Example:
    python scripts/infer_real.py --tier FULL --fast
    python scripts/infer_real.py --tier FULL --studies study_xxx --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.evaluate import regions_from_label  # noqa: E402
from recon.ingest_real import CLI_TIERS, DEFAULT_OUTPUT, load_inventory_rows  # noqa: E402
from recon.io import load_case  # noqa: E402
from recon.qc_real import abc2_largest_stats, qc_study  # noqa: E402
from recon.scene import build_scene  # noqa: E402
from recon.timing import STAGE_NAMES, stay_awake  # noqa: E402
from scripts.inventory_real import DEFAULT_OUT  # noqa: E402
from scripts.reconstruct import default_checkpoint  # noqa: E402


def _voxel_cm3(mask, affine) -> float:
    voxel = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    return float(np.asarray(mask).sum()) * voxel / 1000.0


def _summarize(label, probs, affine) -> dict:
    from recon.infer import CHANNEL_ORDER
    from recon.postprocess import CONN

    regions = regions_from_label(label)
    wt = regions["WT"]
    labeled, n_comp = ndimage.label(wt, structure=CONN)
    wt_index = CHANNEL_ORDER.index("WT")
    if np.any(wt):
        mean_wt = float(np.asarray(probs[wt_index])[wt].mean())
    else:
        mean_wt = float(np.asarray(probs[wt_index]).mean())
    abc2 = abc2_largest_stats(wt, affine)
    return {
        "wt_cm3": _voxel_cm3(regions["WT"], affine),
        "tc_cm3": _voxel_cm3(regions["TC"], affine),
        "et_cm3": _voxel_cm3(regions["ET"], affine),
        "abc2_cm3": abc2["abc2_largest_cm3"],
        "wt_components": int(n_comp),
        "mean_wt_prob": mean_wt,
        "largest_component_cm3": abc2["largest_component_cm3"],
        "abc2_largest_cm3": abc2["abc2_largest_cm3"],
        "ratio": abc2["ratio"],
        "component_volumes_cm3": abc2["component_volumes_cm3"],
    }


def infer_study(
    study_id: str,
    *,
    out_root: Path,
    checkpoint: Path,
    model: str,
    fast: bool,
    anatomy: bool,
) -> dict:
    from recon.infer import CHANNEL_ORDER, apply_model_preprocessing, load_model, predict_probs, resample_probabilities, roi_for
    from recon.postprocess import PostConfig, brain_mask_from_image, postprocess

    ready = out_root / study_id / "04_ready"
    from recon.timing import StageClock, STAGE_TIMEOUT_S, run_with_timeout

    clock = StageClock()
    loaded = load_case(ready, normalize=model != "abhinav_v3")
    loaded["case_id"] = study_id
    loaded["case_dir"] = ready
    affine = np.asarray(loaded["affine"], dtype=np.float64)
    config = PostConfig.for_model(model)
    network = load_model(checkpoint, None, model)
    network_image, network_affine = apply_model_preprocessing(loaded["image"], affine, model)
    window = {"overlap": 0.5} if fast else {}
    tta = "fast" if fast else True
    clock.start("inference")
    probs = run_with_timeout(
        STAGE_TIMEOUT_S,
        predict_probs,
        network,
        network_image,
        roi_for(model),
        tta,
        model_name=model,
        **window,
    )
    clock.stop()
    if probs.shape[1:] != tuple(loaded["image"].shape[1:]):
        probs = resample_probabilities(probs, network_affine, loaded["image"].shape[1:], affine)
    wt_prob = np.asarray(probs[CHANNEL_ORDER.index("WT")], dtype=np.float32)
    nib.save(
        nib.Nifti1Image(wt_prob, affine),
        str(out_root / study_id / "wt_prob.nii.gz"),
    )
    clock.start("postprocess")
    brain = brain_mask_from_image(loaded["image"], config)
    label = run_with_timeout(STAGE_TIMEOUT_S, postprocess, probs, brain, config)
    clock.stop()
    loaded["probs"] = probs
    manifest = build_scene(
        loaded,
        anatomy=anatomy,
        tracts=False,
        checkpoint=checkpoint,
        output_root=out_root,
        model=model,
        config=config,
        tta=tta,
        overlap=0.5 if fast else None,
    )
    stats = _summarize(label, probs, affine)
    stats["manifest"] = out_root / study_id / "manifest.json"
    stats["anatomy_warning"] = manifest.get("anatomy_warning")
    stage_times = _merge_stage_times(ready, manifest, inference_s=clock.times.get("inference"))
    merged = dict(manifest.get("stage_times_s") or {})
    seconds = dict(merged.get("seconds") or {})
    seconds.update(stage_times)
    merged["seconds"] = seconds
    manifest["stage_times_s"] = merged
    (out_root / study_id / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats["stage_times_s"] = stage_times
    return stats


STAGE_CSV = [f"{name}_s" for name in STAGE_NAMES]
CSV_FIELDS = [
    "study_id",
    "qc_status",
    "qc_reasons",
    "wt_cm3",
    "tc_cm3",
    "et_cm3",
    "abc2_cm3",
    "largest_component_cm3",
    "abc2_largest_cm3",
    "ratio",
    "wt_components",
    "component_volumes_cm3",
    "mean_wt_prob",
    "runtime_s",
    *STAGE_CSV,
    "anatomy_warning",
]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _merge_stage_times(ready: Path, manifest: dict | None = None, *, inference_s=None) -> dict:
    times = {}
    if manifest is None:
        manifest = _load_json(Path(ready).parent / "manifest.json")
    for key, value in ((manifest.get("stage_times_s") or {}).get("seconds") or {}).items():
        if value is not None:
            times[key] = value
    prep = _load_json(Path(ready) / "preprocess.json")
    prep_s = ((prep.get("stage_times_s") or {}).get("seconds") or {}).get("preprocess")
    if prep_s is not None:
        times["preprocess"] = prep_s
    if inference_s is not None:
        times["inference"] = round(float(inference_s), 2)
    return times


def _fmt_stage_row(times: dict) -> dict:
    row = {}
    for name in STAGE_NAMES:
        value = (times or {}).get(name)
        row[f"{name}_s"] = "" if value is None else f"{float(value):.1f}"
    return row


def _fmt_volumes(values) -> str:
    return ";".join(f"{float(v):.3f}" for v in (values or []))


def empty_result_row(study_id: str, qc) -> dict:
    row = {name: "" for name in CSV_FIELDS}
    row["study_id"] = study_id
    row["qc_status"] = qc.status
    row["qc_reasons"] = "; ".join(qc.reasons)
    return row


def summarize_existing(study_id: str, *, out_root: Path, previous: dict | None = None) -> dict:
    """Rebuild summary stats from pred_seg / manifests without re-running inference."""
    from recon.evaluate import regions_from_label as _regions

    ready = out_root / study_id / "04_ready"
    pred = out_root / study_id / "pred_seg.nii.gz"
    if not pred.is_file():
        return {}
    image = nib.load(str(pred))
    label = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
    affine = np.asarray(image.affine, dtype=np.float64)
    regions = _regions(label)
    abc2 = abc2_largest_stats(regions["WT"], affine)
    stats = {
        "wt_cm3": _voxel_cm3(regions["WT"], affine),
        "tc_cm3": _voxel_cm3(regions["TC"], affine),
        "et_cm3": _voxel_cm3(regions["ET"], affine),
        "abc2_cm3": abc2["abc2_largest_cm3"],
        "wt_components": abc2["wt_components"],
        "largest_component_cm3": abc2["largest_component_cm3"],
        "abc2_largest_cm3": abc2["abc2_largest_cm3"],
        "ratio": abc2["ratio"],
        "component_volumes_cm3": abc2["component_volumes_cm3"],
        "mean_wt_prob": None,
        "anatomy_warning": (_load_json(out_root / study_id / "manifest.json") or {}).get("anatomy_warning"),
        "stage_times_s": _merge_stage_times(ready),
    }
    if previous and previous.get("mean_wt_prob"):
        try:
            stats["mean_wt_prob"] = float(previous["mean_wt_prob"])
        except (TypeError, ValueError):
            pass
    return stats


def apply_stats(row: dict, stats: dict) -> dict:
    row["wt_cm3"] = f"{stats['wt_cm3']:.3f}"
    row["tc_cm3"] = f"{stats['tc_cm3']:.3f}"
    row["et_cm3"] = f"{stats['et_cm3']:.3f}"
    row["abc2_cm3"] = f"{stats['abc2_cm3']:.3f}"
    row["largest_component_cm3"] = f"{stats['largest_component_cm3']:.3f}"
    row["abc2_largest_cm3"] = f"{stats['abc2_largest_cm3']:.3f}"
    row["ratio"] = "" if stats.get("ratio") is None else f"{float(stats['ratio']):.3f}"
    row["wt_components"] = str(stats["wt_components"])
    row["component_volumes_cm3"] = _fmt_volumes(stats.get("component_volumes_cm3"))
    if stats.get("mean_wt_prob") is not None:
        row["mean_wt_prob"] = f"{float(stats['mean_wt_prob']):.4f}"
    row["anatomy_warning"] = stats.get("anatomy_warning") or ""
    row.update(_fmt_stage_row(stats.get("stage_times_s") or {}))
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def previous_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["study_id"]: row for row in csv.DictReader(handle) if row.get("study_id")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="QC and infer real MRI studies")
    parser.add_argument("--tier", choices=CLI_TIERS, required=True)
    parser.add_argument("--studies", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=ROOT / "reports" / "real_full_summary.csv")
    parser.add_argument("--model", choices=("ours", "abhinav_v3"), default="abhinav_v3")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fast", action="store_true", help="Four flipped views, overlap 0.5")
    parser.add_argument("--no-anatomy", action="store_true")
    parser.add_argument(
        "--from-outputs",
        action="store_true",
        help="Rebuild the summary CSV from existing pred_seg / manifests (no inference)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = load_inventory_rows(args.inventory, args.tier)
    rows.sort(key=lambda row: row.get("study_id") or "")
    study_ids = [row["study_id"] for row in rows if row.get("study_id")]
    if args.studies:
        keep = set(args.studies)
        study_ids = [study_id for study_id in study_ids if study_id in keep]
    if args.limit is not None:
        study_ids = study_ids[: max(0, int(args.limit))]
    previous = previous_rows(args.csv)
    checkpoint = None
    if not args.from_outputs:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
        elif args.model == "abhinav_v3":
            from recon.infer import resolve_checkpoint

            checkpoint = resolve_checkpoint(args.model)
        else:
            checkpoint = default_checkpoint()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with stay_awake():
        for index, study_id in enumerate(study_ids, start=1):
            print(f"[{index}/{len(study_ids)}] {study_id}", flush=True)
            started = time.perf_counter()
            qc = qc_study(study_id, out_root=args.out)
            row = empty_result_row(study_id, qc)
            if args.from_outputs:
                stats = summarize_existing(study_id, out_root=args.out, previous=previous.get(study_id))
                if stats:
                    apply_stats(row, stats)
                else:
                    row.update(_fmt_stage_row(_merge_stage_times(args.out / study_id / "04_ready")))
                prev = previous.get(study_id) or {}
                if stats and prev.get("wt_cm3") and prev.get("runtime_s"):
                    row["runtime_s"] = prev["runtime_s"]
                elif stats and not row.get("runtime_s"):
                    row["runtime_s"] = f"{time.perf_counter() - started:.1f}"
                else:
                    row["runtime_s"] = prev.get("runtime_s") or f"{time.perf_counter() - started:.1f}"
                print(
                    f"  QC {qc.status} WT={row['wt_cm3'] or '-'} ABC/2={row['abc2_largest_cm3'] or '-'} "
                    f"largest={row['largest_component_cm3'] or '-'}",
                    flush=True,
                )
                written.append(row)
                write_csv(args.csv, written)
                continue
            if qc.failed:
                print(f"  QC FAIL — skip inference ({row['qc_reasons']})", flush=True)
                row["runtime_s"] = f"{time.perf_counter() - started:.1f}"
                written.append(row)
                write_csv(args.csv, written)
                continue
            print(f"  QC {qc.status}: {row['qc_reasons'] or 'ok'}", flush=True)
            try:
                stats = infer_study(
                    study_id,
                    out_root=args.out,
                    checkpoint=checkpoint,
                    model=args.model,
                    fast=args.fast,
                    anatomy=not args.no_anatomy,
                )
            except Exception as exc:
                row["qc_status"] = "FAIL"
                row["qc_reasons"] = (row["qc_reasons"] + "; " if row["qc_reasons"] else "") + f"inference error: {exc.__class__.__name__}"
                row["runtime_s"] = f"{time.perf_counter() - started:.1f}"
                print(f"  FAIL {exc}", flush=True)
                written.append(row)
                write_csv(args.csv, written)
                continue
            apply_stats(row, stats)
            row["runtime_s"] = f"{time.perf_counter() - started:.1f}"
            print(
                f"  WT={row['wt_cm3']} TC={row['tc_cm3']} ET={row['et_cm3']} "
                f"ABC/2={row['abc2_largest_cm3']} largest={row['largest_component_cm3']} "
                f"ratio={row['ratio']} components={row['wt_components']} "
                f"mean_pWT={row['mean_wt_prob']} {row['runtime_s']}s",
                flush=True,
            )
            written.append(row)
            write_csv(args.csv, written)

    print(f"Wrote {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
