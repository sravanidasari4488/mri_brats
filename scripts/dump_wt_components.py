"""Print every WT component for a real study (volume, centroid, lobe, TC)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.ingest_real import DEFAULT_OUTPUT  # noqa: E402
from recon.qc_real import report_study_wt_components  # noqa: E402


def _predict_probs(study_id: str, out_root: Path):
    from recon.infer import apply_model_preprocessing, load_model, predict_probs, resample_probabilities, roi_for, resolve_checkpoint
    from recon.io import load_case

    ready = out_root / study_id / "04_ready"
    loaded = load_case(ready, normalize=False)
    affine = loaded["affine"]
    checkpoint = resolve_checkpoint("abhinav_v3")
    network = load_model(checkpoint, None, "abhinav_v3")
    network_image, network_affine = apply_model_preprocessing(loaded["image"], affine, "abhinav_v3")
    probs = predict_probs(network, network_image, roi_for("abhinav_v3"), "fast", model_name="abhinav_v3", overlap=0.5)
    if probs.shape[1:] != tuple(loaded["image"].shape[1:]):
        probs = resample_probabilities(probs, network_affine, loaded["image"].shape[1:], affine)
    return probs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_id")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--predict", action="store_true", help="Run fast inference to fill mean_wt_prob")
    args = parser.parse_args(argv)
    probs = _predict_probs(args.study_id, args.out) if args.predict else None
    rows = report_study_wt_components(args.study_id, out_root=args.out, probs=probs)
    print(f"{args.study_id}: {len(rows)} WT component(s)", flush=True)
    print(
        f"{'rank':>4} {'vol_cm3':>10} {'n_vox':>8} {'contains_TC':>12} "
        f"{'mean_pWT':>10} {'lobe':<18} centroid_ijk  centroid_mm",
        flush=True,
    )
    serial = []
    for row in rows:
        mean_p = row.get("mean_wt_prob")
        mean_s = "-" if mean_p is None else f"{mean_p:.4f}"
        lobe = row.get("nearest_lobe") or "-"
        ijk = ",".join(f"{v:.1f}" for v in row["centroid_ijk"])
        xyz = ",".join(f"{v:.1f}" for v in row["centroid_mm"])
        print(
            f"{row['rank']:4d} {row['volume_cm3']:10.3f} {row['n_voxels']:8d} "
            f"{str(row['contains_tc']):>12} {mean_s:>10} {lobe:<18} ({ijk})  ({xyz})",
            flush=True,
        )
        serial.append(
            {
                "rank": row["rank"],
                "volume_cm3": round(float(row["volume_cm3"]), 3),
                "n_voxels": row["n_voxels"],
                "contains_tc": row["contains_tc"],
                "mean_wt_prob": mean_p,
                "nearest_lobe": lobe,
                "centroid_ijk": [round(float(v), 2) for v in row["centroid_ijk"]],
                "centroid_mm": [round(float(v), 2) for v in row["centroid_mm"]],
            }
        )
    dest = args.json or (ROOT / "reports" / f"{args.study_id}_wt_components.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(f"Wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
