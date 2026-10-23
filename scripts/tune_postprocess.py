"""Grid-search post-processing on cached validation probabilities.

Thresholds are the grid 0.4, 0.5, 0.6, 0.7. A first pass pairs one shared
threshold with each minimum-component triple. A second pass searches every
per-head threshold combination at the winning component sizes.

The saved config is the one with the highest mean precision whose mean Dice
(WT, TC, ET) is at least the raw 0.5-threshold Dice minus 0.01. Raw scores
threshold each head at 0.5 and do not apply hierarchy or component removal.

Example:
    python scripts/tune_postprocess.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.evaluate import (  # noqa: E402
    REGION_NAMES,
    aggregate_scores,
    as_region_masks,
    choose_operating_point,
    region_overlap,
    score_case,
)
from recon.infer import CHANNEL_ORDER  # noqa: E402
from recon.postprocess import PostConfig, postprocess  # noqa: E402

_DEFAULT_CACHE = ROOT / "caches" / "val_probs"
_DEFAULT_CONFIG = ROOT / "configs" / "postprocess.yaml"
_DEFAULT_REPORT = ROOT / "reports" / "postprocess_ablation.csv"

THRESHOLD_VALUES = (0.4, 0.5, 0.6, 0.7)
# (ET, TC, WT) minimum 26-connected component sizes, in voxels.
MIN_COMPONENT_GRID = (
    (0, 0, 0),
    (20, 50, 100),
    (50, 150, 300),
    (100, 300, 600),
)


class CachedCase:
    def __init__(self, case_id, probs, label, brain, spacing):
        self.case_id = case_id
        self.probs = probs
        self.label = label
        self.brain = brain
        self.spacing = spacing


def _bounds(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _crop(probs, label, brain, spacing):
    sl = _bounds(brain)
    if sl is None:
        return probs, label, brain, spacing
    return probs[(slice(None),) + sl], label[sl], brain[sl], spacing


def load_cached_cases(cache_dir: Path) -> list[CachedCase]:
    files = sorted(path for path in cache_dir.glob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No cached probabilities in {cache_dir}. Run scripts/cache_probs.py first."
        )
    cases = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            probs = np.array(data["probs"])
            label = np.array(data["label"])
            brain = np.array(data["brain"]).astype(bool) if "brain" in data.files else np.ones(label.shape, dtype=bool)
            spacing = np.array(data["spacing"], dtype=np.float64) if "spacing" in data.files else np.ones(3)
            if "channel_order" in data.files:
                order = tuple(str(name) for name in data["channel_order"])
            else:
                order = CHANNEL_ORDER
            case_id = str(data["case_id"]) if "case_id" in data.files else path.stem
        if order != tuple(CHANNEL_ORDER):
            permute = [order.index(name) for name in CHANNEL_ORDER]
            probs = probs[permute]
        probs, label, brain, spacing = _crop(probs, label, brain, spacing)
        cases.append(CachedCase(case_id, probs, label, brain, spacing))
    return cases


def raw_region_masks(probs, brain, threshold=0.5) -> dict[str, np.ndarray]:
    """Threshold each head at ``threshold``. No hierarchy and no component filter."""
    masks = {}
    for index, name in enumerate(CHANNEL_ORDER):
        masks[name] = (np.asarray(probs[index]) > threshold) & brain
    return masks


def _config(base: PostConfig, thresholds, min_voxels) -> PostConfig:
    return PostConfig(
        threshold_et=float(thresholds[0]),
        threshold_tc=float(thresholds[1]),
        threshold_wt=float(thresholds[2]),
        min_voxels_et=int(min_voxels[0]),
        min_voxels_tc=int(min_voxels[1]),
        min_voxels_wt=int(min_voxels[2]),
        et_min_voxels=base.et_min_voxels,
        brain_closing_iterations=base.brain_closing_iterations,
    )


def search_configs(base: PostConfig) -> list[PostConfig]:
    """One shared threshold for every head, across the component-size grid."""
    configs = []
    for threshold in THRESHOLD_VALUES:
        for min_voxels in MIN_COMPONENT_GRID:
            configs.append(_config(base, (threshold, threshold, threshold), min_voxels))
    return configs


def threshold_grid(base: PostConfig, min_voxels) -> list[PostConfig]:
    """Every ET/TC/WT threshold in 0.4–0.7 at one component-size triple."""
    configs = []
    for thresholds in itertools.product(THRESHOLD_VALUES, repeat=3):
        configs.append(_config(base, thresholds, min_voxels))
    return configs


def _mean_dice_precision(case_scores: list[dict]) -> tuple[float, float]:
    dices = []
    precisions = []
    for region in REGION_NAMES:
        dices.append(float(np.mean([case[region]["dice"] for case in case_scores])))
        precisions.append(float(np.mean([case[region]["precision"] for case in case_scores])))
    return float(np.mean(dices)), float(np.mean(precisions))


def _overlap_cases(cases: list[CachedCase], masks_for_case) -> list[dict]:
    scored = []
    for case in cases:
        pred = masks_for_case(case)
        gt_label = case.label
        gt = as_region_masks(gt_label)
        row = {}
        for region in REGION_NAMES:
            dice, precision, recall = region_overlap(pred[region], gt[region])
            row[region] = {
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "hd95_mm": float("nan"),
                "lesion_fp": 0,
            }
        row["gt_empty_et_but_pred"] = (not gt["ET"].any()) and bool(np.asarray(pred["ET"]).any())
        scored.append(row)
    return scored


def evaluate_config(cases: list[CachedCase], config: PostConfig) -> dict:
    def masks_for_case(case: CachedCase):
        label = postprocess(np.asarray(case.probs, dtype=np.float32), case.brain, config)
        return as_region_masks(label)

    scores = _overlap_cases(cases, masks_for_case)
    mean_dice, mean_precision = _mean_dice_precision(scores)
    return {"config": config, "mean_dice": mean_dice, "mean_precision": mean_precision, "scores": scores}


def evaluate_raw(cases: list[CachedCase], threshold=0.5) -> dict:
    def masks_for_case(case: CachedCase):
        return raw_region_masks(np.asarray(case.probs, dtype=np.float32), case.brain, threshold)

    scores = _overlap_cases(cases, masks_for_case)
    mean_dice, mean_precision = _mean_dice_precision(scores)
    return {"mean_dice": mean_dice, "mean_precision": mean_precision, "scores": scores}


def full_summary(cases: list[CachedCase], prediction_for_case) -> dict:
    scored = []
    for case in cases:
        prediction = prediction_for_case(case)
        scored.append(score_case(prediction, case.label, case.spacing))
    return aggregate_scores(scored)


def _write_report(path: Path, raw: dict, post: dict, config: PostConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "region",
        "dice",
        "hd95_mm",
        "precision",
        "recall",
        "lesion_fp",
        "n_gt_empty_et_but_pred",
        "threshold_et",
        "threshold_tc",
        "threshold_wt",
        "min_voxels_et",
        "min_voxels_tc",
        "min_voxels_wt",
        "et_min_voxels",
    ]
    rows = []
    for variant, summary, thresholds, mins, et_min in (
        ("raw", raw, (0.5, 0.5, 0.5), ("", "", ""), ""),
        (
            "post",
            post,
            (config.threshold_et, config.threshold_tc, config.threshold_wt),
            (config.min_voxels_et, config.min_voxels_tc, config.min_voxels_wt),
            config.et_min_voxels,
        ),
    ):
        for region in list(REGION_NAMES) + ["mean"]:
            if region == "mean":
                metrics = {
                    "dice": summary["mean_dice"],
                    "hd95_mm": float(np.nanmean([summary[name]["hd95_mm"] for name in REGION_NAMES])),
                    "precision": summary["mean_precision"],
                    "recall": float(np.mean([summary[name]["recall"] for name in REGION_NAMES])),
                    "lesion_fp": int(sum(summary[name]["lesion_fp"] for name in REGION_NAMES)),
                }
            else:
                metrics = summary[region]
            rows.append(
                {
                    "variant": variant,
                    "region": region,
                    "dice": f"{metrics['dice']:.6f}",
                    "hd95_mm": "" if metrics["hd95_mm"] != metrics["hd95_mm"] else f"{metrics['hd95_mm']:.6f}",
                    "precision": f"{metrics['precision']:.6f}",
                    "recall": f"{metrics['recall']:.6f}",
                    "lesion_fp": metrics["lesion_fp"],
                    "n_gt_empty_et_but_pred": summary["n_gt_empty_et_but_pred"],
                    "threshold_et": thresholds[0],
                    "threshold_tc": thresholds[1],
                    "threshold_wt": thresholds[2],
                    "min_voxels_et": mins[0],
                    "min_voxels_tc": mins[1],
                    "min_voxels_wt": mins[2],
                    "et_min_voxels": et_min,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(cache_dir: Path, config_path: Path, report_path: Path, margin: float = 0.01) -> PostConfig | None:
    cases = load_cached_cases(cache_dir)
    base = PostConfig.from_yaml(config_path)
    print(f"cases={len(cases)} margin={margin}", flush=True)
    raw_fast = evaluate_raw(cases)
    print(
        f"raw 0.5  mean_dice={raw_fast['mean_dice']:.4f}  mean_precision={raw_fast['mean_precision']:.4f}",
        flush=True,
    )
    scored = []
    first_pass = search_configs(base)
    for index, config in enumerate(first_pass, start=1):
        result = evaluate_config(cases, config)
        scored.append(result)
        print(
            f"[{index}/{len(first_pass)}] thr=({config.threshold_et:.2f},{config.threshold_tc:.2f},{config.threshold_wt:.2f}) "
            f"min=({config.min_voxels_et},{config.min_voxels_tc},{config.min_voxels_wt}) "
            f"dice={result['mean_dice']:.4f} prec={result['mean_precision']:.4f}",
            flush=True,
        )
    winner = choose_operating_point(scored, raw_fast["mean_dice"], margin)
    if winner is None:
        size_for_grid = (0, 0, 0)
    else:
        chosen_size = winner["config"]
        size_for_grid = (
            chosen_size.min_voxels_et,
            chosen_size.min_voxels_tc,
            chosen_size.min_voxels_wt,
        )
    already = {
        (item["config"].threshold_et, item["config"].threshold_tc, item["config"].threshold_wt,
         item["config"].min_voxels_et, item["config"].min_voxels_tc, item["config"].min_voxels_wt)
        for item in scored
    }
    for config in threshold_grid(base, size_for_grid):
        key = (
            config.threshold_et, config.threshold_tc, config.threshold_wt,
            config.min_voxels_et, config.min_voxels_tc, config.min_voxels_wt,
        )
        if key in already:
            continue
        result = evaluate_config(cases, config)
        scored.append(result)
        print(
            f"[thresholds] thr=({config.threshold_et:.2f},{config.threshold_tc:.2f},{config.threshold_wt:.2f}) "
            f"min=({config.min_voxels_et},{config.min_voxels_tc},{config.min_voxels_wt}) "
            f"dice={result['mean_dice']:.4f} prec={result['mean_precision']:.4f}",
            flush=True,
        )
    winner = choose_operating_point(scored, raw_fast["mean_dice"], margin)
    if winner is None:
        print(
            f"No config kept mean Dice within {margin} of raw {raw_fast['mean_dice']:.4f}. "
            "configs/postprocess.yaml was left unchanged.",
            flush=True,
        )
        return None

    chosen = winner["config"]
    raw_summary = full_summary(
        cases,
        lambda case: raw_region_masks(np.asarray(case.probs, dtype=np.float32), case.brain, 0.5),
    )
    post_summary = full_summary(
        cases,
        lambda case: postprocess(np.asarray(case.probs, dtype=np.float32), case.brain, chosen),
    )
    chosen.write_yaml(config_path)
    _write_report(report_path, raw_summary, post_summary, chosen)
    print(
        f"winner thr=({chosen.threshold_et:.2f},{chosen.threshold_tc:.2f},{chosen.threshold_wt:.2f}) "
        f"min=({chosen.min_voxels_et},{chosen.min_voxels_tc},{chosen.min_voxels_wt}) "
        f"dice={post_summary['mean_dice']:.4f} prec={post_summary['mean_precision']:.4f}",
        flush=True,
    )
    print(f"wrote {config_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BraTS post-processing on cached validation probabilities")
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--margin", type=float, default=0.01)
    args = parser.parse_args()
    run(args.cache_dir, args.config, args.report, args.margin)


if __name__ == "__main__":
    main()
