"""Grid-search post-processing on cached validation probabilities.

Thresholds are the grid 0.4 to 0.9 in steps of 0.1. A first pass pairs one
shared threshold with each minimum-component triple. A second pass searches
every per-head threshold combination at the winning component sizes. A winner
on the largest threshold or component size is reported as being at the edge
of the grid. ``--workers`` evaluates settings in parallel processes.

The saved config is the one with the highest mean precision whose mean Dice
(WT, TC, ET) is at least the raw 0.5-threshold Dice minus 0.01. Raw scores
threshold each head at 0.5 and do not apply hierarchy or component removal.

``--model abhinav_v3`` reads ``caches/val_probs_abhinav_v3``, refuses any
cached case from that checkpoint's training list, and writes
``configs/postprocess_abhinav_v3.yaml``. ``--model ours`` writes
``configs/postprocess_ours.yaml``.

Example:
    python scripts/tune_postprocess.py
    python scripts/tune_postprocess.py --model abhinav_v3 --workers 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
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
    check_reportable,
    choose_operating_point,
    evaluation_case_ids,
    region_overlap,
    score_case,
)
from recon.infer import CHANNEL_ORDER, channel_index, reorder_probabilities  # noqa: E402
from recon.postprocess import PostConfig, config_path_for, postprocess  # noqa: E402
from recon.splits import ABHINAV_V3, MODEL_NAMES, OURS, model_name  # noqa: E402

_HEADS = ("ET", "TC", "WT")

THRESHOLD_VALUES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Minimum 26-connected component sizes, in voxels.
MIN_COMPONENT_GRID = (
    {"ET": 0, "TC": 0, "WT": 0},
    {"ET": 20, "TC": 50, "WT": 100},
    {"ET": 50, "TC": 150, "WT": 300},
    {"ET": 100, "TC": 300, "WT": 600},
    {"ET": 200, "TC": 600, "WT": 1200},
    {"ET": 400, "TC": 1200, "WT": 2400},
)


def default_cache_dir(model: str = OURS) -> Path:
    if model_name(model) == ABHINAV_V3:
        return ROOT / "caches" / "val_probs_abhinav_v3"
    return ROOT / "caches" / "val_probs"


def default_report_path(model: str = OURS) -> Path:
    if model_name(model) == ABHINAV_V3:
        return ROOT / "reports" / "postprocess_ablation_abhinav_v3.csv"
    return ROOT / "reports" / "postprocess_ablation.csv"


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


def _check_cache_model(cache_dir: Path, model: str) -> None:
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return
    cached = json.loads(manifest.read_text(encoding="utf-8")).get("model", OURS)
    if cached != model:
        raise ValueError(f"{cache_dir} was cached with model {cached!r}, not {model!r}")


def load_cached_cases(cache_dir: Path, model: str = OURS) -> list[CachedCase]:
    model = model_name(model)
    files = sorted(path for path in cache_dir.glob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No cached probabilities in {cache_dir}. Run scripts/cache_probs.py first."
        )
    _check_cache_model(cache_dir, model)
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
            cached_model = str(data["model"]) if "model" in data.files else None
        if cached_model is not None and cached_model != model:
            raise ValueError(f"{path.name} was cached with model {cached_model!r}, not {model!r}")
        probs = reorder_probabilities(probs, order)
        probs, label, brain, spacing = _crop(probs, label, brain, spacing)
        cases.append(CachedCase(case_id, probs, label, brain, spacing))
    check_reportable([case.case_id for case in cases], model)
    heldout = evaluation_case_ids(model)
    if heldout is not None:
        missing = sorted(set(heldout) - {case.case_id for case in cases})
        if missing:
            print(f"{len(missing)} of {len(heldout)} held-out cases are not cached yet", flush=True)
    return cases


def raw_region_masks(probs, brain, threshold=0.5) -> dict[str, np.ndarray]:
    """Threshold each head at ``threshold``. No hierarchy and no component filter."""
    return {
        name: (np.asarray(probs[channel_index(name)]) > threshold) & brain
        for name in _HEADS
    }


def _config(base: PostConfig, thresholds: dict, min_voxels: dict) -> PostConfig:
    return PostConfig(
        threshold_et=float(thresholds["ET"]),
        threshold_tc=float(thresholds["TC"]),
        threshold_wt=float(thresholds["WT"]),
        min_voxels_et=int(min_voxels["ET"]),
        min_voxels_tc=int(min_voxels["TC"]),
        min_voxels_wt=int(min_voxels["WT"]),
        et_min_voxels=base.et_min_voxels,
        brain_closing_iterations=base.brain_closing_iterations,
    )


def _min_voxels(config: PostConfig) -> dict:
    return {name: config.min_voxels(name) for name in _HEADS}


def _thresholds(config: PostConfig) -> dict:
    return {name: config.threshold(name) for name in _HEADS}


def search_configs(base: PostConfig) -> list[PostConfig]:
    """One shared threshold for every head, across the component-size grid."""
    configs = []
    for threshold in THRESHOLD_VALUES:
        for min_voxels in MIN_COMPONENT_GRID:
            configs.append(_config(base, {name: threshold for name in _HEADS}, min_voxels))
    return configs


def threshold_grid(base: PostConfig, min_voxels: dict) -> list[PostConfig]:
    """Every ET/TC/WT threshold combination at one component-size setting."""
    configs = []
    for values in itertools.product(THRESHOLD_VALUES, repeat=len(_HEADS)):
        configs.append(_config(base, dict(zip(_HEADS, values)), min_voxels))
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


_WORKER_CASES: list[CachedCase] | None = None


def _init_worker(cases: list[CachedCase]) -> None:
    global _WORKER_CASES
    _WORKER_CASES = cases


def _evaluate_in_worker(config: PostConfig) -> dict:
    return evaluate_config(_WORKER_CASES, config)


def evaluate_configs(cases: list[CachedCase], configs: list[PostConfig], workers: int = 1):
    """Yield ``evaluate_config`` results in ``configs`` order.

    With ``workers > 1`` the cases are shared with child processes: forked
    where the platform allows it, otherwise pickled once per worker.
    """
    if workers <= 1 or len(configs) <= 1:
        for config in configs:
            yield evaluate_config(cases, config)
        return
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else None)
    with context.Pool(min(workers, len(configs)), initializer=_init_worker, initargs=(cases,)) as pool:
        yield from pool.imap(_evaluate_in_worker, configs)


def grid_edges(config: PostConfig) -> list[str]:
    """Parameters of ``config`` that sit on the largest value searched."""
    edges = [
        f"{name} threshold {config.threshold(name):.2f}"
        for name in _HEADS
        if config.threshold(name) >= max(THRESHOLD_VALUES)
    ]
    if _min_voxels(config) == MIN_COMPONENT_GRID[-1]:
        edges.append("component sizes " + "/".join(str(config.min_voxels(name)) for name in _HEADS))
    return edges


def evaluate_raw(cases: list[CachedCase], threshold=0.5) -> dict:
    def masks_for_case(case: CachedCase):
        return raw_region_masks(np.asarray(case.probs, dtype=np.float32), case.brain, threshold)

    scores = _overlap_cases(cases, masks_for_case)
    mean_dice, mean_precision = _mean_dice_precision(scores)
    return {"mean_dice": mean_dice, "mean_precision": mean_precision, "scores": scores}


def full_summary(cases: list[CachedCase], prediction_for_case, model: str = OURS) -> dict:
    scored = []
    for case in cases:
        prediction = prediction_for_case(case)
        scored.append(score_case(prediction, case.label, case.spacing, case_id=case.case_id, model=model))
    return aggregate_scores(scored, model=model)


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
        ("raw", raw, {name: 0.5 for name in _HEADS}, {name: "" for name in _HEADS}, ""),
        ("post", post, _thresholds(config), _min_voxels(config), config.et_min_voxels),
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
                    "threshold_et": thresholds["ET"],
                    "threshold_tc": thresholds["TC"],
                    "threshold_wt": thresholds["WT"],
                    "min_voxels_et": mins["ET"],
                    "min_voxels_tc": mins["TC"],
                    "min_voxels_wt": mins["WT"],
                    "et_min_voxels": et_min,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    cache_dir: Path,
    config_path: Path,
    report_path: Path,
    margin: float = 0.01,
    model: str = OURS,
    workers: int = 1,
) -> PostConfig | None:
    model = model_name(model)
    cases = load_cached_cases(cache_dir, model)
    base = PostConfig.from_yaml(config_path) if Path(config_path).is_file() else PostConfig()
    print(f"model={model} cases={len(cases)} margin={margin} workers={workers}", flush=True)
    raw_fast = evaluate_raw(cases)
    print(
        f"raw 0.5  mean_dice={raw_fast['mean_dice']:.4f}  mean_precision={raw_fast['mean_precision']:.4f}",
        flush=True,
    )
    scored = []
    first_pass = search_configs(base)
    for index, result in enumerate(evaluate_configs(cases, first_pass, workers), start=1):
        config = result["config"]
        scored.append(result)
        print(
            f"[{index}/{len(first_pass)}] thr=({config.threshold_et:.2f},{config.threshold_tc:.2f},{config.threshold_wt:.2f}) "
            f"min=({config.min_voxels_et},{config.min_voxels_tc},{config.min_voxels_wt}) "
            f"dice={result['mean_dice']:.4f} prec={result['mean_precision']:.4f}",
            flush=True,
        )
    winner = choose_operating_point(scored, raw_fast["mean_dice"], margin)
    if winner is None:
        size_for_grid = MIN_COMPONENT_GRID[0]
    else:
        size_for_grid = _min_voxels(winner["config"])
    already = {
        (item["config"].threshold_et, item["config"].threshold_tc, item["config"].threshold_wt,
         item["config"].min_voxels_et, item["config"].min_voxels_tc, item["config"].min_voxels_wt)
        for item in scored
    }
    second_pass = [
        config
        for config in threshold_grid(base, size_for_grid)
        if (
            config.threshold_et, config.threshold_tc, config.threshold_wt,
            config.min_voxels_et, config.min_voxels_tc, config.min_voxels_wt,
        ) not in already
    ]
    print(f"threshold grid: {len(second_pass)} settings", flush=True)
    for result in evaluate_configs(cases, second_pass, workers):
        config = result["config"]
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
            f"{config_path} was left unchanged.",
            flush=True,
        )
        return None

    chosen = winner["config"]
    raw_summary = full_summary(
        cases,
        lambda case: raw_region_masks(np.asarray(case.probs, dtype=np.float32), case.brain, 0.5),
        model,
    )
    post_summary = full_summary(
        cases,
        lambda case: postprocess(np.asarray(case.probs, dtype=np.float32), case.brain, chosen),
        model,
    )
    chosen.write_yaml(config_path)
    _write_report(report_path, raw_summary, post_summary, chosen)
    print(
        f"winner thr=({chosen.threshold_et:.2f},{chosen.threshold_tc:.2f},{chosen.threshold_wt:.2f}) "
        f"min=({chosen.min_voxels_et},{chosen.min_voxels_tc},{chosen.min_voxels_wt}) "
        f"dice={post_summary['mean_dice']:.4f} prec={post_summary['mean_precision']:.4f}",
        flush=True,
    )
    edges = grid_edges(chosen)
    if edges:
        print(f"note: winner is at the edge of the grid ({', '.join(edges)}); a wider grid may do better", flush=True)
    print(f"wrote {config_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BraTS post-processing on cached validation probabilities")
    parser.add_argument("--model", choices=MODEL_NAMES, default=OURS)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Default: caches/val_probs or caches/val_probs_abhinav_v3")
    parser.add_argument("--config", type=Path, default=None, help="Default: configs/postprocess_<model>.yaml")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes for the grid. Each holds the cropped cache in memory on Windows")
    args = parser.parse_args()
    run(
        args.cache_dir or default_cache_dir(args.model),
        args.config or config_path_for(args.model),
        args.report or default_report_path(args.model),
        args.margin,
        args.model,
        args.workers,
    )


if __name__ == "__main__":
    main()
