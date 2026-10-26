"""Synthetic cache: the tuner must raise precision without giving up Dice."""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.postprocess import PostConfig


def _tune():
    path = ROOT / "scripts" / "tune_postprocess.py"
    spec = importlib.util.spec_from_file_location("tune_postprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path, label, probs):
    np.savez_compressed(
        path,
        probs=probs.astype(np.float16),
        label=label.astype(np.uint8),
        brain=np.ones(label.shape, dtype=np.uint8),
        spacing=np.array([1.0, 1.0, 2.0]),
        channel_order=np.array(["ET", "TC", "WT"]),
        case_id=np.array(path.stem),
    )


def _build_cache(cache: Path):
    shape = (40, 40, 20)
    # Slab at probability 0.55 (3200 voxels) is larger than every size cutoff,
    # so only a threshold above 0.5 can remove it. The speck is 10 voxels at
    # 0.95, so only a WT minimum above 10 can remove it.
    slab = np.zeros(shape, dtype=bool)
    slab[:, :, 0:2] = True
    speck = np.zeros(shape, dtype=bool)
    speck[0:2, 0:5, 19] = True
    wt = np.zeros(shape, dtype=bool)
    wt[2:16, 2:16, 4:18] = True
    tc = np.zeros(shape, dtype=bool)
    tc[4:14, 4:14, 6:16] = True
    et = np.zeros(shape, dtype=bool)
    et[5:11, 5:11, 7:13] = True

    def probs_for(et_mask, tc_mask, wt_mask, extra_et=None):
        probs = np.zeros((3,) + shape, dtype=np.float32)
        for index, mask in enumerate((et_mask, tc_mask, wt_mask)):
            probs[index][mask] = 0.95
            probs[index][slab] = 0.55
        probs[2][speck] = 0.95
        if extra_et is not None:
            probs[0][extra_et] = 0.55
        return probs

    label = np.zeros(shape, dtype=np.uint8)
    label[wt] = 2
    label[tc] = 1
    label[et] = 3
    _save(cache / "tumor.npz", label, probs_for(et, tc, wt))

    false_et = np.zeros(shape, dtype=bool)
    false_et[5:8, 5:10, 8:10] = True
    label_no_et = np.zeros(shape, dtype=np.uint8)
    label_no_et[wt] = 2
    label_no_et[tc] = 1
    _save(cache / "no_et.npz", label_no_et, probs_for(np.zeros(shape, dtype=bool), tc, wt, false_et))


def test_grid_search_writes_yaml_and_ablation(tmp_path=None):
    tune = _tune()
    root = Path(tmp_path) if tmp_path is not None else Path(__import__("tempfile").mkdtemp())
    cache = root / "cache"
    cache.mkdir(parents=True)
    config_path = root / "postprocess.yaml"
    report_path = root / "postprocess_ablation.csv"
    _build_cache(cache)
    PostConfig(
        threshold_et=0.5,
        threshold_tc=0.5,
        threshold_wt=0.5,
        min_voxels_et=50,
        min_voxels_tc=150,
        min_voxels_wt=300,
        et_min_voxels=0,
        brain_closing_iterations=1,
    ).write_yaml(config_path)

    try:
        tune.load_cached_cases(root / "missing")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass

    chosen = tune.run(cache, config_path, report_path, margin=0.01)
    assert chosen is not None
    loaded = PostConfig.from_yaml(config_path)
    assert (loaded.threshold_et, loaded.threshold_tc, loaded.threshold_wt) == (0.6, 0.6, 0.6)
    assert (loaded.min_voxels_et, loaded.min_voxels_tc, loaded.min_voxels_wt) == (20, 50, 100)
    assert loaded.et_min_voxels == 0

    with report_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    variants = {row["variant"] for row in rows}
    regions = {row["region"] for row in rows}
    assert variants == {"raw", "post"}
    assert regions == {"WT", "TC", "ET", "mean"}
    raw = next(row for row in rows if row["variant"] == "raw" and row["region"] == "mean")
    post = next(row for row in rows if row["variant"] == "post" and row["region"] == "mean")
    assert float(post["precision"]) > float(raw["precision"])
    assert float(post["dice"]) >= float(raw["dice"]) - 0.01
    assert int(raw["n_gt_empty_et_but_pred"]) == 1
    assert int(post["n_gt_empty_et_but_pred"]) == 0
    assert post["hd95_mm"] != ""


def test_parallel_grid_matches_serial_and_edges_are_reported(tmp_path=None):
    from scripts import tune_postprocess as tune

    root = Path(tmp_path) if tmp_path is not None else Path(__import__("tempfile").mkdtemp())
    _build_cache(root)
    cases = tune.load_cached_cases(root)
    configs = tune.threshold_grid(PostConfig(), tune.MIN_COMPONENT_GRID[1])[:6]
    serial = [(r["mean_dice"], r["mean_precision"]) for r in tune.evaluate_configs(cases, configs, 1)]
    parallel = list(tune.evaluate_configs(cases, configs, 2))
    assert [(r["mean_dice"], r["mean_precision"]) for r in parallel] == serial
    assert [r["config"] for r in parallel] == configs

    top = max(tune.THRESHOLD_VALUES)
    inside = tune._config(PostConfig(), {"ET": 0.7, "TC": 0.7, "WT": 0.7}, tune.MIN_COMPONENT_GRID[3])
    assert tune.grid_edges(inside) == []
    edge = tune._config(PostConfig(), {"ET": top, "TC": 0.7, "WT": 0.7}, tune.MIN_COMPONENT_GRID[-1])
    edges = tune.grid_edges(edge)
    assert edges[0] == f"ET threshold {top:.2f}"
    assert edges[1].startswith("component sizes")


if __name__ == "__main__":
    test_grid_search_writes_yaml_and_ablation()
    test_parallel_grid_matches_serial_and_edges_are_reported()
    print("tune ok")
