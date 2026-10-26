"""Region scores and mesh-versus-voxel volume checks.

Per-region scores use WT = {1, 2, 3}, TC = {1, 3}, ET = {3} (legacy 4 counts as ET).
Dice is 1 when both the prediction and the ground truth are empty, and 0 when
only one of them is. HD95 is the 95th percentile surface distance in millimeters.

With ``model="abhinav_v3"`` every scored case needs a ``case_id``, and cases
from that checkpoint's training list raise ``TrainingCaseError``. The default
evaluation list is ``evaluation_case_ids(model)``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from recon.postprocess import CONN
from recon.splits import (
    OURS,
    TrainingCaseError,
    check_reportable,
    heldout_case_ids,
    model_name,
    training_case_ids,
)

REGION_NAMES = ("WT", "TC", "ET")


def evaluation_case_ids(model: str = OURS) -> list[str] | None:
    """Held-out case ids for ``model``. ``None`` means use the local validation split."""
    return heldout_case_ids(model)


def _guard(case_ids, model: str) -> None:
    if not training_case_ids(model):
        return
    ids = list(case_ids)
    if any(case_id is None for case_id in ids):
        raise TrainingCaseError(f"{model} metrics need a case_id for every case to rule out training cases")
    check_reportable(ids, model)


def check_mesh_volume(mesh, mask, affine, name=""):
    """Mesh volume should be within ~3% of voxel volume; larger gaps = over-smoothing or holes."""
    from recon.mesh import check_mesh

    watertight, err = check_mesh(mesh, mask, affine)
    vox_vol = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    v_vox = float(np.asarray(mask).sum()) * vox_vol
    v_mesh = abs(float(mesh.volume)) if watertight else float("nan")
    print(f"{name:10s} voxels={v_vox/1000:8.2f} cm3  mesh={v_mesh/1000:8.2f} cm3  err={err:+.1f}%  "
          f"watertight={watertight}")
    return err


def regions_from_label(label) -> dict[str, np.ndarray]:
    """Exclusive BraTS labels to WT / TC / ET masks. Label 4 is treated as ET."""
    lab = np.rint(np.squeeze(np.asarray(label))).astype(np.int16, copy=False)
    if lab.ndim != 3:
        raise ValueError(f"Expected a 3D label, got {lab.shape}")
    et = (lab == 3) | (lab == 4)
    tc = (lab == 1) | et
    wt = tc | (lab == 2)
    return {"WT": wt, "TC": tc, "ET": et}


def as_region_masks(value) -> dict[str, np.ndarray]:
    """Accept a label map or a dict that already has WT, TC, and ET masks."""
    if isinstance(value, dict):
        missing = [name for name in REGION_NAMES if name not in value]
        if missing:
            raise KeyError(f"Region masks missing {missing}")
        return {name: np.asarray(value[name], dtype=bool) for name in REGION_NAMES}
    return regions_from_label(value)


def region_overlap(pred, gt) -> tuple[float, float, float]:
    """Dice, precision, and recall.

    Both empty → 1, 1, 1. Only one empty → Dice 0; precision is 0 when the
    prediction is empty or has no true positives, and recall is 0 when the
    ground truth is empty or has no true positives.
    """
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    pred_sum = int(pred_b.sum())
    gt_sum = int(gt_b.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0, 1.0
    tp = int(np.logical_and(pred_b, gt_b).sum())
    dice = 0.0 if pred_sum == 0 or gt_sum == 0 else (2.0 * tp) / (pred_sum + gt_sum)
    precision = 0.0 if pred_sum == 0 else tp / pred_sum
    recall = 0.0 if gt_sum == 0 else tp / gt_sum
    return float(dice), float(precision), float(recall)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=CONN, border_value=0)
    surface = mask & ~eroded
    if not surface.any():
        return mask
    return surface


def hd95_mm(pred, gt, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th percentile symmetric surface distance in millimeters.

    Both empty → 0. Only one empty → NaN (the distance is undefined).
    ``spacing`` is the voxel size along each axis, in mm.
    """
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return float("nan")
    sampling = tuple(float(v) for v in np.asarray(spacing).reshape(-1)[:3])
    if len(sampling) != 3:
        raise ValueError(f"spacing must have 3 values, got {spacing}")
    pred_s = _surface(pred_b)
    gt_s = _surface(gt_b)
    dist_to_gt = ndimage.distance_transform_edt(~gt_s, sampling=sampling)
    dist_to_pred = ndimage.distance_transform_edt(~pred_s, sampling=sampling)
    distances = np.concatenate([dist_to_gt[pred_s], dist_to_pred[gt_s]])
    return float(np.percentile(distances, 95))


def lesion_false_positives(pred, gt) -> int:
    """Predicted 26-connected components that do not overlap the ground truth."""
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    if not pred_b.any():
        return 0
    labels, count = ndimage.label(pred_b, structure=CONN)
    if count == 0:
        return 0
    if not gt_b.any():
        return int(count)
    hit = np.unique(labels[gt_b])
    hit = hit[hit > 0]
    return int(count - hit.size)


def score_case(prediction, ground_truth, spacing=(1.0, 1.0, 1.0), *, case_id=None, model: str = OURS) -> dict:
    """Score one case. ``prediction`` / ``ground_truth`` are labels or region dicts."""
    model = model_name(model)
    _guard([case_id], model)
    pred = as_region_masks(prediction)
    gt = as_region_masks(ground_truth)
    metrics = {}
    for region in REGION_NAMES:
        dice, precision, recall = region_overlap(pred[region], gt[region])
        metrics[region] = {
            "dice": dice,
            "hd95_mm": hd95_mm(pred[region], gt[region], spacing),
            "precision": precision,
            "recall": recall,
            "lesion_fp": lesion_false_positives(pred[region], gt[region]),
        }
    metrics["gt_empty_et_but_pred"] = (not gt["ET"].any()) and bool(pred["ET"].any())
    if case_id is not None:
        metrics["case_id"] = str(case_id)
    return metrics


def aggregate_scores(cases: list[dict], *, model: str = OURS) -> dict:
    """Mean Dice / HD95 / precision / recall, summed lesion false positives.

    HD95 ignores cases where it is undefined (only one side empty).
    ``n_gt_empty_et_but_pred`` counts cases with no enhancing tumor in the
    ground truth and at least one predicted enhancing voxel.
    """
    if not cases:
        raise ValueError("No cases to aggregate")
    _guard([case.get("case_id") for case in cases], model_name(model))
    summary = {}
    for region in REGION_NAMES:
        hd = np.array([case[region]["hd95_mm"] for case in cases], dtype=np.float64)
        summary[region] = {
            "dice": float(np.mean([case[region]["dice"] for case in cases])),
            "hd95_mm": float(np.nanmean(hd)) if np.isfinite(hd).any() else float("nan"),
            "precision": float(np.mean([case[region]["precision"] for case in cases])),
            "recall": float(np.mean([case[region]["recall"] for case in cases])),
            "lesion_fp": int(sum(case[region]["lesion_fp"] for case in cases)),
        }
    summary["n_gt_empty_et_but_pred"] = int(sum(bool(case["gt_empty_et_but_pred"]) for case in cases))
    summary["n_cases"] = len(cases)
    summary["mean_dice"] = float(np.mean([summary[region]["dice"] for region in REGION_NAMES]))
    summary["mean_precision"] = float(np.mean([summary[region]["precision"] for region in REGION_NAMES]))
    return summary


def choose_operating_point(candidates: list[dict], raw_mean_dice: float, margin: float = 0.01):
    """Max mean precision among candidates whose mean Dice is within ``margin`` of raw.

    Each candidate needs ``mean_dice`` and ``mean_precision``. Returns None when
    every candidate drops Dice by more than ``margin``.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    limit = float(raw_mean_dice) - float(margin)
    feasible = [item for item in candidates if item["mean_dice"] >= limit - 1e-9]
    if not feasible:
        return None
    return max(feasible, key=lambda item: (item["mean_precision"], item["mean_dice"]))
