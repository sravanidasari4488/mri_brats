"""Region metrics, empty-ET Dice, and the post-process selection rule."""

import numpy as np

from recon.evaluate import (
    choose_operating_point,
    hd95_mm,
    lesion_false_positives,
    region_overlap,
    regions_from_label,
    score_case,
    aggregate_scores,
)


def test_empty_et_dice_and_false_positive_count():
    both_empty = np.zeros((4, 4, 4), dtype=np.uint8)
    pred_only = both_empty.copy()
    pred_only[1, 1, 1] = 3
    dice_empty, _, _ = region_overlap(np.zeros((4, 4, 4), dtype=bool), np.zeros((4, 4, 4), dtype=bool))
    assert dice_empty == 1.0
    scored = score_case(pred_only, both_empty, spacing=(1, 1, 1))
    assert scored["ET"]["dice"] == 0.0
    assert scored["gt_empty_et_but_pred"]
    quiet = score_case(both_empty, both_empty, spacing=(1, 1, 1))
    assert quiet["ET"]["dice"] == 1.0
    assert not quiet["gt_empty_et_but_pred"]
    summary = aggregate_scores([scored, quiet])
    assert summary["n_gt_empty_et_but_pred"] == 1
    assert summary["ET"]["dice"] == 0.5


def test_overlap_precision_recall_and_label4():
    pred = np.zeros((4, 4, 1), dtype=bool)
    gt = np.zeros((4, 4, 1), dtype=bool)
    pred[:, :2] = True
    gt[:, 1:3] = True
    dice, precision, recall = region_overlap(pred, gt)
    assert abs(dice - 0.5) < 1e-6
    assert abs(precision - 0.5) < 1e-6
    assert abs(recall - 0.5) < 1e-6
    label = np.zeros((3, 3, 3), dtype=np.uint8)
    label[0, 0, 0] = 4
    label[1, 1, 1] = 1
    label[2, 2, 2] = 2
    masks = regions_from_label(label)
    assert masks["ET"][0, 0, 0] and masks["TC"][0, 0, 0] and masks["WT"][0, 0, 0]
    assert masks["TC"][1, 1, 1] and not masks["ET"][1, 1, 1]
    assert masks["WT"][2, 2, 2] and not masks["TC"][2, 2, 2]


def test_lesion_false_positives_ignore_overlapping_component():
    pred = np.zeros((8, 8, 8), dtype=bool)
    gt = np.zeros((8, 8, 8), dtype=bool)
    pred[0:2, 0:2, 0:2] = True
    gt[0:2, 0:2, 0:2] = True
    pred[5:7, 5:7, 5:7] = True
    assert lesion_false_positives(pred, gt) == 1
    assert lesion_false_positives(pred, np.zeros_like(pred)) == 2
    assert lesion_false_positives(np.zeros_like(pred), gt) == 0


def test_hd95_uses_spacing():
    pred = np.zeros((6, 3, 3), dtype=bool)
    gt = np.zeros((6, 3, 3), dtype=bool)
    pred[0, 0, 0] = True
    gt[4, 0, 0] = True
    assert abs(hd95_mm(pred, gt, spacing=(1, 1, 1)) - 4.0) < 1e-6
    assert abs(hd95_mm(pred, gt, spacing=(2, 1, 1)) - 8.0) < 1e-6
    assert hd95_mm(pred, pred, spacing=(1.5, 2, 3)) == 0.0
    assert np.isnan(hd95_mm(pred, np.zeros_like(pred), spacing=(1, 1, 1)))
    # A voxel inside a solid region is not on that region's surface.
    interior = np.zeros((7, 7, 7), dtype=bool)
    interior[3, 3, 3] = True
    solid = np.ones((7, 7, 7), dtype=bool)
    assert hd95_mm(interior, solid, spacing=(1, 1, 1)) > 0.0


def test_choose_operating_point_keeps_dice_within_margin():
    raw = 0.80
    candidates = [
        {"name": "aggressive", "mean_dice": 0.70, "mean_precision": 0.99},
        {"name": "ok", "mean_dice": 0.795, "mean_precision": 0.90},
        {"name": "safer", "mean_dice": 0.80, "mean_precision": 0.85},
    ]
    chosen = choose_operating_point(candidates, raw, margin=0.01)
    assert chosen["name"] == "ok"
    assert choose_operating_point(candidates[:1], raw, margin=0.01) is None


if __name__ == "__main__":
    test_empty_et_dice_and_false_positive_count()
    test_overlap_precision_recall_and_label4()
    test_lesion_false_positives_ignore_overlapping_component()
    test_hd95_uses_spacing()
    test_choose_operating_point_keeps_dice_within_margin()
    print("evaluate ok")
