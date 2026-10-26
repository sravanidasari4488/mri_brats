"""Dice each SegResNet output channel against GT TC, WT, and ET.

Runs three cases with the model's validation preprocessing and
``predict_probs`` (no post-processing). Prints the full channel/region
matrix and the permutation that matches. ``ours`` takes the first sorted
cases; ``abhinav_v3`` takes the first held-out cases.

Example:
    python scripts/verify_channels.py
    python scripts/verify_channels.py --model abhinav_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.infer import (  # noqa: E402
    ABHINAV_V3,
    CHANNEL_ORDER,
    OURS,
    default_device,
    load_model,
    model_name,
    output_order,
    predict_probs,
    resolve_checkpoint,
    roi_for,
    validation_transform,
)
from recon.splits import check_reportable, heldout_case_ids  # noqa: E402

REGIONS = ("TC", "WT", "ET")
# On-disk names for the modalities recorded in the training checkpoint.
_FILE_FOR_MODALITY = {
    "flair": "flair.nii.gz",
    "t2f": "flair.nii.gz",
    "t1": "t1.nii.gz",
    "t1n": "t1.nii.gz",
    "t1c": "t1c.nii.gz",
    "t1ce": "t1c.nii.gz",
    "t2": "t2.nii.gz",
    "t2w": "t2.nii.gz",
}
_DEFAULT_DATA = Path(r"C:\Users\Sravani\Desktop\MRI project\data\processed\brats_nifti")
_DEFAULT_CKPT = Path(
    r"C:\Users\Sravani\Desktop\MRI project\models\checkpoints\brats_pretrain_smoke\best_model.pt"
)


def _training_transform(name: str = OURS) -> Compose:
    """Validation transform for ``name``. Ours is RAS + 1 mm; abhinav_v3 is theirs."""
    return validation_transform(name)


def _case_dirs(root: Path, n: int, model: str = OURS) -> list[Path]:
    """First ``n`` cases. ``abhinav_v3`` takes them from ``splits/abhinav_heldout.txt``."""
    listed = heldout_case_ids(model)
    if listed is not None:
        check_reportable(listed[:n], model)
        cases = [root / case_id for case_id in listed[:n] if (root / case_id / "mask.nii.gz").is_file()]
        if len(cases) < n:
            raise FileNotFoundError(f"Needed {n} held-out cases under {root}, found {len(cases)}")
        return cases
    cases = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if (path / "mask.nii.gz").is_file():
            cases.append(path)
        if len(cases) == n:
            break
    if len(cases) < n:
        raise FileNotFoundError(f"Needed {n} training cases under {root}, found {len(cases)}")
    return cases


def _checkpoint_info(ckpt_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = blob.get("model_info") or {}
    modalities = tuple(info.get("in_modalities") or ("flair", "t1", "t1c", "t2"))
    regions = tuple(str(name).upper() for name in (info.get("out_regions") or ("ET", "TC", "WT")))
    return modalities, regions


def _load_case(case_dir: Path, modalities: tuple[str, ...], transform: Compose) -> tuple[np.ndarray, np.ndarray]:
    image_paths = []
    for name in modalities:
        filename = _FILE_FOR_MODALITY.get(str(name).lower())
        if filename is None:
            raise KeyError(f"No filename for training modality {name!r}")
        path = case_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        image_paths.append(str(path))
    batch = transform({"image": image_paths, "label": str(case_dir / "mask.nii.gz")})
    image = np.asarray(batch["image"], dtype=np.float32)
    label = np.asarray(batch["label"])
    label = np.squeeze(label)
    label = np.rint(label).astype(np.uint8)
    label[label == 4] = 3
    return image, label


def _regions(label: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "TC": (label == 1) | (label == 3),
        "WT": (label == 1) | (label == 2) | (label == 3),
        "ET": label == 3,
    }


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = np.logical_and(pred_b, gt_b).sum(dtype=np.float64)
    denom = pred_b.sum(dtype=np.float64) + gt_b.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def _ranked_mappings(mean_dice: np.ndarray) -> list[tuple[tuple[str, ...], float]]:
    import itertools

    scored = []
    for perm in itertools.permutations(REGIONS):
        score = float(sum(mean_dice[i, REGIONS.index(name)] for i, name in enumerate(perm)))
        scored.append((perm, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SegResNet output channel order against BraTS regions")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", choices=(OURS, ABHINAV_V3), default=OURS)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    kind = model_name(args.model)
    if args.checkpoint is None:
        args.checkpoint = resolve_checkpoint(kind) if kind == ABHINAV_V3 else _DEFAULT_CKPT

    device = default_device()
    modalities, trained_order = _checkpoint_info(args.checkpoint)
    if kind == ABHINAV_V3:
        modalities = ("FLAIR", "T1", "T1c", "T2")
        trained_order = output_order(kind)
    roi = roi_for(kind)
    print(f"model={kind} device={device} roi={roi} modalities={modalities}", flush=True)
    print(f"source heads={output_order(kind)} pipeline={CHANNEL_ORDER}", flush=True)
    print(f"checkpoint out_regions={trained_order}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device, kind)
    transform = _training_transform(kind)

    per_case = []
    for case_dir in _case_dirs(args.data_dir, args.cases, kind):
        print(f"running {case_dir.name}", flush=True)
        image, label = _load_case(case_dir, modalities, transform)
        probs = predict_probs(model, image, roi, tta=True, model_name=kind)
        if probs.shape[1:] != label.shape:
            raise RuntimeError(f"{case_dir.name}: probs {probs.shape} vs label {label.shape}")
        gt = _regions(label)
        matrix = np.zeros((probs.shape[0], len(REGIONS)), dtype=np.float64)
        print(f"  labels={np.unique(label).tolist()} prob_shape={tuple(probs.shape)}")
        for c, channel in enumerate(CHANNEL_ORDER):
            pred = probs[c] > args.threshold
            scores = [_dice(pred, gt[region]) for region in REGIONS]
            matrix[c] = scores
            formatted = "  ".join(f"{region}={score:.3f}" for region, score in zip(REGIONS, scores))
            own = scores[REGIONS.index(channel)]
            print(f"  {channel}  {formatted}  (own={own:.3f})", flush=True)
        per_case.append(matrix)

    mean_dice = np.mean(per_case, axis=0)
    print("mean Dice (rows=CHANNEL_ORDER ET TC WT, cols=TC WT ET)", flush=True)
    for c, (channel, row) in enumerate(zip(CHANNEL_ORDER, mean_dice)):
        formatted = "  ".join(f"{name}={score:.3f}" for name, score in zip(REGIONS, row))
        own = row[REGIONS.index(channel)]
        print(f"  {channel}  {formatted}  (own={own:.3f})", flush=True)

    ranked = _ranked_mappings(mean_dice)
    best, best_score = ranked[0]
    second, second_score = ranked[1]
    print(
        f"best Dice permutation: [{', '.join(best)}] sum={best_score:.3f}; "
        f"next [{', '.join(second)}] sum={second_score:.3f}",
        flush=True,
    )
    assumed = ("TC", "WT", "ET")
    # A 1-epoch checkpoint overlaps WT on every head, so permutation sums sit
    # within a few thousandths. Prefer the order the loss was trained with.
    if best_score - second_score < 0.02 and trained_order in {perm for perm, _ in ranked}:
        mapping = trained_order
        print(
            f"Dice does not separate the heads. Mapping from training: [{', '.join(mapping)}]",
            flush=True,
        )
    else:
        mapping = best
        print(f"correct mapping: [{', '.join(mapping)}]", flush=True)
    if mapping == assumed:
        print("mapping matches [TC, WT, ET]", flush=True)
    else:
        print(f"mapping is not [TC, WT, ET]; CHANNEL_ORDER should be {mapping}", flush=True)


if __name__ == "__main__":
    main()
