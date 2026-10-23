"""Dice each SegResNet output channel against GT TC, WT, and ET.

Runs three training cases with the training preprocessing and
``predict_probs`` (no post-processing). Prints the full channel/region
matrix and the permutation that matches.

Example:
    python scripts/verify_channels.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.infer import ROI_SIZE, default_device, load_model, predict_probs  # noqa: E402

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


def _training_transform() -> Compose:
    """Validation path from training, full volume (no crop): RAS, 1 mm, z-score."""
    keys = ["image", "label"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            MapLabelValued(
                keys=["label"],
                orig_labels=[0, 1, 2, 4],
                target_labels=[0, 1, 2, 3],
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys),
        ]
    )


def _case_dirs(root: Path, n: int) -> list[Path]:
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
        filename = _FILE_FOR_MODALITY.get(name)
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
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = default_device()
    modalities, trained_order = _checkpoint_info(args.checkpoint)
    print(f"device={device} roi={ROI_SIZE} modalities={modalities}", flush=True)
    print(f"checkpoint out_regions={trained_order}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device)
    transform = _training_transform()

    per_case = []
    for case_dir in _case_dirs(args.data_dir, args.cases):
        print(f"running {case_dir.name}", flush=True)
        image, label = _load_case(case_dir, modalities, transform)
        probs = predict_probs(model, image, ROI_SIZE, tta=True)
        if probs.shape[1:] != label.shape:
            raise RuntimeError(f"{case_dir.name}: probs {probs.shape} vs label {label.shape}")
        gt = _regions(label)
        matrix = np.zeros((probs.shape[0], len(REGIONS)), dtype=np.float64)
        print(f"  labels={np.unique(label).tolist()} prob_shape={tuple(probs.shape)}")
        for c in range(probs.shape[0]):
            pred = probs[c] > args.threshold
            scores = [_dice(pred, gt[name]) for name in REGIONS]
            matrix[c] = scores
            formatted = "  ".join(f"{name}={score:.3f}" for name, score in zip(REGIONS, scores))
            print(f"  ch{c}  {formatted}", flush=True)
        per_case.append(matrix)

    mean_dice = np.mean(per_case, axis=0)
    print("mean Dice (rows=output channels, cols=TC WT ET)", flush=True)
    for c, row in enumerate(mean_dice):
        formatted = "  ".join(f"{name}={score:.3f}" for name, score in zip(REGIONS, row))
        print(f"  ch{c}  {formatted}", flush=True)

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
