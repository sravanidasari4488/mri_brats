"""Cache TTA probability maps for a held-out validation split.

Uses the training split: sorted complete case folders, ``random.Random(42)``,
``val_ratio=0.2``. The first cases after the shuffle are validation and were
not in the training half. This script writes about 40 of those validation
cases so threshold tuning can reload probabilities instead of running the
network again.

Each ``.npz`` holds ``probs`` as float16 in ``CHANNEL_ORDER`` (ET, TC, WT)
plus the label on the same RAS 1 mm grid.

Re-running the same command resumes. Each case is written into ``--out-dir``
as soon as it finishes, so a Colab runtime that disconnects keeps every
completed file on Drive. A case whose ``.npz`` already opens is skipped.
A truncated file is computed again.

Example:
    python scripts/cache_probs.py
    python scripts/cache_probs.py --dry-run
    python scripts/cache_probs.py --out-dir "/content/drive/MyDrive/brats/val_probs"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
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

from recon.infer import CHANNEL_ORDER, ROI_SIZE, default_device, load_model, predict_probs  # noqa: E402
from recon.postprocess import brain_mask_from_image  # noqa: E402

_DEFAULT_DATA = Path(r"C:\Users\Sravani\Desktop\MRI project\data\processed\brats_nifti")
_DEFAULT_CKPT = Path(
    r"C:\Users\Sravani\Desktop\MRI project\models\checkpoints\brats_pretrain_smoke\best_model.pt"
)
_DEFAULT_OUT = ROOT / "caches" / "val_probs"
_REQUIRED = ("flair.nii.gz", "t1.nii.gz", "t1c.nii.gz", "t2.nii.gz", "mask.nii.gz")
_MODALITIES = ("flair", "t1", "t1c", "t2")


def _training_transform() -> Compose:
    """Full-volume validation transform: RAS, 1 mm, label remap, nonzero z-score."""
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


def complete_cases(root: Path) -> list[Path]:
    """Sorted case folders that have the four training modalities and a mask."""
    cases = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((path / name).is_file() for name in _REQUIRED):
            cases.append(path)
    return cases


def split_train_val(cases: list[Path], *, val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Same shuffle as ``train_brats`` / ``split_train_val`` (seed 42, 20% val)."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    items = list(cases)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = int(round(len(items) * val_ratio)) if items else 0
    if items and n_val == 0:
        n_val = 1
    if n_val >= len(items) and len(items) > 1:
        n_val = len(items) - 1
    return items[n_val:], items[:n_val]


def held_out_cases(root: Path, n: int, *, val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Return ``n`` validation cases and the training cases they are disjoint from."""
    train, val = split_train_val(complete_cases(root), val_ratio=val_ratio, seed=seed)
    if len(val) < n:
        raise RuntimeError(f"Validation split has {len(val)} cases, need {n}")
    selected = val[:n]
    train_ids = {path.name for path in train}
    leaked = [path.name for path in selected if path.name in train_ids]
    if leaked:
        raise RuntimeError(f"Held-out cases overlap training: {leaked}")
    return selected, train


def _spacing_mm(image) -> np.ndarray:
    affine = getattr(image, "affine", None)
    if affine is None:
        return np.ones(3, dtype=np.float64)
    from nibabel.affines import voxel_sizes

    return np.asarray(voxel_sizes(np.asarray(affine, dtype=np.float64)), dtype=np.float64)


def _cache_exists(path: Path) -> bool:
    """True when ``path`` is a finished cache that can be reloaded."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return "probs" in data.files and "label" in data.files
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _write_npz(out_path: Path, **arrays) -> None:
    """Write one case into ``out_path``'s directory as soon as it is finished.

    The archive is built in memory, flushed to a sibling ``.partial`` file in
    the output directory, then renamed to the ``.npz``. The final name appears
    only after the bytes are on disk, so a killed Colab run cannot skip a
    half-written cache on the next start.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    partial = out_path.with_name(out_path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    try:
        partial.replace(out_path)
    except OSError:
        out_path.write_bytes(payload)
        partial.unlink(missing_ok=True)


def _load_case(case_dir: Path, transform: Compose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = transform(
        {
            "image": [str(case_dir / f"{name}.nii.gz") for name in _MODALITIES],
            "label": str(case_dir / "mask.nii.gz"),
        }
    )
    image_meta = batch["image"]
    spacing = _spacing_mm(image_meta)
    image = np.asarray(image_meta, dtype=np.float32)
    label = np.rint(np.squeeze(np.asarray(batch["label"]))).astype(np.uint8)
    label[label == 4] = 3
    return image, label, spacing


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache TTA probabilities for held-out BraTS validation cases")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--num-cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Print the validation case ids and exit")
    args = parser.parse_args()

    selected, train = held_out_cases(
        args.data_dir, args.num_cases, val_ratio=args.val_ratio, seed=args.seed
    )
    print(
        f"held-out {len(selected)} of validation split "
        f"(train={len(train)}, seed={args.seed}, val_ratio={args.val_ratio})",
        flush=True,
    )
    for path in selected:
        print(f"  {path.name}", flush=True)
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "n_train": len(train),
        "n_cached": len(selected),
        "case_ids": [path.name for path in selected],
        "channel_order": list(CHANNEL_ORDER),
        "checkpoint": str(args.checkpoint),
        "roi": list(ROI_SIZE),
        "tta": True,
        "probs_dtype": "float16",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = len(selected)
    pending: list[tuple[int, Path]] = []
    skipped = 0
    for index, case_dir in enumerate(selected, start=1):
        out_path = args.out_dir / f"{case_dir.name}.npz"
        if _cache_exists(out_path):
            skipped += 1
            print(f"[{index}/{total}] skip {case_dir.name}", flush=True)
            continue
        if out_path.is_file():
            print(f"[{index}/{total}] {case_dir.name} unreadable, recomputing", flush=True)
        pending.append((index, case_dir))
    print(f"cached {skipped}/{total}, remaining {len(pending)}", flush=True)
    if not pending:
        print(f"wrote {args.out_dir}", flush=True)
        return

    device = default_device()
    print(f"device={device} roi={ROI_SIZE}", flush=True)
    model = load_model(args.checkpoint, device)
    transform = _training_transform()
    written = 0
    for index, case_dir in pending:
        out_path = args.out_dir / f"{case_dir.name}.npz"
        print(f"[{index}/{total}] {case_dir.name}", flush=True)
        image, label, spacing = _load_case(case_dir, transform)
        probs = predict_probs(model, image, ROI_SIZE, tta=True)
        if probs.shape[1:] != label.shape:
            raise RuntimeError(f"{case_dir.name}: probs {probs.shape} vs label {label.shape}")
        brain = brain_mask_from_image(image).astype(np.uint8)
        _write_npz(
            out_path,
            probs=np.asarray(probs, dtype=np.float16),
            label=label,
            brain=brain,
            spacing=spacing,
            channel_order=np.array(CHANNEL_ORDER),
            case_id=np.array(case_dir.name),
        )
        written += 1
        print(f"[{index}/{total}] saved {out_path.name}", flush=True)
    print(f"wrote {written}, skipped {skipped}, dir {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
