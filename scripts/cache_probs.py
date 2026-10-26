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

The default pass is the faster one: four views and overlap 0.5. CUDA also
runs that pass in float16. ``--full-tta`` restores eight views and overlap
0.625.

``--model abhinav_v3`` caches every case in ``splits/abhinav_heldout.txt``
into ``caches/val_probs_abhinav_v3`` instead of the seed-42 split, and
refuses any case from that checkpoint's training list. An output directory
already cached with a different model or checkpoint is refused.

Example:
    python scripts/cache_probs.py
    python scripts/cache_probs.py --dry-run
    python scripts/cache_probs.py --model abhinav_v3
    python scripts/cache_probs.py --out-dir "/content/drive/MyDrive/brats/val_probs"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
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
    predict_probs,
    resolve_checkpoint,
    roi_for,
    validation_transform,
)
from recon.splits import check_reportable, heldout_case_ids, read_case_list  # noqa: E402
from recon.postprocess import brain_mask_from_image  # noqa: E402

_DEFAULT_DATA = Path(r"C:\Users\Sravani\Desktop\MRI project\data\processed\brats_nifti")
_DEFAULT_CKPT = Path(
    r"C:\Users\Sravani\Desktop\MRI project\models\checkpoints\brats_pretrain_smoke\best_model.pt"
)
_DEFAULT_OUT = ROOT / "caches" / "val_probs"
_REQUIRED = ("flair.nii.gz", "t1.nii.gz", "t1c.nii.gz", "t2.nii.gz", "mask.nii.gz")
_MODALITIES = ("flair", "t1", "t1c", "t2")


def _training_transform(name: str = OURS) -> Compose:
    """Full-volume validation transform for ``name``."""
    return validation_transform(name)


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


def listed_cases(root: Path, case_ids, model: str, n: int | None = None) -> list[Path]:
    """Case folders for ``case_ids`` under ``root``. Training cases for ``model`` are refused."""
    ids = list(case_ids)
    check_reportable(ids, model)
    complete = {path.name: path for path in complete_cases(root)}
    missing = [case_id for case_id in ids if case_id not in complete]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed cases are missing or incomplete under {root}: {missing[:5]}")
    selected = [complete[case_id] for case_id in ids]
    return selected if n is None else selected[:n]


def select_cases(
    root: Path,
    model: str,
    *,
    num_cases: int | None = None,
    case_list: Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[str]]:
    """Cases to cache or score, and a description of where they came from.

    ``abhinav_v3`` defaults to ``splits/abhinav_heldout.txt``. ``ours``
    defaults to the first 40 of the seed-42 validation split.
    """
    model = model_name(model)
    ids = read_case_list(case_list) if case_list is not None else heldout_case_ids(model)
    if ids is not None:
        source = str(case_list) if case_list is not None else f"splits/abhinav_heldout.txt ({model})"
        return listed_cases(root, ids, model, num_cases), [f"list={source}", f"n_listed={len(ids)}"]
    n = 40 if num_cases is None else num_cases
    selected, train = held_out_cases(root, n, val_ratio=val_ratio, seed=seed)
    return selected, [f"validation split (train={len(train)}, seed={seed}, val_ratio={val_ratio})"]


def default_out_dir(model: str) -> Path:
    if model_name(model) == ABHINAV_V3:
        return ROOT / "caches" / "val_probs_abhinav_v3"
    return _DEFAULT_OUT


def checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_resume(out_dir: Path, model: str, checkpoint: Path, sha256: str) -> None:
    """Refuse to resume into a cache from another model or checkpoint.

    Checkpoints are matched by content hash, so a cache started on Colab can
    be resumed on another machine where the same file lives at another path.
    """
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    old_model = previous.get("model", OURS)
    old_ckpt = previous.get("checkpoint")
    old_sha = previous.get("checkpoint_sha256")
    if old_sha is not None:
        other_checkpoint = old_sha != sha256
    else:
        other_checkpoint = old_ckpt is not None and Path(old_ckpt) != Path(checkpoint)
    if old_model != model or other_checkpoint:
        raise RuntimeError(
            f"{out_dir} holds probabilities from model={old_model} checkpoint={old_ckpt}. "
            "Pass a new --out-dir so the resume does not mix models."
        )


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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", choices=(OURS, ABHINAV_V3), default=OURS)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: caches/val_probs or caches/val_probs_abhinav_v3")
    parser.add_argument("--case-list", type=Path, default=None, help="Case ids to cache. Default for abhinav_v3: splits/abhinav_heldout.txt")
    parser.add_argument("--num-cases", type=int, default=None, help="Default: 40 for ours, every listed case for abhinav_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--full-tta",
        action="store_true",
        help="Eight flips and overlap 0.625. About twice as slow as the default four-view pass",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the validation case ids and exit")
    args = parser.parse_args()
    kind = model_name(args.model)
    if args.checkpoint is None:
        args.checkpoint = resolve_checkpoint(kind) if kind == ABHINAV_V3 else _DEFAULT_CKPT
    if args.out_dir is None:
        args.out_dir = default_out_dir(kind)

    selected, source = select_cases(
        args.data_dir,
        kind,
        num_cases=args.num_cases,
        case_list=args.case_list,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"held-out {len(selected)} cases, model={kind}, {', '.join(source)}", flush=True)
    for path in selected:
        print(f"  {path.name}", flush=True)
    if args.dry_run:
        return

    sha256 = checkpoint_sha256(args.checkpoint)
    _check_resume(args.out_dir, kind, args.checkpoint, sha256)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source": source,
        "n_cached": len(selected),
        "case_ids": [path.name for path in selected],
        "channel_order": list(CHANNEL_ORDER),
        "model": kind,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256,
        "roi": list(roi_for(kind)),
        "tta": "full" if args.full_tta else "fast",
        "tta_views": 8 if args.full_tta else 4,
        "overlap": 0.625 if args.full_tta else 0.5,
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
    tta_mode = True if args.full_tta else "fast"
    overlap = 0.625 if args.full_tta else 0.5
    roi = roi_for(kind)
    print(
        f"model={kind} device={device} roi={roi} tta={tta_mode} overlap={overlap}",
        flush=True,
    )
    model = load_model(args.checkpoint, device, kind)
    transform = _training_transform(kind)
    written = 0
    for index, case_dir in pending:
        out_path = args.out_dir / f"{case_dir.name}.npz"
        print(f"[{index}/{total}] {case_dir.name}", flush=True)
        image, label, spacing = _load_case(case_dir, transform)
        probs = predict_probs(model, image, roi, tta_mode, overlap=overlap, model_name=kind)
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
            model=np.array(kind),
        )
        written += 1
        print(f"[{index}/{total}] saved {out_path.name}", flush=True)
    print(f"wrote {written}, skipped {skipped}, dir {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
