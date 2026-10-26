"""Reconstruct one BraTS case to glTF, a NIfTI segmentation, and a manifest.

Example:
    python scripts/reconstruct.py --case path/to/BraTS-00001
    python scripts/reconstruct.py --case path/to/BraTS-00001 --anatomy --tracts
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.scene import build_scene, default_output_root  # noqa: E402


def default_checkpoint() -> Path:
    """First existing checkpoint: ``BRATS_CHECKPOINT``, the repo, then the smoke run."""
    candidates = []
    env = os.environ.get("BRATS_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "models" / "best_model.pt")
    candidates.append(
        Path.home() / "Desktop" / "MRI project" / "models" / "checkpoints" / "brats_pretrain_smoke" / "best_model.pt"
    )
    for path in candidates:
        if path.is_file():
            return path
    lines = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(f"No SegResNet checkpoint found. Pass --checkpoint.\nLooked in:\n{lines}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inference, post-processing, and scene export for one BraTS case")
    parser.add_argument("--case", type=Path, required=True, help="BraTS case folder (t1n, t1c, t2w, t2f, seg)")
    parser.add_argument("--anatomy", action="store_true", help="Add the brain shell and SynthSeg structures")
    parser.add_argument("--tracts", action="store_true", help="Warp and mesh the JHU atlas tracts")
    parser.add_argument("--model", choices=("ours", "abhinav_v3"), default="ours", help="ours keeps the local preprocessing. abhinav_v3 uses the teammate checkpoint and its validation preprocessing")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SegResNet checkpoint. Default: BRATS_CHECKPOINT or the smoke checkpoint; abhinav_v3 uses checkpoints/brats_scale_full/best_model.pt")
    parser.add_argument("--synthseg", type=Path, default=None, help="SynthSeg parcellation NIfTI. Default: *synthseg* in the case folder")
    parser.add_argument("--output", type=Path, default=None, help="Output root. Default: outputs/")
    args = parser.parse_args(argv)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    elif args.model == "abhinav_v3":
        from recon.infer import resolve_checkpoint

        checkpoint = resolve_checkpoint(args.model)
    else:
        checkpoint = default_checkpoint()
    manifest = build_scene(
        args.case,
        anatomy=args.anatomy,
        tracts=args.tracts,
        checkpoint=checkpoint,
        output_root=args.output,
        synthseg=args.synthseg,
        model=args.model,
    )
    root = args.output if args.output is not None else default_output_root()
    print(f"wrote {root / manifest['case_id']}", flush=True)


if __name__ == "__main__":
    main()
