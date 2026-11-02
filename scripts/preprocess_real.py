"""Preprocess inventoried real MRI studies into outputs/real/<hashed_id>/04_ready.

Example:
    python scripts/preprocess_real.py --tier FULL
    python scripts/preprocess_real.py --tier NO_CONTRAST --limit 5
    python scripts/preprocess_real.py --tier ALL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inventory_real import DEFAULT_DATA, DEFAULT_OUT  # noqa: E402
from recon.ingest_real import (  # noqa: E402
    CLI_TIERS,
    DEFAULT_OUTPUT,
    print_summary,
    run,
)
from recon.timing import stay_awake  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess real MRI studies (resumable)")
    parser.add_argument("--tier", choices=CLI_TIERS, required=True, help="FULL, NO_CONTRAST, FLAIR_T2_ONLY, or ALL")
    parser.add_argument("--limit", type=int, default=None, help="Max studies to consider (after sorting by hashed id)")
    parser.add_argument("--force", action="store_true", help="Redo studies even if 04_ready already exists")
    parser.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Hashed study ids to process (within the selected tier)",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA, help="DICOM/NIfTI root")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUT, help="Inventory CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output root (hashed study folders)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    with stay_awake():
        summary = run(
            root=args.root,
            inventory=args.inventory,
            out_root=args.out,
            tier=args.tier,
            limit=args.limit,
            force=args.force,
            studies=args.studies,
        )
    print_summary(summary)
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
