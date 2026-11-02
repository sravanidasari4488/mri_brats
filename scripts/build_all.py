"""Rebuild every existing case with anatomy.

Example:
    python scripts/build_all.py --model abhinav_v3
    python scripts/build_all.py --model abhinav_v3 --cases data/cases
    python scripts/build_all.py --model abhinav_v3 --cases BraTS20_Training_002 BraTS20_Training_004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.io import case_sequence_paths  # noqa: E402
from recon.scene import build_scene, default_output_root, scene_is_up_to_date  # noqa: E402


def default_checkpoint() -> Path:
    from scripts.reconstruct import default_checkpoint as reconstruct_checkpoint

    return reconstruct_checkpoint()


def _looks_like_case(path: Path) -> bool:
    try:
        case_sequence_paths(path)
        return True
    except (FileNotFoundError, NotADirectoryError, OSError):
        return False


def _case_dir_from_manifest(out_dir: Path) -> Path | None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = meta.get("case_dir")
    if recorded and Path(recorded).is_dir():
        return Path(recorded)
    return None


def _discover_cases(cases, output_root: Path) -> list[tuple[str, Path | None, Path]]:
    """(case_id, case_dir, out_dir) for each case to consider."""
    found: list[tuple[str, Path | None, Path]] = []
    seen: set[str] = set()

    def add(case_id: str, case_dir: Path | None, out_dir: Path) -> None:
        if case_id in seen:
            return
        seen.add(case_id)
        found.append((case_id, case_dir, out_dir))

    if not cases:
        if not output_root.is_dir():
            return []
        for path in sorted(output_root.iterdir()):
            if path.is_dir() and (path / "manifest.json").is_file():
                add(path.name, _case_dir_from_manifest(path), path)
        return found

    for item in cases:
        path = Path(item)
        if not path.is_absolute():
            for candidate in (Path.cwd() / path, ROOT / path, output_root / path, ROOT / "data" / "cases" / path):
                if candidate.exists():
                    path = candidate
                    break
        if path.is_dir() and _looks_like_case(path):
            add(path.name, path, output_root / path.name)
            continue
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir() and _looks_like_case(child):
                    add(child.name, child, output_root / child.name)
            continue
        name = path.name
        out_dir = output_root / name
        case_dir = _case_dir_from_manifest(out_dir)
        fallback = ROOT / "data" / "cases" / name
        if case_dir is None and fallback.is_dir():
            case_dir = fallback
        add(name, case_dir, out_dir)
    return found


def _volumes(manifest: dict) -> tuple[float | None, float | None, float | None]:
    structures = manifest.get("structures") or {}
    gm = (structures.get("Grey matter total") or {}).get("volume_cm3")
    wm = (structures.get("White Matter") or {}).get("volume_cm3")
    wt = 0.0
    present = False
    for name in ("edema", "necrosis", "enhancing"):
        value = (structures.get(name) or {}).get("volume_cm3")
        if value is not None:
            wt += float(value)
            present = True
    return (
        None if gm is None else float(gm),
        None if wm is None else float(wm),
        wt if present else None,
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _status(manifest: dict | None, skipped: bool, error: str | None) -> str:
    if error:
        return f"failed: {error}"
    if skipped:
        return "skipped"
    if manifest and manifest.get("anatomy_warning"):
        return str(manifest["anatomy_warning"])
    return "rebuilt"


def _print_table(rows: list[tuple[str, float | None, float | None, float | None, str]]) -> None:
    headers = ("case", "GM", "WM", "WT", "status")
    cols = list(zip(*([headers] + [(r[0], _fmt(r[1]), _fmt(r[2]), _fmt(r[3]), r[4]) for r in rows]))) if rows else [
        headers
    ]
    if not rows:
        print(f"{headers[0]:<28} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10}  {headers[4]}")
        print("(no cases)")
        return
    widths = [max(len(str(cell)) for cell in col) for col in cols]
    widths[0] = max(widths[0], 12)
    line = (
        f"{headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]:>{widths[2]}}  "
        f"{headers[3]:>{widths[3]}}  {headers[4]}"
    )
    print(line)
    print("-" * len(line))
    for case_id, gm, wm, wt, status in rows:
        print(
            f"{case_id:<{widths[0]}}  {_fmt(gm):>{widths[1]}}  {_fmt(wm):>{widths[2]}}  "
            f"{_fmt(wt):>{widths[3]}}  {status}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild every case in outputs/ with anatomy")
    parser.add_argument(
        "--model",
        choices=("ours", "abhinav_v3"),
        default="abhinav_v3",
        help="Tumor model. Default: abhinav_v3",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Case folder, a directory of cases, or case ids. Default: every folder in outputs/",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="SegResNet checkpoint")
    parser.add_argument("--output", type=Path, default=None, help="Output root. Default: outputs/")
    parser.add_argument("--fast", action="store_true", help="Four flipped views and overlap 0.5")
    args = parser.parse_args(argv)

    output_root = args.output if args.output is not None else default_output_root()
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    elif args.model == "abhinav_v3":
        from recon.infer import resolve_checkpoint

        checkpoint = resolve_checkpoint(args.model)
    else:
        checkpoint = default_checkpoint()

    rows = []
    extras = {"tta": "fast", "overlap": 0.5} if args.fast else {}
    for case_id, case_dir, out_dir in _discover_cases(args.cases, output_root):
        if case_dir is None or not case_dir.is_dir():
            rows.append((case_id, None, None, None, "failed: no scan folder"))
            continue
        if scene_is_up_to_date(out_dir, case_dir, anatomy=True, model=args.model):
            meta = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            gm, wm, wt = _volumes(meta)
            rows.append((case_id, gm, wm, wt, "skipped"))
            print(f"skip {case_id}", flush=True)
            continue
        try:
            manifest = build_scene(
                case_dir,
                anatomy=True,
                tracts=False,
                checkpoint=checkpoint,
                output_root=output_root,
                model=args.model,
                **extras,
            )
            gm, wm, wt = _volumes(manifest)
            rows.append((case_id, gm, wm, wt, _status(manifest, False, None)))
            print(f"wrote {out_dir}", flush=True)
        except Exception as exc:
            rows.append((case_id, None, None, None, f"failed: {exc}"))
            print(f"fail {case_id}: {exc}", flush=True)
    _print_table(rows)
    return 0 if all(not status.startswith("failed:") for *_, status in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
