"""Capture the default glass-brain view to ``outputs/<case>/preview.png``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _chrome() -> Path:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Google Chrome is required to render preview.png")


def render_preview(case_id: str, host: str = "http://127.0.0.1:8000") -> Path:
    dest = ROOT / "outputs" / case_id / "preview.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    chrome = _chrome()
    profile = Path(tempfile.mkdtemp(prefix="brats-preview-"))
    url = f"{host}/viewer?case={case_id}&preview=1"
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        "--window-size=1600,900",
        "--virtual-time-budget=40000",
        f"--screenshot={dest}",
        url,
    ]
    subprocess.run(cmd, check=True)
    if not dest.is_file():
        raise RuntimeError(f"Chrome did not write {dest}")
    return dest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Render the default glass-brain preview")
    parser.add_argument("--case", default="BraTS20_Training_002")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    path = render_preview(args.case, args.host)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
