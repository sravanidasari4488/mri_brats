"""Rebuild-all skip logic and summary table."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)


def test_build_all_skips_up_to_date_and_rebuilds_stale(tmp_path):
    import scripts.build_all as build_all

    cases = tmp_path / "cases"
    outputs = tmp_path / "outputs"
    fresh = cases / "fresh"
    stale = cases / "stale"
    for folder in (fresh, stale):
        for suffix in ("t2f", "t1n", "t1c", "t2w"):
            _write(folder / f"{suffix}.nii.gz", np.ones((4, 4, 4)))

    from recon.io import input_files_hash

    fresh_hash = input_files_hash(fresh)
    fresh_out = outputs / "fresh"
    fresh_out.mkdir(parents=True)
    (fresh_out / "scene.glb").write_bytes(b"glb")
    (fresh_out / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "fresh",
                "case_dir": str(fresh),
                "model": "abhinav_v3",
                "input_hash": fresh_hash,
                "structures": {
                    "Left Cortex": {"group": "Tissue", "volume_cm3": 10.5},
                    "White Matter": {"group": "Tissue", "volume_cm3": 20.25},
                    "Grey matter total": {"group": "Tissue", "volume_cm3": 12.0},
                    "edema": {"group": "Tumor", "volume_cm3": 1.5},
                },
            }
        ),
        encoding="utf-8",
    )
    stale_out = outputs / "stale"
    stale_out.mkdir(parents=True)
    (stale_out / "scene.glb").write_bytes(b"glb")
    (stale_out / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "stale",
                "case_dir": str(stale),
                "model": "abhinav_v3",
                "structures": {"edema": {"group": "Tumor", "volume_cm3": 3.0}},
            }
        ),
        encoding="utf-8",
    )

    rebuilt = []

    def fake_build(case, anatomy=True, model="ours", **kwargs):
        rebuilt.append(Path(case).name)
        out = Path(kwargs["output_root"]) / Path(case).name
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.glb").write_bytes(b"glb")
        manifest = {
            "case_id": Path(case).name,
            "case_dir": str(case),
            "model": model,
            "anatomy": anatomy,
            "structures": {
                "Left Cortex": {"group": "Tissue", "volume_cm3": 8.0},
                "White Matter": {"group": "Tissue", "volume_cm3": 9.0},
                "Grey matter total": {"group": "Tissue", "volume_cm3": 11.0},
                "edema": {"group": "Tumor", "volume_cm3": 2.0},
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    original = build_all.build_scene
    build_all.build_scene = fake_build
    try:
        code = build_all.main(
            ["--model", "abhinav_v3", "--cases", str(cases), "--output", str(outputs), "--checkpoint", "ckpt.pt"]
        )
    finally:
        build_all.build_scene = original
    assert code == 0
    assert rebuilt == ["stale"]
    assert json.loads((fresh_out / "manifest.json").read_text(encoding="utf-8"))["structures"]["Grey matter total"][
        "volume_cm3"
    ] == 12.0
    assert json.loads((stale_out / "manifest.json").read_text(encoding="utf-8"))["structures"]["White Matter"][
        "volume_cm3"
    ] == 9.0


if __name__ == "__main__":
    import tempfile

    test_build_all_skips_up_to_date_and_rebuilds_stale(Path(tempfile.mkdtemp()))
    print("build_all ok")
