"""Scene export: colors, sources, original-space segmentation, CLI flags."""

import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.postprocess import PostConfig
from recon.scene import SCENE_COLORS, build_scene, tract_color
from recon.tracts import AtlasTract


def _write(path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), path)


def test_build_scene_exports_named_nodes_and_original_space():
    shape = (20, 20, 20)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = -1.0
    affine[1, 1] = -1.0
    affine[:3, 3] = (4.0, -6.0, 2.0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / "BraTS_001"
        _write(case_dir / "t1n.nii.gz", np.ones(shape, np.float32), affine)
        canon = nib.as_closest_canonical(nib.load(case_dir / "t1n.nii.gz"))
        probs = np.zeros((3,) + shape, dtype=np.float32)
        probs[2, 6:14, 6:14, 6:14] = 0.95  # WT
        probs[1, 7:13, 7:13, 7:13] = 0.95  # TC
        probs[0, 8:13, 8:13, 8:13] = 0.95  # ET
        parc = np.zeros(shape, dtype=np.int16)
        parc[0:4, 0:4, 0:4] = 16
        case = {
            "image": np.ones((4,) + shape, dtype=np.float32),
            "affine": np.asarray(canon.affine, dtype=np.float64),
            "case_id": "BraTS_001",
            "case_dir": case_dir,
            "probs": probs,
            "synthseg": parc,
        }

        def fake_warp(t1_path, label, **kwargs):
            mesh = trimesh.Trimesh(
                vertices=np.zeros((3, 3)),
                faces=[[0, 1, 2]],
                process=False,
            )
            mask = np.zeros(label.shape, dtype=bool)
            mask[0, 0, 0] = True
            tract = AtlasTract(
                name="corticospinal_tract_L",
                mesh=mesh,
                mask=mask,
                threshold=0.25,
                affine=case["affine"],
            )
            return {tract.name: tract}

        import recon.tracts as tracts

        original = tracts.warp_atlas_tracts
        tracts.warp_atlas_tracts = fake_warp
        try:
            manifest = build_scene(
                case,
                anatomy=True,
                tracts=True,
                output_root=root / "out",
                config=PostConfig(
                    threshold_et=0.5,
                    threshold_tc=0.5,
                    threshold_wt=0.5,
                    min_voxels_et=0,
                    min_voxels_tc=0,
                    min_voxels_wt=0,
                    et_min_voxels=0,
                    brain_closing_iterations=1,
                ),
            )
        finally:
            tracts.warp_atlas_tracts = original

        out = root / "out" / "BraTS_001"
        assert (out / "scene.glb").is_file()
        assert (out / "pred_seg.nii.gz").is_file()
        for name in ("edema", "necrosis", "enhancing", "brain", "brainstem", "corticospinal_tract_L"):
            assert (out / f"{name}.glb").is_file()
            assert name in manifest["structures"]

        edema = manifest["structures"]["edema"]
        assert edema["source"] == "model"
        assert edema["color"] == list(SCENE_COLORS["edema"])
        assert edema["color"][3] == int(round(0.35 * 255))
        assert edema["volume_cm3"] > 0
        assert isinstance(edema["watertight"], bool)
        assert manifest["structures"]["necrosis"]["color"][:3] == [220, 40, 40]
        assert manifest["structures"]["enhancing"]["color"][:3] == [60, 120, 255]
        brain = manifest["structures"]["brain"]
        assert brain["source"] == "image"
        assert brain["color"][3] == int(round(0.10 * 255))
        assert manifest["structures"]["brainstem"]["source"] == "synthseg"
        tract = manifest["structures"]["corticospinal_tract_L"]
        assert tract["source"] == "atlas-estimated"
        assert tract["color"] == list(tract_color("corticospinal_tract_L"))
        assert tract["color"][3] == int(round(0.60 * 255))
        assert set(tract["proximity"]) == {"wt_distance_mm", "et_distance_mm", "wt_overlaps", "et_overlaps"}
        assert "threshold_et" in manifest["postprocess"]

        saved = nib.load(out / "pred_seg.nii.gz")
        assert np.allclose(saved.affine, affine)
        roundtrip = np.rint(nib.as_closest_canonical(saved).get_fdata()).astype(np.int16)
        pred = nib.load(out / "pred_seg.nii.gz")
        native = np.rint(pred.get_fdata()).astype(np.int16)
        assert native.shape == shape
        assert roundtrip.shape == shape
        assert int(roundtrip.max()) == 3
        scene = trimesh.load(out / "scene.glb")
        assert "necrosis" in scene.geometry
        written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert written["case_id"] == "BraTS_001"


def test_cli_anatomy_and_tracts_default_off():
    import importlib.util

    spec = importlib.util.spec_from_file_location("reconstruct", ROOT / "scripts" / "reconstruct.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}

    def fake(case, anatomy=False, tracts=False, **kwargs):
        seen["case"] = str(case)
        seen["anatomy"] = anatomy
        seen["tracts"] = tracts
        return {"case_id": "x"}

    module.build_scene = fake
    module.main(["--case", "BraTS_001"])
    assert seen["anatomy"] is False
    assert seen["tracts"] is False
    module.main(["--case", "BraTS_001", "--anatomy", "--tracts"])
    assert seen["anatomy"] is True
    assert seen["tracts"] is True


if __name__ == "__main__":
    test_build_scene_exports_named_nodes_and_original_space()
    test_cli_anatomy_and_tracts_default_off()
    print("scene ok")
