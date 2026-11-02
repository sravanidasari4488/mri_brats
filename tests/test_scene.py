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
        brain = np.zeros(shape, dtype=np.float32)
        brain[0:18, 0:18, 0:18] = 1.0
        _write(case_dir / "t1n.nii.gz", brain, affine)
        canon = nib.as_closest_canonical(nib.load(case_dir / "t1n.nii.gz"))
        probs = np.zeros((3,) + shape, dtype=np.float32)
        probs[2, 6:14, 6:14, 6:14] = 0.95  # WT
        probs[1, 7:13, 7:13, 7:13] = 0.95  # TC
        probs[0, 8:13, 8:13, 8:13] = 0.95  # ET
        parc = np.zeros(shape, dtype=np.int16)
        parc[0:4, 0:4, 0:4] = 16
        parc[0:6, 14:20, 0:6] = 3
        parc[0:6, 14:20, 6:8] = 2
        parc[14:20, 14:20, 0:6] = 42
        parc[14:20, 14:20, 6:8] = 41
        parc[0:5, 14:19, 0:4] = 101
        parc[1:3, 1:3, 8:10] = 10
        parc[16:18, 1:3, 8:10] = 49
        image = np.zeros((4,) + shape, dtype=np.float32)
        image[:, 0:18, 0:18, 0:18] = 1.0
        case = {
            "image": image,
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
        for name in ("edema", "necrosis", "enhancing", "Left Cortex", "Right Cortex", "Brainstem", "Left Corticospinal Tract"):
            assert (out / f"{name}.glb").is_file()
            assert name in manifest["structures"]
        assert "brain" not in manifest["structures"]
        assert "Frontal Lobe" not in manifest["structures"]

        edema = manifest["structures"]["edema"]
        assert edema["source"] == "model"
        assert edema["color"] == list(SCENE_COLORS["edema"])
        assert edema["color"][3] == int(round(0.25 * 255))
        assert edema["volume_cm3"] > 0
        assert isinstance(edema["watertight"], bool)
        assert manifest["structures"]["necrosis"]["color"][:3] == [192, 57, 43]
        assert manifest["structures"]["enhancing"]["color"][:3] == [59, 130, 246]
        cortex = manifest["structures"]["Left Cortex"]
        assert cortex["glass"] is True
        assert cortex["lobe_tints"] is True
        assert cortex["group"] == "Tissue"
        assert cortex["full_name"] == "Cortical grey matter (left)"
        assert cortex["legend"] == "Cortical grey matter (left)"
        assert cortex["color"][:3] == [159, 184, 214]
        assert cortex["volume_cm3"] > 0
        assert "White Matter" in manifest["structures"]
        assert manifest["structures"]["White Matter"]["hidden_by_default"] is True
        assert manifest["structures"]["White Matter"]["glass"] is True
        assert manifest["structures"]["White Matter"]["full_name"] == "White matter"
        assert manifest["structures"]["White Matter"]["legend"] == "White matter"
        assert manifest["structures"]["White Matter"]["color"][:3] == [232, 220, 200]
        assert manifest["structures"]["White Matter"]["volume_cm3"] > 0
        total = manifest["structures"]["Grey matter total"]
        assert total["volume_only"] is True
        assert total["group"] == "Tissue"
        assert total["legend"] == "Grey matter total"
        assert total["volume_cm3"] >= cortex["volume_cm3"] + manifest["structures"]["Right Cortex"]["volume_cm3"] - 1e-6
        assert manifest["structures"]["Brainstem"]["source"] == "synthseg"
        assert manifest["structures"]["Brainstem"]["legend"] == "Brainstem"
        assert manifest["structures"]["Brainstem"]["group"] == "Brainstem & Cerebellum"
        assert manifest["structures"]["Brainstem"]["glass"] is True
        assert manifest["structures"]["Brainstem"]["volume_cm3"] > 0
        assert len(manifest["structures"]["Brainstem"]["centroid_mm"]) == 3
        assert manifest["structures"]["Brainstem"]["color"][:3] == [159, 184, 214]
        assert isinstance(manifest["structures"]["Brainstem"]["watertight"], bool)
        assert "volume_error_pct" in manifest["structures"]["Brainstem"]
        tract = manifest["structures"]["Left Corticospinal Tract"]
        assert tract["source"] == "atlas-estimated"
        assert tract["color"] == list(tract_color("corticospinal_tract_L"))
        assert tract["color"][3] == int(round(0.60 * 255))
        assert set(tract["proximity"]) == {"wt_distance_mm", "et_distance_mm", "wt_overlaps", "et_overlaps"}
        legend = {entry["name"]: entry for entry in manifest["legend"]}
        assert legend["Corticospinal Tract"]["meshes"] == ["Left Corticospinal Tract"]
        assert legend["Corticospinal Tract"]["source"] == "atlas-estimated"
        assert "Brain" not in legend
        assert "Frontal Lobe" not in legend
        assert legend["Cortical grey matter (left)"]["meshes"] == ["Left Cortex"]
        assert legend["Cortical grey matter (left)"]["volume_cm3"] == cortex["volume_cm3"]
        assert legend["Cortical grey matter (right)"]["meshes"] == ["Right Cortex"]
        assert legend["White matter"]["meshes"] == ["White Matter"]
        assert legend["White matter"]["volume_cm3"] == manifest["structures"]["White Matter"]["volume_cm3"]
        assert legend["Grey matter total"]["volume_cm3"] == total["volume_cm3"]
        assert "threshold_et" in manifest["postprocess"]

        saved = nib.load(out / "pred_seg.nii.gz")
        assert np.allclose(saved.affine, affine)
        anatomy_seg = nib.load(out / "anatomy_seg.nii.gz")
        assert anatomy_seg.shape[:3] == saved.shape[:3]
        assert np.allclose(anatomy_seg.affine, saved.affine)
        roundtrip = np.rint(nib.as_closest_canonical(saved).get_fdata()).astype(np.int16)
        pred = nib.load(out / "pred_seg.nii.gz")
        native = np.rint(pred.get_fdata()).astype(np.int16)
        assert native.shape == shape
        assert roundtrip.shape == shape
        assert int(roundtrip.max()) == 3
        scene = trimesh.load(out / "scene.glb")
        assert "necrosis" in scene.geometry
        parents = scene.graph.transforms.parents
        assert parents["necrosis"] == "Tumor"
        assert parents["Brainstem"] == "Brainstem & Cerebellum"
        assert parents["Left Cortex"] == "Tissue"
        assert parents["Left Corticospinal Tract"] == "Tracts"
        written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert written["case_id"] == "BraTS_001"


def test_cli_anatomy_defaults_on():
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
    assert seen["anatomy"] is True
    assert seen["tracts"] is False
    module.main(["--case", "BraTS_001", "--no-anatomy", "--tracts"])
    assert seen["anatomy"] is False
    assert seen["tracts"] is True


def test_anatomy_is_built_before_the_tumor_and_overlap_is_remeshed():
    shape = (20, 20, 20)
    affine = np.eye(4, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / "BraTS_002"
        brain = np.zeros(shape, dtype=np.float32)
        brain[0:18, 0:18, 0:18] = 1.0
        _write(case_dir / "t1n.nii.gz", brain, affine)
        probs = np.zeros((3,) + shape, dtype=np.float32)
        probs[2, 6:14, 6:14, 6:14] = 0.95
        probs[1, 7:13, 7:13, 7:13] = 0.95
        probs[0, 8:13, 8:13, 8:13] = 0.95
        parc = np.zeros(shape, dtype=np.int16)
        parc[0:4, 0:4, 0:4] = 16  # brainstem, clear of the tumor
        parc[8:12, 8:12, 8:12] = 10  # thalamus, inside the tumor
        image = np.zeros((4,) + shape, dtype=np.float32)
        image[:, 0:18, 0:18, 0:18] = 1.0
        case = {
            "image": image,
            "affine": affine,
            "case_id": "BraTS_002",
            "case_dir": case_dir,
            "probs": probs,
            "synthseg": parc,
        }
        order = []
        import importlib

        import recon.anatomy as anatomy_mod
        import recon.scene as scene_mod

        post_mod = importlib.import_module("recon.postprocess")
        real_mesh = scene_mod.mesh_parcellation_labels
        real_post = post_mod.postprocess
        real_glass = anatomy_mod.iter_glass_shells

        def mesh_wrap(parc_volume, grid, **kwargs):
            order.append(("mesh", int(np.count_nonzero(parc_volume))))
            return real_mesh(parc_volume, grid, **kwargs)

        def post_wrap(probabilities, brain_mask, config=None):
            order.append(("tumor", 0))
            return real_post(probabilities, brain_mask, config)

        def glass_wrap(parc_volume, tumor_label, grid):
            order.append(("glass", int(np.count_nonzero(parc_volume))))
            return real_glass(parc_volume, tumor_label, grid)

        scene_mod.mesh_parcellation_labels = mesh_wrap
        post_mod.postprocess = post_wrap
        anatomy_mod.iter_glass_shells = glass_wrap
        try:
            manifest = build_scene(
                case,
                anatomy=True,
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
            scene_mod.mesh_parcellation_labels = real_mesh
            post_mod.postprocess = real_post
            anatomy_mod.iter_glass_shells = real_glass

    kinds = [step for step, _count in order]
    assert kinds[0] == "tumor"
    assert "glass" in kinds
    assert kinds.index("tumor") < kinds.index("glass")
    assert "Brainstem" in manifest["structures"]
    assert manifest["structures"]["Brainstem"]["removed_voxels"] == 0
    assert "brain" not in manifest["structures"]
    assert "Frontal Lobe" not in manifest["structures"]
    assert "Left Thalamus" not in manifest["structures"]
    assert manifest["removed_voxels"]["Left Thalamus"] == 64
    assert len(manifest["structures"]["Brainstem"]["centroid_mm"]) == 3


def test_tumor_location_lists_overlap_and_ten_millimetres():
    from recon.scene import tumor_location_summary

    shape = (32, 12, 12)
    parc = np.zeros(shape, dtype=np.int16)
    parc[2:10, 2:10, 2:10] = 10
    parc[18:20, 2:6, 2:6] = 11
    parc[30, 10, 10] = 16
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[8:16, 2:10, 2:10] = 2
    tumor[8, 2:10, 2:10] = 3
    summary = tumor_location_summary(parc, tumor, np.eye(4))
    whole = {item["name"]: item for item in summary["WT"]}
    enhancing = {item["name"]: item for item in summary["ET"]}
    assert summary["margin_mm"] == 10.0
    assert summary["min_overlap_voxels"] == 50
    assert whole["Left Thalamus"]["relation"] == "overlaps"
    assert whole["Left Thalamus"]["distance_mm"] == 0.0
    assert whole["Left Thalamus"]["contact_voxels"] >= 50
    assert whole["Left Caudate"]["relation"] == "within_10mm"
    assert 0 < whole["Left Caudate"]["distance_mm"] <= 10
    assert "Brainstem" not in whole
    assert enhancing["Left Thalamus"]["relation"] == "overlaps"
    assert "Brainstem" not in enhancing


def test_anatomy_failure_keeps_tumor_and_warns():
    shape = (16, 16, 16)
    affine = np.eye(4, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / "fail_case"
        brain = np.zeros(shape, dtype=np.float32)
        brain[1:15, 1:15, 1:15] = 1.0
        for suffix in ("t2f", "t1n", "t1c", "t2w"):
            _write(case_dir / f"{suffix}.nii.gz", brain, affine)
        probs = np.zeros((3,) + shape, dtype=np.float32)
        probs[2, 4:12, 4:12, 4:12] = 0.95
        image = np.zeros((4,) + shape, dtype=np.float32)
        image[:, 1:15, 1:15, 1:15] = 1.0
        import recon.scene as scene_mod

        original = scene_mod._raw_parcellation

        def boom(*args, **kwargs):
            raise RuntimeError("SynthSeg crashed")

        scene_mod._raw_parcellation = boom
        try:
            manifest = build_scene(
                {
                    "image": image,
                    "affine": affine,
                    "case_id": "fail_case",
                    "case_dir": case_dir,
                    "probs": probs,
                },
                anatomy=True,
                output_root=root / "out",
                config=PostConfig(
                    min_voxels_et=0,
                    min_voxels_tc=0,
                    min_voxels_wt=0,
                    et_min_voxels=0,
                    brain_closing_iterations=1,
                ),
            )
        finally:
            scene_mod._raw_parcellation = original
        assert manifest["anatomy_warning"].startswith("Anatomy unavailable:")
        assert "SynthSeg crashed" in manifest["anatomy_warning"]
        assert "enhancing" in manifest["structures"] or "edema" in manifest["structures"]
        assert "brain" in manifest["structures"]
        assert "Left Cortex" not in manifest["structures"]
        assert (root / "out" / "fail_case" / "scene.glb").is_file()


def test_anatomy_cache_skips_synthseg_on_reopen():
    shape = (12, 12, 12)
    affine = np.eye(4, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / "cached"
        brain = np.ones(shape, dtype=np.float32)
        for suffix in ("t2f", "t1n", "t1c", "t2w"):
            _write(case_dir / f"{suffix}.nii.gz", brain, affine)
        parc = np.zeros(shape, dtype=np.int16)
        parc[1:6, 1:6, 1:6] = 3
        parc[6:11, 1:6, 1:6] = 42
        parc[1:11, 6:11, 1:6] = 2
        image = np.ones((4,) + shape, dtype=np.float32)
        probs = np.zeros((3,) + shape, dtype=np.float32)
        calls = {"n": 0}

        def fake_parc(*args, **kwargs):
            calls["n"] += 1
            return parc, parc, {}

        import recon.anatomy as anatomy
        import recon.scene as scene_mod

        original = anatomy.parcellate_case
        original_cache = scene_mod.default_anatomy_cache_root
        anatomy.parcellate_case = fake_parc
        scene_mod.default_anatomy_cache_root = lambda: root / "anatomy_cache"
        case = {
            "image": image,
            "affine": affine,
            "case_id": "cached",
            "case_dir": case_dir,
            "probs": probs,
        }
        config = PostConfig(
            min_voxels_et=0,
            min_voxels_tc=0,
            min_voxels_wt=0,
            et_min_voxels=0,
            brain_closing_iterations=1,
        )
        try:
            first = build_scene(case, anatomy=True, output_root=root / "out", config=config)
            second = build_scene(case, anatomy=True, output_root=root / "out", config=config)
        finally:
            anatomy.parcellate_case = original
            scene_mod.default_anatomy_cache_root = original_cache
        assert calls["n"] == 1
        assert first["input_hash"] == second["input_hash"]
        assert first["input_hash"]
        assert "Left Cortex" in second["structures"]
        assert "White Matter" in second["structures"]
        assert second["structures"]["Left Cortex"]["volume_cm3"] > 0
        assert second["structures"]["White Matter"]["volume_cm3"] > 0
        assert second.get("anatomy_warning") is None


def test_scene_is_up_to_date_requires_tissue_or_warning():
    from recon.scene import scene_is_up_to_date

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "case"
        out.mkdir()
        (out / "scene.glb").write_bytes(b"glb")
        (out / "manifest.json").write_text(
            json.dumps({"structures": {"enhancing": {"group": "Tumor"}}, "model": "abhinav_v3"}),
            encoding="utf-8",
        )
        assert scene_is_up_to_date(out, anatomy=True) is False
        meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        meta["anatomy_warning"] = "Anatomy unavailable: test"
        (out / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
        assert scene_is_up_to_date(out, anatomy=True) is True
        meta.pop("anatomy_warning")
        meta["structures"]["Left Cortex"] = {"group": "Tissue"}
        (out / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
        assert scene_is_up_to_date(out, anatomy=True, model="abhinav_v3") is True


def test_tumor_location_ignores_labels_only_inside_the_tumor():
    from recon.scene import tumor_location_summary

    shape = (16, 8, 8)
    parc = np.zeros(shape, dtype=np.int16)
    parc[2:6, 2:6, 2:6] = 12
    tumor = np.zeros(shape, dtype=np.uint8)
    tumor[2:6, 2:6, 2:6] = 2
    summary = tumor_location_summary(parc, tumor, np.eye(4))
    assert summary["WT"] == []


if __name__ == "__main__":
    test_build_scene_exports_named_nodes_and_original_space()
    test_anatomy_is_built_before_the_tumor_and_overlap_is_remeshed()
    test_tumor_location_lists_overlap_and_ten_millimetres()
    test_tumor_location_ignores_labels_only_inside_the_tumor()
    test_cli_anatomy_defaults_on()
    test_anatomy_failure_keeps_tumor_and_warns()
    test_anatomy_cache_skips_synthseg_on_reopen()
    test_scene_is_up_to_date_requires_tissue_or_warning()
    print("scene ok")
