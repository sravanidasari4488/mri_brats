"""Slice PNGs, case routes, and the viewer page."""

import io
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.app import _slice_png, create_app


def _nifti(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine),
        path,
    )


def test_slice_png_places_the_voxel_on_the_second_free_axis():
    from recon.slices import brain_window, orient_ras_slice, window_to_uint8

    volume = np.zeros((10, 6, 12), dtype=np.float32)
    volume[5, 2, 8] = 100
    image = Image.open(io.BytesIO(_slice_png(volume, axis=0, index=5)))
    lo, hi = brain_window(volume)
    expected = window_to_uint8(orient_ras_slice(volume, 0, 5), lo, hi)
    np.testing.assert_array_equal(np.asarray(image), expected)


def test_api_serves_scene_manifest_and_slices(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    outputs = root / "outputs"
    cases = root / "cases"
    shape = (8, 7, 6)
    data = np.zeros(shape, dtype=np.float32)
    data[1, 2, 3] = 50
    _nifti(cases / "demo" / "t2f.nii.gz", data)
    _nifti(cases / "demo" / "t1n.nii.gz", data + 2)
    _nifti(cases / "demo" / "t1c.nii.gz", data + 1)
    _nifti(cases / "demo" / "t2w.nii.gz", data + 3)
    pred = np.zeros(shape, dtype=np.uint8)
    pred[1, 2, 3] = 1
    case_out = outputs / "demo"
    case_out.mkdir(parents=True, exist_ok=True)
    _nifti(outputs / "demo" / "pred_seg.nii.gz", pred)
    (case_out / "scene.glb").write_bytes(b"glb")
    (case_out / "manifest.json").write_text(
        json.dumps({"case_id": "demo", "case_dir": str(cases / "demo"), "structures": {}, "tract_proximity": {}}),
        encoding="utf-8",
    )
    client = create_app(outputs, cases).test_client()

    manifest = client.get("/api/cases/demo/manifest.json")
    assert manifest.status_code == 200
    assert manifest.json["case_id"] == "demo"
    assert client.get("/api/cases/demo/scene.glb").data == b"glb"

    volume = client.get("/api/cases/demo/volume?modality=flair")
    assert volume.status_code == 200
    assert volume.json["shape"] == list(shape)
    assert volume.json["modality"] == "t2f"
    assert volume.json["bbox"]["lo"] == [1, 2, 3]
    assert volume.json["bbox"]["hi"] == [1, 2, 3]
    assert volume.json["letters"]["2"]["right"] == "L"
    assert volume.json["letters"]["0"]["left"] == "A"
    assert "extent_mm" in volume.json

    png = client.get("/api/cases/demo/slice?modality=t1c&axis=0&index=1")
    assert png.status_code == 200
    assert png.mimetype == "image/png"
    image = Image.open(io.BytesIO(png.data))
    assert image.size == (shape[1], shape[2])
    pixel = image.getpixel((4, 2))
    assert pixel == 255 or pixel[:3] == (255, 255, 255)
    assert client.get("/api/cases/demo/slice?modality=t1&axis=0&index=1").status_code == 200
    assert client.get("/api/cases/demo/slice?modality=t2&axis=0&index=1").status_code == 200
    overlay = client.get("/api/cases/demo/overlay?axis=0&index=1")
    assert overlay.status_code == 200
    assert overlay.mimetype == "image/png"
    cached = outputs / "demo" / "slices"
    assert cached.is_dir()
    assert any(cached.glob("*.png"))

    missing = client.get("/api/cases/demo/slice?modality=flair&axis=2&index=99")
    assert missing.status_code == 400
    assert "outside" in missing.json["error"]
    assert client.get("/api/cases/nope/manifest.json").status_code == 404


def test_viewer_page_has_groups_clipping_and_tract_note():
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    assert "GLTFLoader" in script
    assert "OrbitControls" in script
    assert "CSS2DRenderer" in script
    assert "tumor_location" in script
    assert "Tracts are atlas-estimated, not patient tractography." in html
    assert "Hovered only" in html and 'id="label-mode"' in html
    assert 'id="screenshot"' in html
    assert "Tumor" in script and "Anatomy" in script and "Tracts" in script
    assert "Tumor focus" in script and "Tumor + Tracts" in script
    assert "Glass + tumor" in script
    assert 'name="anatomy" type="checkbox" checked' in html
    assert "enabled: false" in script
    assert "applyPreset(\"glass-tumor\"" in script
    assert "volumeBox" in script
    assert "depthWrite" in script
    assert "enhancing" in script and "renderOrder" in script
    assert "RoomEnvironment" in script
    assert "ACESFilmicToneMapping" in script
    assert "ShaderMaterial" in script
    assert "pow(1.0 - abs(dot(normalize(vNormal), normalize(vViewDir))), 2.5)" in script
    assert "TUMOR_LABELS" in script
    assert "preview" in script
    assert "MeshPhysicalMaterial" in script
    assert "sortObjects = true" in script
    assert "setPixelRatio(Math.min" in script
    assert 'id="glass-opacity"' in html
    assert 'id="auto-rotate"' in html
    page = create_app().test_client().get("/viewer")
    assert page.status_code == 200
    assert b"viewer.js?v=anatomy1" in page.data
    assert b'id="job-progress"' in page.data
    assert b'id="anatomy-warning"' in page.data
    assert "Cortical grey matter (left)" in script
    assert "Grey matter total" in script or "structureLabel" in script
    assert b'data-edge="top"' in page.data
    assert "ORIENT_LETTERS" in script
    assert "renderOrientationGizmo" in script
    assert "sliceTextureCorners" in script
    assert "orient_ras_slice" not in script
    assert b'id="slices-panel"' in page.data
    assert b'id="slice-grid"' in page.data
    assert b'value="t1"' in page.data
    assert b'value="t2"' in page.data
    assert "slice-grid" in script
    assert "overlay-opacity" in html
    assert "Possible blood products" in script
    assert "hemorrhage" not in script.lower()
    assert b'value="hover" selected' in page.data
    css = (ROOT / "web" / "static" / "viewer.css").read_text(encoding="utf-8")
    assert "color-scheme: light" in css
    assert "background: #f4f6f8" in css
    assert "background: #14171a" not in css
    assert "setClearColor(0x0b0f17)" in script
    assert "orientNormals" in script
    assert "structureName" in script
    assert "replace(/_/g, \" \")" in script
    assert "tintMix" in script
    assert "frameTumorView" in script
    assert "pollJob" in script
    assert "renderJobProgress" in script
    assert "showAnatomyWarning" in script
    assert "Anatomy unavailable" not in html or "anatomy-warning" in html


def _wait_job(client, job_url, timeout=8):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(job_url)
        if last.status_code == 200 and last.json.get("status") in {"done", "error"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {None if last is None else last.json}")


def test_upload_calls_build_scene(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    app = create_app(root / "outputs", root / "cases")
    calls = {}

    def fake_build(case, anatomy=False, tracts=False, checkpoint=None, output_root=None, synthseg=None, model="ours",
                   tta=True, overlap=None, **kwargs):
        calls["case"] = str(case)
        calls["anatomy"] = anatomy
        calls["tracts"] = tracts
        calls["checkpoint"] = checkpoint
        calls["model"] = model
        calls["tta"] = (tta, overlap)
        return {"case_id": "scan1"}

    import recon.scene as scene
    import scripts.reconstruct as reconstruct

    original_build = scene.build_scene
    original_ckpt = reconstruct.default_checkpoint
    scene.build_scene = fake_build
    reconstruct.default_checkpoint = lambda: Path("checkpoint.pt")
    try:
        client = app.test_client()
        buffers = {}
        for suffix in ("t2f", "t1n", "t1c", "t2w"):
            path = root / f"{suffix}.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), path)
            buffers[suffix] = io.BytesIO(path.read_bytes())
        response = client.post(
            "/api/cases",
            data={
                "case_id": "scan1",
                "anatomy": "1",
                "tracts": "1",
                **{suffix: (buf, f"{suffix}.nii.gz") for suffix, buf in buffers.items()},
            },
        )
        assert response.status_code == 202
        assert response.json["job"] == "/api/cases/scan1/job"
        job = _wait_job(client, response.json["job"])
        assert job.json["status"] == "done"
        assert job.json["viewer"] == "/viewer?case=scan1"
        assert "stages" in job.json
        assert set(job.json["stages"]) >= {"Loading", "Tumor", "Anatomy", "Lobes", "Meshing", "Done"}
    finally:
        scene.build_scene = original_build
        reconstruct.default_checkpoint = original_ckpt

    assert response.json["viewer"] == "/viewer?case=scan1"
    assert calls["anatomy"] is True
    assert calls["tracts"] is True
    assert calls["tta"] == ("fast", 0.5)
    from recon.infer import ABHINAV_CHECKPOINT

    if Path(ABHINAV_CHECKPOINT).is_file():
        assert calls["model"] == "abhinav_v3"
        assert Path(calls["checkpoint"]) == Path(ABHINAV_CHECKPOINT)
    assert Path(calls["case"]).name == "scan1"
    assert (root / "cases" / "scan1" / "seg.nii.gz").is_file()

    defaulted = {}

    def fake_default(case, anatomy=False, tracts=False, **kwargs):
        defaulted["anatomy"] = anatomy
        defaulted["tracts"] = tracts
        return {"case_id": "scan2"}

    scene.build_scene = fake_default
    reconstruct.default_checkpoint = lambda: Path("checkpoint.pt")
    try:
        buffers = {suffix: io.BytesIO((root / f"{suffix}.nii.gz").read_bytes()) for suffix in ("t2f", "t1n", "t1c", "t2w")}
        response = client.post(
            "/api/cases",
            data={"case_id": "scan2", **{suffix: (buf, f"{suffix}.nii.gz") for suffix, buf in buffers.items()}},
        )
        assert response.status_code == 202
        job = _wait_job(client, response.json["job"])
        assert job.json["status"] == "done"
    finally:
        scene.build_scene = original_build
        reconstruct.default_checkpoint = original_ckpt
    assert defaulted["anatomy"] is True
    assert defaulted["tracts"] is False


def test_file_names_are_matched_to_scans():
    from recon.io import guess_case_id, identify_modality

    expected = {
        # Processed cache layout
        "flair.nii.gz": "t2f", "t1.nii.gz": "t1n", "t1c.nii.gz": "t1c", "t2.nii.gz": "t2w", "mask.nii.gz": "seg",
        # BraTS 2020
        "BraTS20_Training_002_flair.nii": "t2f", "BraTS20_Training_002_t1.nii": "t1n",
        "BraTS20_Training_002_t1ce.nii": "t1c", "BraTS20_Training_002_t2.nii": "t2w",
        "BraTS20_Training_002_seg.nii": "seg",
        # BraTS 2021+
        "BraTS-GLI-00001-000-t2f.nii.gz": "t2f", "BraTS-GLI-00001-000-t1n.nii.gz": "t1n",
        "BraTS-GLI-00001-000-t1c.nii.gz": "t1c", "BraTS-GLI-00001-000-t2w.nii.gz": "t2w",
        # Scanner / BIDS style
        "sub-01_ses-1_FLAIR.nii.gz": "t2f", "sub-01_T1w.nii.gz": "t1n", "sub-01_ce-gd_T1w.nii.gz": "t1c",
        "T1 post contrast.nii": "t1c", "AX_T1_GD.nii.gz": "t1c", "sub-01_T2w.nii.gz": "t2w",
        "patient_synthseg.nii.gz": "synthseg", "folder/sub/T2_FLAIR.nii.gz": "t2f",
        "notes.nii.gz": None, "DWI_b1000.nii.gz": None,
    }
    wrong = {name: identify_modality(name) for name, role in expected.items() if identify_modality(name) != role}
    assert not wrong, wrong

    assert guess_case_id(["t1.nii.gz"], "BraTS20_Training_002") == "BraTS20_Training_002"
    names = ["BraTS20_Training_002_flair.nii", "BraTS20_Training_002_t1.nii"]
    assert guess_case_id(names) == "BraTS20_Training_002"
    assert guess_case_id(["t1.nii.gz", "flair.nii.gz"]) == "scan"


def test_identify_endpoint_and_plain_missing_message(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    client = create_app(root / "outputs", root / "cases").test_client()
    names = ["flair.nii.gz", "t1.nii.gz", "t1c.nii.gz", "t2.nii.gz", "readme.nii"]
    response = client.post("/api/identify", json={"names": names, "folder": "BraTS20_Training_002"})
    assert response.status_code == 200
    assert response.json["roles"] == ["t2f", "t1n", "t1c", "t2w", None]
    assert response.json["required"] == ["t2f", "t1n", "t1c", "t2w"]
    assert response.json["case_id"] == "BraTS20_Training_002"

    path = root / "flair.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), path)
    partial = client.post("/api/cases", data={"case_id": "partial", "t2f": (io.BytesIO(path.read_bytes()), "flair.nii.gz")})
    assert partial.status_code == 400
    assert partial.json["error"] == "Missing scans: T1, T1 with contrast, T2"

    import web.app as web_app

    web_app._BUILDING.add("busy")
    try:
        busy = client.post("/api/cases", data={"case_id": "busy", "t2f": (io.BytesIO(path.read_bytes()), "flair.nii.gz")})
    finally:
        web_app._BUILDING.discard("busy")
    assert busy.status_code == 409
    assert "already being built" in busy.json["error"]
    assert not (root / "cases" / "busy").exists()


def test_upload_page_guides_without_modality_jargon():
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "static" / "viewer.css").read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in css
    assert 'id="drop-zone"' in html and "webkitdirectory" in html
    assert "Choose folder" in html and "Build 3D model" in html
    assert "<details id=\"advanced\">" in html
    assert "(t2f)" not in html and "(t1n)" not in html
    assert "/api/identify" in script and "webkitGetAsEntry" in script
    assert "T1 with contrast" in script and "DICOM" in script
    assert 'id="job-progress"' in html
    assert "Loading" in script and "Meshing" in script


def test_upload_rejects_mismatched_grid(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    client = create_app(root / "outputs", root / "cases").test_client()
    buffers = {}
    for suffix in ("t2f", "t1n", "t1c"):
        path = root / f"{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), path)
        buffers[suffix] = io.BytesIO(path.read_bytes())
    path = root / "t2w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 4, 4), np.float32), np.eye(4)), path)
    buffers["t2w"] = io.BytesIO(path.read_bytes())
    response = client.post(
        "/api/cases",
        data={"case_id": "mismatch", **{suffix: (buf, f"{suffix}.nii.gz") for suffix, buf in buffers.items()}},
    )
    assert response.status_code == 400
    assert "T2" in response.json["error"]
    job = client.get("/api/cases/mismatch/job")
    assert job.status_code == 404


def test_upload_two_cases_include_tissue_gm_wm(tmp_path=None):
    import tempfile

    from recon.postprocess import PostConfig

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    app = create_app(root / "outputs", root / "cases")
    shape = (16, 16, 16)
    affine = np.eye(4, dtype=np.float64)

    def fake_predict(network, image, *args, **kwargs):
        volume = np.asarray(image)
        grid = volume.shape[1:]
        probs = np.zeros((3,) + grid, dtype=np.float32)
        probs[2, 5:11, 5:11, 5:11] = 0.95
        return probs

    import recon.infer as infer
    import recon.scene as scene
    import scripts.reconstruct as reconstruct
    from recon.postprocess import PostConfig as Config

    originals = {
        "predict": infer.predict_probs,
        "load": infer.load_model,
        "pre": infer.apply_model_preprocessing,
        "config": Config.for_model,
        "ckpt": reconstruct.default_checkpoint,
        "cache": scene.default_anatomy_cache_root,
    }
    infer.predict_probs = fake_predict
    infer.load_model = lambda *args, **kwargs: object()
    infer.apply_model_preprocessing = lambda image, affine, model: (image, affine)
    Config.for_model = classmethod(
        lambda cls, model=None: PostConfig(
            min_voxels_et=0,
            min_voxels_tc=0,
            min_voxels_wt=0,
            et_min_voxels=0,
            brain_closing_iterations=1,
        )
    )
    reconstruct.default_checkpoint = lambda: Path("checkpoint.pt")
    scene.default_anatomy_cache_root = lambda: root / "anatomy_cache"

    def parc_for(seed):
        parc = np.zeros(shape, dtype=np.int16)
        parc[1:8, 1:8, 1:8] = 3
        parc[8:15, 1:8, 1:8] = 42
        parc[1:15, 8:15, 1:8] = 2
        parc[2:4, 2:4, 10:12] = 10 + seed
        return parc

    client = app.test_client()
    manifests = {}
    try:
        for case_id, seed in (("BraTS20_Training_004", 0), ("BraTS20_Training_022", 1)):
            brain = np.ones(shape, dtype=np.float32) * (seed + 1)
            data = {
                "case_id": case_id,
                "anatomy": "1",
            }
            files = {}
            for suffix in ("t2f", "t1n", "t1c", "t2w"):
                path = root / f"{case_id}_{suffix}.nii.gz"
                nib.save(nib.Nifti1Image(brain, affine), path)
                files[suffix] = (io.BytesIO(path.read_bytes()), f"{suffix}.nii.gz")
            synth = root / f"{case_id}_synthseg.nii.gz"
            nib.save(nib.Nifti1Image(parc_for(seed).astype(np.float32), affine), synth)
            files["synthseg"] = (io.BytesIO(synth.read_bytes()), "synthseg.nii.gz")
            data.update(files)
            response = client.post("/api/cases", data=data)
            assert response.status_code == 202, response.json
            job = _wait_job(client, response.json["job"], timeout=60)
            assert job.json["status"] == "done", job.json
            manifest = client.get(f"/api/cases/{case_id}/manifest.json")
            assert manifest.status_code == 200
            manifests[case_id] = manifest.json
    finally:
        infer.predict_probs = originals["predict"]
        infer.load_model = originals["load"]
        infer.apply_model_preprocessing = originals["pre"]
        Config.for_model = originals["config"]
        reconstruct.default_checkpoint = originals["ckpt"]
        scene.default_anatomy_cache_root = originals["cache"]

    for case_id, meta in manifests.items():
        structures = meta["structures"]
        assert structures["Left Cortex"]["group"] == "Tissue", case_id
        assert structures["White Matter"]["group"] == "Tissue", case_id
        assert structures["Left Cortex"]["volume_cm3"] > 0, case_id
        assert structures["White Matter"]["volume_cm3"] > 0, case_id
        assert structures["Grey matter total"]["volume_cm3"] > 0, case_id
        assert meta.get("anatomy_warning") is None, case_id


if __name__ == "__main__":
    test_slice_png_places_the_voxel_on_the_second_free_axis()
    test_api_serves_scene_manifest_and_slices()
    test_viewer_page_has_groups_clipping_and_tract_note()
    test_upload_calls_build_scene()
    test_file_names_are_matched_to_scans()
    test_identify_endpoint_and_plain_missing_message()
    test_upload_page_guides_without_modality_jargon()
    test_upload_rejects_mismatched_grid()
    test_upload_two_cases_include_tissue_gm_wm()
    print("viewer ok")
