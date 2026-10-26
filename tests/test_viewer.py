"""Slice PNGs, case routes, and the viewer page."""

import io
import json
import sys
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
    volume = np.zeros((10, 6, 12), dtype=np.float32)
    volume[5, 2, 8] = 100
    image = Image.open(io.BytesIO(_slice_png(volume, axis=0, index=5)))
    assert image.size == (6, 12)
    assert image.getpixel((2, 8)) == 255
    assert image.getpixel((0, 0)) == 0


def test_api_serves_scene_manifest_and_slices(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    outputs = root / "outputs"
    cases = root / "cases"
    shape = (8, 7, 6)
    data = np.zeros(shape, dtype=np.float32)
    data[1, 2, 3] = 50
    _nifti(cases / "demo" / "t2f.nii.gz", data)
    _nifti(cases / "demo" / "t1c.nii.gz", data + 1)
    case_out = outputs / "demo"
    case_out.mkdir(parents=True)
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

    png = client.get("/api/cases/demo/slice?modality=t1c&axis=0&index=1")
    assert png.status_code == 200
    assert png.mimetype == "image/png"
    image = Image.open(io.BytesIO(png.data))
    assert image.size == (shape[1], shape[2])
    assert image.getpixel((2, 3)) == 255

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
    assert "depthWrite" in script
    assert "enhancing" in script and "renderOrder" in script
    page = create_app().test_client().get("/viewer")
    assert page.status_code == 200
    assert b"viewer.js" in page.data


def test_upload_calls_build_scene(tmp_path=None):
    import tempfile

    root = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    app = create_app(root / "outputs", root / "cases")
    calls = {}

    def fake_build(case, anatomy=False, tracts=False, checkpoint=None, output_root=None, synthseg=None):
        calls["case"] = str(case)
        calls["anatomy"] = anatomy
        calls["tracts"] = tracts
        calls["checkpoint"] = checkpoint
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
    finally:
        scene.build_scene = original_build
        reconstruct.default_checkpoint = original_ckpt

    assert response.status_code == 200
    assert response.json["viewer"] == "/viewer?case=scan1"
    assert calls["anatomy"] is True
    assert calls["tracts"] is True
    assert Path(calls["case"]).name == "scan1"
    assert (root / "cases" / "scan1" / "seg.nii.gz").is_file()


if __name__ == "__main__":
    test_slice_png_places_the_voxel_on_the_second_free_axis()
    test_api_serves_scene_manifest_and_slices()
    test_viewer_page_has_groups_clipping_and_tract_note()
    test_upload_calls_build_scene()
    print("viewer ok")
