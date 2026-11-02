"""Series selection, anonymized headers, and resume checks for real-MRI ingest."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.ingest_real import (
    choose_channels,
    choose_reference,
    coverage_reject_reasons,
    mi_accepted,
    pick_channel_series,
    public_series,
    series_extent,
    strip_nifti_identity,
    study_is_done,
)
from scripts.inventory_real import _public_series


def _series(**kwargs):
    item = {
        "usable_as_input": True,
        "source_path": Path("unused"),
        "source_kind": "dicom",
        "series_description": "eT1W_SE",
        "acquisition": "2D",
        "slice_thickness_mm": 5.0,
        "n_slices": 23,
        "orientation": "axial",
        "modality": "T1",
    }
    item.update(kwargs)
    return item


def test_reference_prefers_thinnest_3d_t1_over_t1c():
    series = [
        _series(modality="T1", acquisition="2D", slice_thickness_mm=5.0, n_slices=23),
        _series(
            modality="T1",
            acquisition="3D",
            slice_thickness_mm=1.6,
            n_slices=170,
            series_description="sT1W_3D_HR",
        ),
        _series(
            modality="T1",
            acquisition="3D",
            slice_thickness_mm=1.1,
            n_slices=265,
            series_description="eT1W_3D_TFE_ref",
        ),
        _series(modality="T1c", acquisition="2D", slice_thickness_mm=5.0, series_description="T1W_SPIR CE"),
        _series(modality="FLAIR", acquisition="2D", slice_thickness_mm=5.0, series_description="eT2W_FLAIR"),
        _series(modality="T2", acquisition="2D", slice_thickness_mm=5.0, series_description="eT2W_TSE"),
    ]
    chosen = choose_channels(series)
    assert chosen["T1"]["series_description"] == "eT1W_3D_TFE_ref"
    modality, reason = choose_reference(chosen)
    assert modality == "T1"
    assert reason == "3D T1 (thinnest)"


def test_reference_falls_back_t1c_then_2d_t1_then_flair():
    t1c = _series(modality="T1c", series_description="T1W_SPIR CE")
    t1 = _series(modality="T1", acquisition="2D")
    flair = _series(modality="FLAIR", series_description="eT2W_FLAIR")
    t2 = _series(modality="T2")

    modality, reason = choose_reference({"T1c": t1c, "T1": t1, "FLAIR": flair, "T2": t2})
    assert modality == "T1c"
    assert "T1c" in reason

    modality, reason = choose_reference({"T1": t1, "FLAIR": flair, "T2": t2})
    assert modality == "T1"
    assert reason.startswith("2D T1")

    modality, reason = choose_reference({"FLAIR": flair, "T2": t2})
    assert modality == "FLAIR"


def test_pick_channel_ignores_unusable_and_missing_source():
    usable = _series(modality="FLAIR", n_slices=25, series_description="keep")
    skipped = _series(modality="FLAIR", n_slices=99, usable_as_input=False, series_description="skip")
    nosrc = _series(modality="FLAIR", n_slices=40, source_path=None, series_description="nosrc")
    picked = pick_channel_series([skipped, nosrc, usable], "FLAIR")
    assert picked["series_description"] == "keep"


def test_public_series_has_no_paths_or_ids():
    item = _series(source_path=Path(r"C:\Users\Someone\MRI DATA\PATIENT_NAME"))
    item["study_uid_hash"] = "study_abc"
    published = public_series(item)
    blob = json.dumps(published)
    assert "PATIENT_NAME" not in blob
    assert "source_path" not in published
    assert "MRI DATA" not in blob
    inventory = _public_series(item)
    assert "source_path" not in inventory
    assert "source_kind" not in inventory


def test_strip_nifti_identity_clears_header_text(tmp_path):
    affine = np.eye(4)
    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine)
    image.header["descrip"] = b"John Doe 50Y 2020-01-01"
    image.header["aux_file"] = b"secret.img"
    image.header["db_name"] = b"HOSPITALID"
    path = tmp_path / "raw.nii.gz"
    nib.save(image, str(path))
    strip_nifti_identity(path)
    cleaned = nib.load(str(path))
    for key in ("descrip", "aux_file", "db_name"):
        value = bytes(cleaned.header[key]).replace(b"\x00", b"").strip()
        assert value == b"", (key, value)


def test_mi_accepted_is_strictly_better_than_threshold():
    assert mi_accepted(-0.30) is True
    assert mi_accepted(-0.16) is True
    assert mi_accepted(-0.15) is False
    assert mi_accepted(-0.046) is False
    assert mi_accepted(None) is False


def _geom(*, fov_mm, n_slices, thickness, orientation="axial", rows=256, **kwargs):
    spacing = fov_mm / float(rows)
    item = _series(
        n_slices=n_slices,
        slice_thickness_mm=thickness,
        orientation=orientation,
        in_plane_spacing_mm=f"{spacing:.4f}x{spacing:.4f}",
        matrix=f"{rows}x{rows}",
        **kwargs,
    )
    return item


def test_rejects_small_fov_drive_and_sella():
    drive = _geom(
        fov_mm=150,
        n_slices=140,
        thickness=1.2,
        orientation="sagittal",
        modality="T2",
        series_description="T2W_3D_DRIVE",
        acquisition="3D",
    )
    sella = _geom(
        fov_mm=111,
        n_slices=20,
        thickness=2.0,
        orientation="sagittal",
        modality="T1c",
        series_description="eT1W_3D_TFE sella",
        acquisition="3D",
    )
    axial_t2 = _geom(
        fov_mm=226,
        n_slices=23,
        thickness=5.0,
        orientation="axial",
        modality="T2",
        series_description="eT2W_TSE SENSE",
    )
    rejected = []
    picked = pick_channel_series([drive, axial_t2], "T2", ref_si_mm=140.0, rejected=rejected)
    assert picked["series_description"] == "eT2W_TSE SENSE"
    assert any("T2W_3D_DRIVE" in item["series_description"] for item in rejected)
    assert any("FOV" in item["reason"] for item in rejected)

    rejected_t1c = []
    picked_t1c = pick_channel_series([sella], "T1c", ref_si_mm=140.0, rejected=rejected_t1c)
    assert picked_t1c is None
    assert rejected_t1c and "FOV" in rejected_t1c[0]["reason"]
    assert series_extent(drive)["inplane_fov_mm"] < 180


def test_rejects_thin_si_slab():
    slab = _geom(
        fov_mm=200,
        n_slices=16,
        thickness=2.0,
        orientation="axial",
        modality="T2",
        series_description="PITUITARY",
    )
    reasons = coverage_reject_reasons(slab, ref_si_mm=140.0)
    assert any("SI" in reason for reason in reasons)
    whole = _geom(
        fov_mm=220,
        n_slices=23,
        thickness=5.0,
        orientation="axial",
        modality="T2",
        series_description="eT2W_TSE SENSE",
    )
    assert coverage_reject_reasons(whole, ref_si_mm=140.0) == []


def test_prefers_axial_then_thinnest_after_fov_filter():
    sag = _geom(
        fov_mm=220,
        n_slices=40,
        thickness=1.0,
        orientation="sagittal",
        modality="T2",
        series_description="T2W_SPIR_sag",
    )
    axial_thick = _geom(
        fov_mm=220,
        n_slices=23,
        thickness=5.0,
        orientation="axial",
        modality="T2",
        series_description="eT2W_TSE 5mm",
    )
    axial_thin = _geom(
        fov_mm=220,
        n_slices=46,
        thickness=3.0,
        orientation="axial",
        modality="T2",
        series_description="eT2W_TSE 3mm",
    )
    picked = pick_channel_series([sag, axial_thick, axial_thin], "T2", ref_si_mm=120.0)
    assert picked["series_description"] == "eT2W_TSE 3mm"


def test_study_is_done_requires_json_mask_png_and_chosen_volumes(tmp_path):
    ready = tmp_path / "04_ready"
    ready.mkdir()
    assert study_is_done(ready) is False
    (ready / "preprocess.json").write_text(
        json.dumps({"series_chosen": {"flair": {}, "t1": {}, "t1c": {}, "t2": {}}}),
        encoding="utf-8",
    )
    affine = np.eye(4)
    empty = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine)
    for name in ("brain_mask", "flair", "t1", "t1c", "t2"):
        nib.save(empty, str(ready / f"{name}.nii.gz"))
    (ready / "registration_check.png").write_bytes(b"png")
    assert study_is_done(ready) is True
    (ready / "t1c.nii.gz").unlink()
    assert study_is_done(ready) is False


if __name__ == "__main__":
    import tempfile

    test_reference_prefers_thinnest_3d_t1_over_t1c()
    test_reference_falls_back_t1c_then_2d_t1_then_flair()
    test_pick_channel_ignores_unusable_and_missing_source()
    test_public_series_has_no_paths_or_ids()
    test_strip_nifti_identity_clears_header_text(Path(tempfile.mkdtemp()))
    test_mi_accepted_is_strictly_better_than_threshold()
    test_study_is_done_requires_json_mask_png_and_chosen_volumes(Path(tempfile.mkdtemp()))
    test_rejects_small_fov_drive_and_sella()
    test_rejects_thin_si_slab()
    test_prefers_axial_then_thinnest_after_fov_filter()
    print("ingest_real ok")
