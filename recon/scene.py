"""Scene export: tumor compartments, optional anatomy, optional atlas tracts.

``reconstruct_tumor`` still builds the nested WT / TC / ET shells.
``build_scene`` runs inference and post-processing and writes the case folder
under ``outputs/<case_id>/``. Tumor nodes in that scene are the exclusive
compartments: edema (label 2), necrosis (label 1), and enhancing (label 3).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

from recon.anatomy import ANATOMY_COLORS, add_anatomy_meshes
from recon.evaluate import check_mesh_volume
from recon.mesh import check_mesh, mesh_from_mask, region_shells

REGION_COLORS = {
    "edema": (255, 220, 0, 90),
    "core": (220, 40, 40, 255),
    "enhancing": (60, 120, 255, 255),
}


def reconstruct_tumor(
    label,
    affine,
    out_path="tumor.glb",
    *,
    include_anatomy=False,
    brain_mask=None,
    synthseg=None,
):
    """Export tumor shells. Anatomy is opt-in and off by default.

    ``include_anatomy`` is the UI toggle. When it is set, ``brain_mask`` is
    meshed as a transparent outer shell and ``synthseg`` (a SynthSeg label
    volume on the same grid) supplies the deep structures.
    """
    scene = trimesh.Scene()
    for name, mask in region_shells(label).items():
        mesh = mesh_from_mask(mask, affine)
        if mesh is None:
            continue
        check_mesh_volume(mesh, mask, affine, name)
        mesh.visual.face_colors = REGION_COLORS[name]
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    if include_anatomy:
        add_anatomy_meshes(scene, affine, label, brain_mask=brain_mask, synthseg=synthseg)
    scene.export(out_path)
    return scene


def _rgba(rgb, opacity: float) -> tuple[int, int, int, int]:
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(round(float(opacity) * 255)))


# Exclusive tumor compartments in the exported scene.
SCENE_COLORS = {
    "edema": _rgba((255, 220, 0), 0.35),
    "necrosis": (220, 40, 40, 255),
    "enhancing": (60, 120, 255, 255),
    "brain": _rgba((180, 180, 180), 0.10),
}

_TRACT_RGB = (
    (220, 60, 60),
    (40, 160, 90),
    (50, 110, 220),
    (230, 140, 30),
    (150, 70, 190),
    (30, 170, 170),
    (200, 70, 130),
    (170, 170, 40),
    (120, 90, 60),
    (70, 130, 190),
)


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


def tumor_compartment_masks(label) -> dict[str, np.ndarray]:
    """Edema, necrosis, and enhancing as exclusive labels. Label 4 counts as enhancing."""
    lab = np.rint(np.squeeze(np.asarray(label))).astype(np.int16)
    if lab.ndim != 3:
        raise ValueError(f"Expected a 3D label, got {lab.shape}")
    return {
        "edema": lab == 2,
        "necrosis": lab == 1,
        "enhancing": (lab == 3) | (lab == 4),
    }


def tract_color(name: str) -> tuple[int, int, int, int]:
    """Stable distinct color at 60% opacity."""
    slot = sum(name.encode("utf-8")) % len(_TRACT_RGB)
    return _rgba(_TRACT_RGB[slot], 0.60)


def save_pred_seg(label, reference_path, out_path) -> None:
    """Write the RAS label back into the reference NIfTI's voxel grid and affine."""
    orig = nib.load(str(reference_path))
    canon = nib.as_closest_canonical(orig)
    transform = ornt_transform(io_orientation(canon.affine), io_orientation(orig.affine))
    native = apply_orientation(np.asarray(label), transform)
    if native.shape != orig.shape[:3]:
        raise ValueError(
            f"Reoriented prediction shape {native.shape} does not match {reference_path} shape {orig.shape[:3]}"
        )
    nib.save(nib.Nifti1Image(np.asarray(native, dtype=np.uint8), orig.affine), str(out_path))


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _structure_record(name, mesh, mask, affine, color, source, proximity=None, extra=None):
    watertight, err = check_mesh(mesh, mask, affine)
    voxel_mm3 = float(np.asarray(mask).sum()) * abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    mesh_mm3 = abs(float(mesh.volume)) if watertight else None
    record = {
        "volume_cm3": voxel_mm3 / 1000.0,
        "mesh_volume_cm3": None if mesh_mm3 is None else mesh_mm3 / 1000.0,
        "volume_error_pct": _json_number(err),
        "watertight": bool(watertight),
        "color": [int(channel) for channel in color],
        "source": source,
        "proximity": proximity,
        "file": f"{name}.glb",
    }
    if extra:
        record.update(extra)
    return record


def _paint(mesh, color):
    mesh.visual.face_colors = color
    return mesh


def _add_mesh(scene, out_dir, name, mesh):
    scene.add_geometry(mesh, geom_name=name, node_name=name)
    mesh.export(out_dir / f"{name}.glb")


def _load_synthseg(path, shape):
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
    if data.shape != tuple(shape):
        raise ValueError(f"SynthSeg shape {data.shape} does not match the case grid {tuple(shape)}")
    return data


def _find_synthseg(case_dir: Path | None, explicit):
    if explicit is not None:
        return Path(explicit)
    if case_dir is None:
        return None
    hits = sorted(case_dir.glob("*synthseg*.nii.gz")) + sorted(case_dir.glob("*synthseg*.nii"))
    return hits[0] if hits else None


def _write_canonical_t1(case_dir: Path, dest: Path):
    from recon.io import _find_volume, _read_canonical

    raw, affine = _read_canonical(_find_volume(case_dir, "t1n"), label=False)
    nib.save(nib.Nifti1Image(np.asarray(raw, dtype=np.float32), affine), str(dest))
    return affine, raw.shape


def build_scene(
    case,
    *,
    anatomy=False,
    tracts=False,
    checkpoint=None,
    device=None,
    output_root=None,
    synthseg=None,
    config=None,
):
    """Run the case and write ``outputs/<case_id>/``.

    ``case`` is a BraTS folder or a ``load_case`` dict. Anatomy and tracts stay
    off unless the flags are set. Tumor nodes are edema (yellow, 35% opacity),
    necrosis (red), and enhancing (blue). The brain shell is gray at 10%
    opacity. Tract meshes use distinct colors at 60% opacity.
    """
    from recon.postprocess import PostConfig, brain_mask_from_image, postprocess

    if isinstance(case, (str, Path)):
        from recon.io import load_case

        case_dir = Path(case)
        loaded = load_case(case_dir)
        loaded["case_dir"] = case_dir
        case = loaded
    else:
        case_dir = Path(case["case_dir"]) if case.get("case_dir") else None

    if config is None:
        config = PostConfig.from_yaml()
    if "probs" in case:
        probs = np.asarray(case["probs"], dtype=np.float32)
    else:
        if checkpoint is None:
            raise ValueError("checkpoint is required when the case has no 'probs'")
        from recon.infer import load_model, predict_probs

        model = load_model(checkpoint, device)
        probs = predict_probs(model, case["image"])

    brain = brain_mask_from_image(case["image"], config)
    label = postprocess(probs, brain, config)
    affine = np.asarray(case["affine"], dtype=np.float64)
    case_id = str(case["case_id"])
    out_dir = Path(output_root) if output_root is not None else default_output_root()
    out_dir = out_dir / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if case_dir is None:
        raise ValueError("case_dir is required to write pred_seg.nii.gz in the original image space")
    from recon.io import _find_volume

    reference = _find_volume(case_dir, "t1n")
    save_pred_seg(label, reference, out_dir / "pred_seg.nii.gz")

    scene = trimesh.Scene()
    structures = {}

    for name, mask in tumor_compartment_masks(label).items():
        if not np.any(mask):
            continue
        mesh = mesh_from_mask(mask, affine)
        if mesh is None:
            continue
        color = SCENE_COLORS[name]
        _paint(mesh, color)
        _add_mesh(scene, out_dir, name, mesh)
        structures[name] = _structure_record(name, mesh, mask, affine, color, "model")

    removed = {}
    anatomy_removed = {}
    if anatomy:
        color = SCENE_COLORS["brain"]
        brain_mesh = None
        from recon.anatomy import mesh_brain_surface

        brain_mesh = mesh_brain_surface(brain, affine)
        if brain_mesh is not None:
            _paint(brain_mesh, color)
            _add_mesh(scene, out_dir, "brain", brain_mesh)
            structures["brain"] = _structure_record("brain", brain_mesh, brain, affine, color, "image")

        parc = case.get("synthseg")
        if parc is None:
            found = _find_synthseg(case_dir, synthseg)
            if found is not None:
                parc = _load_synthseg(found, label.shape)
        anatomy_removed = {}
        if parc is None:
            from recon.anatomy import parcellate_case

            with tempfile.TemporaryDirectory() as tmp:
                t1_ras = Path(tmp) / "t1n_ras.nii.gz"
                _write_canonical_t1(case_dir, t1_ras)
                parc, cleaned, anatomy_removed = parcellate_case(
                    t1_ras,
                    label,
                    affine,
                    case_id=case_id,
                    device=device,
                )
        else:
            from recon.anatomy import excise_tumor

            parc = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
            cleaned, anatomy_removed = excise_tumor(parc, label)
        from recon.anatomy import cleaned_structure_masks, save_anatomy_seg

        anatomy_path = out_dir / "anatomy_seg.nii.gz"
        save_anatomy_seg(cleaned, reference, anatomy_path, out_dir / "pred_seg.nii.gz")
        masks, removed = cleaned_structure_masks(parc, label)
        if parc is not None:
            for name, mask in masks.items():
                if not np.any(mask):
                    continue
                mesh = mesh_from_mask(mask, affine)
                if mesh is None:
                    continue
                color = ANATOMY_COLORS[name]
                _paint(mesh, color)
                _add_mesh(scene, out_dir, name, mesh)
                structures[name] = _structure_record(
                    name,
                    mesh,
                    mask,
                    affine,
                    color,
                    "synthseg",
                    extra={"removed_voxels": int(removed.get(name, 0))},
                )

    proximity = {}
    if tracts:
        from recon.tracts import compute_tract_proximity, warp_atlas_tracts

        with tempfile.TemporaryDirectory() as tmp:
            t1_ras = Path(tmp) / "t1n_ras.nii.gz"
            _write_canonical_t1(case_dir, t1_ras)
            warped = warp_atlas_tracts(t1_ras, label, case_id=case_id)
        tract_affine = affine
        for tract in warped.values():
            if tract.affine is not None:
                tract_affine = np.asarray(tract.affine, dtype=np.float64)
                break
        proximity = compute_tract_proximity(label, warped, tract_affine)
        for name, tract in warped.items():
            if tract.mesh is None:
                continue
            color = tract_color(name)
            _paint(tract.mesh, color)
            _add_mesh(scene, out_dir, name, tract.mesh)
            structures[name] = _structure_record(
                name,
                tract.mesh,
                tract.mask,
                tract_affine,
                color,
                "atlas-estimated",
                proximity=proximity[name],
            )

    scene.export(out_dir / "scene.glb")
    manifest = {
        "case_id": case_id,
        "case_dir": None if case_dir is None else str(case_dir),
        "postprocess": asdict(config),
        "structures": structures,
        "tract_proximity": proximity,
        "removed_voxels": {name: int(count) for name, count in removed.items()},
        "anatomy_removed_voxels": {name: int(count) for name, count in anatomy_removed.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
