"""Scene export: labelled brain first, then tumor, then the carved anatomy.

``build_scene`` loads the four RAS sequences, meshes the skull-stripped brain
and the SynthSeg parcellation, and only then runs tumor detection. Structures
the tumor overlaps are carved and meshed again. ``reconstruct_tumor`` still
builds the nested WT / TC / ET shells.
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

from recon.anatomy import add_anatomy_meshes, centroid_mm, ensure_scene_group, mesh_parcellation_labels
from recon.evaluate import check_mesh_volume
from recon.labels import (
    BRAIN,
    COMPARTMENTS,
    EDEMA,
    ENHANCING,
    NECROSIS,
    SHELLS,
    get,
    legend_entries,
    rgba,
)
from recon.mesh import check_mesh, mesh_from_mask, region_shells

# Nested WT / TC / ET shells. RGB and names come from ``recon.labels``.
REGION_COLORS = {spec.mesh_name: rgba(spec) for spec in SHELLS}


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
    volume on the same grid) supplies one mesh per label. Colors and names
    come from ``recon.labels``.
    """
    scene = trimesh.Scene()
    for name, mask in region_shells(label).items():
        mesh = mesh_from_mask(mask, affine)
        if mesh is None:
            continue
        check_mesh_volume(mesh, mask, affine, name)
        mesh.visual.face_colors = REGION_COLORS[name]
        spec = get(name)
        ensure_scene_group(scene, spec.group)
        scene.add_geometry(mesh, geom_name=name, node_name=name, parent_node_name=spec.group)
    if include_anatomy:
        add_anatomy_meshes(scene, affine, label, brain_mask=brain_mask, synthseg=synthseg)
    scene.export(out_path)
    return scene


# Exclusive tumor compartments plus the outer brain shell.
SCENE_COLORS = {spec.mesh_name: rgba(spec) for spec in (*COMPARTMENTS, BRAIN)}


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


def tumor_compartment_masks(label) -> dict[str, np.ndarray]:
    """Edema, necrosis, and enhancing as exclusive labels. Label 4 counts as enhancing."""
    lab = np.rint(np.squeeze(np.asarray(label))).astype(np.int16)
    if lab.ndim != 3:
        raise ValueError(f"Expected a 3D label, got {lab.shape}")
    return {
        EDEMA: lab == 2,
        NECROSIS: lab == 1,
        ENHANCING: (lab == 3) | (lab == 4),
    }


def tract_color(name: str) -> tuple[int, int, int, int]:
    """Tract color from ``recon.labels``. Every tract is atlas-estimated."""
    spec = get(name)
    if spec.source != "atlas-estimated":
        raise KeyError(f"{name} is not an atlas tract")
    return rgba(spec)


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


def _catalog_extra(spec, **more):
    extra = {
        "short_name": spec.short_name,
        "group": spec.group,
        "legend": spec.legend_name,
        "label_id": spec.label_id,
    }
    extra.update(more)
    return extra


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
        "centroid_mm": centroid_mm(mask, affine),
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


def _add_mesh(scene, out_dir, name, mesh, group=None):
    """Add a named node. ``group`` is the parent in ``scene.glb``."""
    ensure_scene_group(scene, group)
    scene.add_geometry(mesh, geom_name=name, node_name=name, parent_node_name=group)
    mesh.export(out_dir / f"{name}.glb")


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


def _scan_shape(case) -> tuple[int, ...]:
    image = np.asarray(case["image"])
    if image.ndim != 4:
        raise ValueError(f"Expected four MRI sequences shaped (4, H, W, D), got {image.shape}")
    return tuple(int(v) for v in image.shape[1:])


def _record_mesh(scene, out_dir, structures, exported, spec, mesh, mask, affine, *, proximity=None, extra=None):
    """Paint, parent the node by anatomical group, and store name, color, and centroid."""
    color = rgba(spec)
    _paint(mesh, color)
    _add_mesh(scene, out_dir, spec.mesh_name, mesh, spec.group)
    payload = _catalog_extra(spec)
    if extra:
        payload.update(extra)
    structures[spec.mesh_name] = _structure_record(
        spec.mesh_name,
        mesh,
        mask,
        affine,
        color,
        spec.source,
        proximity=proximity,
        extra=payload,
    )
    for index, item in enumerate(exported):
        if item.mesh_name == spec.mesh_name:
            exported[index] = spec
            return
    exported.append(spec)


def _remove_structure(scene, out_dir, structures, exported, name):
    if name in scene.geometry:
        scene.delete_geometry(name)
    path = out_dir / f"{name}.glb"
    if path.is_file():
        path.unlink()
    structures.pop(name, None)
    exported[:] = [item for item in exported if item.mesh_name != name]


def _raw_parcellation(case, case_dir, synthseg, shape, affine, device, case_id):
    """SynthSeg labels on the scan grid, before the tumor is removed.

    A file on another grid is resampled with nearest neighbours. A volume that
    is already on the scan grid is kept. SynthSeg itself runs only when no
    parcellation was supplied. The tumor is not known yet, so nothing is carved.
    """
    from recon.anatomy import parcellate_case, resample_parcellation

    parc = case.get("synthseg")
    source_affine = case.get("synthseg_affine")
    if parc is None and case_dir is not None:
        found = _find_synthseg(case_dir, synthseg)
        if found is not None:
            image = nib.as_closest_canonical(nib.load(str(found)))
            parc = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
            source_affine = np.asarray(image.affine, dtype=np.float64)
    if parc is not None:
        parc = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
        if parc.shape == tuple(shape) and source_affine is None:
            return parc
        if source_affine is None:
            raise ValueError(
                f"SynthSeg shape {parc.shape} does not match the scan grid {tuple(shape)}"
            )
        return resample_parcellation(parc, source_affine, shape, affine)
    if case_dir is None:
        raise ValueError("case_dir is required to run SynthSeg")
    with tempfile.TemporaryDirectory() as tmp:
        t1_ras = Path(tmp) / "t1n_ras.nii.gz"
        _write_canonical_t1(case_dir, t1_ras)
        # Empty tumor: registration and parcellation happen before detection.
        empty = np.zeros(shape, dtype=np.uint8)
        raw, _cleaned, _removed = parcellate_case(
            t1_ras,
            empty,
            affine,
            case_id=case_id,
            device=device,
        )
    return raw


def _mesh_intact_anatomy(scene, out_dir, structures, exported, parc, affine):
    """One FreeSurfer-coloured mesh per label. The tumor is not in this model yet."""
    for structure, mask, mesh in mesh_parcellation_labels(parc, affine):
        _record_mesh(
            scene,
            out_dir,
            structures,
            exported,
            structure,
            mesh,
            mask,
            affine,
            extra={"removed_voxels": 0},
        )


def _carve_and_remesh(scene, out_dir, structures, exported, raw, tumor_label, affine):
    """Cut the tumor out of the parcellation and replace meshes that changed."""
    from recon.anatomy import excise_tumor
    from recon.labels import anatomy as anatomy_by_id

    cleaned, anatomy_removed = excise_tumor(raw, tumor_label)
    raw_labels = np.rint(np.squeeze(np.asarray(raw))).astype(np.int16)
    cleaned_labels = np.rint(np.squeeze(np.asarray(cleaned))).astype(np.int16)
    removed = {}
    affected = []
    for label_id in sorted(int(value) for value in np.unique(raw_labels) if int(value) != 0):
        before = int(np.count_nonzero(raw_labels == label_id))
        after = int(np.count_nonzero(cleaned_labels == label_id))
        spec = anatomy_by_id(label_id)
        if spec is None:
            continue
        removed[spec.mesh_name] = before - after
        if spec.mesh_name in structures:
            structures[spec.mesh_name]["removed_voxels"] = before - after
        if before != after:
            affected.append(label_id)
    refreshed = set()
    if affected:
        subset = np.where(np.isin(cleaned_labels, affected), cleaned_labels, 0).astype(np.int16)
        for structure, mask, mesh in mesh_parcellation_labels(subset, affine):
            _remove_structure(scene, out_dir, structures, exported, structure.mesh_name)
            _record_mesh(
                scene,
                out_dir,
                structures,
                exported,
                structure,
                mesh,
                mask,
                affine,
                extra={"removed_voxels": int(removed.get(structure.mesh_name, 0))},
            )
            refreshed.add(structure.mesh_name)
        for label_id in affected:
            spec = anatomy_by_id(label_id)
            if spec is not None and spec.mesh_name not in refreshed:
                _remove_structure(scene, out_dir, structures, exported, spec.mesh_name)
    return cleaned, removed, anatomy_removed


def tumor_location_summary(parc, tumor_label, affine, *, margin_mm=10.0) -> dict:
    """Lobes and structures the whole tumor or enhancing tumor overlaps or nears.

    Distance is the minimum spacing-aware distance from the structure to that
    tumor region. Overlap is recorded as 0 mm. Anything farther than
    ``margin_mm`` is omitted. The parcellation must still contain the tumor
    voxels, so pass the labels from before they are carved out.
    """
    from nibabel.affines import voxel_sizes
    from scipy import ndimage

    from recon.evaluate import regions_from_label
    from recon.labels import anatomy as anatomy_by_id

    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    if labels.ndim != 3:
        raise ValueError(f"Parcellation must be 3D, got {labels.shape}")
    regions = regions_from_label(tumor_label)
    if regions["WT"].shape != labels.shape:
        raise ValueError(
            f"Tumor shape {regions['WT'].shape} does not match the parcellation {labels.shape}"
        )
    spacing = tuple(float(abs(value)) for value in voxel_sizes(np.asarray(affine, dtype=np.float64)))
    fields = {}
    for name in ("WT", "ET"):
        mask = regions[name]
        fields[name] = ndimage.distance_transform_edt(~mask, sampling=spacing) if np.any(mask) else None

    summary = {"margin_mm": float(margin_mm), "WT": [], "ET": []}
    for label_id in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        spec = anatomy_by_id(label_id)
        if spec is None:
            continue
        structure = labels == label_id
        for region in ("WT", "ET"):
            field = fields[region]
            if field is None:
                continue
            distance = float(field[structure].min())
            if distance > float(margin_mm):
                continue
            overlaps = distance <= 1e-3
            summary[region].append(
                {
                    "name": spec.mesh_name,
                    "group": spec.group,
                    "label_id": int(spec.label_id),
                    "relation": "overlaps" if overlaps else "within_10mm",
                    "distance_mm": 0.0 if overlaps else round(distance, 3),
                }
            )
    for region in ("WT", "ET"):
        summary[region].sort(key=lambda item: (item["distance_mm"], item["name"]))
    return summary


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
    model="ours",
):
    """Write ``outputs/<case_id>/`` with the brain model ahead of the tumor.

    The scan's four sequences are already RAS when ``load_case`` reads them,
    and the returned affine is the millimetre grid for every mesh. With
    ``anatomy``, the nonzero brain mask and SynthSeg labels are meshed before
    tumor detection. Overlapping structures are then carved and meshed again.
    """
    from recon.postprocess import PostConfig, brain_mask_from_image, postprocess

    # 1. Load the scan. Four sequences, RAS orientation, one affine in millimetres.
    if isinstance(case, (str, Path)):
        from recon.infer import ABHINAV_V3
        from recon.io import load_case

        case_dir = Path(case)
        loaded = load_case(case_dir, normalize=model != ABHINAV_V3)
        loaded["case_dir"] = case_dir
        case = loaded
    else:
        case_dir = Path(case["case_dir"]) if case.get("case_dir") else None

    if case_dir is None:
        raise ValueError("case_dir is required to write pred_seg.nii.gz in the original image space")
    from recon.io import _find_volume

    reference = _find_volume(case_dir, "t1n")
    if config is None:
        config = PostConfig.for_model(model)
    affine = np.asarray(case["affine"], dtype=np.float64)
    shape = _scan_shape(case)
    case_id = str(case["case_id"])
    out_dir = Path(output_root) if output_root is not None else default_output_root()
    out_dir = out_dir / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    structures = {}
    exported = []
    raw_parc = None
    brain = None

    if anatomy:
        from recon.anatomy import mesh_brain_surface

        # 2. Skull-stripped brain: every nonzero voxel, then a heavily smoothed shell.
        brain = brain_mask_from_image(case["image"], config)
        brain_mesh = mesh_brain_surface(brain, affine)
        if brain_mesh is not None:
            _record_mesh(scene, out_dir, structures, exported, BRAIN, brain_mesh, brain, affine)

        # 3-4. Parcellate on the scan grid and mesh the labelled brain with no tumor.
        raw_parc = _raw_parcellation(case, case_dir, synthseg, shape, affine, device, case_id)
        _mesh_intact_anatomy(scene, out_dir, structures, exported, raw_parc, affine)

    # 5. Tumor detection.
    if "probs" in case:
        probs = np.asarray(case["probs"], dtype=np.float32)
    else:
        if checkpoint is None:
            raise ValueError("checkpoint is required when the case has no 'probs'")
        from recon.infer import (
            apply_model_preprocessing,
            load_model,
            predict_probs,
            resample_probabilities,
            roi_for,
        )

        network = load_model(checkpoint, device, model)
        network_image, network_affine = apply_model_preprocessing(case["image"], affine, model)
        probs = predict_probs(network, network_image, roi_for(model), model_name=model)
        if probs.shape[1:] != tuple(shape):
            probs = resample_probabilities(probs, network_affine, shape, affine)
    if brain is None:
        brain = brain_mask_from_image(case["image"], config)
    label = postprocess(probs, brain, config)
    save_pred_seg(label, reference, out_dir / "pred_seg.nii.gz")
    for name, mask in tumor_compartment_masks(label).items():
        if not np.any(mask):
            continue
        mesh = mesh_from_mask(mask, affine)
        if mesh is None:
            continue
        _record_mesh(scene, out_dir, structures, exported, get(name), mesh, mask, affine)

    # Location uses the intact parcellation, before the tumor is carved out.
    tumor_location = (
        tumor_location_summary(raw_parc, label, affine)
        if raw_parc is not None
        else {"margin_mm": 10.0, "WT": [], "ET": []}
    )

    # 6. Carve the tumor out of the anatomy and re-mesh only the structures it cut.
    removed = {}
    anatomy_removed = {}
    if anatomy and raw_parc is not None:
        from recon.anatomy import save_anatomy_seg

        cleaned, removed, anatomy_removed = _carve_and_remesh(
            scene, out_dir, structures, exported, raw_parc, label, affine
        )
        save_anatomy_seg(cleaned, reference, out_dir / "anatomy_seg.nii.gz", out_dir / "pred_seg.nii.gz")

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
            spec = get(name)
            _record_mesh(
                scene,
                out_dir,
                structures,
                exported,
                spec,
                tract.mesh,
                tract.mask,
                tract_affine,
                proximity=proximity[name],
            )

    scene.export(out_dir / "scene.glb")
    manifest = {
        "case_id": case_id,
        "case_dir": None if case_dir is None else str(case_dir),
        "postprocess": asdict(config),
        "structures": structures,
        "legend": legend_entries(exported),
        "tract_proximity": proximity,
        "tumor_location": tumor_location,
        "removed_voxels": {name: int(count) for name, count in removed.items()},
        "anatomy_removed_voxels": {name: int(count) for name, count in anatomy_removed.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
