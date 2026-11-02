"""Scene export: tumor compartments, then glass hemispheres, then interiors.

``build_scene`` loads the four RAS sequences, runs tumor detection, then
meshes a folded glass cortex from SynthSeg tissue (tumor filled so the
shell has no hole). Lobe ids stay on the cortex vertices and in the
location summary; they are not separate solid meshes.
``reconstruct_tumor`` still builds the nested WT / TC / ET shells.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

from recon.anatomy import add_anatomy_meshes, centroid_mm, ensure_scene_group, mesh_parcellation_labels
from recon.mesh import mesh_compartment
from recon.evaluate import check_mesh_volume
from recon.labels import (
    BLOOD_MESH,
    BLOOD_NOTE,
    BLOOD_PRODUCTS,
    BRAIN,
    COMPARTMENTS,
    EDEMA,
    ENHANCING,
    GREY_MATTER_TOTAL,
    LEFT_CORTEX,
    NECROSIS,
    RIGHT_CORTEX,
    SHELLS,
    WHITE_MATTER,
    get,
    legend_entries,
    rgba,
)

from recon.mesh import check_mesh, mesh_from_mask, region_shells

MIN_OVERLAP_VOXELS = 50
logger = logging.getLogger(__name__)

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
        mesh = mesh_compartment(name, mask, affine)
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


def scene_is_up_to_date(out_dir, case_dir=None, *, anatomy=True, model=None) -> bool:
    """True when ``scene.glb`` already has the requested anatomy for these files."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not (out_dir / "scene.glb").is_file() or not manifest_path.is_file():
        return False
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    structures = meta.get("structures") or {}
    has_tissue = any((entry or {}).get("group") == "Tissue" for entry in structures.values())
    if anatomy and not has_tissue and not meta.get("anatomy_warning"):
        return False
    stored_model = meta.get("model")
    if model is not None and stored_model not in (None, model):
        return False
    if case_dir is None:
        recorded = meta.get("case_dir")
        case_dir = Path(recorded) if recorded else None
    if case_dir is not None:
        try:
            from recon.io import input_files_hash

            current = input_files_hash(case_dir)
        except (FileNotFoundError, NotADirectoryError, ValueError, OSError):
            current = None
        stored = meta.get("input_hash")
        if stored and current and stored != current:
            return False
    return True


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
        "full_name": spec.full_name,
        "group": spec.group,
        "legend": spec.legend_name,
        "label_id": spec.label_id,
    }
    extra.update(more)
    return extra


def _legend_volumes(structures) -> dict[str, float]:
    return {
        name: float(entry["volume_cm3"])
        for name, entry in structures.items()
        if entry.get("volume_cm3") is not None
    }


def _record_grey_matter_total(structures, exported, parc, tumor_label, affine):
    """Volume-only Tissue row: cortical GM + deep GM."""
    from recon.anatomy import grey_matter_total_volume_mask

    volume = _voxel_cm3(grey_matter_total_volume_mask(parc, tumor_label), affine)
    payload = _catalog_extra(
        GREY_MATTER_TOTAL,
        volume_cm3=volume,
        color=list(rgba(GREY_MATTER_TOTAL)),
        source=GREY_MATTER_TOTAL.source,
        file=None,
        volume_only=True,
        glass=False,
        hidden_by_default=False,
    )
    ordered = {}
    inserted = False
    for name, entry in structures.items():
        ordered[name] = entry
        if name == WHITE_MATTER.mesh_name:
            ordered[GREY_MATTER_TOTAL.mesh_name] = payload
            inserted = True
    if not inserted:
        ordered[GREY_MATTER_TOTAL.mesh_name] = payload
    structures.clear()
    structures.update(ordered)
    if GREY_MATTER_TOTAL not in exported:
        wm_at = next((i for i, spec in enumerate(exported) if spec is WHITE_MATTER), None)
        if wm_at is None:
            exported.append(GREY_MATTER_TOTAL)
        else:
            exported.insert(wm_at + 1, GREY_MATTER_TOTAL)


def _voxel_cm3(mask, affine) -> float:
    voxel_mm3 = float(np.asarray(mask).sum()) * abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    return voxel_mm3 / 1000.0


def _glass_record_extra(structure) -> dict:
    hidden = structure is WHITE_MATTER
    return {
        "glass": True,
        "hidden_by_default": hidden,
        "lobe_tints": structure in (LEFT_CORTEX, RIGHT_CORTEX),
        "removed_voxels": 0,
    }


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
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(np.asarray(colors)):
        return mesh
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

    from recon.postprocess import foreground

    raw, affine = _read_canonical(_find_volume(case_dir, "t1n"), label=False)
    raw = np.asarray(raw, dtype=np.float32)
    # ANTs and the atlas mask need a zero background; normalized copies have a negative one.
    raw = np.where(foreground(raw), raw - raw.min(), 0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(raw, affine), str(dest))
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


def default_anatomy_cache_root() -> Path:
    return Path(__file__).resolve().parents[1] / "caches" / "anatomy"


def _load_cached_parcellation(path: Path, shape) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    image = nib.as_closest_canonical(nib.load(str(path)))
    parc = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
    if parc.shape != tuple(shape):
        return None
    return parc


def _store_cached_parcellation(path: Path, parc, affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(parc, dtype=np.int16), np.asarray(affine, dtype=np.float64)), str(path))


def _raw_parcellation(case, case_dir, synthseg, shape, affine, device, case_id, *, progress=None, file_hash=None, tumor_label=None, clock=None):
    """SynthSeg labels on the scan grid, before the tumor is removed.

    A hash-keyed cache under ``caches/anatomy/<hash>/`` stores the painted
    parcellation so a second open of the same files skips SynthSeg and ANTs.
    """
    from recon.anatomy import parcellate_case, resample_parcellation

    def _stage(name: str) -> None:
        if progress is not None:
            progress.set_stage(name)
        if clock is not None:
            if name == "Anatomy":
                clock.start("synthseg")
            elif name == "Lobes":
                clock.start("lobe_warp")

    cache_dir = default_anatomy_cache_root() / file_hash if file_hash else None
    cached = _load_cached_parcellation(cache_dir / "parc_raw.nii.gz" if cache_dir else None, shape)
    if cached is not None:
        _stage("Anatomy")
        _stage("Lobes")
        return cached

    parc = case.get("synthseg")
    source_affine = case.get("synthseg_affine")
    if parc is None and case_dir is not None:
        found = _find_synthseg(case_dir, synthseg)
        if found is not None:
            image = nib.as_closest_canonical(nib.load(str(found)))
            parc = np.rint(np.squeeze(image.get_fdata(dtype=np.float32))).astype(np.int16)
            source_affine = np.asarray(image.affine, dtype=np.float64)
    if parc is not None:
        _stage("Anatomy")
        parc = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
        if parc.shape == tuple(shape) and source_affine is None:
            _stage("Lobes")
            return parc
        if source_affine is None:
            raise ValueError(
                f"SynthSeg shape {parc.shape} does not match the scan grid {tuple(shape)}"
            )
        _stage("Lobes")
        return resample_parcellation(parc, source_affine, shape, affine)
    if case_dir is None:
        raise ValueError("case_dir is required to run SynthSeg")
    _stage("Anatomy")
    tumor = np.zeros(shape, dtype=np.uint8) if tumor_label is None else np.asarray(tumor_label)
    cache_key = file_hash or case_id
    with tempfile.TemporaryDirectory() as tmp:
        t1_ras = Path(tmp) / "t1n_ras.nii.gz"
        _write_canonical_t1(case_dir, t1_ras)

        def on_lobes():
            _stage("Lobes")

        raw, _cleaned, _removed = parcellate_case(
            t1_ras,
            tumor,
            affine,
            case_id=cache_key,
            device=device,
            on_lobes=on_lobes,
        )
    if cache_dir is not None:
        _store_cached_parcellation(cache_dir / "parc_raw.nii.gz", raw, affine)
        (cache_dir / "meta.json").write_text(
            json.dumps({"hash": file_hash, "case_id": case_id}, indent=2),
            encoding="utf-8",
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


def tumor_location_summary(parc, tumor_label, affine, *, margin_mm=10.0, min_overlap_voxels=MIN_OVERLAP_VOXELS) -> dict:
    """Lobes and structures the tumor overlaps or nears, after carving.

    Anatomy voxels that fall inside the whole-tumor mask are ignored — those
    labels were tumor, not structure. Overlap requires at least
    ``min_overlap_voxels`` remaining structure voxels in contact with the
    region; otherwise the distance in millimetres is reported. Anything
    farther than ``margin_mm`` is omitted.
    """
    from nibabel.affines import voxel_sizes
    from scipy import ndimage

    from recon.evaluate import regions_from_label
    from recon.labels import anatomy as anatomy_by_id
    from recon.postprocess import CONN

    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    if labels.ndim != 3:
        raise ValueError(f"Parcellation must be 3D, got {labels.shape}")
    regions = regions_from_label(tumor_label)
    if regions["WT"].shape != labels.shape:
        raise ValueError(
            f"Tumor shape {regions['WT'].shape} does not match the parcellation {labels.shape}"
        )
    spacing = tuple(float(abs(value)) for value in voxel_sizes(np.asarray(affine, dtype=np.float64)))
    whole = regions["WT"]
    fields = {}
    dilated = {}
    for name in ("WT", "ET"):
        mask = regions[name]
        fields[name] = ndimage.distance_transform_edt(~mask, sampling=spacing) if np.any(mask) else None
        dilated[name] = ndimage.binary_dilation(mask, structure=CONN) if np.any(mask) else None

    summary = {
        "margin_mm": float(margin_mm),
        "min_overlap_voxels": int(min_overlap_voxels),
        "WT": [],
        "ET": [],
    }
    for label_id in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        spec = anatomy_by_id(label_id)
        if spec is None or int(label_id) == 24:
            continue
        remainder = (labels == label_id) & ~whole
        if not np.any(remainder):
            continue
        for region in ("WT", "ET"):
            field = fields[region]
            grown = dilated[region]
            if field is None or grown is None:
                continue
            contact = int(np.count_nonzero(remainder & grown))
            distance = float(field[remainder].min())
            overlaps = contact >= int(min_overlap_voxels)
            if not overlaps and distance > float(margin_mm):
                continue
            summary[region].append(
                {
                    "name": spec.mesh_name,
                    "group": spec.group,
                    "label_id": int(spec.label_id),
                    "relation": "overlaps" if overlaps else "within_10mm",
                    "distance_mm": 0.0 if overlaps else round(distance, 3),
                    "contact_voxels": contact,
                }
            )
    for region in ("WT", "ET"):
        summary[region].sort(key=lambda item: (0 if item["relation"] == "overlaps" else 1, item["distance_mm"], item["name"]))
    return summary


def build_scene(
    case,
    *,
    anatomy=True,
    tracts=False,
    checkpoint=None,
    device=None,
    output_root=None,
    synthseg=None,
    config=None,
    model="ours",
    tta=True,
    overlap=None,
    progress=None,
):
    """Write ``outputs/<case_id>/``: load RAS, tumor, anatomy, meshes.

    Anatomy (GM/WM, subcortical, ventricles, cerebellum, brainstem, lobes) is
    on by default. Pass ``anatomy=False`` to skip it. SynthSeg and ANTs
    transforms are cached by a hash of the four input files.
    """
    from recon.progress import BuildProgress

    if progress is None:
        progress = BuildProgress()
    try:
        return _export_scene(
            case,
            anatomy=anatomy,
            tracts=tracts,
            checkpoint=checkpoint,
            device=device,
            output_root=output_root,
            synthseg=synthseg,
            config=config,
            model=model,
            tta=tta,
            overlap=overlap,
            progress=progress,
        )
    except Exception as exc:
        progress.fail(str(exc))
        raise


def _export_scene(
    case,
    *,
    anatomy,
    tracts,
    checkpoint,
    device,
    output_root,
    synthseg,
    config,
    model,
    tta,
    overlap,
    progress,
):
    from recon.io import input_files_hash, validate_case_grid
    from recon.postprocess import PostConfig, brain_mask_from_image, postprocess

    anatomy_warning = None
    file_hash = None
    progress.set_stage("Loading")
    if isinstance(case, (str, Path)):
        from recon.infer import ABHINAV_V3
        from recon.io import load_case

        case_dir = Path(case)
        validate_case_grid(case_dir)
        loaded = load_case(case_dir, normalize=model != ABHINAV_V3)
        loaded["case_dir"] = case_dir
        case = loaded
    else:
        case_dir = Path(case["case_dir"]) if case.get("case_dir") else None

    if case_dir is None:
        raise ValueError("case_dir is required to write pred_seg.nii.gz in the original image space")
    from recon.io import _find_volume

    reference = _find_volume(case_dir, "t1n")
    try:
        file_hash = input_files_hash(case_dir)
    except (FileNotFoundError, ValueError, OSError):
        file_hash = None
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

    from recon.timing import STAGE_TIMEOUT_S, StageClock, run_with_timeout

    clock = StageClock()
    progress.set_stage("Tumor")
    clock.start("postprocess")
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
        window = {} if overlap is None else {"overlap": overlap}
        probs = predict_probs(network, network_image, roi_for(model), tta, model_name=model, **window)
        if probs.shape[1:] != tuple(shape):
            probs = resample_probabilities(probs, network_affine, shape, affine)
    if brain is None:
        brain = brain_mask_from_image(case["image"], config)
    label = run_with_timeout(STAGE_TIMEOUT_S, postprocess, probs, brain, config)
    save_pred_seg(label, reference, out_dir / "pred_seg.nii.gz")

    clock.start("synthseg")
    if anatomy:
        try:
            raw_parc = _raw_parcellation(
                case,
                case_dir,
                synthseg,
                shape,
                affine,
                device,
                case_id,
                progress=progress,
                file_hash=file_hash,
                tumor_label=label,
                clock=clock,
            )
        except Exception as exc:
            anatomy_warning = f"Anatomy unavailable: {exc}"
            logger.exception("anatomy failed for %s", case_id)
            raw_parc = None
            progress.skip("Lobes")
            clock.skip("lobe_warp")
    else:
        progress.skip("Anatomy")
        progress.skip("Lobes")
        clock.skip("synthseg")
        clock.skip("lobe_warp")

    progress.set_stage("Meshing")
    clock.start("meshing")
    for name, mask in tumor_compartment_masks(label).items():
        if not np.any(mask):
            continue
        mesh = mesh_compartment(name, mask, affine)
        if mesh is None:
            continue
        _record_mesh(scene, out_dir, structures, exported, get(name), mesh, mask, affine)

    # Location uses uncarved labels, but only voxels that sit outside the tumor.
    tumor_location = (
        tumor_location_summary(raw_parc, label, affine)
        if raw_parc is not None
        else {"margin_mm": 10.0, "min_overlap_voxels": MIN_OVERLAP_VOXELS, "WT": [], "ET": []}
    )

    # GM glass covers the tumor (no hole). WM and interiors are carved.
    removed = {}
    anatomy_removed = {}
    if anatomy and raw_parc is not None:
        from recon.anatomy import (
            excise_tumor,
            grey_matter_volume_mask,
            iter_glass_shells,
            possible_blood_products_mask,
            save_anatomy_seg,
        )
        from recon.io import _find_volume, _read_canonical
        from recon.labels import anatomy as anatomy_by_id
        from recon.mesh import mesh_from_mask

        save_anatomy_seg(raw_parc, reference, out_dir / "synthseg_raw.nii.gz", out_dir / "pred_seg.nii.gz")
        for structure, mask, mesh in iter_glass_shells(raw_parc, label, affine):
            extra = _glass_record_extra(structure)
            if structure in (LEFT_CORTEX, RIGHT_CORTEX):
                side = "left" if structure is LEFT_CORTEX else "right"
                gm = grey_matter_volume_mask(raw_parc, label, side)
                extra["volume_cm3"] = _voxel_cm3(gm, affine)
            _record_mesh(
                scene,
                out_dir,
                structures,
                exported,
                structure,
                mesh,
                mask,
                affine,
                extra=extra,
            )
        _record_grey_matter_total(structures, exported, raw_parc, label, affine)
        cleaned, anatomy_removed = excise_tumor(raw_parc, label)
        save_anatomy_seg(cleaned, reference, out_dir / "anatomy_seg.nii.gz", out_dir / "pred_seg.nii.gz")
        raw_labels = np.rint(np.squeeze(np.asarray(raw_parc))).astype(np.int16)
        cleaned_labels = np.rint(np.squeeze(np.asarray(cleaned))).astype(np.int16)
        for label_id in sorted(int(value) for value in np.unique(raw_labels) if int(value) != 0):
            spec = anatomy_by_id(label_id)
            if spec is None:
                continue
            before = int(np.count_nonzero(raw_labels == label_id))
            after = int(np.count_nonzero(cleaned_labels == label_id))
            removed[spec.mesh_name] = before - after
        for structure, mask, mesh in mesh_parcellation_labels(cleaned, affine):
            extra = {
                "glass": structure.group == "Ventricles",
                "hidden_by_default": True,
                "removed_voxels": int(removed.get(structure.mesh_name, 0)),
            }
            _record_mesh(
                scene,
                out_dir,
                structures,
                exported,
                structure,
                mesh,
                mask,
                affine,
                extra=extra,
            )
        if case_dir is not None:
            try:
                t1_native, _ = _read_canonical(_find_volume(case_dir, "t1n"), label=False)
                t1c_native, _ = _read_canonical(_find_volume(case_dir, "t1c"), label=False)
                blood = possible_blood_products_mask(t1_native, t1c_native, label, raw_parc)
            except (FileNotFoundError, ValueError):
                blood = None
            if blood is not None and np.any(blood):
                mesh = mesh_from_mask(blood, affine, sigma=0.5, taubin_iters=5)
                if mesh is not None:
                    _record_mesh(
                        scene,
                        out_dir,
                        structures,
                        exported,
                        BLOOD_PRODUCTS,
                        mesh,
                        blood,
                        affine,
                        extra={"hidden_by_default": True, "note": BLOOD_NOTE, "experimental": True},
                    )
    elif anatomy and raw_parc is None:
        from recon.anatomy import mesh_brain_surface

        shell = mesh_brain_surface(brain, affine)
        if shell is not None:
            _record_mesh(
                scene,
                out_dir,
                structures,
                exported,
                BRAIN,
                shell,
                brain,
                affine,
                extra={"glass": True, "hidden_by_default": False, "warning": anatomy_warning},
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

    clock.start("export")
    scene.export(out_dir / "scene.glb")
    clock.stop()
    manifest = {
        "case_id": case_id,
        "case_dir": None if case_dir is None else str(case_dir),
        "model": model,
        "input_hash": file_hash,
        "anatomy": bool(anatomy),
        "postprocess": asdict(config),
        "structures": structures,
        "legend": legend_entries(exported, _legend_volumes(structures)),
        "tract_proximity": proximity,
        "tumor_location": tumor_location,
        "removed_voxels": {name: int(count) for name, count in removed.items()},
        "anatomy_removed_voxels": {name: int(count) for name, count in anatomy_removed.items()},
        "blood_products_note": BLOOD_NOTE if BLOOD_MESH in structures else None,
        "anatomy_warning": anatomy_warning,
        "stage_times_s": clock.as_dict(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress.finish(warning=anatomy_warning)
    return manifest


def rebuild_glass_meshes(out_dir):
    """Remesh glass hemispheres and tumor from existing ``pred_seg`` / ``anatomy_seg``.

    Voxel metrics, tumor location, and removed-voxel counts stay as they were.
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    pred = nib.load(str(out_dir / "pred_seg.nii.gz"))
    raw_path = out_dir / "synthseg_raw.nii.gz"
    cleaned_path = out_dir / "anatomy_seg.nii.gz"
    parc_path = raw_path if raw_path.is_file() else cleaned_path
    parc_img = nib.load(str(parc_path))
    label = np.rint(np.squeeze(pred.get_fdata(dtype=np.float32))).astype(np.int16)
    parc = np.rint(np.squeeze(parc_img.get_fdata(dtype=np.float32))).astype(np.int16)
    affine = np.asarray(parc_img.affine, dtype=np.float64)
    if cleaned_path.is_file() and raw_path.is_file():
        cleaned = np.rint(np.squeeze(nib.load(str(cleaned_path)).get_fdata(dtype=np.float32))).astype(np.int16)
    else:
        from recon.anatomy import excise_tumor

        cleaned, _removed = excise_tumor(parc, label)

    from recon.anatomy import (
        grey_matter_volume_mask,
        iter_glass_shells,
        mesh_parcellation_labels,
        possible_blood_products_mask,
    )
    from recon.io import _find_volume, _read_canonical
    from recon.mesh import mesh_from_mask

    scene = trimesh.Scene()
    structures = {}
    exported = []
    print(f"rebuilding glass meshes from {parc_path.name}", flush=True)
    for name, mask in tumor_compartment_masks(label).items():
        print(f"meshing tumor {name}", flush=True)
        if not np.any(mask):
            continue
        mesh = mesh_compartment(name, mask, affine)
        if mesh is None:
            continue
        previous = manifest.get("structures", {}).get(name, {})
        extra = {key: previous[key] for key in ("volume_cm3",) if key in previous}
        _record_mesh(scene, out_dir, structures, exported, get(name), mesh, mask, affine, extra=extra)
        if "volume_cm3" in previous:
            structures[name]["volume_cm3"] = previous["volume_cm3"]

    print("meshing grey matter, white matter, cerebellum, and brainstem", flush=True)
    for structure, mask, mesh in iter_glass_shells(parc, label, affine):
        faces = 0 if mesh is None else len(mesh.faces)
        print(f"  glass {structure.mesh_name}: {int(np.count_nonzero(mask))} voxels, {faces} faces", flush=True)
        extra = _glass_record_extra(structure)
        if structure in (LEFT_CORTEX, RIGHT_CORTEX):
            side = "left" if structure is LEFT_CORTEX else "right"
            extra["volume_cm3"] = _voxel_cm3(grey_matter_volume_mask(parc, label, side), affine)
        _record_mesh(
            scene,
            out_dir,
            structures,
            exported,
            structure,
            mesh,
            mask,
            affine,
            extra=extra,
        )
    _record_grey_matter_total(structures, exported, parc, label, affine)

    for structure, mask, mesh in mesh_parcellation_labels(cleaned, affine):
        previous = manifest.get("structures", {}).get(structure.mesh_name, {})
        _record_mesh(
            scene,
            out_dir,
            structures,
            exported,
            structure,
            mesh,
            mask,
            affine,
            extra={
                "glass": structure.group == "Ventricles",
                "hidden_by_default": True,
                "removed_voxels": int(previous.get("removed_voxels", 0)),
            },
        )

    case_dir = manifest.get("case_dir")
    if not case_dir or not Path(case_dir).is_dir():
        fallback = Path(__file__).resolve().parents[1] / "data" / "cases" / out_dir.name
        case_dir = str(fallback) if fallback.is_dir() else None
    if case_dir and Path(case_dir).is_dir():
        try:
            t1_native, _ = _read_canonical(_find_volume(Path(case_dir), "t1n"), label=False)
            t1c_native, _ = _read_canonical(_find_volume(Path(case_dir), "t1c"), label=False)
            blood = possible_blood_products_mask(t1_native, t1c_native, label, parc)
        except (FileNotFoundError, ValueError) as exc:
            print(f"skipping blood-products heuristic: {exc}", flush=True)
            blood = None
        if blood is not None and np.any(blood):
            print(f"meshing possible blood products ({int(np.count_nonzero(blood))} voxels)", flush=True)
            mesh = mesh_from_mask(blood, affine, sigma=0.5, taubin_iters=5)
            if mesh is not None:
                _record_mesh(
                    scene,
                    out_dir,
                    structures,
                    exported,
                    BLOOD_PRODUCTS,
                    mesh,
                    blood,
                    affine,
                    extra={"hidden_by_default": True, "note": BLOOD_NOTE, "experimental": True},
                )

    manifest["tumor_location"] = tumor_location_summary(parc, label, affine)
    manifest["blood_products_note"] = BLOOD_NOTE if BLOOD_MESH in structures else None

    for name, entry in manifest.get("structures", {}).items():
        if entry.get("source") == "atlas-estimated" and (out_dir / f"{name}.glb").is_file():
            mesh = trimesh.load(out_dir / f"{name}.glb", force="mesh")
            spec = get(name)
            scene_mesh = mesh if isinstance(mesh, trimesh.Trimesh) else None
            if scene_mesh is None:
                continue
            mask = np.zeros((1, 1, 1), dtype=bool)
            mask[0, 0, 0] = True
            _add_mesh(scene, out_dir, spec.mesh_name, scene_mesh, spec.group)
            structures[name] = dict(entry)
            exported.append(spec)

    scene.export(out_dir / "scene.glb")
    manifest["structures"] = structures
    manifest["legend"] = legend_entries(exported, _legend_volumes(structures))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
