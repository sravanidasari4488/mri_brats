"""One catalog for every mesh: id, name, group, and color.

SynthSeg and Desikan-Killiany colors are the FreeSurferColorLUT values
(FreeSurfer ``FreeSurferColorLUT.txt``). Reviewers already know those colors.
BraTS compartments keep the usual convention: enhancing blue, necrosis red,
edema yellow. Tract colors are separate, and every tract is marked
``atlas-estimated``.

Left and right stay separate meshes (``Left Thalamus``, ``Right Thalamus``).
:func:`legend_entries` collapses each pair to one legend row. The swatch is
the left mesh when both sides are present.

BraTS labels 1, 2, and 3 are not FreeSurfer ids. Anatomy lookup and tumor
lookup are separate so white matter (FreeSurfer 2) is not edema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUPS = (
    "Cortex",
    "Subcortical",
    "Ventricles",
    "Brainstem & Cerebellum",
    "Tumor",
    "Tracts",
)

EDEMA = "edema"
NECROSIS = "necrosis"
ENHANCING = "enhancing"
CORE = "core"
BRAIN_MESH = "brain"

_TUMOR_EDEMA_OPACITY = 0.35
_TUMOR_OPACITY = 1.0
_BRAIN_OPACITY = 0.10
_TRACT_OPACITY = 0.60
_ANATOMY_OPACITY = 220 / 255

_SIDE_ORDER = {"left": 0, "right": 1, None: 2}


@dataclass(frozen=True)
class Structure:
    """One mesh, or one side of a left/right pair."""

    label_id: int | None
    full_name: str
    short_name: str
    group: str
    rgb: tuple[int, int, int]
    legend_name: str
    side: str | None
    source: str
    mesh_name: str
    key: str
    opacity: float
    in_legend: bool = True

    def rgba(self) -> tuple[int, int, int, int]:
        return _pack(self.rgb, self.opacity)


BY_KEY: dict[str, Structure] = {}
BY_MESH: dict[str, Structure] = {}
ANATOMY_BY_ID: dict[int, Structure] = {}
TUMOR_BY_ID: dict[int, Structure] = {}
STRUCTURES: list[Structure] = []


def _pack(rgb, opacity: float) -> tuple[int, int, int, int]:
    red, green, blue = (int(channel) for channel in rgb)
    return (red, green, blue, int(round(float(opacity) * 255)))


def _add(
    *,
    label_id: int | None,
    full_name: str,
    short_name: str,
    group: str,
    rgb: tuple[int, int, int],
    legend_name: str,
    side: str | None,
    source: str,
    mesh_name: str,
    key: str,
    opacity: float,
    in_legend: bool = True,
    id_map: dict[int, Structure] | None = None,
) -> Structure:
    if group not in GROUPS and not (group == "Brain" and not in_legend):
        raise ValueError(f"{full_name} group {group!r} is not one of {GROUPS}")
    if key in BY_KEY:
        raise ValueError(f"Duplicate structure key {key}")
    if mesh_name in BY_MESH:
        raise ValueError(f"Duplicate mesh name {mesh_name}")
    if id_map is not None and label_id is not None and label_id in id_map:
        raise ValueError(f"Duplicate label id {label_id} for {full_name}")
    spec = Structure(
        label_id=None if label_id is None else int(label_id),
        full_name=full_name,
        short_name=short_name,
        group=group,
        rgb=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
        legend_name=legend_name,
        side=side,
        source=source,
        mesh_name=mesh_name,
        key=key,
        opacity=float(opacity),
        in_legend=in_legend,
    )
    BY_KEY[key] = spec
    BY_MESH[mesh_name] = spec
    STRUCTURES.append(spec)
    if id_map is not None and spec.label_id is not None:
        id_map[spec.label_id] = spec
    return spec


def _pair(
    left_id: int,
    right_id: int | None,
    legend_name: str,
    short: str,
    group: str,
    left_rgb: tuple[int, int, int],
    right_rgb: tuple[int, int, int] | None = None,
    *,
    left_full: str | None = None,
    right_full: str | None = None,
    source: str = "synthseg",
    key: str | None = None,
    opacity: float = _ANATOMY_OPACITY,
) -> None:
    """Register one side, or both sides, under a single legend name."""
    stem = key or legend_name.lower().replace(" ", "_").replace("-", "_")
    _add(
        label_id=left_id,
        full_name=left_full or f"Left {legend_name}",
        short_name=f"{short}-L",
        group=group,
        rgb=left_rgb,
        legend_name=legend_name,
        side="left",
        source=source,
        mesh_name=left_full or f"Left {legend_name}",
        key=f"{stem}_l",
        opacity=opacity,
        id_map=ANATOMY_BY_ID,
    )
    if right_id is None:
        return
    _add(
        label_id=right_id,
        full_name=right_full or f"Right {legend_name}",
        short_name=f"{short}-R",
        group=group,
        rgb=left_rgb if right_rgb is None else right_rgb,
        legend_name=legend_name,
        side="right",
        source=source,
        mesh_name=right_full or f"Right {legend_name}",
        key=f"{stem}_r",
        opacity=opacity,
        id_map=ANATOMY_BY_ID,
    )


def _midline(
    label_id: int,
    full_name: str,
    short_name: str,
    group: str,
    rgb: tuple[int, int, int],
    *,
    key: str | None = None,
) -> None:
    _add(
        label_id=label_id,
        full_name=full_name,
        short_name=short_name,
        group=group,
        rgb=rgb,
        legend_name=full_name,
        side=None,
        source="synthseg",
        mesh_name=full_name,
        key=key or full_name.lower().replace(" ", "_"),
        opacity=_ANATOMY_OPACITY,
        id_map=ANATOMY_BY_ID,
    )


# FreeSurferColorLUT. Cerebral white matter is subcortical tissue in this legend.
_pair(2, 41, "Cerebral White Matter", "WM", "Subcortical", (245, 245, 245))
_pair(3, 42, "Cerebral Cortex", "Ctx", "Cortex", (205, 62, 78))
_pair(4, 43, "Lateral Ventricle", "LV", "Ventricles", (120, 18, 134))
_pair(5, 44, "Inferior Lateral Ventricle", "ILV", "Ventricles", (196, 58, 250))
_pair(7, 46, "Cerebellar White Matter", "CbWM", "Brainstem & Cerebellum", (220, 248, 164))
_pair(8, 47, "Cerebellar Cortex", "CbCtx", "Brainstem & Cerebellum", (230, 148, 34))
_pair(10, 49, "Thalamus", "Thal", "Subcortical", (0, 118, 14))
_pair(11, 50, "Caudate", "Caud", "Subcortical", (122, 186, 220))
_pair(12, 51, "Putamen", "Put", "Subcortical", (236, 13, 176))
# Left pallidum is 12,48,255 and right is 13,48,255 in the LUT.
_pair(13, 52, "Pallidum", "Pall", "Subcortical", (12, 48, 255), (13, 48, 255))
_midline(14, "Third Ventricle", "3V", "Ventricles", (204, 182, 142))
_midline(15, "Fourth Ventricle", "4V", "Ventricles", (42, 204, 164))
_midline(16, "Brainstem", "BS", "Brainstem & Cerebellum", (119, 159, 176))
_pair(17, 53, "Hippocampus", "Hip", "Subcortical", (220, 216, 20))
_pair(18, 54, "Amygdala", "Amy", "Subcortical", (103, 255, 255))
_midline(24, "CSF", "CSF", "Ventricles", (60, 60, 60))
_pair(26, 58, "Accumbens", "Acc", "Subcortical", (255, 165, 0))
# Left ventral DC is 145,42,42 and right is 165,42,42 in the LUT.
_pair(28, 60, "Ventral Diencephalon", "VDC", "Subcortical", (145, 42, 42), (165, 42, 42))

# MNI lobe fallback (ids 101-105). RGB is the FreeSurfer color of the
# Desikan-Killiany region used as the stand-in for that lobe.
_LOBE_COLORS = {
    "Frontal Lobe": (20, 220, 160),  # ctx superior frontal
    "Parietal Lobe": (20, 180, 140),  # ctx superior parietal
    "Temporal Lobe": (160, 100, 50),  # ctx middle temporal
    "Occipital Lobe": (20, 30, 140),  # ctx lateral occipital
    "Insular Lobe": (255, 192, 32),  # ctx insula
}
for _lobe_id, _lobe_name, _lobe_short in (
    (101, "Frontal Lobe", "Fro"),
    (102, "Parietal Lobe", "Par"),
    (103, "Temporal Lobe", "Tem"),
    (104, "Occipital Lobe", "Occ"),
    (105, "Insular Lobe", "InsL"),
):
    _midline(_lobe_id, _lobe_name, _lobe_short, "Cortex", _LOBE_COLORS[_lobe_name], key=_lobe_name.lower().replace(" ", "_"))

# Desikan-Killiany. ctx-lh is 1000+offset and ctx-rh is 2000+offset.
# FreeSurfer uses the same RGB for both hemispheres of each region.
_DK = (
    ("Unknown", "Unk", (25, 5, 25)),
    ("Banks of the Superior Temporal Sulcus", "BSTS", (25, 100, 40)),
    ("Caudal Anterior Cingulate", "cACC", (125, 100, 160)),
    ("Caudal Middle Frontal", "cMFG", (100, 25, 0)),
    ("Corpus Callosum", "CC", (120, 70, 50)),
    ("Cuneus", "Cun", (220, 20, 100)),
    ("Entorhinal", "Ent", (220, 20, 10)),
    ("Fusiform", "Fus", (180, 220, 140)),
    ("Inferior Parietal", "IPL", (220, 60, 220)),
    ("Inferior Temporal", "ITG", (180, 40, 120)),
    ("Isthmus Cingulate", "IstC", (140, 20, 140)),
    ("Lateral Occipital", "LOG", (20, 30, 140)),
    ("Lateral Orbitofrontal", "lOFC", (35, 75, 50)),
    ("Lingual", "Lin", (225, 140, 140)),
    ("Medial Orbitofrontal", "mOFC", (200, 35, 75)),
    ("Middle Temporal", "MTG", (160, 100, 50)),
    ("Parahippocampal", "PHG", (20, 220, 60)),
    ("Paracentral", "ParaC", (60, 220, 60)),
    ("Pars Opercularis", "Op", (220, 180, 140)),
    ("Pars Orbitalis", "Orb", (20, 100, 50)),
    ("Pars Triangularis", "Tri", (220, 60, 20)),
    ("Pericalcarine", "Calc", (120, 100, 60)),
    ("Postcentral", "PostC", (220, 20, 20)),
    ("Posterior Cingulate", "PCC", (220, 180, 220)),
    ("Precentral", "PreC", (60, 20, 220)),
    ("Precuneus", "PCun", (160, 140, 180)),
    ("Rostral Anterior Cingulate", "rACC", (80, 20, 140)),
    ("Rostral Middle Frontal", "rMFG", (75, 50, 125)),
    ("Superior Frontal", "SFG", (20, 220, 160)),
    ("Superior Parietal", "SPL", (20, 180, 140)),
    ("Superior Temporal", "STG", (140, 220, 220)),
    ("Supramarginal", "SMG", (80, 160, 20)),
    ("Frontal Pole", "FPol", (100, 0, 100)),
    ("Temporal Pole", "TPol", (70, 70, 70)),
    ("Transverse Temporal", "TTG", (150, 150, 200)),
    ("Insula", "Ins", (255, 192, 32)),
)
for _offset, (_pretty, _short, _rgb) in enumerate(_DK):
    _pair(
        1000 + _offset,
        2000 + _offset,
        _pretty,
        _short,
        "Cortex",
        _rgb,
        key=f"dk_{_short.lower()}",
    )

# BraTS convention. ``core`` is the nested tumor-core shell, not its own label.
EDEMA_STRUCTURE = _add(
    label_id=2,
    full_name="Edema",
    short_name="ED",
    group="Tumor",
    rgb=(255, 220, 0),
    legend_name="Edema",
    side=None,
    source="model",
    mesh_name=EDEMA,
    key=EDEMA,
    opacity=_TUMOR_EDEMA_OPACITY,
    id_map=TUMOR_BY_ID,
)
NECROSIS_STRUCTURE = _add(
    label_id=1,
    full_name="Necrosis",
    short_name="NCR",
    group="Tumor",
    rgb=(220, 40, 40),
    legend_name="Necrosis",
    side=None,
    source="model",
    mesh_name=NECROSIS,
    key=NECROSIS,
    opacity=_TUMOR_OPACITY,
    id_map=TUMOR_BY_ID,
)
ENHANCING_STRUCTURE = _add(
    label_id=3,
    full_name="Enhancing Tumor",
    short_name="ET",
    group="Tumor",
    rgb=(60, 120, 255),
    legend_name="Enhancing Tumor",
    side=None,
    source="model",
    mesh_name=ENHANCING,
    key=ENHANCING,
    opacity=_TUMOR_OPACITY,
    id_map=TUMOR_BY_ID,
)
CORE_STRUCTURE = _add(
    label_id=None,
    full_name="Tumor Core",
    short_name="TC",
    group="Tumor",
    rgb=(220, 40, 40),
    legend_name="Tumor Core",
    side=None,
    source="model",
    mesh_name=CORE,
    key=CORE,
    opacity=_TUMOR_OPACITY,
    in_legend=False,
)
SHELLS = (EDEMA_STRUCTURE, CORE_STRUCTURE, ENHANCING_STRUCTURE)
COMPARTMENTS = (EDEMA_STRUCTURE, NECROSIS_STRUCTURE, ENHANCING_STRUCTURE)
SHELL_NAMES = tuple(spec.mesh_name for spec in SHELLS)

# Outer brain shell. It is not a parcellation class, so it stays out of the legend.
BRAIN = _add(
    label_id=None,
    full_name="Brain",
    short_name="Brain",
    group="Brain",
    rgb=(180, 180, 180),
    legend_name="Brain",
    side=None,
    source="image",
    mesh_name=BRAIN_MESH,
    key=BRAIN_MESH,
    opacity=_BRAIN_OPACITY,
    in_legend=False,
)

# JHU tracts. Each key matches ``recon.tracts.JHU_PROB_VOLUMES``.
for _key, _legend, _short, _rgb_l, _rgb_r in (
    ("corticospinal_tract", "Corticospinal Tract", "CST", (0, 83, 138), (214, 96, 0)),
    ("arcuate_slf", "Arcuate Fasciculus / SLF", "SLF", (0, 140, 90), (130, 70, 180)),
    ("ifof", "Inferior Fronto-occipital Fasciculus", "IFOF", (180, 70, 130), (110, 70, 40)),
    ("optic_radiation", "Optic Radiation", "OR", (0, 150, 150), (150, 150, 40)),
    ("uncinate", "Uncinate Fasciculus", "UF", (190, 110, 170), (70, 110, 60)),
):
    _add(
        label_id=None,
        full_name=f"Left {_legend}",
        short_name=f"{_short}-L",
        group="Tracts",
        rgb=_rgb_l,
        legend_name=_legend,
        side="left",
        source="atlas-estimated",
        mesh_name=f"Left {_legend}",
        key=f"{_key}_L",
        opacity=_TRACT_OPACITY,
    )
    _add(
        label_id=None,
        full_name=f"Right {_legend}",
        short_name=f"{_short}-R",
        group="Tracts",
        rgb=_rgb_r,
        legend_name=_legend,
        side="right",
        source="atlas-estimated",
        mesh_name=f"Right {_legend}",
        key=f"{_key}_R",
        opacity=_TRACT_OPACITY,
    )


def get(name: str) -> Structure:
    """Look up a mesh name or a stable key such as ``corticospinal_tract_L``."""
    if name in BY_KEY:
        return BY_KEY[name]
    if name in BY_MESH:
        return BY_MESH[name]
    raise KeyError(f"Unknown structure {name!r}")


def rgba(name) -> tuple[int, int, int, int]:
    """RGBA for a :class:`Structure`, mesh name, or key."""
    spec = name if isinstance(name, Structure) else get(name)
    return spec.rgba()


def anatomy(label_id: int) -> Structure | None:
    """FreeSurfer, SynthSeg, lobe, or Desikan-Killiany label. Not a BraTS id."""
    return ANATOMY_BY_ID.get(int(label_id))


def tumor(label_id: int) -> Structure | None:
    """BraTS compartment for label 1, 2, or 3."""
    return TUMOR_BY_ID.get(int(label_id))


def structures_in_volume(parc) -> list[tuple[Structure, np.ndarray]]:
    """Known anatomical labels in ``parc``, one boolean mask per mesh."""
    labels = np.rint(np.squeeze(np.asarray(parc))).astype(np.int16)
    if labels.ndim != 3:
        raise ValueError(f"Parcellation must be 3D, got {labels.shape}")
    found = []
    for label_id in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        spec = ANATOMY_BY_ID.get(label_id)
        if spec is None:
            continue
        mask = labels == label_id
        if np.any(mask):
            found.append((spec, mask))
    return found


def legend_entries(structures) -> list[dict]:
    """One row per left/right pair. Mesh names stay on the row."""
    buckets: dict[tuple[str, str], list[Structure]] = {}
    order: list[tuple[str, str]] = []
    for spec in structures:
        if not spec.in_legend:
            continue
        key = (spec.group, spec.legend_name)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(spec)

    def _sort_key(item: tuple[str, str]) -> tuple[int, int]:
        return (GROUPS.index(item[0]), order.index(item))

    entries = []
    for group, legend_name in sorted(order, key=_sort_key):
        members = sorted(buckets[(group, legend_name)], key=lambda spec: _SIDE_ORDER[spec.side])
        swatch = next((spec for spec in members if spec.side == "left"), members[0])
        entries.append(
            {
                "name": legend_name,
                "group": group,
                "rgb": [int(channel) for channel in swatch.rgb],
                "source": swatch.source,
                "meshes": [spec.mesh_name for spec in members],
            }
        )
    return entries
