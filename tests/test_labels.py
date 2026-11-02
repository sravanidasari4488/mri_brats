"""Label catalog: FreeSurfer colors, BraTS tumor colors, and left/right legend rows."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.labels import GROUPS, anatomy, get, legend_entries, tumor
from recon.mesh import SHELL_NAMES


def test_freesurfer_colors_and_separate_meshes():
    left = anatomy(10)
    right = anatomy(49)
    assert left.full_name == "Left Thalamus"
    assert right.full_name == "Right Thalamus"
    assert left.mesh_name == "Left Thalamus"
    assert right.mesh_name == "Right Thalamus"
    assert left.rgb == right.rgb == (0, 118, 14)
    assert left.group == right.group == "Subcortical"
    assert anatomy(3).rgb == (205, 62, 78)
    assert anatomy(1028).full_name == "Left Superior Frontal"
    assert anatomy(1028).rgb == anatomy(2028).rgb == (20, 220, 160)
    assert anatomy(13).rgb == (12, 48, 255)
    assert anatomy(52).rgb == (13, 48, 255)
    assert anatomy(16).full_name == "Brainstem"
    assert anatomy(16).group == "Brainstem & Cerebellum"
    assert {spec.group for spec in (left, right, anatomy(3), anatomy(16))} <= set(GROUPS)
    assert "Brain" in GROUPS
    assert get("brain").in_legend is False
    assert get("brain").group == "Brain"
    assert get("brain").rgb == (159, 184, 214)
    assert get("Left Cortex").rgb == (159, 184, 214)
    assert get("Left Cortex").full_name == "Cortical grey matter (left)"
    assert get("Left Cortex").legend_name == "Cortical grey matter (left)"
    assert get("Right Cortex").group == "Tissue"
    assert get("Right Cortex").full_name == "Cortical grey matter (right)"
    assert get("White Matter").rgb == (232, 220, 200)
    assert get("White Matter").group == "Tissue"
    assert get("White Matter").full_name == "White matter"
    assert get("White Matter").legend_name == "White matter"
    assert get("Grey matter total").group == "Tissue"
    assert get("grey_matter_total").full_name == "Grey matter total"
    assert anatomy(4).rgb == (95, 198, 232)
    assert anatomy(14).rgb == (95, 198, 232)
    assert "Tissue" in GROUPS
    assert get("blood_products").source == "heuristic"
    assert "hemorrhage" not in get("blood_products").mesh_name.lower()
    assert "Possible blood products" in get("blood_products").mesh_name
    assert anatomy(101).rgb == (236, 118, 142)
    assert anatomy(101).opacity == 0.35
    assert anatomy(104).rgb == (122, 118, 214)
    assert anatomy(8).rgb == (159, 184, 214)
    assert anatomy(16).rgb == (159, 184, 214)


def test_tumor_colors_and_tracts_are_atlas_estimated():
    assert tumor(1).rgb == (192, 57, 43)
    assert tumor(2).rgb == (245, 197, 66)
    assert tumor(3).rgb == (59, 130, 246)
    assert tumor(2).opacity == 0.25
    assert tumor(1).opacity == 1.0
    assert tumor(2).mesh_name == "edema"
    assert SHELL_NAMES == ("edema", "core", "enhancing")
    left = get("corticospinal_tract_L")
    right = get("corticospinal_tract_R")
    assert left.source == right.source == "atlas-estimated"
    assert left.mesh_name == "Left Corticospinal Tract"
    assert right.mesh_name == "Right Corticospinal Tract"
    assert left.rgb != right.rgb
    assert left.group == "Tracts"


def test_legend_merges_a_pair_and_keeps_both_meshes():
    rows = legend_entries(
        [anatomy(10), anatomy(49), anatomy(16), get("corticospinal_tract_L"), get("corticospinal_tract_R")],
        volumes={"Left Thalamus": 1.25, "Right Thalamus": 1.5},
    )
    by_name = {row["name"]: row for row in rows}
    assert list(by_name) == ["Thalamus", "Brainstem", "Corticospinal Tract"]
    assert by_name["Thalamus"]["meshes"] == ["Left Thalamus", "Right Thalamus"]
    assert by_name["Thalamus"]["rgb"] == [0, 118, 14]
    assert by_name["Thalamus"]["volume_cm3"] == 2.75
    assert by_name["Corticospinal Tract"]["meshes"] == ["Left Corticospinal Tract", "Right Corticospinal Tract"]
    assert len(rows) == 3
    volume = np.zeros((2, 2, 2), dtype=np.int16)
    volume[0, 0, 0] = 10
    volume[1, 1, 1] = 49
    from recon.labels import structures_in_volume

    found = structures_in_volume(volume)
    assert [spec.mesh_name for spec, _mask in found] == ["Left Thalamus", "Right Thalamus"]


if __name__ == "__main__":
    test_freesurfer_colors_and_separate_meshes()
    test_tumor_colors_and_tracts_are_atlas_estimated()
    test_legend_merges_a_pair_and_keeps_both_meshes()
    print("labels ok")
