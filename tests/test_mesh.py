"""Sphere volume and nested WT / TC / ET shells."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.mesh import check_mesh, mesh_from_mask, mesh_shells, region_shells


def _sphere(shape, center, radius):
    coords = np.indices(shape)
    delta = coords - np.asarray(center, dtype=np.float64).reshape(3, 1, 1, 1)
    return np.sum(delta * delta, axis=0) <= radius * radius


def _affine():
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = (100.0, -20.0, 5.0)
    return affine


def test_sphere_is_watertight_within_3_percent():
    shape = (48, 48, 48)
    mask = _sphere(shape, (23.5, 23.5, 23.5), 20)
    affine = _affine()
    mesh = mesh_from_mask(mask, affine)
    watertight, err = check_mesh(mesh, mask, affine)
    assert watertight
    assert abs(err) < 3.0
    # The RAS affine moves vertices out of voxel index space.
    assert mesh.bounds[0, 0] > 50.0


def test_empty_mask_returns_none():
    affine = np.eye(4)
    assert mesh_from_mask(np.zeros((8, 8, 8), dtype=bool), affine) is None


def test_shells_are_nested_wt_tc_et():
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[1:7, 1:7, 1:7] = 2
    label[2:6, 2:6, 2:6] = 1
    label[3:5, 3:5, 3:5] = 4  # legacy ET
    shells = region_shells(label)
    assert set(shells) == {"edema", "core", "enhancing"}
    assert int(shells["edema"].sum()) == int((label > 0).sum())
    assert int(shells["core"].sum()) == int(((label == 1) | (label == 4)).sum())
    assert int(shells["enhancing"].sum()) == int((label == 4).sum())
    assert np.all(shells["enhancing"] <= shells["core"])
    assert np.all(shells["core"] <= shells["edema"])


def test_nested_shell_volumes_rank_and_stay_within_3_percent():
    shape = (48, 48, 48)
    center = (23.5, 23.5, 23.5)
    label = np.zeros(shape, dtype=np.uint8)
    label[_sphere(shape, center, 18)] = 2
    label[_sphere(shape, center, 12)] = 1
    label[_sphere(shape, center, 6)] = 3
    affine = _affine()
    shells = region_shells(label)
    meshes = mesh_shells(label, affine)
    volumes = {}
    for name in ("edema", "core", "enhancing"):
        watertight, _err = check_mesh(meshes[name], shells[name], affine)
        assert watertight
        volumes[name] = abs(float(meshes[name].volume))
    assert volumes["edema"] > volumes["core"] > volumes["enhancing"]


if __name__ == "__main__":
    test_sphere_is_watertight_within_3_percent()
    test_empty_mask_returns_none()
    test_shells_are_nested_wt_tc_et()
    test_nested_shell_volumes_rank_and_stay_within_3_percent()
    print("mesh ok")
