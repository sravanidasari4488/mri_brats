"""Hierarchy, component removal, ET-to-necrosis, channel order, and the yaml config."""

import importlib
import itertools

import numpy as np

from recon.postprocess import PostConfig, brain_mask_from_image, config_path_for, postprocess


def test_yaml_loads_region_cutoffs():
    config = PostConfig.from_yaml()
    assert config.threshold("ET") == 0.5
    assert config.threshold("TC") == 0.5
    assert config.threshold("WT") == 0.5
    assert config.min_voxels("ET") == 50
    assert config.min_voxels("TC") == 150
    assert config.min_voxels("WT") == 300
    assert config.et_min_voxels == 100
    assert config.brain_closing_iterations == 2


def test_brain_mask_closes_and_fills_holes():
    image = np.zeros((4, 12, 12, 12), dtype=np.float32)
    image[:, 2:10, 2:10, 2:10] = 1
    image[:, 5:7, 5:7, 5:7] = 0
    config = PostConfig.from_yaml()
    mask = brain_mask_from_image(image, config)
    assert mask[6, 6, 6]
    assert not mask[0, 0, 0]


def test_brain_mask_keeps_a_border_touching_brain_from_filling_the_box():
    image = np.zeros((4, 12, 12, 12), dtype=np.float32)
    image[:, 0:9, 2:10, 2:10] = 1
    image[:, 4:6, 5:7, 5:7] = 0
    image[:, 11, 11, 11] = 0.4
    config = PostConfig.from_yaml()
    mask = brain_mask_from_image(image, config)
    assert mask[0, 5, 5]
    assert mask[5, 6, 6]
    assert not mask[11, 11, 11]
    assert not mask[11, 0, 0]


def test_brain_mask_on_zscored_volumes_uses_nonzero_tissue():
    rng = np.random.default_rng(0)
    image = np.zeros((4, 16, 16, 16), dtype=np.float32)
    tissue = rng.normal(0, 1, (8, 8, 8)).astype(np.float32)
    tissue[3:5, 3:5, 3:5] = 0
    image[:, 2:10, 2:10, 2:10] = tissue
    mask = brain_mask_from_image(image, PostConfig.from_yaml())
    assert mask[6, 6, 6]
    assert mask[2, 2, 2]
    assert not mask[0, 0, 0]
    assert int(mask.sum()) < 16 * 16 * 16 // 2


def test_brain_mask_ignores_negative_fov_fill():
    image = np.full((4, 16, 16, 16), -0.2, dtype=np.float32)
    image[:, 3:12, 3:12, 3:12] = 1.5
    image[:, 6:8, 6:8, 6:8] = 0
    mask = brain_mask_from_image(image, PostConfig.from_yaml())
    assert mask[4, 4, 4]
    assert mask[7, 7, 7]
    assert not mask[0, 0, 0]
    assert int(mask.sum()) < 16 * 16 * 16 // 2


def _volume(config: PostConfig):
    """WT slab, TC inside it, ET core, plus a speck and an out-of-hierarchy ET."""
    shape = (16, 16, 16)
    probs = np.zeros((3, *shape), dtype=np.float32)
    # channel 0 ET, 1 TC, 2 WT
    probs[2, 2:12, 2:12, 2:10] = 0.9
    probs[1, 4:10, 4:10, 4:8] = 0.9
    probs[0, 5:8, 5:8, 5:7] = 0.9
    probs[2, 14:16, 14:16, 14:15] = 0.9  # speck, smaller than min WT
    probs[0, 0:2, 0:2, 0:2] = 0.9  # ET outside TC/WT
    brain = np.ones(shape, dtype=bool)
    return postprocess(probs, brain, config)


def test_hierarchy_component_filter_and_et_to_necrosis():
    config = PostConfig(
        threshold_et=0.5,
        threshold_tc=0.5,
        threshold_wt=0.5,
        min_voxels_et=4,
        min_voxels_tc=4,
        min_voxels_wt=20,
        et_min_voxels=100,
        brain_closing_iterations=1,
    )
    label = _volume(config)
    assert label[15, 15, 14] == 0
    assert label[0, 0, 0] == 0
    assert label[3, 3, 3] == 2  # edema outside tumor core
    assert label[4, 4, 4] == 1  # necrotic core
    assert label[6, 6, 5] == 1  # ET dropped; voxel stays NCR
    assert 3 not in label


def test_et_kept_when_large_enough():
    config = PostConfig(
        threshold_et=0.5,
        threshold_tc=0.5,
        threshold_wt=0.5,
        min_voxels_et=4,
        min_voxels_tc=4,
        min_voxels_wt=20,
        et_min_voxels=1,
        brain_closing_iterations=1,
    )
    label = _volume(config)
    assert label[6, 6, 5] == 3
    assert label[4, 4, 4] == 1
    assert label[3, 3, 3] == 2


def test_shuffled_channel_order_gives_the_same_label_map():
    infer = importlib.import_module("recon.infer")
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    heads = {"ET": np.zeros(shape, np.float32), "TC": np.zeros(shape, np.float32), "WT": np.zeros(shape, np.float32)}
    heads["WT"][2:12, 2:12, 2:10] = 0.7
    heads["TC"][4:10, 4:10, 4:8] = 0.8
    heads["ET"][5:8, 5:8, 5:7] = 0.9
    for head in heads.values():
        head += rng.uniform(0.0, 0.3, shape).astype(np.float32)
    brain = np.ones(shape, dtype=bool)
    # ET 0.85, TC 0.75, WT 0.65: the background (< 0.3) never passes, every
    # region does, and a head scored under another head's name moves voxels.
    config = PostConfig(0.85, 0.75, 0.65, 4, 4, 20, 1, 1)

    original = tuple(infer.CHANNEL_ORDER)
    reference = postprocess(np.stack([heads[name] for name in original]), brain, config)
    assert set(np.unique(reference)) == {0, 1, 2, 3}

    try:
        for order in itertools.permutations(("ET", "TC", "WT")):
            probs = np.stack([heads[name] for name in order])
            passed = postprocess(probs, brain, config, channel_order=order)
            assert np.array_equal(passed, reference), order
            infer.CHANNEL_ORDER = order
            patched = postprocess(probs, brain, config)
            assert np.array_equal(patched, reference), order
    finally:
        infer.CHANNEL_ORDER = original

    try:
        postprocess(np.stack([heads["ET"]] * 3), brain, config, channel_order=("ET", "ET", "WT"))
        raise AssertionError("expected ValueError for a repeated head")
    except ValueError:
        pass


def test_config_paths_are_per_model():
    assert config_path_for("ours").name == "postprocess_ours.yaml"
    assert config_path_for("abhinav_v3").name == "postprocess_abhinav_v3.yaml"
    assert config_path_for("ours").is_file()


if __name__ == "__main__":
    test_yaml_loads_region_cutoffs()
    test_brain_mask_closes_and_fills_holes()
    test_brain_mask_keeps_a_border_touching_brain_from_filling_the_box()
    test_brain_mask_on_zscored_volumes_uses_nonzero_tissue()
    test_brain_mask_ignores_negative_fov_fill()
    test_hierarchy_component_filter_and_et_to_necrosis()
    test_et_kept_when_large_enough()
    test_shuffled_channel_order_gives_the_same_label_map()
    test_config_paths_are_per_model()
    print("postprocess ok")
