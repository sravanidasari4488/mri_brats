"""Hierarchy, component removal, ET-to-necrosis, and the yaml config."""

import numpy as np

from recon.postprocess import PostConfig, brain_mask_from_image, postprocess


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
