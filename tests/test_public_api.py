"""Import surface of the split package matches the root script."""

import brats_recon
import recon

PUBLIC = [
    "CONN",
    "REGION_COLORS",
    "brain_mask_from_image",
    "check_mesh_volume",
    "mesh_from_mask",
    "postprocess",
    "predict_probs",
    "reconstruct_tumor",
    "remove_small_components",
    "warp_atlas_tracts",
]


def test_root_script_reexports_package():
    for name in PUBLIC:
        assert getattr(brats_recon, name) is getattr(recon, name)
