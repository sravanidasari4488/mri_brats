"""BraTS reconstruction pipeline split out of the root ``brats_recon`` script.

Inference, post-processing, meshing, QA, scene export, and atlas tracts live in
submodules. Anatomy shells are opt-in via ``reconstruct_tumor(include_anatomy=True)``.
"""

from recon.evaluate import check_mesh_volume
from recon.infer import predict_probs
from recon.io import load_case
from recon.mesh import mesh_from_mask
from recon.postprocess import (
    CONN,
    PostConfig,
    brain_mask_from_image,
    postprocess,
    remove_small_components,
)
from recon.scene import REGION_COLORS, build_scene, reconstruct_tumor
from recon.tracts import compute_tract_proximity, warp_atlas_tracts

__all__ = [
    "CONN",
    "PostConfig",
    "REGION_COLORS",
    "brain_mask_from_image",
    "build_scene",
    "check_mesh_volume",
    "load_case",
    "mesh_from_mask",
    "postprocess",
    "predict_probs",
    "reconstruct_tumor",
    "compute_tract_proximity",
    "remove_small_components",
    "warp_atlas_tracts",
]
