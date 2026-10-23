"""
BraTS 3D reconstruction stage: high-precision inference -> FP-reducing post-processing -> meshes.

Assumes a trained MONAI SegResNet with 3 sigmoid output channels.
This checkpoint's heads are ET, TC, WT (``recon.infer.CHANNEL_ORDER``), which is
not ConvertToMultiChannelBasedOnBratsClassesd's [TC, WT, ET].

Label convention (BraTS 2021+): 1 = NCR, 2 = ED, 3 = ET   (older BraTS used 4 for ET)

pip install monai nibabel scipy scikit-image trimesh antspyx

Implementations live in the ``recon`` package. Imports below stay on this module
so existing ``import brats_recon`` / ``from brats_recon import ...`` call sites
keep working, including the example CLI's ``nib`` name.
"""

import numpy as np
import torch
import nibabel as nib
import trimesh
from scipy import ndimage
from skimage import measure
from monai.inferers import sliding_window_inference

from recon import (
    CONN,
    REGION_COLORS,
    brain_mask_from_image,
    build_scene,
    check_mesh_volume,
    mesh_from_mask,
    postprocess,
    predict_probs,
    reconstruct_tumor,
    compute_tract_proximity,
    remove_small_components,
    warp_atlas_tracts,
)

__all__ = [
    "np",
    "torch",
    "nib",
    "trimesh",
    "ndimage",
    "measure",
    "sliding_window_inference",
    "CONN",
    "REGION_COLORS",
    "brain_mask_from_image",
    "build_scene",
    "check_mesh_volume",
    "mesh_from_mask",
    "postprocess",
    "predict_probs",
    "reconstruct_tumor",
    "compute_tract_proximity",
    "remove_small_components",
    "warp_atlas_tracts",
]


# --------------------------------------------------------------------------
# Example
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # model = ...load your best SegResNet checkpoint...
    # image_t: (4,H,W,D) tensor after YOUR training preprocessing (same orientation/normalization)
    # ref = nib.load("BraTS-XXX-t1n.nii.gz"); affine = ref.affine
    #
    # probs = predict_probs(model, image_t)
    # brain = brain_mask_from_image(image_t.numpy())
    # label = postprocess(probs, brain)
    # nib.save(nib.Nifti1Image(label, affine), "pred_seg.nii.gz")   # open in ITK-SNAP to QA
    # reconstruct_tumor(label, affine, "tumor.glb")
    pass
