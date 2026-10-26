"""Inference: Gaussian sliding window plus flip test-time averaging.

The sliding-window ROI is the training crop, ``(96, 96, 96)``. Probabilities
are averaged in probability space (sigmoid, then mean over the original view
and seven axis flips). ``CHANNEL_ORDER`` names the three output heads and is
checked by ``scripts/verify_channels.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from recon.splits import ABHINAV_V3, MODEL_NAMES, OURS, model_name

# train_brats(spatial_size=...) / RandCropByPosNegLabeld roi.
ROI_SIZE = (96, 96, 96)

# Sigmoid heads in the order the training loss supervises. exclusive_labels_to_regions
# stacks ET, TC, WT, and the checkpoint records out_regions in that order.
# Hard Dice on the 1-epoch smoke checkpoint does not support [TC, WT, ET]: every
# channel overlaps WT most, and the best permutation is only 0.006 above ET, TC, WT.
CHANNEL_ORDER = ("ET", "TC", "WT")

# third_party/abhinav_mri src/segmentation/model.py BRATS_REGIONS:
# enhancing_tumor, tumor_core, whole_tumor. Same order as CHANNEL_ORDER.
# Their train_config roi_size and pixdim are 96^3 and 1 mm.
ABHINAV_CHANNEL_ORDER = ("ET", "TC", "WT")
ABHINAV_ROI = (96, 96, 96)
ABHINAV_MODALITIES = ("FLAIR", "T1", "T1c", "T2")
ABHINAV_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "brats_scale_full" / "best_model.pt"

_OVERLAP = 0.625
_SW_BATCH_SIZE = 2
_SW_BATCH_SIZE_CUDA = 4
# Axis pairs, then the three double flips, then all three. ``fast`` uses the first three.
_TTA_FLIPS = ((2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4))


def default_device() -> torch.device:
    """``cuda`` when a GPU is available, then ``mps``, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_device(device) -> torch.device:
    if device is None:
        return default_device()
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return resolved


def channel_index(name: str, order=None) -> int:
    """Position of region ``name`` (ET, TC, WT) in ``order``, default ``CHANNEL_ORDER``."""
    names = tuple(CHANNEL_ORDER if order is None else order)
    if name not in names:
        raise KeyError(f"Region {name!r} is not in channel order {names}")
    return names.index(name)


def output_order(name: str) -> tuple[str, ...]:
    """Sigmoid-head order produced by ``name`` before it is aligned to ``CHANNEL_ORDER``."""
    if model_name(name) == ABHINAV_V3:
        return ABHINAV_CHANNEL_ORDER
    return CHANNEL_ORDER


def roi_for(name: str) -> tuple[int, int, int]:
    """Sliding-window crop. Abhinav v3 was trained at ``(96, 96, 96)``."""
    if model_name(name) == ABHINAV_V3:
        return ABHINAV_ROI
    return ROI_SIZE


def resolve_checkpoint(name: str, explicit=None) -> Path:
    """Checkpoint path for ``name``. An explicit path wins.

    Abhinav v3 is ``checkpoints/brats_scale_full/best_model.pt``.
    """
    if explicit is not None:
        return Path(explicit)
    if model_name(name) != ABHINAV_V3:
        raise ValueError("Pass a checkpoint path for the local SegResNet")
    path = ABHINAV_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(f"No abhinav_v3 checkpoint at {path}")
    return path


def load_model(ckpt_path, device, name: str = OURS) -> SegResNet:
    """Load a training SegResNet checkpoint and move it to ``device``.

    ``ours`` accepts ``best_model.pt`` dictionaries (``model_state`` /
    ``model_state_dict``) or a raw state dict. Input and output channel counts
    come from the checkpoint. Architecture matches training: init filters 16,
    blocks ``(1, 2, 2, 4)`` / ``(1, 1, 1)``, dropout 0.2.

    ``abhinav_v3`` builds that same SegResNet with 4 inputs and 3 outputs and
    loads ``ckpt["model_state"]`` with ``strict=True``.
    """
    kind = model_name(name)
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    dev = _as_device(device)
    blob = torch.load(path, map_location=dev, weights_only=False)
    if kind == ABHINAV_V3:
        if not isinstance(blob, dict) or "model_state" not in blob:
            raise KeyError(f"abhinav_v3 checkpoint {path} has no 'model_state'")
        state = blob["model_state"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            init_filters=16,
            dropout_prob=0.2,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
        model.load_state_dict(state, strict=True)
    else:
        if isinstance(blob, dict) and "model_state" in blob:
            state = blob["model_state"]
        elif isinstance(blob, dict) and "model_state_dict" in blob:
            state = blob["model_state_dict"]
        elif isinstance(blob, dict) and "convInit.conv.weight" in blob:
            state = blob
        else:
            raise KeyError(f"No SegResNet weights in {path}")

        state = {k.removeprefix("module."): v for k, v in state.items()}
        in_channels = int(state["convInit.conv.weight"].shape[1])
        out_channels = int(state["conv_final.2.conv.weight"].shape[0])
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=16,
            dropout_prob=0.2,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
        model.load_state_dict(state)
    model.to(dev)
    model.eval()
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return model


def reorder_probabilities(probs, source_order) -> np.ndarray:
    """Permute ``probs`` from ``source_order`` into ``CHANNEL_ORDER``."""
    source = tuple(source_order)
    target = tuple(CHANNEL_ORDER)
    array = np.asarray(probs)
    if source == target:
        return array
    missing = [name for name in target if name not in source]
    if missing or len(source) != array.shape[0]:
        raise ValueError(f"Cannot reorder {source} into {target} for shape {array.shape}")
    return array[[source.index(name) for name in target]]


def validation_transform(name: str = OURS):
    """Full-volume validation ``Compose`` for ``name``.

    ``ours`` is RAS, 1 mm, legacy label 4 rewritten to 3, then nonzero z-score.
    ``abhinav_v3`` is ``get_val_transforms`` from their ``src/segmentation/dataset.py``:
    no orientation change, 1 mm spacing, nonzero per-channel z-score.
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        MapLabelValued,
        NormalizeIntensityd,
        Orientationd,
        Spacingd,
    )

    kind = model_name(name)
    keys = ["image", "label"]
    if kind == ABHINAV_V3:
        return Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                EnsureTyped(keys=keys),
            ]
        )
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            MapLabelValued(keys=["label"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys),
        ]
    )


def apply_model_preprocessing(image, affine, name: str = OURS):
    """Network input for a case this repo already loaded in RAS.

    ``ours`` expects the nonzero z-score stack ``load_case`` already built.
    ``abhinav_v3`` expects raw intensities in FLAIR, T1, T1c, T2 order. It
    resamples to 1 mm and applies their nonzero per-channel z-score. The
    second return value is the affine of that network grid.
    """
    array = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    grid = np.asarray(affine, dtype=np.float64)
    if model_name(name) != ABHINAV_V3:
        return array, grid
    if array.ndim != 4 or array.shape[0] != 4:
        raise ValueError(f"abhinav_v3 expects (4, H, W, D) raw intensities, got {array.shape}")
    from monai.data import MetaTensor
    from monai.transforms import NormalizeIntensity, Spacing

    meta = MetaTensor(array, affine=grid)
    spaced = Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear")(meta)
    normalized = NormalizeIntensity(nonzero=True, channel_wise=True)(spaced)
    out_affine = np.asarray(getattr(spaced, "affine", grid), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(normalized), dtype=np.float32), out_affine


def resample_probabilities(probs, src_affine, dst_shape, dst_affine) -> np.ndarray:
    """Linearly resample channel maps onto the original scan grid."""
    array = np.asarray(probs, dtype=np.float32)
    target_shape = tuple(int(value) for value in dst_shape)
    if tuple(array.shape[1:]) == target_shape:
        return array
    import nibabel as nib
    from nibabel.processing import resample_from_to

    target = (target_shape, np.asarray(dst_affine, dtype=np.float64))
    src = np.asarray(src_affine, dtype=np.float64)
    channels = []
    for channel in array:
        image = nib.Nifti1Image(np.ascontiguousarray(channel, dtype=np.float32), src)
        moved = resample_from_to(image, target, order=1, cval=0.0)
        channels.append(np.asarray(moved.dataobj, dtype=np.float32))
    return np.stack(channels, axis=0)


def _as_batch(image, device: torch.device) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach()
        array = tensor.float().cpu().numpy() if tensor.ndim == 4 else None
        if array is None:
            raise ValueError(f"image must be (C, H, W, D), got {tuple(image.shape)}")
    else:
        array = np.asarray(image)
    if array.ndim != 4:
        raise ValueError(f"image must be (C, H, W, D), got {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None].to(device)


def _tta_flips(tta):
    """``True`` is eight views, ``"fast"`` is four, ``False`` is the original view only."""
    if tta == "fast":
        return _TTA_FLIPS[:3]
    if tta:
        return _TTA_FLIPS
    return ()


@torch.no_grad()
def predict_probs(model, image, roi=ROI_SIZE, tta=True, *, overlap=_OVERLAP, sw_batch_size=None, model_name: str = OURS):
    """Sliding-window probabilities, shape ``(3, H, W, D)``, order ``CHANNEL_ORDER``.

    ``image`` is a float array ``(C, H, W, D)`` normalized as in training.
    Gaussian window, overlap 0.625 unless ``overlap`` is set. ``tta=True``
    averages eight flips. ``tta="fast"`` averages the three single-axis flips.
    CUDA runs the window in float16 and uses a larger batch.

    ``model_name="abhinav_v3"`` uses their head order and permutes it into
    ``CHANNEL_ORDER`` (ET, TC, WT). Pass ``roi_for(model_name)``; that crop
    is ``(96, 96, 96)``.
    """
    device = next(model.parameters()).device
    model.eval()
    x = _as_batch(image, device)
    if sw_batch_size is None:
        sw_batch_size = _SW_BATCH_SIZE_CUDA if device.type == "cuda" else _SW_BATCH_SIZE
    flips = _tta_flips(tta)
    use_amp = device.type == "cuda"

    def run(t):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = sliding_window_inference(
                t,
                roi,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
        return torch.sigmoid(logits.float())

    probs = run(x)
    for dims in flips:
        flipped = torch.flip(run(torch.flip(x, dims)), dims)
        probs = probs + flipped
    if flips:
        probs = probs / (1 + len(flips))
    array = probs[0].float().cpu().numpy()
    return reorder_probabilities(array, output_order(model_name))
