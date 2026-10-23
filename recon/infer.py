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

# train_brats(spatial_size=...) / RandCropByPosNegLabeld roi.
ROI_SIZE = (96, 96, 96)

# Sigmoid heads in the order the training loss supervises. exclusive_labels_to_regions
# stacks ET, TC, WT, and the checkpoint records out_regions in that order.
# Hard Dice on the 1-epoch smoke checkpoint does not support [TC, WT, ET]: every
# channel overlaps WT most, and the best permutation is only 0.006 above ET, TC, WT.
CHANNEL_ORDER = ("ET", "TC", "WT")

_OVERLAP = 0.625
_SW_BATCH_SIZE = 2
_TTA_FLIPS = ((2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4))


def default_device() -> torch.device:
    """``mps`` when that backend is available, otherwise CPU."""
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_device(device) -> torch.device:
    if device is None:
        return default_device()
    resolved = torch.device(device)
    if resolved.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return resolved


def load_model(ckpt_path, device) -> SegResNet:
    """Load a training SegResNet checkpoint and move it to ``device``.

    Accepts ``best_model.pt`` dictionaries (``model_state`` / ``model_state_dict``)
    or a raw state dict. Input and output channel counts come from the checkpoint.
    Architecture matches training: init filters 16, blocks ``(1, 2, 2, 4)`` /
    ``(1, 1, 1)``, dropout 0.2.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    dev = _as_device(device)
    blob = torch.load(path, map_location=dev, weights_only=False)
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
    return model


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


@torch.no_grad()
def predict_probs(model, image, roi=ROI_SIZE, tta=True):
    """Sliding-window probabilities, shape ``(3, H, W, D)``, order ``CHANNEL_ORDER``.

    ``image`` is a float array ``(C, H, W, D)`` normalized as in training.
    Gaussian window, overlap 0.625. With ``tta=True``, eight flips are averaged
    after the sigmoid.
    """
    device = next(model.parameters()).device
    model.eval()
    x = _as_batch(image, device)

    def run(t):
        logits = sliding_window_inference(
            t,
            roi,
            sw_batch_size=_SW_BATCH_SIZE,
            predictor=model,
            overlap=_OVERLAP,
            mode="gaussian",
        )
        return torch.sigmoid(logits)

    probs = run(x)
    if tta:
        for dims in _TTA_FLIPS:
            flipped = torch.flip(run(torch.flip(x, dims)), dims)
            probs = probs + flipped
        probs = probs / (1 + len(_TTA_FLIPS))
    return probs[0].float().cpu().numpy()
