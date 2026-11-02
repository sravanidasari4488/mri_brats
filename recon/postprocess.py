"""Post-processing: brain mask, hierarchy, small-component removal, ET rule.

Probability channels are looked up by name in ``recon.infer.CHANNEL_ORDER``.
Thresholds and size cutoffs come from ``configs/postprocess_<model>.yaml`` via
``PostConfig``: ``postprocess_ours.yaml`` for the local SegResNet and
``postprocess_abhinav_v3.yaml`` for the teammate checkpoint.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from recon.splits import OURS, model_name

CONN = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity

_REGION_NAMES = ("ET", "TC", "WT")


def _yaml_number(value: float) -> str:
    text = f"{round(float(value), 4):.4f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def config_path_for(model: str = OURS) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / f"postprocess_{model_name(model)}.yaml"


def default_config_path() -> Path:
    return config_path_for(OURS)


@dataclass(frozen=True)
class PostConfig:
    """Every tunable cutoff used by the post-processing steps."""

    threshold_et: float = 0.5
    threshold_tc: float = 0.5
    threshold_wt: float = 0.5
    min_voxels_et: int = 50
    min_voxels_tc: int = 150
    min_voxels_wt: int = 300
    et_min_voxels: int = 100
    brain_closing_iterations: int = 2

    def threshold(self, region: str) -> float:
        return {
            "ET": self.threshold_et,
            "TC": self.threshold_tc,
            "WT": self.threshold_wt,
        }[region]

    def min_voxels(self, region: str) -> int:
        return {
            "ET": self.min_voxels_et,
            "TC": self.min_voxels_tc,
            "WT": self.min_voxels_wt,
        }[region]

    @classmethod
    def for_model(cls, model: str = OURS) -> PostConfig:
        """``configs/postprocess_<model>.yaml``, or the untuned defaults when it is missing."""
        path = config_path_for(model)
        if path.is_file():
            return cls.from_yaml(path)
        warnings.warn(
            f"{path.name} not found; using untuned defaults. "
            f"Run scripts/tune_postprocess.py --model {model_name(model)}.",
            stacklevel=2,
        )
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, *, model: str = OURS) -> PostConfig:
        """Load cutoffs from ``path``, default ``configs/postprocess_<model>.yaml``."""
        config_path = Path(path) if path is not None else config_path_for(model)
        if not config_path.is_file():
            raise FileNotFoundError(f"Post-process config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        thresholds = raw["thresholds"]
        min_voxels = raw["min_voxels"]
        return cls(
            threshold_et=float(thresholds["ET"]),
            threshold_tc=float(thresholds["TC"]),
            threshold_wt=float(thresholds["WT"]),
            min_voxels_et=int(min_voxels["ET"]),
            min_voxels_tc=int(min_voxels["TC"]),
            min_voxels_wt=int(min_voxels["WT"]),
            et_min_voxels=int(raw["et_min_voxels"]),
            brain_closing_iterations=int(raw["brain_closing_iterations"]),
        )

    def write_yaml(self, path: str | Path | None = None, *, model: str = OURS) -> Path:
        """Write this config to ``path``, default ``configs/postprocess_<model>.yaml``."""
        config_path = Path(path) if path is not None else config_path_for(model)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "# SegResNet head thresholds, keyed by region name (ET, TC, WT).\n"
            "# Updated by scripts/tune_postprocess.py.\n"
            "thresholds:\n"
            f"  ET: {_yaml_number(self.threshold_et)}\n"
            f"  TC: {_yaml_number(self.threshold_tc)}\n"
            f"  WT: {_yaml_number(self.threshold_wt)}\n"
            "\n"
            "# Minimum 26-connected component size, in voxels (mm^3 when the volume is 1 mm isotropic).\n"
            "min_voxels:\n"
            f"  ET: {int(self.min_voxels_et)}\n"
            f"  TC: {int(self.min_voxels_tc)}\n"
            f"  WT: {int(self.min_voxels_wt)}\n"
            "\n"
            "# If the remaining enhancing tumor is smaller than this, drop it. Those voxels stay NCR.\n"
            f"et_min_voxels: {int(self.et_min_voxels)}\n"
            "\n"
            "# binary_closing iterations on the nonzero-voxel brain mask before hole filling.\n"
            f"brain_closing_iterations: {int(self.brain_closing_iterations)}\n"
        )
        config_path.write_text(text, encoding="utf-8")
        return config_path


def foreground(volume) -> np.ndarray:
    """Brain voxels, excluding padding.

    Raw BraTS padding is 0. Some processed copies z-score only tissue and leave
    that padding at 0, so tissue is ``!= 0``. Others fill the field of view with
    small negatives; there ``!= 0`` is the whole box and tissue is ``> 0``.
    """
    volume = np.asarray(volume)
    positive = volume > 0
    nonzero = volume != 0
    if float(np.mean(nonzero)) > 0.5:
        return positive
    return nonzero


def brain_mask_from_image(image_np, config: PostConfig | None = None):
    """BraTS is skull-stripped: brain = any modality above its background. image_np: (C, H, W, D).

    The mask is padded before hole-filling so a brain that touches the volume
    border does not fill the whole box. Only the largest 26-connected component
    is kept.
    """
    if config is None:
        config = PostConfig.from_yaml()
    mask = np.stack([foreground(channel) for channel in np.asarray(image_np)]).any(0)
    pad = max(int(config.brain_closing_iterations), 0) + 1
    padded = np.pad(mask, pad)
    padded = ndimage.binary_closing(padded, CONN, iterations=config.brain_closing_iterations)
    filled = ndimage.binary_fill_holes(padded)[pad:-pad, pad:-pad, pad:-pad]
    labeled, count = ndimage.label(filled, structure=CONN)
    if count <= 1:
        return filled
    sizes = ndimage.sum(filled, labeled, index=np.arange(1, count + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def remove_small_components(mask, min_vox):
    lab, n = ndimage.label(mask, structure=CONN)
    if n == 0:
        return np.asarray(mask, dtype=bool)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep_ids = np.where(sizes >= min_vox)[0] + 1
    return np.isin(lab, keep_ids)


def postprocess(probs, brain_mask, config: PostConfig | None = None, channel_order=None):
    """Threshold heads, enforce ET ⊂ TC ⊂ WT, drop specks, and apply the ET rule.

    ``probs`` is ``(3, H, W, D)``. Each head is found by name in
    ``channel_order``, default ``CHANNEL_ORDER``. Labels are 1 = NCR,
    2 = ED, 3 = ET. When enhancing tumor is smaller than ``et_min_voxels``,
    it is cleared and those voxels stay necrotic core.
    """
    if config is None:
        config = PostConfig.from_yaml()
    from recon.infer import CHANNEL_ORDER

    order = tuple(CHANNEL_ORDER if channel_order is None else channel_order)
    if sorted(order) != sorted(_REGION_NAMES):
        raise ValueError(f"Channel order {order} must name each of {_REGION_NAMES} once")
    if np.shape(probs)[0] != len(order):
        raise ValueError(f"Expected {len(order)} probability channels, got {np.shape(probs)[0]}")

    def idx(name):
        return order.index(name)

    regions = {
        name: (probs[idx(name)] > config.threshold(name)) & brain_mask
        for name in _REGION_NAMES
    }
    wt = remove_small_components(regions["WT"], config.min_voxels("WT"))
    # enforce ET ⊂ TC ⊂ WT  (a voxel violating this cannot be correct)
    tc = remove_small_components(regions["TC"] & wt, config.min_voxels("TC"))
    et = remove_small_components(regions["ET"] & tc, config.min_voxels("ET"))

    if et.sum() < config.et_min_voxels:
        et = np.zeros(et.shape, dtype=bool)  # those voxels stay in TC -> labelled NCR below

    label = np.zeros(wt.shape, dtype=np.uint8)
    label[wt] = 2  # edema
    label[tc] = 1  # necrotic core
    label[et] = 3  # enhancing
    return label
