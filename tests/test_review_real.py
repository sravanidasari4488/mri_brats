"""Review-sheet slice selection, black background, and FAIL contours."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.review_real import (
    _gray_masked,
    brain_si_slices,
    brightest_flair_slice,
    empty_pred_slices,
)
from recon.slices import ORIENT_LETTERS, brain_window, orient_ras_slice


def test_brain_si_slices_at_30_50_70():
    mask = np.zeros((20, 20, 100), dtype=bool)
    mask[4:16, 4:16, 10:90] = True
    picks = brain_si_slices(mask)
    assert picks == [34, 50, 65]


def test_empty_pred_includes_brightest_flair():
    mask = np.zeros((12, 12, 40), dtype=bool)
    mask[2:10, 2:10, 5:35] = True
    flair = np.zeros((12, 12, 40), dtype=np.float32)
    flair[mask] = 10.0
    flair[6, 6, 22] = 400.0
    picks = empty_pred_slices(flair, mask)
    assert brightest_flair_slice(flair, mask) == 22
    assert 22 in picks
    lo, hi = 5, 34
    assert int(round(lo + 0.3 * (hi - lo))) in picks
    assert int(round(lo + 0.5 * (hi - lo))) in picks
    assert int(round(lo + 0.7 * (hi - lo))) in picks


def test_windowed_background_is_black():
    volume = np.zeros((16, 16, 8), dtype=np.float32)
    mask = np.zeros((16, 16, 8), dtype=bool)
    mask[4:12, 4:12, :] = True
    volume[mask] = np.linspace(-2, 4, int(mask.sum()), dtype=np.float32)
    lo, hi = brain_window(volume, mask)
    rgb = _gray_masked(volume, mask, 4, lo, hi)
    keep = orient_ras_slice(mask, 2, 4)
    assert rgb[~keep].max() == 0
    assert rgb[keep].max() > 0


def test_axial_letters_are_radiological():
    letters = ORIENT_LETTERS[2]
    assert letters["left"] == "R"
    assert letters["right"] == "L"
    assert letters["top"] == "A"
    assert letters["bottom"] == "P"


if __name__ == "__main__":
    test_brain_si_slices_at_30_50_70()
    test_empty_pred_includes_brightest_flair()
    test_windowed_background_is_black()
    test_axial_letters_are_radiological()
    print("review_real ok")
