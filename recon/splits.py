"""Model names and the case lists each checkpoint may be scored on.

``splits/abhinav_heldout.txt`` is the 60-case test set from
abhinavgadde1/MRI ``splits/heldout.txt``. ``abhinav_train.txt`` (289) and
``abhinav_val.txt`` (20) are ``checkpoints/brats_scale_full/train_cases.txt``
and ``train_run_val_cases.txt`` from that run. The three lists are disjoint
and cover all 369 BraTS 2020 training cases.
"""

from __future__ import annotations

import warnings
from pathlib import Path

OURS = "ours"
ABHINAV_V3 = "abhinav_v3"
MODEL_NAMES = (OURS, ABHINAV_V3)

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
ABHINAV_HELDOUT = SPLITS_DIR / "abhinav_heldout.txt"
ABHINAV_TRAIN = SPLITS_DIR / "abhinav_train.txt"
ABHINAV_VAL = SPLITS_DIR / "abhinav_val.txt"


class TrainingCaseError(ValueError):
    """Raised when metrics are requested on a case the model was trained on."""


def model_name(name: str) -> str:
    """Return ``ours`` or ``abhinav_v3``."""
    key = str(name).strip().lower()
    if key not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}")
    return key


def read_case_list(path) -> list[str]:
    """Case ids, one per line. Blank lines and ``#`` comments are ignored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Case list not found: {path}")
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            ids.append(text)
    return ids


def heldout_case_ids(model: str) -> list[str] | None:
    """Default evaluation cases for ``model``. ``None`` means use the local split."""
    if model_name(model) == ABHINAV_V3:
        return read_case_list(ABHINAV_HELDOUT)
    return None


def training_case_ids(model: str) -> frozenset[str]:
    if model_name(model) == ABHINAV_V3:
        return frozenset(read_case_list(ABHINAV_TRAIN))
    return frozenset()


def selection_case_ids(model: str) -> frozenset[str]:
    """Cases used to pick the checkpoint. Not trained on, so allowed with a warning."""
    if model_name(model) == ABHINAV_V3:
        return frozenset(read_case_list(ABHINAV_VAL))
    return frozenset()


def check_reportable(case_ids, model: str) -> None:
    """Refuse metrics on any case in ``model``'s training list."""
    ids = [str(case_id) for case_id in case_ids]
    trained = sorted(set(ids) & training_case_ids(model))
    if trained:
        shown = ", ".join(trained[:5]) + (" ..." if len(trained) > 5 else "")
        raise TrainingCaseError(
            f"{len(trained)} case(s) are in the {model} training list and cannot be scored: {shown}. "
            f"Use {ABHINAV_HELDOUT.relative_to(SPLITS_DIR.parent).as_posix()}."
        )
    selected = sorted(set(ids) & selection_case_ids(model))
    if selected:
        warnings.warn(
            f"{len(selected)} case(s) were used to select the {model} checkpoint: "
            + ", ".join(selected[:5]),
            stacklevel=2,
        )
