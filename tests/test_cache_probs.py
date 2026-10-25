"""Resume helpers: a finished .npz is kept, a truncated one is not."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cache():
    path = ROOT / "scripts" / "cache_probs.py"
    spec = importlib.util.spec_from_file_location("cache_probs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_npz_lands_in_the_output_dir_and_reloads(tmp_path):
    cache = _cache()
    out_path = tmp_path / "BraTS20_Training_001.npz"
    probs = np.zeros((3, 2, 2, 2), dtype=np.float16)
    probs[0, 0, 0, 0] = np.float16(0.5)
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    cache._write_npz(out_path, probs=probs, label=label, case_id=np.array(out_path.stem))

    assert out_path.is_file()
    assert not (tmp_path / "BraTS20_Training_001.npz.partial").exists()
    assert cache._cache_exists(out_path)
    with np.load(out_path, allow_pickle=False) as data:
        assert data["probs"].shape == (3, 2, 2, 2)
        assert data["label"].shape == (2, 2, 2)


def test_truncated_and_empty_caches_are_not_skipped(tmp_path):
    cache = _cache()
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(b"not a zip")
    missing = tmp_path / "missing.npz"

    assert not cache._cache_exists(empty)
    assert not cache._cache_exists(truncated)
    assert not cache._cache_exists(missing)
