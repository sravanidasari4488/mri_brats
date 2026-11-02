"""Stage clock and timeout helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.timing import STAGE_TIMEOUT_S, StageClock, run_with_timeout, stay_awake


def test_stage_clock_records_named_times():
    clock = StageClock()
    clock.start("inference")
    clock.start("postprocess")
    clock.stop()
    assert clock.times["inference"] is not None
    assert clock.times["postprocess"] is not None
    blob = clock.as_dict()
    assert "seconds" in blob and "status" in blob
    assert blob["timeout_s"] == STAGE_TIMEOUT_S


def test_run_with_timeout_raises():
    try:
        run_with_timeout(0.05, lambda: __import__("time").sleep(2))
    except TimeoutError as exc:
        assert "exceeded" in str(exc)
    else:
        raise AssertionError("expected TimeoutError")


def test_stay_awake_context():
    with stay_awake():
        pass


if __name__ == "__main__":
    test_stage_clock_records_named_times()
    test_run_with_timeout_raises()
    test_stay_awake_context()
    print("timing ok")
