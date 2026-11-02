"""Stage clocks, per-stage timeouts, and keeping Windows awake during batch runs."""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import contextmanager

STAGE_TIMEOUT_S = 20 * 60
STAGE_NAMES = (
    "preprocess",
    "inference",
    "postprocess",
    "synthseg",
    "lobe_warp",
    "meshing",
    "export",
)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def stay_awake():
    """Prevent Windows sleep while a long batch is running."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


class StageClock:
    """Record wall times for named pipeline stages."""

    def __init__(self, timeout_s: float = STAGE_TIMEOUT_S):
        self.timeout_s = float(timeout_s)
        self.times: dict[str, float | None] = {name: None for name in STAGE_NAMES}
        self.status: dict[str, str] = {name: "pending" for name in STAGE_NAMES}
        self._name: str | None = None
        self._t0: float | None = None
        self._lock = threading.Lock()

    def start(self, name: str) -> None:
        self.stop()
        with self._lock:
            self._name = name
            self._t0 = time.perf_counter()
            if name in self.status:
                self.status[name] = "running"

    def stop(self) -> float | None:
        with self._lock:
            if self._name is None or self._t0 is None:
                return None
            elapsed = time.perf_counter() - self._t0
            name = self._name
            self.times[name] = elapsed
            self.status[name] = "done"
            self._name = None
            self._t0 = None
            return elapsed

    def fail(self, name: str | None = None) -> None:
        with self._lock:
            key = name or self._name
            if key:
                if self._name == key and self._t0 is not None:
                    self.times[key] = time.perf_counter() - self._t0
                self.status[key] = "timeout" if self.status.get(key) == "running" else "error"
            self._name = None
            self._t0 = None

    def skip(self, name: str) -> None:
        with self._lock:
            if self.status.get(name) == "pending":
                self.status[name] = "skipped"
                self.times[name] = 0.0

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "seconds": {name: None if value is None else round(float(value), 2) for name, value in self.times.items()},
                "status": dict(self.status),
                "timeout_s": self.timeout_s,
            }


def run_with_timeout(seconds: float, fn, *args, **kwargs):
    """Run ``fn`` and raise TimeoutError if it exceeds ``seconds``.

    The worker thread is abandoned on timeout (it cannot be killed). The
    caller should skip the stage and continue the batch.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=float(seconds))
        except FuturesTimeout as exc:
            raise TimeoutError(f"{getattr(fn, '__name__', 'stage')} exceeded {seconds:.0f}s") from exc
