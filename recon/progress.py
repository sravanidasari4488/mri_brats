"""Staged reconstruction progress for the viewer job poller."""

from __future__ import annotations

import threading
import time

STAGES = ("Loading", "Tumor", "Anatomy", "Lobes", "Meshing", "Done")


class BuildProgress:
    """One reconstruction's stage times. Safe to snapshot from another thread."""

    def __init__(self):
        now = time.monotonic()
        self._lock = threading.Lock()
        self.stage = "Loading"
        self.status = "running"
        self.error = None
        self.warning = None
        self.started = now
        self._stage_started = now
        self.stages = {name: {"seconds": None, "status": "pending"} for name in STAGES}
        self.stages["Loading"]["status"] = "running"

    def set_stage(self, name: str) -> None:
        if name not in self.stages:
            raise ValueError(f"Unknown stage {name!r}")
        with self._lock:
            if self.stage == name and self.stages[name]["status"] == "running":
                return
            self._close_current()
            self.stage = name
            self._stage_started = time.monotonic()
            if name != "Done":
                self.stages[name]["status"] = "running"
                self.stages[name]["seconds"] = None

    def skip(self, name: str) -> None:
        with self._lock:
            if self.stages[name]["status"] == "pending":
                self.stages[name]["status"] = "skipped"
                self.stages[name]["seconds"] = 0.0

    def finish(self, warning: str | None = None) -> None:
        with self._lock:
            self.warning = warning
            self._close_current()
            self.stage = "Done"
            self.status = "done"
            self.stages["Done"]["status"] = "done"
            self.stages["Done"]["seconds"] = round(time.monotonic() - self.started, 2)

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = str(message)
            self.status = "error"
            self._close_current()
            if self.stage in self.stages:
                self.stages[self.stage]["status"] = "error"

    def snapshot(self) -> dict:
        with self._lock:
            if self.status == "running" and self.stage in self.stages:
                current = dict(self.stages[self.stage])
                current["seconds"] = round(time.monotonic() - self._stage_started, 2)
                stages = {name: dict(info) for name, info in self.stages.items()}
                stages[self.stage] = current
            else:
                stages = {name: dict(info) for name, info in self.stages.items()}
            return {
                "status": self.status,
                "stage": self.stage,
                "stages": stages,
                "error": self.error,
                "warning": self.warning,
                "elapsed": round(time.monotonic() - self.started, 2),
            }

    def _close_current(self) -> None:
        name = self.stage
        if name not in self.stages:
            return
        if self.stages[name]["status"] == "running":
            self.stages[name]["status"] = "done"
            self.stages[name]["seconds"] = round(time.monotonic() - self._stage_started, 2)
