"""Rolling screenshot saver for inspecting the rendered GUI frame.

Enabled with ``--screenshot``. Writes the overlaid video frame (skeleton +
correction cues, exactly what the user sees) to ``screenshots/`` about once a
second, keeping only the newest few. The folder is wiped at the start of every
run so it always reflects the current session. Intended as a way to share what
the app is showing without a live screen.
"""
from __future__ import annotations

import glob
import os

import cv2

from core.config import PROJECT_ROOT

SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")


class ScreenshotSaver:
    def __init__(self, max_shots: int = 10, interval_sec: float = 1.0,
                 directory: str = SCREENSHOT_DIR) -> None:
        self.dir = directory
        self.max_shots = max_shots
        self.interval = interval_sec
        self._last = -1e9
        self._seq = 0
        self._reset_dir()

    def _reset_dir(self) -> None:
        """Clear any screenshots from a previous run (and create the folder)."""
        os.makedirs(self.dir, exist_ok=True)
        for path in glob.glob(os.path.join(self.dir, "shot_*.png")):
            try:
                os.remove(path)
            except OSError:
                pass

    def maybe_save(self, frame_bgr, now: float) -> None:
        """Save a BGR numpy frame if ``interval`` has passed (console path)."""
        if frame_bgr is None or (now - self._last) < self.interval:
            return
        self._last = now
        if cv2.imwrite(self._next_path(), frame_bgr):
            self._seq += 1
            self._prune()

    def maybe_save_widget(self, widget, now: float) -> None:
        """Grab a whole Qt widget/window to PNG (full-GUI screenshot path)."""
        if widget is None or (now - self._last) < self.interval:
            return
        self._last = now
        if widget.grab().save(self._next_path()):
            self._seq += 1
            self._prune()

    def _next_path(self) -> str:
        return os.path.join(self.dir, f"shot_{self._seq:06d}.png")

    def _prune(self) -> None:
        shots = sorted(glob.glob(os.path.join(self.dir, "shot_*.png")))
        for old in shots[:-self.max_shots]:
            try:
                os.remove(old)
            except OSError:
                pass
