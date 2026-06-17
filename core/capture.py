"""Webcam capture loop with a single shared frame buffer (Section 3).

A background thread continuously grabs frames and stores the most recent one
under a lock. Consumers (inference) read the latest frame without queues or
per-frame copies piling up, keeping memory flat.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("capture")


def _candidate_backends() -> List[int]:
    """OS-appropriate capture backends to try, most-preferred first.

    Falls back to CAP_ANY everywhere so the app still works on OpenCV builds
    that lack a given backend constant.
    """
    system = platform.system()
    names = {
        "Windows": ["CAP_DSHOW", "CAP_MSMF"],
        "Darwin": ["CAP_AVFOUNDATION"],
        "Linux": ["CAP_V4L2"],
    }.get(system, [])
    backends: List[int] = []
    for name in names:
        val = getattr(cv2, name, None)
        if isinstance(val, int):
            backends.append(val)
    backends.append(getattr(cv2, "CAP_ANY", 0))
    # De-dupe while preserving order.
    seen, ordered = set(), []
    for b in backends:
        if b not in seen:
            seen.add(b)
            ordered.append(b)
    return ordered


class Camera:
    """Threaded camera reader exposing the latest downscaled BGR frame."""

    def __init__(
        self,
        index: int = 0,
        width: int = 320,
        height: int = 240,
        crop_top_fraction: float = 1.0,
        mirror: bool = True,
        fail_threshold: int = 30,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.crop_top_fraction = max(0.1, min(1.0, crop_top_fraction))
        self.mirror = mirror
        # Consecutive failed reads before we tear down and re-open the device
        # (recovers from sleep/wake invalidating the capture handle).
        self.fail_threshold = max(5, fail_threshold)

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_id = 0

        # Health/recovery bookkeeping (read by the pipeline watchdog).
        self._consecutive_failures = 0
        self._last_success = time.monotonic()
        self.reopen_count = 0
        self._force_reopen = threading.Event()

        # Diagnostics: which device/backend actually worked + native resolution.
        self.active_index: Optional[int] = None
        self.active_backend: Optional[int] = None
        self.native_width: int = 0
        self.native_height: int = 0

    # -- lifecycle ---------------------------------------------------------
    def _try_open(self, index: int, backend: int) -> Optional[cv2.VideoCapture]:
        """Open one (index, backend) combo and validate it yields a frame."""
        try:
            cap = cv2.VideoCapture(index, backend)
        except Exception:
            return None
        if not cap or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # isOpened() can lie on some drivers — confirm we can actually read.
        ok, frame = cap.read()
        if not ok or frame is None:
            try:
                cap.release()
            except Exception:
                pass
            return None
        self.active_index = index
        self.active_backend = backend
        self.native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
        self.native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        return cap

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open the camera, trying OS-appropriate backends and index fallbacks.

        Makes the app portable across Windows (DirectShow/MSMF), macOS
        (AVFoundation) and Linux (V4L2), and tolerant of the configured index
        being wrong (built-in vs. external webcams).
        """
        backends = _candidate_backends()
        # Preferred index first, then a small range of common fallbacks.
        indices = [self.index] + [i for i in range(0, 4) if i != self.index]
        for index in indices:
            for backend in backends:
                cap = self._try_open(index, backend)
                if cap is not None:
                    if index != self.index or backend != backends[0]:
                        log.info("camera opened on index %d via backend %s",
                                 index, backend)
                    return cap
        return None

    def start(self) -> "Camera":
        if self._thread is not None:
            return self
        self._cap = self._open_capture()
        if self._cap is None:
            raise RuntimeError(
                "No working camera found. Tried indices 0-3 with the "
                f"{platform.system()} capture backends. Check a webcam is "
                "connected, not in use by another app, and that camera "
                "permissions are granted to your terminal/app."
            )
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Honour an externally requested reopen (e.g. after sleep/wake).
            if self._force_reopen.is_set():
                self._force_reopen.clear()
                self._reopen()

            cap = self._cap
            ok, frame = (False, None) if cap is None else cap.read()
            if not ok or frame is None:
                self._consecutive_failures += 1
                # After enough failures the handle is almost certainly stale
                # (camera released during sleep) — re-open it.
                if self._consecutive_failures >= self.fail_threshold:
                    self._reopen()
                time.sleep(0.03)
                continue

            self._consecutive_failures = 0
            self._last_success = time.monotonic()
            frame = self._preprocess(frame)
            with self._lock:
                self._latest = frame
                self._frame_id += 1
            # Tiny sleep yields the GIL; the inference thread paces real rate.
            time.sleep(0.001)

    def _reopen(self) -> None:
        """Release and re-open the capture device. Runs on the capture thread."""
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        time.sleep(0.3)  # give the OS a moment to release the device
        cap = self._open_capture()
        if cap is not None:
            self._cap = cap
            self._consecutive_failures = 0
            self._last_success = time.monotonic()
            self.reopen_count += 1
            log.info("camera re-opened (reopen #%d)", self.reopen_count)
        else:
            # Couldn't grab it yet; back off and let the loop retry.
            time.sleep(0.5)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Downscale to inference resolution (Section 3).
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Mirror to a selfie view so on-screen left/right match the user's body
        # (done before inference so landmark coords stay consistent downstream).
        if self.mirror:
            frame = cv2.flip(frame, 1)
        if self.crop_top_fraction < 1.0:
            h = int(frame.shape[0] * self.crop_top_fraction)
            frame = frame[:h, :, :]
        return frame

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """Return (copy of latest frame, frame_id). frame is None until ready."""
        with self._lock:
            if self._latest is None:
                return None, self._frame_id
            return self._latest.copy(), self._frame_id

    # -- health / recovery -------------------------------------------------
    def force_reopen(self) -> None:
        """Request the capture thread to tear down and re-open the device."""
        self._force_reopen.set()

    def frame_age(self) -> float:
        """Seconds since the last successfully grabbed frame."""
        return time.monotonic() - self._last_success

    def is_healthy(self) -> bool:
        """True when we have a recent, fresh frame to work with."""
        with self._lock:
            has_frame = self._latest is not None
        return has_frame and self.frame_age() < 1.5

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
