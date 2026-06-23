"""Processing pipeline tying capture -> engines -> scoring -> state together.

A single ``Processor`` is shared by every run mode (console, background tray,
full GUI). Each mode just calls :meth:`process_once` at its target frame rate,
or runs :meth:`run_loop` in a background thread.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .breathing import BreathingEstimator, BreathingMetrics
from .calibration import compute_baseline
from .capture import Camera
from .near_work import NearWorkEstimator, NearWorkMetrics
from .config import save_baseline
from .face_engine import FaceEngine, FaceMetrics
from .gaze import is_looking_at_screen
from .health_log import HealthLogger
from .pose_engine import PoseEngine, PoseMetrics
from .posture_score import ScoreResult, compute_score
from .state_manager import Event, Snapshot, StateManager

log = logging.getLogger("pipeline")


@dataclass
class FrameResult:
    """Everything one processed frame produces (consumed by the GUI overlay)."""

    frame: Optional[np.ndarray] = None
    pose: Optional[PoseMetrics] = None
    face: Optional[FaceMetrics] = None
    score: Optional[ScoreResult] = None
    snapshot: Optional[Snapshot] = None
    breathing: Optional[BreathingMetrics] = None
    near_work: Optional[NearWorkMetrics] = None
    gaze_on_screen: bool = True
    recalibrating: bool = False


class Processor:
    def __init__(self, cfg: dict, baseline: Optional[dict],
                 on_events: Optional[Callable[[List[Event]], None]] = None,
                 on_status: Optional[Callable[[str, str], None]] = None):
        self.cfg = cfg
        self.baseline = baseline
        self.on_events = on_events
        # on_status(title, message): user-facing status notifications that
        # bypass the rate-limited reminder events (e.g. "recording new
        # baseline"). on_recalibration_start/end let a GUI un/re-minimise.
        self.on_status = on_status
        self.on_recalibration_start: Optional[Callable[[], None]] = None
        self.on_recalibration_end: Optional[Callable[[], None]] = None

        cam = cfg["camera"]
        self.camera = Camera(
            index=cam["index"], width=cam["width"], height=cam["height"],
            crop_top_fraction=cam["crop_top_fraction"], mirror=cam.get("mirror", True),
            fail_threshold=cfg.get("recovery", {}).get("camera_fail_threshold", 30),
        )
        self.pose = PoseEngine(cfg["models"]["pose_task"])
        self.face = FaceEngine(cfg, cfg["models"]["face_task"])
        self.state = StateManager(cfg)
        self.health = HealthLogger(cfg)
        self.breathing = (
            BreathingEstimator(cfg) if cfg["breathing"]["enabled"] else None
        )
        self.near_work = (
            NearWorkEstimator(cfg) if cfg.get("near_work", {}).get("enabled", False)
            else None
        )

        # Optional IMU bridge — lazy-loaded only when enabled (Section 3/8).
        self.imu = None
        if cfg["imu"]["enabled"]:
            try:
                from .imu_bridge import IMUBridge
                self.imu = IMUBridge(cfg).start()
                log.info("IMU bridge enabled (%s)", cfg["imu"]["connection"])
            except Exception as exc:
                log.warning("IMU init failed, camera-only: %s", exc)

        # EMA state for temporal smoothing of scalar metrics (anti-jitter).
        self._smoothing = float(cfg["posture"].get("smoothing", 0.0) or 0.0)
        self._ema: dict = {}

        self._latest = FrameResult()
        self._latest_lock = threading.Lock()
        self._ts0 = time.time()
        self._last_ts_ms = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Recovery / recalibration bookkeeping (Issue 1/2).
        self._last_wall: Optional[float] = None
        self._last_reopen_count = self.camera.reopen_count
        self._pending_recalib = False
        self._inference_errors = 0
        self.recalibrating = False
        self._recalib_lock = threading.Lock()
        self._recalib_thread: Optional[threading.Thread] = None
        self._pause_before_recalib = False

        # Performance instrumentation (read by the hardware diagnostics).
        self._perf_lock = threading.Lock()
        self._proc_ms_ema: Optional[float] = None
        self._frame_wall: deque = deque(maxlen=90)
        self._frames_processed = 0

    # ------------------------------------------------------------------
    def start_camera(self) -> None:
        self.camera.start()

    def process_once(self) -> Optional[FrameResult]:
        # Watchdog: detect sleep/wake + camera reconnects before reading.
        self._check_recovery()

        frame, _ = self.camera.read()
        if frame is None:
            return None
        ts_ms = self._next_ts()
        t_start = time.perf_counter()

        try:
            pose_m = self.pose.process(frame, ts_ms)
            face_m = self.face.process(frame, ts_ms)
        except Exception as exc:
            # MediaPipe can throw on a corrupted/stale stream after resume.
            # Count errors and rebuild the engines rather than crashing.
            self._inference_errors += 1
            log.warning("inference error (%d): %s", self._inference_errors, exc)
            if self._inference_errors >= self.cfg["recovery"].get(
                    "inference_error_threshold", 10):
                self._recreate_engines()
                self._inference_errors = 0
            return None
        self._inference_errors = 0
        self._apply_smoothing(pose_m, face_m)

        fused = None
        if self.imu is not None and self.baseline is not None and pose_m.detected:
            cam_dev = pose_m.fha_deg - self.baseline.get("pose", {}).get("fha_deg", 0.0)
            fused = self.imu.fused_fha_dev(cam_dev, self.baseline)

        now = time.monotonic()
        breath_m = self.breathing.update(pose_m, now) if self.breathing else None
        near_m = self.near_work.update(pose_m, self.baseline, now) if self.near_work else None

        score = compute_score(pose_m, face_m, self.baseline, self.cfg, fused)
        events = self.state.update(pose_m, face_m, score, breathing=breath_m,
                                   near_work=near_m, now=now)
        snap = self.state.snapshot()
        self.health.observe(snap)

        if events and self.on_events:
            self.on_events(events)

        result = FrameResult(
            frame=frame, pose=pose_m, face=face_m, score=score, snapshot=snap,
            breathing=breath_m, near_work=near_m,
            gaze_on_screen=is_looking_at_screen(face_m, self.cfg) if face_m else False,
            recalibrating=self.recalibrating,
        )
        with self._latest_lock:
            self._latest = result

        # Record processing cost + arrival time for the diagnostics check.
        proc_ms = (time.perf_counter() - t_start) * 1000.0
        with self._perf_lock:
            self._proc_ms_ema = (proc_ms if self._proc_ms_ema is None
                                 else 0.8 * self._proc_ms_ema + 0.2 * proc_ms)
            self._frame_wall.append(time.monotonic())
            self._frames_processed += 1
        return result

    def get_perf(self) -> dict:
        """Snapshot of achieved throughput for the hardware diagnostics."""
        with self._perf_lock:
            times = list(self._frame_wall)
            proc_ms = self._proc_ms_ema
            n = self._frames_processed
        fps = 0.0
        if len(times) >= 2:
            span = times[-1] - times[0]
            if span > 0:
                fps = (len(times) - 1) / span
        return {"proc_ms": proc_ms or 0.0, "fps": fps, "frames": n}

    # -- timestamps & recovery ------------------------------------------
    def _next_ts(self) -> int:
        """Strictly increasing millisecond timestamp for MediaPipe VIDEO mode.

        After a sleep/wake the wall clock jumps; clamping to be monotonic
        avoids the 'timestamp must be monotonically increasing' crash.
        """
        ts = int((time.time() - self._ts0) * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def _check_recovery(self) -> None:
        """Detect sleep/wake or a camera reconnect and schedule recovery."""
        rc = self.cfg["recovery"]
        wall = time.time()
        if self._last_wall is not None:
            gap = wall - self._last_wall
            if gap > rc.get("sleep_resume_gap_sec", 20):
                log.warning("Resumed after %.0fs gap (sleep/wake) — "
                            "re-initialising camera and engines", gap)
                self.camera.force_reopen()
                self._recreate_engines()
                self._pending_recalib = True
        self._last_wall = wall

        # Camera self-healed (e.g. handle went stale) -> fresh baseline.
        if self.camera.reopen_count != self._last_reopen_count:
            self._last_reopen_count = self.camera.reopen_count
            self._pending_recalib = True

        # Fire the recalibration once the camera is producing frames again.
        if (self._pending_recalib and not self.recalibrating
                and self.camera.is_healthy()):
            self._pending_recalib = False
            if rc.get("recalibrate_on_resume", True) and self.baseline is not None:
                self.trigger_recalibration("camera restart")

    def _recreate_engines(self) -> None:
        """Tear down and rebuild the MediaPipe engines (post-resume safety)."""
        for eng in (getattr(self, "pose", None), getattr(self, "face", None)):
            try:
                if eng is not None:
                    eng.close()
            except Exception:
                pass
        try:
            self.pose = PoseEngine(self.cfg["models"]["pose_task"])
            self.face = FaceEngine(self.cfg, self.cfg["models"]["face_task"])
        except Exception as exc:
            log.error("engine reinit failed: %s", exc)
        # New engines start a fresh timestamp series.
        self._ts0 = time.time()
        self._last_ts_ms = -1
        self._ema.clear()

    # -- recalibration ---------------------------------------------------
    def trigger_recalibration(self, reason: str = "manual") -> bool:
        """Start a non-blocking baseline re-record using the shared camera.

        Returns False if one is already running. The video feed keeps updating
        because process_once continues to run on its caller's thread; this only
        spins a sampler that reads the latest metrics (Issue 2).
        """
        with self._recalib_lock:
            if self.recalibrating:
                return False
            self.recalibrating = True

        # Avoid spurious posture nudges while the user settles into position.
        self._pause_before_recalib = self.state.paused
        self.state.set_paused(True)

        if self.on_status:
            self.on_status(
                "Recalibrating posture",
                "Recording a new baseline — sit upright for a few seconds.")
        self._safe_call(self.on_recalibration_start)

        self._recalib_thread = threading.Thread(
            target=self._recalibration_worker, args=(reason,),
            name="recalib", daemon=True)
        self._recalib_thread.start()
        return True

    def _recalibration_worker(self, reason: str) -> None:
        dur = float(self.cfg["recovery"].get("recalibration_duration_sec", 5))
        pose_samples: List = []
        yaw: List[float] = []
        head_pitch: List[float] = []
        imu_samples: List = []

        # Wait for the camera to come back before sampling (bounded).
        t_wait = time.monotonic()
        while not self._stop.is_set() and not self.camera.is_healthy():
            if time.monotonic() - t_wait > 15:
                break
            time.sleep(0.1)

        last_id = None
        start = time.monotonic()
        while not self._stop.is_set() and time.monotonic() - start < dur:
            res = self.latest()
            if res is not None and id(res) != last_id:
                last_id = id(res)
                pm, fm = res.pose, res.face
                if pm is not None and pm.detected and getattr(pm, "reliable", False):
                    pose_samples.append(pm)
                if fm is not None and fm.detected:
                    yaw.append(fm.head_yaw_deg)
                    head_pitch.append(fm.head_pitch_deg)
                if self.imu is not None:
                    r = self.imu.read()
                    if r.fresh:
                        imu_samples.append((r.pitch, r.roll, r.yaw))
            time.sleep(0.05)

        try:
            baseline = compute_baseline(pose_samples, yaw, head_pitch,
                                        imu_samples, min_samples=5)
            save_baseline(baseline)
            self.baseline = baseline
            log.info("Recalibration complete (%s): %d samples",
                     reason, baseline["samples"])
            if self.on_status:
                self.on_status("Posture baseline updated",
                               "Back to monitoring your posture.")
        except Exception as exc:
            log.warning("Recalibration skipped (%s): %s", reason, exc)
            if self.on_status:
                self.on_status(
                    "Recalibration skipped",
                    "Couldn't see you clearly — keeping your previous baseline.")
        finally:
            self.state.set_paused(self._pause_before_recalib)
            self.recalibrating = False
            self._safe_call(self.on_recalibration_end)

    @staticmethod
    def _safe_call(cb: Optional[Callable[[], None]]) -> None:
        if cb is None:
            return
        try:
            cb()
        except Exception as exc:  # never let a UI callback break recovery
            log.warning("recalibration callback error: %s", exc)

    def _apply_smoothing(self, pose_m, face_m) -> None:
        """EMA-smooth the scalar metrics used for scoring/display in place.

        Landmark coordinates are left raw so the overlay stays accurate and the
        breathing estimator still sees the true shoulder oscillation.
        """
        a = self._smoothing
        if a <= 0.0:
            return

        def ema(obj, key: str) -> None:
            cur = getattr(obj, key)
            prev = self._ema.get(key, cur)
            val = a * prev + (1.0 - a) * cur
            self._ema[key] = val
            setattr(obj, key, val)

        pose_keys = ("fha_deg", "forward_head", "neck_ratio", "torso_incline_deg",
                     "shoulder_tilt_deg", "head_roll_deg", "head_lateral",
                     "ear_mid_y_norm", "face_width")
        if pose_m is not None and pose_m.detected:
            for k in pose_keys:
                ema(pose_m, k)
        else:
            for k in pose_keys:
                self._ema.pop(k, None)  # reset so we don't fade in stale values

        if face_m is not None and face_m.detected:
            ema(face_m, "head_yaw_deg")
            ema(face_m, "head_pitch_deg")
        else:
            self._ema.pop("head_yaw_deg", None)
            self._ema.pop("head_pitch_deg", None)

    def latest(self) -> FrameResult:
        with self._latest_lock:
            return self._latest

    # ------------------------------------------------------------------
    def run_loop(self, fps_getter: Callable[[], float]) -> None:
        """Run process_once in this thread at a dynamic frame rate."""
        self.start_camera()
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.process_once()
            except Exception as exc:  # keep the daemon alive
                log.exception("frame processing error: %s", exc)
            period = 1.0 / max(0.5, fps_getter())
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def start_background(self, fps_getter: Callable[[], float]) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_loop, args=(fps_getter,), name="processor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._recalib_thread is not None:
            self._recalib_thread.join(timeout=2.0)
            self._recalib_thread = None
        self.camera.stop()
        self.pose.close()
        self.face.close()
        if self.imu is not None:
            self.imu.stop()
        self.health.flush(self.state.snapshot())
