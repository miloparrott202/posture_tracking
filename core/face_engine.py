"""MediaPipe Face Landmarker wrapper: blink (EAR), gaze, head pose (Section 5)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from . import geometry
from .models import ensure_model

# Face Mesh landmark indices (468 mesh + iris when refine on -> 478 total).
# Eyelid/corner points for the Eye Aspect Ratio.
LEFT_EYE = {"h": (33, 133), "v1": (159, 145), "v2": (158, 153)}
RIGHT_EYE = {"h": (362, 263), "v1": (386, 374), "v2": (385, 380)}
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


@dataclass
class FaceMetrics:
    detected: bool = False
    ear: float = 0.35                  # current eye aspect ratio (avg of both eyes)
    is_blinking: bool = False
    blink_rate: float = 0.0            # blinks/min over the rolling window
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    gaze_h_ratio: float = 0.5          # 0=left corner, 1=right corner of eye
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class BlinkCounter:
    """Counts blinks via EAR threshold + rolling 60s window (Section 5.1)."""

    def __init__(self, ear_threshold: float, consec_frames: int, window_seconds: float):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.window_seconds = window_seconds
        self._below = 0
        self._blink_times: Deque[float] = deque()
        self._first_sample: Optional[float] = None

    def update(self, ear: float, now: float) -> bool:
        """Feed one EAR sample; return True if a blink completed this frame."""
        if self._first_sample is None:
            self._first_sample = now
        blinked = False
        if ear < self.ear_threshold:
            self._below += 1
        else:
            if self._below >= self.consec_frames:
                self._blink_times.append(now)
                blinked = True
            self._below = 0
        # Drop samples outside the rolling window.
        cutoff = now - self.window_seconds
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        return blinked

    def rate_per_min(self, now: float) -> float:
        cutoff = now - self.window_seconds
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        # Until the rolling window has actually filled, divide by elapsed time
        # so the rate isn't under-reported for the first minute of a session.
        elapsed = now - self._first_sample if self._first_sample is not None else 0.0
        effective = min(self.window_seconds, max(elapsed, 1.0))
        return len(self._blink_times) * (60.0 / effective)


class FaceEngine:
    """Face Mesh wrapper producing blink/gaze/head-pose metrics."""

    def __init__(self, cfg: dict, task_file: str = "face_landmarker.task") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_model(task_file)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,  # head pose
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        b = cfg["blink"]
        self._blinks = BlinkCounter(b["ear_threshold"], b["consec_frames"], b["window_seconds"])

    @staticmethod
    def _ear(lm, eye: Dict[str, Tuple[int, int]]) -> float:
        def p(i: int) -> Tuple[float, float]:
            return (lm[i].x, lm[i].y)

        h = geometry.euclidean(p(eye["h"][0]), p(eye["h"][1]))
        v1 = geometry.euclidean(p(eye["v1"][0]), p(eye["v1"][1]))
        v2 = geometry.euclidean(p(eye["v2"][0]), p(eye["v2"][1]))
        return (v1 + v2) / (2.0 * h + 1e-9)

    @staticmethod
    def _gaze_h_ratio(lm) -> float:
        # Horizontal iris position within the left eye (inner=133, outer=33).
        inner = lm[133].x
        outer = lm[33].x
        iris = lm[LEFT_IRIS_CENTER].x
        denom = (inner - outer)
        if abs(denom) < 1e-6:
            return 0.5
        return float(np.clip((iris - outer) / denom, 0.0, 1.0))

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int, now: Optional[float] = None) -> FaceMetrics:
        import cv2

        now = time.monotonic() if now is None else now
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            # Still report the (decaying) blink rate so timers behave.
            return FaceMetrics(detected=False, blink_rate=self._blinks.rate_per_min(now))

        lm = result.face_landmarks[0]
        ear = (self._ear(lm, LEFT_EYE) + self._ear(lm, RIGHT_EYE)) / 2.0
        is_blinking = self._blinks.update(ear, now)
        rate = self._blinks.rate_per_min(now)

        pitch = yaw = roll = 0.0
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = geometry.rotation_matrix_to_euler(
                result.facial_transformation_matrixes[0]
            )

        gaze_h = self._gaze_h_ratio(lm)

        return FaceMetrics(
            detected=True,
            ear=ear,
            is_blinking=is_blinking,
            blink_rate=rate,
            head_yaw_deg=yaw,
            head_pitch_deg=pitch,
            head_roll_deg=roll,
            gaze_h_ratio=gaze_h,
            landmarks={
                "left_iris": (lm[LEFT_IRIS_CENTER].x, lm[LEFT_IRIS_CENTER].y),
                "right_iris": (lm[RIGHT_IRIS_CENTER].x, lm[RIGHT_IRIS_CENTER].y),
            },
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
