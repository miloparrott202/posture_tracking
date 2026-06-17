"""MediaPipe Pose Landmarker wrapper + advanced posture features (Section 4).

Goes well beyond a single forward-head angle: it pulls eyes, ears, nose, mouth,
shoulders and hips from the 33-point BlazePose topology and derives a set of
posture features tuned for the two things that matter most at a desk —
**slouching/hunching** and **head craning** — plus left/right **asymmetry**.

All features are normalised by shoulder width (a stable, always-present scale)
so they're resolution- and distance-independent, and several use the landmark
z-depth to catch forward-head motion a frontal camera can't see in 2D alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import geometry
from .models import ensure_model

# MediaPipe Pose landmark indices (BlazePose 33-point topology).
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# Landmarks we require to be confidently visible to trust the metrics.
_KEY_POINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR, NOSE)


@dataclass
class PoseMetrics:
    """Per-frame posture features (Section 4.2), normalised by shoulder width."""

    detected: bool = False
    reliable: bool = False             # key landmarks confidently visible
    well_framed: bool = True           # shoulders sit in view with margin below

    # --- forward head / craning (priority) -------------------------------
    fha_deg: float = 0.0               # 2D forward-head angle (ear vs shoulder)
    forward_head: float = 0.0          # z-depth: ears ahead of shoulders (+ = craning)

    # --- slouch / hunch (priority) ---------------------------------------
    neck_ratio: float = 0.0            # vertical ear->shoulder gap / shoulder width
    torso_incline_deg: float = 0.0     # hip->shoulder line vs vertical (if hips seen)
    torso_valid: bool = False

    # --- asymmetry --------------------------------------------------------
    shoulder_tilt_deg: float = 0.0     # +ve = right-image shoulder lower
    head_roll_deg: float = 0.0         # eye-line tilt (head cocked sideways)
    head_lateral: float = 0.0          # nose offset from shoulder midline / SW

    # --- bookkeeping / overlay -------------------------------------------
    ear_mid_y_norm: float = 0.0
    shoulder_width: float = 1.0
    face_width: float = 1.0
    landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class PoseEngine:
    """Thin wrapper around the MediaPipe Tasks Pose Landmarker (VIDEO mode)."""

    def __init__(self, task_file: str = "pose_landmarker_lite.task") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_model(task_file)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseMetrics:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseMetrics(detected=False)
        return self._compute(result.pose_landmarks[0])

    # ------------------------------------------------------------------
    @staticmethod
    def _compute(lm) -> PoseMetrics:
        def xy(i: int) -> Tuple[float, float]:
            return (lm[i].x, lm[i].y)

        def z(i: int) -> float:
            return lm[i].z

        def vis(i: int) -> float:
            return getattr(lm[i], "visibility", 1.0)

        l_ear, r_ear = xy(LEFT_EAR), xy(RIGHT_EAR)
        l_sh, r_sh = xy(LEFT_SHOULDER), xy(RIGHT_SHOULDER)
        l_eye, r_eye = xy(LEFT_EYE), xy(RIGHT_EYE)
        l_hip, r_hip = xy(LEFT_HIP), xy(RIGHT_HIP)
        nose = xy(NOSE)

        ear_mid = geometry.midpoint(l_ear, r_ear)
        sh_mid = geometry.midpoint(l_sh, r_sh)
        eye_mid = geometry.midpoint(l_eye, r_eye)
        hip_mid = geometry.midpoint(l_hip, r_hip)

        shoulder_width = max(geometry.euclidean(l_sh, r_sh), 1e-3)
        face_width = max(geometry.euclidean(l_ear, r_ear), 1e-3)

        # Forward head (craning): 2D angle + z-depth of ears ahead of shoulders.
        fha = geometry.angle_from_vertical(sh_mid, ear_mid)
        ear_z = (z(LEFT_EAR) + z(RIGHT_EAR)) / 2.0
        sh_z = (z(LEFT_SHOULDER) + z(RIGHT_SHOULDER)) / 2.0
        forward_head = (sh_z - ear_z) / shoulder_width  # +ve = ears closer to cam

        # Slouch/hunch: vertical neck gap (ear above shoulder) over shoulder width.
        neck_ratio = (sh_mid[1] - ear_mid[1]) / shoulder_width

        # Torso incline from vertical (only when hips are actually visible).
        torso_valid = min(vis(LEFT_HIP), vis(RIGHT_HIP)) > 0.5
        torso_incline = geometry.angle_from_vertical(hip_mid, sh_mid) if torso_valid else 0.0

        # Asymmetry.
        shoulder_tilt = geometry.signed_tilt_deg(l_sh, r_sh)
        head_roll = geometry.signed_tilt_deg(l_eye, r_eye)
        head_lateral = (nose[0] - sh_mid[0]) / shoulder_width

        reliable = min(vis(i) for i in _KEY_POINTS) > 0.5
        # Only flag framing when the shoulders are genuinely at the very bottom
        # edge (clipped). The neck_ratio is relative, so it stays usable as long
        # as the shoulders are visible at all.
        well_framed = (sh_mid[1] < 0.96
                       and min(vis(LEFT_SHOULDER), vis(RIGHT_SHOULDER)) > 0.5)

        return PoseMetrics(
            detected=True,
            reliable=reliable,
            well_framed=well_framed,
            fha_deg=fha,
            forward_head=forward_head,
            neck_ratio=neck_ratio,
            torso_incline_deg=torso_incline,
            torso_valid=torso_valid,
            shoulder_tilt_deg=shoulder_tilt,
            head_roll_deg=head_roll,
            head_lateral=head_lateral,
            ear_mid_y_norm=ear_mid[1],
            shoulder_width=shoulder_width,
            face_width=face_width,
            landmarks={
                "nose": nose,
                "left_eye": l_eye, "right_eye": r_eye,
                "left_ear": l_ear, "right_ear": r_ear,
                "mouth_left": xy(MOUTH_LEFT), "mouth_right": xy(MOUTH_RIGHT),
                "left_shoulder": l_sh, "right_shoulder": r_sh,
                "left_hip": l_hip, "right_hip": r_hip,
                "ear_mid": ear_mid, "shoulder_mid": sh_mid,
                "eye_mid": eye_mid, "hip_mid": hip_mid,
                "_hips_valid": (1.0, 1.0) if torso_valid else (0.0, 0.0),
            },
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
