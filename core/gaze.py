"""Gaze / 'looking at screen' classification (Section 5.2)."""
from __future__ import annotations


def is_looking_at_screen(face_metrics, cfg: dict) -> bool:
    """True if head pose + iris position suggest attention on the screen.

    Away if head yaw/pitch exceeds threshold OR the iris sits at an extreme
    horizontal corner of the eye.
    """
    if face_metrics is None or not face_metrics.detected:
        return False

    g = cfg["gaze"]
    if abs(face_metrics.head_yaw_deg) > g["yaw_threshold_deg"]:
        return False
    if abs(face_metrics.head_pitch_deg) > g["pitch_threshold_deg"]:
        return False

    # gaze_h_ratio: 0 = outer corner, 1 = inner corner. Extreme either end
    # means the eyes are cut hard to one side.
    extreme = g["iris_extreme_ratio"]
    if face_metrics.gaze_h_ratio < extreme or face_metrics.gaze_h_ratio > (1.0 - extreme):
        return False

    return True
