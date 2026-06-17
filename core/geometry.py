"""Small geometry helpers used by the pose/face engines and scoring."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_from_vertical(p_lower: Tuple[float, float], p_upper: Tuple[float, float]) -> float:
    """Angle (degrees) between the vertical axis and the lower->upper vector.

    0deg means ``p_upper`` is directly above ``p_lower``. Larger = more
    forward/back lean. Image coords have y increasing downward, so "up" is
    a negative dy; we account for that.
    """
    dx = p_upper[0] - p_lower[0]
    dy = p_upper[1] - p_lower[1]  # negative when upper is higher on screen
    # angle off the straight-up direction (0, -1)
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))


def signed_tilt_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Tilt (degrees) of the line through a,b relative to horizontal, in [-90, 90].

    The two points are ordered left-to-right by image x first, so the result is
    ~0 for a level line regardless of which argument is the left/right landmark
    (MediaPipe's LEFT_/RIGHT_ indices are anatomical, not image-ordered, so a
    naive ``atan2(dy, dx)`` would land near +/-180 for shoulders). Positive =
    the right-hand (larger x) point sits lower on screen.
    """
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return math.degrees(math.atan2(dy, dx + 1e-9))


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rotation_matrix_to_euler(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 4x4 (or 3x3) rotation matrix to (pitch, yaw, roll) degrees.

    Used with MediaPipe's facial transformation matrix to recover head pose.
    """
    r = np.asarray(matrix, dtype=float)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
