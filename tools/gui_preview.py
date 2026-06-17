#!/usr/bin/env python3
"""Render the GUI offscreen with synthetic poses for visual review.

No camera or person needed — fabricates a few posture states and grabs the whole
window to screenshots/preview_*.png. Run with QT_QPA_PLATFORM=offscreen.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.config import load_config  # noqa: E402
from core.pipeline import FrameResult  # noqa: E402
from core.pose_engine import PoseMetrics  # noqa: E402
from core.posture_score import COLORS, ScoreResult  # noqa: E402
from core.state_manager import Snapshot  # noqa: E402
from ui.full_view import FullView  # noqa: E402
from ui.overlay import draw_overlay  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")


def landmarks(shift_y=0.0, fwd=0.0, tilt=0.0):
    """A plausible upright pose; tweak to simulate slouch/tilt."""
    sh_y = 0.72 - fwd * 0.5
    ear_y = 0.40 + shift_y
    lm = {
        "nose": (0.5, ear_y + 0.04), "left_eye": (0.455, ear_y - 0.02),
        "right_eye": (0.545, ear_y - 0.02), "left_ear": (0.44, ear_y),
        "right_ear": (0.56, ear_y), "mouth_left": (0.47, ear_y + 0.09),
        "mouth_right": (0.53, ear_y + 0.09),
        "left_shoulder": (0.37, sh_y + tilt), "right_shoulder": (0.63, sh_y - tilt),
        "left_hip": (0.40, 0.99), "right_hip": (0.60, 0.99),
    }
    lm["ear_mid"] = (0.5, ear_y)
    lm["shoulder_mid"] = (0.5, sh_y)
    lm["eye_mid"] = (0.5, ear_y - 0.02)
    lm["hip_mid"] = (0.5, 0.99)
    lm["_hips_valid"] = (0.0, 0.0)
    return lm


def make_result(level, gp, cue, lm=None, blink=16, brk=540):
    if lm is None:
        lm = landmarks()
    score = ScoreResult(score={"good": 96, "drifting": 74, "poor": 41}[level],
                        level=level, color=COLORS[level], group_penalties=gp,
                        worst_group=(max(gp, key=gp.get) if any(gp.values()) else None),
                        correction_text=cue)
    pose = PoseMetrics(detected=True, reliable=True, landmarks=lm)
    snap = Snapshot(score=score, blink_rate=blink, seconds_to_break=brk)
    frame = np.full((480, 640, 3), (44, 40, 38), np.uint8)
    return FrameResult(frame=frame, pose=pose, score=score, snapshot=snap,
                       gaze_on_screen=True)


def main():
    cfg = load_config()
    app = QApplication(sys.argv)
    win = FullView(types.SimpleNamespace(), cfg, notifier=None)
    win._timer.stop()  # we drive rendering manually with synthetic data
    win.resize(1060, 720)

    states = {
        "good": make_result("good", {"crane": 0, "slouch": 0, "asym": 0}, ""),
        "slouch": make_result("poor", {"crane": 6, "slouch": 34, "asym": 3},
                              "Sit up tall and lengthen your spine",
                              landmarks(shift_y=0.10, fwd=0.06), blink=9, brk=120),
        "crane": make_result("drifting", {"crane": 20, "slouch": 4, "asym": 2},
                             "Ease your head back over your shoulders",
                             landmarks(shift_y=0.05)),
        "asym": make_result("drifting", {"crane": 1, "slouch": 2, "asym": 16},
                            "Level your shoulders (raise your left)",
                            landmarks(tilt=0.05)),
    }
    os.makedirs(OUT, exist_ok=True)
    for name, res in states.items():
        win._set_video_pixmap(draw_overlay(res.frame.copy(), res))
        win._update_sidebar(res)
        app.processEvents()
        path = os.path.join(OUT, f"preview_{name}.png")
        win.grab().save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
