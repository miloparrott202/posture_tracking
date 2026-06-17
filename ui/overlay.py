"""OpenCV overlay for the full-screen view (Section 6.2).

Drawn on the BGR frame before it reaches Qt. Renders the posture skeleton
(face points, ears, neck/spine line, shoulder line, torso line when hips are
visible), coloured by which problem group is worst, plus a compact score badge
and a correction cue — so a single frame reads on its own.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (40, 40, 255)
AMBER = (0, 200, 255)
WHITE = (245, 245, 245)
GREY = (170, 170, 170)
DARK = (32, 28, 26)
FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


def _px(pt: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
    return int(pt[0] * w), int(pt[1] * h)


def _panel(frame, x1, y1, x2, y2, alpha=0.55) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(sub, DARK), alpha, sub, 1 - alpha, 0, sub)


def draw_overlay(frame: np.ndarray, frame_result) -> np.ndarray:
    pose = frame_result.pose
    score = frame_result.score

    if pose is None or not pose.detected or score is None:
        _panel(frame, 8, 8, 286, 46)
        cv2.putText(frame, "Searching for you...", (18, 35), FONT, 0.62, AMBER, 2, AA)
        return frame
    if not getattr(pose, "reliable", True):
        _panel(frame, 8, 8, 360, 46)
        cv2.putText(frame, "Move so head & shoulders show", (18, 35),
                    FONT, 0.6, AMBER, 2, AA)
        return frame

    h, w = frame.shape[:2]
    lm = pose.landmarks
    worst = score.worst_group

    def gcolor(group: str) -> Tuple[int, int, int]:
        if score.level == "good":
            return GREEN
        return RED if group == worst else AMBER

    p = {k: _px(v, w, h) for k, v in lm.items() if not k.startswith("_")}

    # --- skeleton ----------------------------------------------------------
    if lm.get("_hips_valid", (0, 0))[0] > 0.5:                      # torso (slouch)
        cv2.line(frame, p["hip_mid"], p["shoulder_mid"], gcolor("slouch"), 3, AA)
        for pt in (p["left_hip"], p["right_hip"]):
            cv2.circle(frame, pt, 5, gcolor("slouch"), -1, AA)
    cv2.line(frame, p["left_shoulder"], p["right_shoulder"], gcolor("asym"), 3, AA)
    cv2.line(frame, p["shoulder_mid"], p["ear_mid"], gcolor("crane"), 3, AA)

    for pt in (p["left_shoulder"], p["right_shoulder"]):
        cv2.circle(frame, pt, 7, gcolor("asym"), -1, AA)
        cv2.circle(frame, pt, 7, DARK, 1, AA)
    for pt in (p["left_ear"], p["right_ear"]):
        cv2.circle(frame, pt, 6, gcolor("crane"), -1, AA)
    for key in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
        if key in p:
            cv2.circle(frame, p[key], 3, WHITE, -1, AA)
    cv2.circle(frame, p["shoulder_mid"], 4, WHITE, -1, AA)

    if score.needs_correction:
        _draw_correction_arrow(frame, p, worst)

    _draw_badge(frame, score)
    if not getattr(pose, "well_framed", True):
        _panel(frame, 200, 8, 200 + 360, 40)
        cv2.putText(frame, "Sit back - keep shoulders in view", (210, 30),
                    FONT, 0.55, AMBER, 1, AA)
    _draw_cue(frame, score)
    return frame


def _draw_badge(frame, score) -> None:
    _panel(frame, 8, 8, 188, 78)
    cv2.putText(frame, f"{score.score:.0f}", (18, 60), FONT, 1.6, score.color, 3, AA)
    cv2.putText(frame, score.level.upper(), (104, 34), FONT, 0.6, score.color, 2, AA)


def _draw_cue(frame, score) -> None:
    if score.level == "good" or not score.correction_text:
        return
    h, w = frame.shape[:2]
    _panel(frame, 0, h - 42, w, h, 0.62)
    color = WHITE if score.level == "drifting" else (60, 60, 255)
    text = score.correction_text.encode("ascii", "ignore").decode()  # cv2 is ASCII-only
    cv2.putText(frame, text, (16, h - 15), FONT, 0.6, color, 2, AA)


def _draw_correction_arrow(frame, p, worst) -> None:
    ear_mid, sh_mid = p["ear_mid"], p["shoulder_mid"]
    if worst == "slouch":
        base = sh_mid
        tip = (sh_mid[0], sh_mid[1] - 60)                  # sit up
    elif worst == "crane":
        base = ear_mid
        tip = (ear_mid[0] - 46, ear_mid[1] - 30)           # head back
    else:
        base = ear_mid
        tip = (ear_mid[0], ear_mid[1] - 46)
    cv2.arrowedLine(frame, base, tip, (60, 60, 255), 3, AA, tipLength=0.35)
