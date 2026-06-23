"""OpenCV overlay for the full-screen view (Section 6.2).

Drawn on the BGR frame before it reaches Qt. A clean, subtle posture rig: a
faint face constellation, thin neck/spine + shoulder + torso lines and small
joint dots, coloured by which problem group is worst, plus a compact score
badge and a correction cue.

Kept deliberately low-key — every stroke gets a thin dark casing so it stays
legible over a busy webcam image without resorting to glow.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Calm, semantic palette (BGR).
GOOD = (120, 200, 110)      # green
DRIFT = (40, 190, 230)      # amber
BAD = (70, 70, 220)         # red
FACE = (155, 150, 145)      # neutral grey for the face constellation
WHITE = (235, 235, 235)
AMBER = (0, 200, 255)
CASING = (20, 20, 20)       # thin dark outline under strokes
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


def _link(frame, a, b, color, thick=2) -> None:
    cv2.line(frame, a, b, CASING, thick + 2, AA)   # dark casing for contrast
    cv2.line(frame, a, b, color, thick, AA)


def _node(frame, c, color, r, worst=False) -> None:
    cv2.circle(frame, c, r + 1, CASING, -1, AA)     # casing
    cv2.circle(frame, c, r, color, -1, AA)          # core
    if worst:                                       # subtle ring on the flagged joint
        cv2.circle(frame, c, r + 3, color, 1, AA)


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
            return GOOD
        return BAD if group == worst else DRIFT

    p = {k: _px(v, w, h) for k, v in lm.items() if not k.startswith("_")}
    hips = lm.get("_hips_valid", (0, 0))[0] > 0.5

    # --- face constellation (subtle, behind everything) --------------------
    for a, b in _face_links(p):
        cv2.line(frame, a, b, FACE, 1, AA)
    for k in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
        if k in p:
            cv2.circle(frame, p[k], 2, WHITE, -1, AA)

    # --- skeleton ----------------------------------------------------------
    if hips:                                                         # torso (slouch)
        _link(frame, p["hip_mid"], p["shoulder_mid"], gcolor("slouch"))
        for pt in (p["left_hip"], p["right_hip"]):
            _node(frame, pt, gcolor("slouch"), 4, worst == "slouch")
    _link(frame, p["left_shoulder"], p["right_shoulder"], gcolor("asym"))
    _link(frame, p["shoulder_mid"], p["ear_mid"], gcolor("crane"))

    for pt in (p["left_shoulder"], p["right_shoulder"]):
        _node(frame, pt, gcolor("asym"), 5, worst == "asym")
    for pt in (p["left_ear"], p["right_ear"]):
        _node(frame, pt, gcolor("crane"), 4, worst == "crane")
    _node(frame, p["shoulder_mid"], WHITE, 3)

    if score.needs_correction:
        _draw_correction_arrow(frame, p, worst)

    _draw_badge(frame, score)
    if not getattr(pose, "well_framed", True):
        _panel(frame, 200, 8, 200 + 360, 40)
        cv2.putText(frame, "Sit back - keep shoulders in view", (210, 30),
                    FONT, 0.55, AMBER, 1, AA)
    _draw_cue(frame, score)
    return frame


# ----------------------------------------------------------------------
def _face_links(p: Dict[str, Tuple[int, int]]) -> List[Tuple]:
    """Minimal eye/nose/mouth rig, only for landmarks that are present."""
    pairs = [("left_eye", "right_eye"), ("left_eye", "nose"), ("right_eye", "nose"),
             ("nose", "mouth_left"), ("nose", "mouth_right"),
             ("mouth_left", "mouth_right")]
    return [(p[a], p[b]) for a, b in pairs if a in p and b in p]


def _draw_correction_arrow(frame, p, worst) -> None:
    ear_mid, sh_mid = p["ear_mid"], p["shoulder_mid"]
    if worst == "slouch":
        base, tip = sh_mid, (sh_mid[0], sh_mid[1] - 56)            # sit up
    elif worst == "crane":
        base, tip = ear_mid, (ear_mid[0] - 44, ear_mid[1] - 28)   # head back
    else:
        base, tip = ear_mid, (ear_mid[0], ear_mid[1] - 44)
    cv2.arrowedLine(frame, base, tip, CASING, 4, AA, tipLength=0.38)
    cv2.arrowedLine(frame, base, tip, BAD, 2, AA, tipLength=0.38)


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
