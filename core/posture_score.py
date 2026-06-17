"""Posture scoring (Section 4.4).

Combines the rich pose features into a 0-100 score against the calibrated
baseline. Penalties are grouped into three buckets so we can give a targeted
cue, with weighting that reflects the priorities: **slouching** and **head
craning** dominate, left/right **asymmetry** matters but counts for less.

Every feature uses a deadband (tolerance): only deviation *beyond* the
tolerance is penalised, so ordinary micro-movement never erodes the score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ScoreResult:
    score: float = 100.0
    level: str = "good"                         # good | drifting | poor
    color: Tuple[int, int, int] = (0, 200, 0)   # BGR for overlays
    deviations: Dict[str, float] = field(default_factory=dict)
    group_penalties: Dict[str, float] = field(default_factory=dict)  # crane/slouch/asym
    worst_group: Optional[str] = None
    worst_component: Optional[str] = None
    correction_text: str = ""

    @property
    def needs_correction(self) -> bool:
        return self.level == "poor"


COLORS = {
    "good": (0, 200, 0),
    "drifting": (0, 200, 255),   # amber (BGR)
    "poor": (0, 0, 255),         # red (BGR)
}

# Which group each feature belongs to (for choosing a cue).
_GROUP = {
    "fha": "crane", "forward_head": "crane",
    "neck_ratio": "slouch", "torso_incline": "slouch", "head_drop": "slouch",
    "head_pitch": "slouch",
    "shoulder_tilt": "asym", "head_roll": "asym", "head_lateral": "asym",
}


def compute_score(pose_metrics, face_metrics, baseline: Optional[dict],
                  cfg: dict, fused_fha_dev: Optional[float] = None) -> ScoreResult:
    if (not pose_metrics.detected or not getattr(pose_metrics, "reliable", True)
            or baseline is None):
        # Can't assess (no body, partly out of frame, or not calibrated yet).
        return ScoreResult(score=100.0, level="good", color=COLORS["good"])

    w = cfg["posture"]["weights"]
    tol = cfg["posture"]["tolerances"]
    thresholds = cfg["posture"]["thresholds"]
    base = baseline.get("pose", baseline)

    def b(key: str, default: float = 0.0) -> float:
        return base.get(key, default)

    # --- raw signed deviations from baseline ------------------------------
    fha_dev = pose_metrics.fha_deg - b("fha_deg")
    if fused_fha_dev is not None:
        fha_dev = fused_fha_dev
    fwd_dev = pose_metrics.forward_head - b("forward_head")
    # neck_ratio drops when slouching/hunching -> measure the drop (>=0).
    neck_drop = max(0.0, b("neck_ratio") - pose_metrics.neck_ratio)
    torso_dev = (pose_metrics.torso_incline_deg - b("torso_incline_deg")
                 if pose_metrics.torso_valid else 0.0)
    head_drop = max(0.0, pose_metrics.ear_mid_y_norm - b("ear_mid_y_norm"))
    tilt_dev = pose_metrics.shoulder_tilt_deg - b("shoulder_tilt_deg")
    roll_dev = pose_metrics.head_roll_deg - b("head_roll_deg")
    lat_dev = pose_metrics.head_lateral - b("head_lateral")
    # Head pitch change (chin tilting up/down) catches reclining/slumping even
    # when the head stays oriented at the screen.
    pitch_dev = 0.0
    if face_metrics is not None and face_metrics.detected:
        pitch_dev = face_metrics.head_pitch_deg - base.get("head_pitch_deg", 0.0)

    deviations = {
        "fha": fha_dev, "forward_head": fwd_dev,
        "neck_ratio": -neck_drop, "torso_incline": torso_dev, "head_drop": head_drop,
        "head_pitch": pitch_dev,
        "shoulder_tilt": tilt_dev, "head_roll": roll_dev, "head_lateral": lat_dev,
    }

    def excess(value: float, t: float) -> float:
        return max(0.0, abs(value) - t)

    # --- per-feature penalties (deadbanded, weighted) ---------------------
    # Forward-head/craning is one-sided: only the forward direction is bad.
    # Note: head *yaw* (turning to look away) is an attention signal, not
    # posture, and is noisy while slouching — it's handled by gaze/break logic,
    # not penalised here, so it can't hijack the posture cue.
    sub = {
        "fha": w["fha"] * max(0.0, fha_dev - tol["fha_deg"]),
        "forward_head": w["forward_head"] * max(0.0, fwd_dev - tol["forward_head"]),
        "neck_ratio": w["neck_ratio"] * max(0.0, neck_drop - tol["neck_ratio"]),
        "torso_incline": w["torso_incline"] * excess(torso_dev, tol["torso_incline_deg"]),
        "head_drop": w["head_drop"] * max(0.0, head_drop - tol["head_drop"]),
        "head_pitch": w["head_pitch"] * excess(pitch_dev, tol["head_pitch_deg"]),
        "shoulder_tilt": w["shoulder_tilt"] * excess(tilt_dev, tol["shoulder_tilt_deg"]),
        "head_roll": w["head_roll"] * excess(roll_dev, tol["head_roll_deg"]),
        "head_lateral": w["head_lateral"] * excess(lat_dev, tol["head_lateral"]),
    }

    groups: Dict[str, float] = {"crane": 0.0, "slouch": 0.0, "asym": 0.0}
    for feat, pen in sub.items():
        groups[_GROUP[feat]] += pen

    penalty = clamp(sum(sub.values()), 0.0, 100.0)
    score = 100.0 - penalty

    if score >= thresholds["good"]:
        level = "good"
    elif score >= thresholds["drifting"]:
        level = "drifting"
    else:
        level = "poor"

    worst_group = max(groups, key=groups.get) if penalty > 0 else None
    # Pick the worst feature *within* the worst group so the cue always matches
    # the group being flagged (otherwise the cue could fall back to generic).
    worst_comp = None
    if worst_group:
        members = [f for f in sub if _GROUP[f] == worst_group]
        worst_comp = max(members, key=lambda f: sub[f])
    text = _cue(worst_group, worst_comp, deviations) if level != "good" else ""

    return ScoreResult(
        score=round(score, 1), level=level, color=COLORS[level],
        deviations=deviations, group_penalties=groups,
        worst_group=worst_group, worst_component=worst_comp, correction_text=text,
    )


# Short, ASCII-only cues so they fit the on-video banner and OS banners alike.
def _cue(group: Optional[str], comp: Optional[str], dev: Dict[str, float]) -> str:
    if group == "slouch":
        return "Sit up tall and lengthen your spine"
    if group == "crane":
        return "Ease your head back over your shoulders"
    if group == "asym":
        if comp == "shoulder_tilt":
            side = "right" if dev["shoulder_tilt"] > 0 else "left"
            return f"Level your shoulders (raise your {side})"
        if comp == "head_roll":
            side = "right" if dev["head_roll"] > 0 else "left"
            return f"Straighten your head ({side} tilt)"
        if comp == "head_lateral":
            return "Centre your head over your shoulders"
    return "Adjust your posture"
