"""Near-work / vision-strain signals from pose (screen distance + chin-jut).

Tuned for convergence insufficiency / post-concussion vision syndrome, where
symptoms scale with how close and how long you do near-work, and a sudden
"chin jut" (head lunging toward the screen) is a documented eye-strain tell.

Two per-frame signals, both derived from the pose the tracker already computes:

  * **Screen distance** — apparent face width (ear-to-ear) shrinks with distance.
    Compared against the calibrated baseline face width it gives a relative
    distance ratio (and a rough cm estimate), and a sustained "lean-in" flag.
    Robust to head turn: yaw only *shrinks* apparent width, so it can cause a
    false "farther" but never a false "leaning in".

  * **Chin-jut** — a fast rise in forward-head depth above a slow-adapting
    reference, i.e. the head transiently pushing toward the screen. Rate-limited
    by a cooldown so a sustained lean doesn't fire it repeatedly.

The break-budget accumulation and all user nudges live in the StateManager;
this module only produces the raw signals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class NearWorkMetrics:
    valid: bool = False                 # distance signal usable (baseline + reliable pose)
    distance_ratio: float = 1.0         # baseline_fw / current_fw (>1 farther, <1 closer)
    closeness_pct: float = 0.0          # how much closer than baseline, % (+ = closer)
    distance_cm: Optional[float] = None  # rough absolute distance (approx)
    lean_in: bool = False               # currently closer than the lean-in threshold
    chin_jut: bool = False              # transient head-toward-screen lunge this frame


class NearWorkEstimator:
    def __init__(self, cfg: dict):
        nw = cfg.get("near_work", {})
        self.baseline_distance_cm = float(nw.get("baseline_distance_cm", 60.0))
        self.lean_in_ratio = float(nw.get("lean_in_ratio", 1.15))
        self.min_baseline_fw = float(nw.get("min_baseline_face_width", 0.02))
        self.jut_delta = float(nw.get("chin_jut_delta", 0.05))
        self.jut_cooldown = float(nw.get("chin_jut_cooldown_sec", 8.0))
        self.ref_tau = float(nw.get("chin_jut_ref_tau_sec", 4.0))  # ref adaptation time

        self._fh_ref: Optional[float] = None   # slow-adapting forward-head reference
        self._last_t: Optional[float] = None
        self._last_jut: float = -1e9

    def update(self, pose_metrics, baseline: Optional[dict], now: float) -> NearWorkMetrics:
        m = NearWorkMetrics()
        if pose_metrics is None or not pose_metrics.detected \
                or not getattr(pose_metrics, "reliable", False):
            # Don't adapt the reference on unreliable frames.
            self._last_t = now
            return m

        m = self._distance(pose_metrics, baseline, m)
        m.chin_jut = self._chin_jut(pose_metrics, now)
        return m

    # ------------------------------------------------------------------
    def _distance(self, pose, baseline, m: NearWorkMetrics) -> NearWorkMetrics:
        base_fw = None
        if baseline:
            base_fw = baseline.get("pose", {}).get("face_width")
        if not base_fw or base_fw < self.min_baseline_fw:
            return m  # no usable baseline -> leave distance signal invalid

        cur = max(float(pose.face_width), 1e-6)
        m.valid = True
        m.distance_ratio = base_fw / cur
        m.closeness_pct = (cur / base_fw - 1.0) * 100.0
        m.distance_cm = self.baseline_distance_cm * m.distance_ratio
        m.lean_in = cur >= base_fw * self.lean_in_ratio
        return m

    def _chin_jut(self, pose, now: float) -> bool:
        fh = float(pose.forward_head)
        if self._fh_ref is None:
            self._fh_ref = fh
            self._last_t = now
            return False

        jut = False
        if (fh - self._fh_ref) > self.jut_delta and (now - self._last_jut) >= self.jut_cooldown:
            jut = True
            self._last_jut = now

        # Slow, time-based adaptation of the reference toward the current value
        # so a held lean settles in (won't re-fire) but a quick jut still spikes.
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        alpha = 1.0 - math.exp(-dt / max(self.ref_tau, 1e-3)) if dt > 0 else 0.0
        self._fh_ref += alpha * (fh - self._fh_ref)
        return jut

    def reset(self) -> None:
        self._fh_ref = None
        self._last_t = None
        self._last_jut = -1e9
