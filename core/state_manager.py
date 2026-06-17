"""Shared application state, timers, and notification debouncing (Section 5.3, 7).

The StateManager is the single source of truth read by every UI mode. It is
fed one bundle of metrics per processed frame and emits *events* (break /
posture / blink reminders) when the relevant conditions persist. All event
emission is rate-limited here, independent of the notification backend.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .gaze import is_looking_at_screen
from .posture_score import ScoreResult


@dataclass
class Event:
    """A reminder the app should surface to the user."""

    type: str           # "break" | "posture" | "blink"
    title: str
    message: str


@dataclass
class Snapshot:
    """Immutable-ish view of current state for the UI/health log."""

    score: ScoreResult = field(default_factory=ScoreResult)
    blink_rate: float = 0.0
    gaze_on_screen: bool = True
    face_detected: bool = False
    pose_detected: bool = False
    session_seconds: float = 0.0
    continuous_screen_seconds: float = 0.0
    seconds_to_break: float = 0.0
    paused: bool = False
    breaks_taken: int = 0
    breathing_rate: float = 0.0
    breathing_pattern: str = "unknown"   # chest | relaxed | unknown
    pct_good: float = 0.0                 # % of scored time in each posture band
    pct_drifting: float = 0.0
    pct_poor: float = 0.0


class StateManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lock = threading.Lock()

        self._start = time.monotonic()
        self._paused = False

        # Cumulative time spent in each posture band (only while pose detected).
        self._level_times: Dict[str, float] = {"good": 0.0, "drifting": 0.0, "poor": 0.0}
        self._last_stat_t = self._start

        # Screen-time / break tracking (20-20-20).
        self._screen_start: Optional[float] = None
        self._last_on_screen: float = self._start
        self._away_since: Optional[float] = None
        self._no_face_since: Optional[float] = None
        self._breaks_taken = 0

        # Poor-posture debounce.
        self._poor_since: Optional[float] = None

        # Low blink-rate sustain tracking.
        self._low_blink_since: Optional[float] = None

        # Breathing: periodic reminder timer + chest-breathing sustain.
        self._last_breath_reminder: Optional[float] = None
        self._chest_since: Optional[float] = None

        # Per-type rate limiting for emitted events.
        self._last_emit: Dict[str, float] = {}

        self._snapshot = Snapshot()

    # ------------------------------------------------------------------
    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
            if paused:
                self._screen_start = None
                self._poor_since = None
                self._chest_since = None

    @property
    def paused(self) -> bool:
        return self._paused

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------
    def update(self, pose_metrics, face_metrics, score: ScoreResult,
               breathing=None, now: Optional[float] = None) -> List[Event]:
        """Advance timers with the latest frame metrics; return due events."""
        now = time.monotonic() if now is None else now
        events: List[Event] = []
        with self._lock:
            # Time delta since the last frame, capped so gaps (pause/away) don't
            # dump a big lump into the stats.
            dt = min(max(0.0, now - self._last_stat_t), 1.0)
            self._last_stat_t = now

            if self._paused:
                self._snapshot = self._build_snapshot(score, face_metrics,
                                                       pose_metrics, now,
                                                       breathing=breathing)
                return events

            face_detected = bool(face_metrics and face_metrics.detected)
            blink_rate = face_metrics.blink_rate if face_metrics else 0.0

            # --- user-away detection (pause timers when no face) ---------
            if not face_detected:
                if self._no_face_since is None:
                    self._no_face_since = now
                away_for = now - self._no_face_since
                if away_for > self.cfg["breaks"]["away_pause_sec"]:
                    # Treat as user-away: freeze the continuous screen timer.
                    self._screen_start = None
                    self._poor_since = None
                    self._low_blink_since = None
                    self._chest_since = None
                    self._snapshot = self._build_snapshot(score, face_metrics,
                                                          pose_metrics, now,
                                                          breathing=breathing)
                    return events
            else:
                self._no_face_since = None

            on_screen = face_detected and is_looking_at_screen(face_metrics, self.cfg)

            # Accumulate posture-band time only while the body is actually seen.
            if pose_metrics and pose_metrics.detected and score is not None \
                    and score.level in self._level_times:
                self._level_times[score.level] += dt

            events += self._update_break_timer(on_screen, now)
            events += self._update_posture(score, now)
            events += self._update_blink(blink_rate, on_screen, now)
            events += self._update_breathing(breathing, now)
            events = self._apply_reminder_toggles(events)

            self._snapshot = self._build_snapshot(score, face_metrics,
                                                  pose_metrics, now,
                                                  gaze_on_screen=on_screen,
                                                  breathing=breathing)
        return events

    # ------------------------------------------------------------------
    def _update_break_timer(self, on_screen: bool, now: float) -> List[Event]:
        b = self.cfg["breaks"]
        events: List[Event] = []
        if on_screen:
            self._last_on_screen = now
            self._away_since = None
            if self._screen_start is None:
                self._screen_start = now
            elapsed = now - self._screen_start
            if elapsed >= b["screen_interval_sec"] and self._rate_ok("break", now):
                events.append(Event(
                    "break",
                    "Time to look away",
                    "Focus on something ~20ft away for 20 seconds (20-20-20).",
                ))
                self._breaks_taken += 1
                self._screen_start = now  # restart the interval
        else:
            if self._away_since is None:
                self._away_since = now
            # Reset the continuous timer once away long enough.
            if now - self._away_since >= b["look_away_reset_sec"]:
                self._screen_start = None
        return events

    def _update_posture(self, score: ScoreResult, now: float) -> List[Event]:
        events: List[Event] = []
        sustain = self.cfg["notifications"]["posture_sustain_sec"]
        if score.level == "poor":
            if self._poor_since is None:
                self._poor_since = now
            if (now - self._poor_since) >= sustain and self._rate_ok("posture", now):
                events.append(Event(
                    "posture",
                    "Posture check",
                    score.correction_text or "Adjust your posture.",
                ))
        else:
            self._poor_since = None
        return events

    def _update_blink(self, blink_rate: float, on_screen: bool, now: float) -> List[Event]:
        events: List[Event] = []
        b = self.cfg["blink"]
        if on_screen and blink_rate < b["low_rate_threshold"]:
            if self._low_blink_since is None:
                self._low_blink_since = now
            if (now - self._low_blink_since) >= b["low_rate_sustain_sec"] \
                    and self._rate_ok("blink", now):
                events.append(Event(
                    "blink",
                    "Low blink rate",
                    "Your blink rate is low — try a few deliberate blinks.",
                ))
        else:
            self._low_blink_since = None
        return events

    def _apply_reminder_toggles(self, events: List[Event]) -> List[Event]:
        """Drop events the user has switched off (master + per-type toggles)."""
        rem = self.cfg.get("reminders", {})
        if not rem.get("enabled", True):
            return []
        # breathing has no dedicated sub-toggle -> governed by the master switch.
        sub = {"break": "break", "posture": "posture", "blink": "blink"}
        return [e for e in events if rem.get(sub.get(e.type, ""), True)]

    def _update_breathing(self, breathing, now: float) -> List[Event]:
        """Periodic diaphragmatic reminder + chest-breathing escalation."""
        events: List[Event] = []
        bc = self.cfg["breathing"]
        if not bc.get("enabled", True):
            return events

        # 1) Fixed-interval nudge (works regardless of detection).
        if self._last_breath_reminder is None:
            self._last_breath_reminder = now  # first nudge one interval from start
        elif (now - self._last_breath_reminder) >= bc["reminder_interval_sec"]:
            events.append(Event(
                "breathing",
                "Breathe",
                "Take a few slow breaths into your belly — relax your shoulders and jaw.",
            ))
            self._last_breath_reminder = now

        # 2) Detector-driven escalation when shoulder breathing persists.
        is_chest = bool(breathing is not None and breathing.valid
                        and breathing.pattern == "chest")
        if is_chest:
            if self._chest_since is None:
                self._chest_since = now
            if (now - self._chest_since) >= bc["chest_sustain_sec"] \
                    and self._rate_ok("breathing_chest", now):
                events.append(Event(
                    "breathing",
                    "Shoulder breathing",
                    "You're chest-breathing — drop your shoulders and breathe low "
                    "into your belly.",
                ))
                # Reset the periodic timer so the two don't stack up.
                self._last_breath_reminder = now
        else:
            self._chest_since = None
        return events

    # ------------------------------------------------------------------
    def _rate_ok(self, etype: str, now: float) -> bool:
        limit = self.cfg["notifications"]["rate_limit_sec"].get(etype, 300)
        last = self._last_emit.get(etype)
        if last is None or (now - last) >= limit:
            self._last_emit[etype] = now
            return True
        return False

    def _build_snapshot(self, score, face_metrics, pose_metrics, now,
                        gaze_on_screen: bool = False, breathing=None) -> Snapshot:
        b = self.cfg["breaks"]
        if self._screen_start is not None:
            cont = now - self._screen_start
            to_break = max(0.0, b["screen_interval_sec"] - cont)
        else:
            cont = 0.0
            to_break = b["screen_interval_sec"]
        total = sum(self._level_times.values())
        if total > 0:
            pct_good = 100.0 * self._level_times["good"] / total
            pct_drifting = 100.0 * self._level_times["drifting"] / total
            pct_poor = 100.0 * self._level_times["poor"] / total
        else:
            pct_good = pct_drifting = pct_poor = 0.0
        return Snapshot(
            score=score,
            blink_rate=face_metrics.blink_rate if face_metrics else 0.0,
            gaze_on_screen=gaze_on_screen,
            face_detected=bool(face_metrics and face_metrics.detected),
            pose_detected=bool(pose_metrics and pose_metrics.detected),
            session_seconds=now - self._start,
            continuous_screen_seconds=cont,
            seconds_to_break=to_break,
            paused=self._paused,
            breaks_taken=self._breaks_taken,
            breathing_rate=breathing.rate_bpm if (breathing and breathing.valid) else 0.0,
            breathing_pattern=breathing.pattern if breathing else "unknown",
            pct_good=round(pct_good, 1),
            pct_drifting=round(pct_drifting, 1),
            pct_poor=round(pct_poor, 1),
        )
