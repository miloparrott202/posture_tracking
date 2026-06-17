"""Periodic health-summary logging (Section 11).

Appends timestamped JSON lines that an external pipeline can ingest as a new
data domain without touching the tracker.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import List, Optional


class HealthLogger:
    def __init__(self, cfg: dict):
        hc = cfg["health_log"]
        self.enabled = hc.get("enabled", True)
        self.path = hc.get("path", "posture_health_log.jsonl")
        self.interval = hc.get("interval_sec", 300)
        self._last_write = time.monotonic()
        self._scores: List[float] = []
        self._blink_rates: List[float] = []
        self._breath_rates: List[float] = []

    def observe(self, snapshot) -> None:
        """Accumulate per-frame samples between flushes."""
        if not self.enabled:
            return
        self._scores.append(snapshot.score.score)
        if snapshot.face_detected:
            self._blink_rates.append(snapshot.blink_rate)
        if getattr(snapshot, "breathing_rate", 0.0) > 0.0:
            self._breath_rates.append(snapshot.breathing_rate)
        now = time.monotonic()
        if now - self._last_write >= self.interval:
            self.flush(snapshot)
            self._last_write = now

    def flush(self, snapshot) -> Optional[dict]:
        if not self.enabled or not self._scores:
            return None
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "avg_posture_score": round(sum(self._scores) / len(self._scores), 1),
            "blink_rate": round(
                sum(self._blink_rates) / len(self._blink_rates), 1
            ) if self._blink_rates else 0.0,
            "breathing_rate": round(
                sum(self._breath_rates) / len(self._breath_rates), 1
            ) if self._breath_rates else 0.0,
            "pct_good_posture": round(getattr(snapshot, "pct_good", 0.0), 1),
            "screen_time_min": round(snapshot.continuous_screen_seconds / 60.0, 1),
            "breaks_taken": snapshot.breaks_taken,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self._scores.clear()
        self._blink_rates.clear()
        self._breath_rates.clear()
        return record
