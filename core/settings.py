"""Persisted user toggles (``settings.json``).

Distinct from ``config.yaml`` (hand-edited tuning) and ``baseline.json``
(calibration data): this file remembers the *toggles the user flips in the UI*
so they survive between sessions — reminder switches, pause state, and the
"show window during recalibration" option.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from .config import PROJECT_ROOT

log = logging.getLogger("settings")

DEFAULT_SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.json")

# Defaults for every persisted toggle. New toggles default here.
DEFAULTS: Dict[str, Any] = {
    # Issue 3: when on, the GUI un-minimises while a post-wake recalibration
    # runs, then re-minimises. Default OFF.
    "show_during_recalibration": False,
    # Mirrors cfg["reminders"]; restored over config on startup.
    "reminders": {"enabled": True, "posture": True, "blink": True, "break": True},
    # Whether monitoring was paused when the app last closed.
    "paused": False,
    # Set True after the first-use hardware/performance check has run, so the
    # warning popup only appears once.
    "hardware_checked": False,
}


class Settings:
    """Tiny JSON-backed settings store with dotted-key access and autosave."""

    def __init__(self, path: str | None = None):
        self.path = path or DEFAULT_SETTINGS_PATH
        self._data: Dict[str, Any] = json.loads(json.dumps(DEFAULTS))  # deep copy
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                saved = json.load(fh) or {}
            self._merge(self._data, saved)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s (%s) — using defaults", self.path, exc)

    @staticmethod
    def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> None:
        for key, value in over.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                Settings._merge(base[key], value)
            else:
                base[key] = value

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
        except OSError as exc:
            log.warning("Could not write %s (%s)", self.path, exc)

    # ------------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        self.save()

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    def apply_to_config(self, cfg: Dict[str, Any]) -> None:
        """Restore persisted toggles onto a freshly loaded config in place."""
        reminders = self.get("reminders", {})
        if isinstance(reminders, dict):
            cfg.setdefault("reminders", {}).update(reminders)
