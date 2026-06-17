"""Cross-platform notification wrapper (Section 7.2).

Prefers ``plyer`` for portability; on macOS falls back to ``osascript`` for
richer/native formatting. Rate-limiting lives in the StateManager, so this
class simply dispatches whatever it is given.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess

log = logging.getLogger("notifier")


class Notifier:
    def __init__(self, cfg: dict, app_name: str = "Posture Tracker"):
        self.app_name = app_name
        self.backend = cfg["notifications"].get("backend", "auto")
        self.sound = bool(cfg["notifications"].get("sound", True))
        self._is_mac = platform.system() == "Darwin"
        self._plyer = self._load_plyer()

    @staticmethod
    def _load_plyer():
        try:
            from plyer import notification
            return notification
        except Exception:
            return None

    def notify_event(self, event) -> None:
        """Dispatch an Event: OS notification, plus an audible alert for posture
        warnings (which are easy to miss as a silent banner)."""
        self.notify(event.title, event.message)
        if event.type == "posture":
            self.beep()

    def beep(self) -> None:
        """Play a short audible alert (best-effort, never raises)."""
        if not self.sound:
            return
        try:
            if self._is_mac and shutil.which("afplay"):
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Funk.aiff"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        except Exception:
            pass
        print("\a", end="", flush=True)  # terminal bell fallback

    def notify(self, title: str, message: str) -> None:
        # Always log so nudges are visible in the terminal even if the OS banner
        # is suppressed by notification permissions.
        log.info("nudge: %s — %s", title, message)

        backend = self.backend
        if backend == "auto":
            backend = "osascript" if self._is_mac and shutil.which("osascript") else "plyer"

        if backend == "osascript" and self._is_mac:
            if self._notify_osascript(title, message):
                return
        if self._plyer is not None:
            try:
                self._plyer.notify(title=title, message=message,
                                   app_name=self.app_name, timeout=8)
                return
            except Exception as exc:
                log.warning("plyer notification failed: %s", exc)
        # Last resort: console.
        print(f"[NOTIFY] {title}: {message}")

    def _notify_osascript(self, title: str, message: str) -> bool:
        try:
            safe_t = title.replace('"', "'")
            safe_m = message.replace('"', "'")
            script = f'display notification "{safe_m}" with title "{self.app_name}" subtitle "{safe_t}"'
            subprocess.run(["osascript", "-e", script], check=True,
                           capture_output=True, timeout=5)
            return True
        except Exception as exc:
            log.warning("osascript notification failed: %s", exc)
            return False
