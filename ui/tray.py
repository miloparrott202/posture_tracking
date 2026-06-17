"""System-tray / menu-bar icon for background mode (Section 7.3).

Uses pystray (cross-platform). The processor runs in its own background thread
at the reduced frame rate; this module only renders the icon, updates the
score label periodically, and dispatches notifications for emitted events.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from core.state_manager import Event


def _make_icon_image(score: Optional[float]):
    """Render a small colour-coded dot with the score number."""
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if score is None:
        color = (120, 120, 120, 255)
    elif score >= 85:
        color = (40, 200, 60, 255)
    elif score >= 60:
        color = (220, 200, 30, 255)
    else:
        color = (220, 50, 50, 255)
    draw.ellipse((4, 4, size - 4, size - 4), fill=color)
    label = "--" if score is None else f"{int(score)}"
    draw.text((size / 2 - 6 * len(label) / 2, size / 2 - 10), label, fill=(0, 0, 0, 255))
    return img


class TrayApp:
    """Background tray controller wrapping a Processor + Notifier."""

    def __init__(self, processor, notifier, cfg: dict,
                 on_open_full: Optional[Callable] = None,
                 on_recalibrate: Optional[Callable] = None,
                 on_quit: Optional[Callable] = None,
                 settings=None):
        self.processor = processor
        self.notifier = notifier
        self.cfg = cfg
        self.on_open_full = on_open_full
        self.on_recalibrate = on_recalibrate
        self.on_quit = on_quit
        self.settings = settings
        # Restore persisted pause state (Issue 4).
        self._paused = bool(settings.get("paused", False)) if settings else False
        self.processor.state.set_paused(self._paused)
        self._icon = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    def dispatch_events(self, events) -> None:
        """Processor callback: surface events as OS notifications + sound."""
        for ev in events:
            self.notifier.notify_event(ev)

    # ------------------------------------------------------------------
    def run(self) -> None:
        import pystray

        self.processor.on_events = self.dispatch_events
        # Background frame rate (Section 3/7.1).
        self.processor.start_background(
            fps_getter=lambda: 0.5 if self._paused else self.cfg["framerate"]["background_fps"]
        )

        menu = pystray.Menu(
            pystray.MenuItem(lambda item: self._score_text(), None, enabled=False),
            pystray.MenuItem("Open full view", self._open_full),
            pystray.MenuItem(
                lambda item: "Resume monitoring" if self._paused else "Pause monitoring",
                self._toggle_pause),
            pystray.MenuItem("Recalibrate baseline", self._recalibrate),
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon = pystray.Icon(
            "posture_tracker", _make_icon_image(None), "Posture Tracker", menu)

        threading.Thread(target=self._label_updater, daemon=True).start()
        self._icon.run()  # blocks until icon.stop()

    # ------------------------------------------------------------------
    def _score_text(self) -> str:
        snap = self.processor.state.snapshot()
        if self._paused:
            return "Paused"
        return f"Posture: {snap.score.score:.0f} ({snap.score.level})"

    def _label_updater(self) -> None:
        while not self._stop.is_set():
            if self._icon is not None:
                snap = self.processor.state.snapshot()
                try:
                    self._icon.icon = _make_icon_image(
                        None if self._paused else snap.score.score)
                    self._icon.title = self._score_text()
                    self._icon.update_menu()
                except Exception:
                    pass
            time.sleep(5)

    def _open_full(self, *_):
        if self.on_open_full:
            self.on_open_full()

    def _toggle_pause(self, *_):
        self._paused = not self._paused
        self.processor.state.set_paused(self._paused)
        if self.settings is not None:
            self.settings.set("paused", self._paused)  # persist (Issue 4)

    def _recalibrate(self, *_):
        if self.on_recalibrate:
            self.on_recalibrate()

    def _quit(self, *_):
        self._stop.set()
        if self._icon is not None:
            self._icon.stop()
        if self.on_quit:
            self.on_quit()

    def stop(self) -> None:
        self._stop.set()
        if self._icon is not None:
            self._icon.stop()
