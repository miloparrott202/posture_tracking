"""PyQt6 full-screen correction window + system tray (Sections 6 & 7).

Deliberately minimal: a live camera view with the posture skeleton on the left,
and on the right only what helps you fix your posture — the score, the single
most useful correction cue, a Head / Back / Shoulders breakdown, and the
eye-care readouts (blink rate, next break). Closing the window keeps it running
quietly in the menu-bar/tray at a reduced frame rate.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
    QPushButton, QSpinBox, QSystemTrayIcon, QVBoxLayout, QWidget,
)

from .overlay import draw_overlay

GREEN = "#2ec27e"
AMBER = "#e5a50a"
RED = "#e01b24"
MUTED = "#9aa0a6"


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _tray_icon(score: Optional[float]) -> QIcon:
    size = 64
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if score is None:
        color = QColor(140, 140, 140)
    elif score >= 85:
        color = QColor(46, 194, 126)
    elif score >= 60:
        color = QColor(229, 165, 10)
    else:
        color = QColor(224, 27, 36)
    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, size - 8, size - 8)
    painter.setPen(QColor(255, 255, 255))
    font = painter.font(); font.setPixelSize(30); font.setBold(True)
    painter.setFont(font)
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter,
                     "--" if score is None else f"{int(round(score))}")
    painter.end()
    return QIcon(pix)


class _StatusRow(QWidget):
    """A 'Head / Back / Shoulders' label with a coloured OK / Adjust pill."""

    def __init__(self, name: str):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        self.name = QLabel(name)
        self.name.setStyleSheet("color:#e8e8e8; font-size:15px;")
        self.pill = QLabel("—")
        self.pill.setStyleSheet(f"color:{MUTED}; font-size:14px; font-weight:600;")
        self.pill.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.name)
        row.addStretch(1)
        row.addWidget(self.pill)

    def set_state(self, penalty: float) -> None:
        if penalty < 5:
            self.pill.setText("OK"); color = GREEN
        elif penalty < 20:
            self.pill.setText("drifting"); color = AMBER
        else:
            self.pill.setText("fix"); color = RED
        self.pill.setStyleSheet(f"color:{color}; font-size:14px; font-weight:700;")


class FullView(QMainWindow):
    # Emitted from the processor's recalibration worker thread; Qt delivers the
    # slots on the GUI thread so window show/hide is safe.
    recalibStarted = pyqtSignal()
    recalibFinished = pyqtSignal()
    # First-use hardware/performance warning (emitted from a worker thread).
    diagnosticsWarning = pyqtSignal(str)

    def __init__(self, processor, cfg: dict, notifier=None,
                 on_recalibrate: Optional[Callable] = None,
                 on_quit: Optional[Callable] = None,
                 screenshot: bool = False, settings=None):
        super().__init__()
        self.processor = processor
        self.cfg = cfg
        self.notifier = notifier
        self.on_recalibrate = on_recalibrate
        self.on_quit = on_quit
        self.settings = settings
        self._quitting = False
        self._restore_minimized_after_recal = False

        self._shots = None
        if screenshot:
            from .screenshot import ScreenshotSaver
            self._shots = ScreenshotSaver()
            print("[screenshot] saving frames to screenshots/ (~1/s, newest 10)")

        if notifier is not None:
            self.processor.on_events = lambda evs: [notifier.notify_event(e) for e in evs]
            # Status banners (e.g. "recording new baseline") bypass rate limits.
            self.processor.on_status = lambda title, msg: notifier.notify(title, msg)

        # Drive the un/re-minimise behaviour from the recovery recalibration.
        self.processor.on_recalibration_start = self.recalibStarted.emit
        self.processor.on_recalibration_end = self.recalibFinished.emit
        self.recalibStarted.connect(self._on_recalib_started)
        self.recalibFinished.connect(self._on_recalib_finished)
        self.diagnosticsWarning.connect(self._on_diagnostics_warning)
        self._diag_box = None

        self.setWindowTitle("Posture Tracker")
        self.resize(1060, 720)
        self.setStyleSheet("QMainWindow{background:#16191c;}")

        # --- video ------------------------------------------------------
        self.video = QLabel("Starting camera…")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(720, 540)
        self.video.setStyleSheet("background:#0c0e10; color:#666; border-radius:8px;")

        # --- sidebar ----------------------------------------------------
        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level_label = QLabel("")
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level_label.setStyleSheet(f"color:{MUTED}; font-size:15px; letter-spacing:2px;")
        self.cue_label = QLabel("You're all set. Sit naturally.")
        self.cue_label.setWordWrap(True)
        self.cue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue_label.setStyleSheet("color:#f2f2f2; font-size:16px; padding:6px;")

        # Shown only while a (re)calibration is recording.
        self.recal_banner = QLabel("● Recording new posture baseline…")
        self.recal_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recal_banner.setWordWrap(True)
        self.recal_banner.setStyleSheet(
            f"color:{AMBER}; font-size:14px; font-weight:700; padding:4px;")
        self.recal_banner.setVisible(False)

        self.row_head = _StatusRow("Head")
        self.row_back = _StatusRow("Back")
        self.row_shoulders = _StatusRow("Shoulders")

        self.blink_label = QLabel("Blink rate —")
        self.break_label = QLabel("Next break —:—")
        for lbl in (self.blink_label, self.break_label):
            lbl.setStyleSheet("color:#cfd2d6; font-size:14px;")

        # --- reminder toggles (mirrored in the tray menu) ----------------
        rem = self.cfg.setdefault("reminders", {})
        self._checks = {}
        self._tray_acts = {}
        self.reminders_header = QLabel("Reminders")
        self.reminders_header.setStyleSheet(
            f"color:{MUTED}; font-size:13px; letter-spacing:1px;")
        self._chk_master = self._make_check("enabled", "Enable reminders", rem)
        self._chk_posture = self._make_check("posture", "  Posture", rem)
        self._chk_blink = self._make_check("blink", "  Low blink rate", rem)
        self._chk_break = self._make_check("break", "  Screen breaks", rem)

        # --- timing thresholds (seconds/minutes), applied live --------------
        self.timing_header = QLabel("Timing")
        self.timing_header.setStyleSheet(
            f"color:{MUTED}; font-size:13px; letter-spacing:1px;")
        notif, breaks, blink = (cfg["notifications"], cfg["breaks"], cfg["blink"])
        self.timing_rows = [
            self._make_spin("Nudge after slouch", notif["posture_sustain_sec"],
                            5, 600, " s",
                            lambda v: notif.__setitem__("posture_sustain_sec", v)),
            self._make_spin("Screen break every", breaks["screen_interval_sec"] // 60,
                            1, 120, " min", self._set_break_minutes),
            self._make_spin("Low-blink nudge after", blink["low_rate_sustain_sec"] // 60,
                            1, 60, " min",
                            lambda v: blink.__setitem__("low_rate_sustain_sec", v * 60)),
        ]

        # Issue 3: persisted toggle — un-minimise the window while a post-wake
        # recalibration runs, then re-minimise. Default off.
        show_recal = bool(self.settings.get("show_during_recalibration", False)
                          if self.settings else False)
        self._chk_show_recal = QCheckBox("Show window while recalibrating")
        self._chk_show_recal.setChecked(show_recal)
        self._chk_show_recal.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chk_show_recal.setStyleSheet(
            "QCheckBox{color:#d7dadf; font-size:14px; spacing:8px;}")
        self._chk_show_recal.toggled.connect(self._set_show_during_recal)

        self.recal_btn = QPushButton("Recalibrate posture")
        self.recal_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recal_btn.setStyleSheet(
            "QPushButton{background:#23272b; color:#e8e8e8; border:none;"
            "padding:10px; border-radius:6px; font-size:14px;}"
            "QPushButton:hover{background:#2c3136;}")
        self.recal_btn.clicked.connect(self._recalibrate)

        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(18, 14, 18, 14)
        sidebar.setSpacing(8)
        sidebar.addWidget(self.score_label)
        sidebar.addWidget(self.level_label)
        sidebar.addWidget(self.cue_label)
        sidebar.addWidget(self.recal_banner)
        sidebar.addWidget(self._divider())
        sidebar.addWidget(self.row_head)
        sidebar.addWidget(self.row_back)
        sidebar.addWidget(self.row_shoulders)
        sidebar.addWidget(self._divider())
        sidebar.addWidget(self.blink_label)
        sidebar.addWidget(self.break_label)
        sidebar.addWidget(self._divider())
        sidebar.addWidget(self.reminders_header)
        for c in (self._chk_master, self._chk_posture, self._chk_blink, self._chk_break):
            sidebar.addWidget(c)
        sidebar.addWidget(self._divider())
        sidebar.addWidget(self.timing_header)
        for r in self.timing_rows:
            sidebar.addWidget(r)
        sidebar.addStretch(1)
        sidebar.addWidget(self._chk_show_recal)
        sidebar.addWidget(self.recal_btn)
        self._sync_reminder_widgets()
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(300)
        sidebar_widget.setStyleSheet("background:#1c2024; border-radius:8px;")

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self.video, stretch=1)
        layout.addWidget(sidebar_widget)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self._tray = self._build_tray()
        self._announced_tray = False
        self._sync_reminder_widgets()  # apply enabled/greyed state to tray items

        self._full_ms = int(1000 / max(1, cfg["framerate"]["full_fps"]))
        self._bg_ms = int(1000 / max(0.5, cfg["framerate"]["background_fps"]))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._full_ms)

        self._maybe_run_first_use_check()

    # ------------------------------------------------------------------
    def _maybe_run_first_use_check(self) -> None:
        """On first ever launch, measure throughput and warn if hardware is weak."""
        if self.settings is None:
            return
        if not self.cfg.get("diagnostics", {}).get("enabled", True):
            return
        if self.settings.get("hardware_checked", False):
            return
        threading.Thread(target=self._first_use_check, daemon=True).start()

    def _first_use_check(self) -> None:
        from core.diagnostics import assess_running
        # Let the pipeline warm up so the fps measurement is meaningful.
        time.sleep(float(self.cfg.get("diagnostics", {}).get("warmup_sec", 5)))
        try:
            report = assess_running(self.cfg, self.processor)
        except Exception as exc:
            print(f"[diagnostics] check failed: {exc}")
            self.settings.set("hardware_checked", True)
            return
        self.settings.set("hardware_checked", True)  # only ever run once
        print("[diagnostics]\n" + report.full())
        if report.has_issues:
            if self.notifier is not None:
                self.notifier.notify("Posture Tracker — hardware check", report.short())
            self.diagnosticsWarning.emit(report.full())

    def _on_diagnostics_warning(self, detail: str) -> None:
        # Non-modal so it never blocks the live view.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Posture Tracker — hardware check")
        box.setText("Your hardware may affect tracking performance:")
        box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setModal(False)
        self._diag_box = box  # keep a reference so it isn't garbage-collected
        box.show()

    # ------------------------------------------------------------------
    def _make_check(self, key: str, label: str, rem: dict) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(bool(rem.get(key, True)))
        cb.setCursor(Qt.CursorShape.PointingHandCursor)
        cb.setStyleSheet("QCheckBox{color:#d7dadf; font-size:14px; spacing:8px;}")
        cb.toggled.connect(lambda checked, k=key: self._set_reminder(k, checked))
        self._checks[key] = cb
        return cb

    def _make_spin(self, label: str, value: int, lo: int, hi: int,
                   suffix: str, on_change) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        lab.setStyleSheet("color:#d7dadf; font-size:13px;")
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(int(value))
        spin.setSuffix(suffix)
        spin.setFixedWidth(78)
        spin.setStyleSheet(
            "QSpinBox{background:#23272b; color:#e8e8e8; border:1px solid #2f343a;"
            "border-radius:4px; padding:2px 4px;}")
        spin.valueChanged.connect(on_change)
        h.addWidget(lab)
        h.addStretch(1)
        h.addWidget(spin)
        return row

    def _set_break_minutes(self, minutes: int) -> None:
        # Keep the break interval and its repeat cooldown in step.
        self.cfg["breaks"]["screen_interval_sec"] = minutes * 60
        self.cfg["notifications"]["rate_limit_sec"]["break"] = minutes * 60

    def _set_reminder(self, key: str, value: bool) -> None:
        self.cfg.setdefault("reminders", {})[key] = bool(value)
        if self.settings is not None:
            self.settings.set(f"reminders.{key}", bool(value))  # persist (Issue 4)
        self._sync_reminder_widgets()

    def _set_show_during_recal(self, value: bool) -> None:
        if self.settings is not None:
            self.settings.set("show_during_recalibration", bool(value))

    def _sync_reminder_widgets(self) -> None:
        """Keep the checkboxes, tray menu and config in agreement."""
        rem = self.cfg.setdefault("reminders", {})
        master = bool(rem.get("enabled", True))
        for key, cb in self._checks.items():
            cb.blockSignals(True)
            cb.setChecked(bool(rem.get(key, True)))
            if key != "enabled":
                cb.setEnabled(master)        # grey out sub-toggles when off
            cb.blockSignals(False)
        for key, act in self._tray_acts.items():
            act.blockSignals(True)
            act.setChecked(bool(rem.get(key, True)))
            if key != "enabled":
                act.setEnabled(master)
            act.blockSignals(False)

    # ------------------------------------------------------------------
    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#2a2f34; background:#2a2f34; max-height:1px;")
        return line

    def _build_tray(self) -> Optional[QSystemTrayIcon]:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QSystemTrayIcon(_tray_icon(None), self)
        tray.setToolTip("Posture Tracker")
        menu = QMenu()
        act_open = QAction("Open", self); act_open.triggered.connect(self.show_from_tray)
        self._act_pause = QAction("Pause monitoring", self)
        self._act_pause.triggered.connect(self._toggle_pause)

        rem_menu = menu.addMenu("Reminders")
        for key, label in (("enabled", "Enable reminders"), ("posture", "Posture"),
                           ("blink", "Low blink rate"), ("break", "Screen breaks")):
            act = QAction(label, self, checkable=True)
            act.setChecked(bool(self.cfg.get("reminders", {}).get(key, True)))
            act.toggled.connect(lambda checked, k=key: self._set_reminder(k, checked))
            self._tray_acts[key] = act
            rem_menu.addAction(act)

        act_recal = QAction("Recalibrate", self); act_recal.triggered.connect(self._recalibrate)
        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.quit_app)
        menu.addAction(act_open)
        menu.addAction(self._act_pause)
        menu.addAction(act_recal)
        menu.addAction(act_quit)
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        return tray

    def _on_tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_from_tray()

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        try:
            result = self.processor.process_once()
        except Exception as exc:  # never let a frame error abort the Qt loop
            print(f"[gui] frame error: {exc}")
            return
        if result is None:
            # Camera is momentarily unavailable (e.g. reconnecting after wake).
            if self.isVisible() and not self.processor.camera.is_healthy():
                self.video.setText("Reconnecting camera…")
            return
        now = time.monotonic()
        if self.isVisible() and result.frame is not None:
            self._set_video_pixmap(draw_overlay(result.frame.copy(), result))
            if self._shots is not None:
                self._shots.maybe_save_widget(self, now)
        elif self._shots is not None and result.frame is not None:
            self._shots.maybe_save(draw_overlay(result.frame.copy(), result), now)
        self._update_sidebar(result)
        self._update_tray(result)

    def _set_video_pixmap(self, frame) -> None:
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.width(), self.video.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _update_sidebar(self, result) -> None:
        score = result.score
        snap = result.snapshot
        if score is None or snap is None:
            return
        color = GREEN if score.level == "good" else (
            AMBER if score.level == "drifting" else RED)
        self.score_label.setText(f"{score.score:.0f}")
        self.score_label.setStyleSheet(
            f"color:{color}; font-size:84px; font-weight:800;")
        self.level_label.setText(score.level.upper())
        self.cue_label.setText(score.correction_text or "You're all set. Sit naturally.")
        self.cue_label.setStyleSheet(
            f"color:{'#f2f2f2' if score.level=='good' else color}; "
            f"font-size:16px; padding:6px;")

        gp = score.group_penalties or {}
        self.row_head.set_state(gp.get("crane", 0.0))
        self.row_back.set_state(gp.get("slouch", 0.0))
        self.row_shoulders.set_state(gp.get("asym", 0.0))

        low = self.cfg["blink"]["low_rate_threshold"]
        warn = snap.blink_rate and snap.blink_rate < low
        self.blink_label.setText(f"Blink rate {snap.blink_rate:.0f}/min"
                                 + ("  ⚠ low" if warn else ""))
        self.blink_label.setStyleSheet(
            f"color:{AMBER if warn else '#cfd2d6'}; font-size:14px;")
        self.break_label.setText(f"Next break {_fmt_mmss(snap.seconds_to_break)}")

    def _update_tray(self, result) -> None:
        if self._tray is None or result.snapshot is None:
            return
        snap = result.snapshot
        if snap.paused:
            self._tray.setIcon(_tray_icon(None))
            self._tray.setToolTip("Posture Tracker — paused")
        else:
            self._tray.setIcon(_tray_icon(snap.score.score))
            self._tray.setToolTip(f"Posture {snap.score.score:.0f} ({snap.score.level})")

    # ------------------------------------------------------------------
    def hide_to_tray(self) -> None:
        if self._tray is None:
            self.showMinimized(); return
        self.hide()
        self._timer.setInterval(self._bg_ms)
        print("[tray] Still monitoring in the menu bar. Quit from the tray icon "
              "or press Ctrl-C here.")
        # One-time confirmation that background notifications are live.
        if self.notifier is not None and not self._announced_tray:
            self._announced_tray = True
            self.notifier.notify(
                "Posture Tracker is running",
                "I'll nudge you here about posture, breaks and blinking. "
                "Toggle these from the menu-bar icon.")

    def show_from_tray(self) -> None:
        self._timer.setInterval(self._full_ms)
        self.showNormal(); self.raise_(); self.activateWindow()

    def _toggle_pause(self) -> None:
        paused = not self.processor.state.paused
        self.processor.state.set_paused(paused)
        if self.settings is not None:
            self.settings.set("paused", paused)  # persist (Issue 4)
        if hasattr(self, "_act_pause"):
            self._act_pause.setText("Resume monitoring" if paused else "Pause monitoring")

    def _recalibrate(self) -> None:
        # Use the in-pipeline recalibration so the live video keeps updating
        # (no second camera, no freeze — Issue 2). Falls back to the blocking
        # CLI routine only if the processor doesn't support it.
        if hasattr(self.processor, "trigger_recalibration"):
            self.processor.trigger_recalibration("manual")
        elif self.on_recalibrate:
            self.on_recalibrate()

    # -- recalibration window behaviour (Issue 3) -----------------------
    def _on_recalib_started(self) -> None:
        self.recal_banner.setVisible(True)
        if self.settings is not None and self.settings.get(
                "show_during_recalibration", False):
            # Remember to re-hide afterwards only if we were hidden/minimised.
            self._restore_minimized_after_recal = (
                self.isMinimized() or not self.isVisible())
            if self._restore_minimized_after_recal:
                self.show_from_tray()

    def _on_recalib_finished(self) -> None:
        self.recal_banner.setVisible(False)
        if self._restore_minimized_after_recal:
            self._restore_minimized_after_recal = False
            self.hide_to_tray()

    def quit_app(self) -> None:
        from PyQt6.QtWidgets import QApplication
        self._quitting = True
        if self.on_quit:
            self.on_quit()
        if self._tray is not None:
            self._tray.hide()
        QApplication.quit()

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        if self._quitting or self._tray is None:
            if self._tray is None and not self._quitting:
                event.accept(); self.quit_app(); return
            super().closeEvent(event); return
        event.ignore()
        self.hide_to_tray()
