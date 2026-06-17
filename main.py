#!/usr/bin/env python3
"""Posture & Eye-Strain Tracker — entry point and mode dispatch (Section 9/10).

Usage examples (easy terminal prototyping):

    python main.py --calibrate      # record your ideal-posture baseline
    python main.py --console        # headless: stream metrics to the terminal
    python main.py --background      # tray icon + OS notifications only
    python main.py                   # full-screen GUI (default)

    python main.py --config myconfig.yaml --console
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from core.config import DEFAULT_CONFIG_PATH, load_baseline, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _require_baseline(baseline):
    if baseline is None:
        print("No baseline found. Run calibration first:\n"
              "    python main.py --calibrate", file=sys.stderr)
        return False
    return True


def _spawn_hardware_check(cfg, proc, notifier, settings) -> None:
    """First-use only: warm up, then warn (notification) if hardware is weak.

    Used by the non-GUI modes. The full GUI runs its own check (with a dialog)
    inside FullView.
    """
    if settings is None or not cfg.get("diagnostics", {}).get("enabled", True):
        return
    if settings.get("hardware_checked", False):
        return

    def run():
        from core.diagnostics import assess_running
        time.sleep(float(cfg.get("diagnostics", {}).get("warmup_sec", 5)))
        try:
            report = assess_running(cfg, proc)
        except Exception as exc:
            print(f"[diagnostics] check failed: {exc}")
            settings.set("hardware_checked", True)
            return
        settings.set("hardware_checked", True)
        print("[diagnostics]\n" + report.full())
        if report.has_issues and notifier is not None:
            notifier.notify("Posture Tracker — hardware check", report.short())

    threading.Thread(target=run, daemon=True).start()


def run_diagnostics(cfg) -> int:
    """Standalone hardware/performance report (python main.py --diagnostics)."""
    from core.diagnostics import run_standalone
    from notifications.notifier import Notifier

    print("Running hardware/performance diagnostics "
          f"(~{cfg.get('diagnostics', {}).get('warmup_sec', 5)}s)…\n")
    report = run_standalone(cfg)
    print(report.full())
    if report.has_issues:
        Notifier(cfg).notify("Posture Tracker — hardware check", report.short())
    return 0 if not report.critical else 2


# ----------------------------------------------------------------------
def run_console(cfg, baseline, screenshot: bool = False, settings=None) -> int:
    """Milestone-1 prototype: stream metrics to the terminal, no GUI."""
    from core.pipeline import Processor
    from notifications.notifier import Notifier

    notifier = Notifier(cfg)
    proc = Processor(cfg, baseline, on_events=lambda evs: [
        print(f"  >> EVENT [{e.type}] {e.title}: {e.message}") for e in evs])
    try:
        proc.start_camera()
    except RuntimeError as exc:
        print(f"\nCamera error: {exc}", file=sys.stderr)
        notifier.notify("Posture Tracker — no camera", str(exc))
        return 1
    _spawn_hardware_check(cfg, proc, notifier, settings)

    shots = overlay = None
    if screenshot:
        from ui.overlay import draw_overlay as overlay
        from ui.screenshot import ScreenshotSaver
        shots = ScreenshotSaver()
        print("[screenshot] saving frames to screenshots/ (~1/s, newest 10)")
    print("Streaming metrics (Ctrl-C to stop)...\n")
    try:
        target_fps = cfg["framerate"]["full_fps"]
        period = 1.0 / max(1, target_fps)
        while True:
            t0 = time.monotonic()
            res = proc.process_once()
            if res is not None and shots is not None and res.frame is not None:
                shots.maybe_save(overlay(res.frame.copy(), res), t0)
            if res is not None and res.score is not None:
                s, snap = res.score, res.snapshot
                dev = s.deviations
                breath = (f"{snap.breathing_rate:2.0f}/min {snap.breathing_pattern}"
                          if snap.breathing_pattern != "unknown" else "--")
                print(
                    f"\rscore={s.score:5.1f} [{s.level:8s}] "
                    f"FHA_dev={dev.get('fha', 0):+5.1f} "
                    f"tilt_dev={dev.get('shoulder_tilt', 0):+5.1f} "
                    f"blink={snap.blink_rate:4.0f}/min "
                    f"breath={breath:14s} "
                    f"gaze={'ON ' if res.gaze_on_screen else 'AWAY'} "
                    f"break_in={snap.seconds_to_break/60:4.1f}m   ",
                    end="", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        proc.stop()
    return 0


def run_test_notify(cfg) -> int:
    """Send test notifications and explain the macOS permission setting."""
    from notifications.notifier import Notifier

    notifier = Notifier(cfg)
    print("Sending 2 test notifications now — watch the top-right of your screen...")
    notifier.notify("Posture Tracker", "Test 1 of 2 — if you see this, nudges work.")
    time.sleep(2.0)
    notifier.notify("Posture check", "Sit up tall and lengthen your spine.")
    print("\nDid a banner appear? If NOT, macOS is blocking it. Enable it:")
    print("  System Settings  ->  Notifications")
    print("  ->  find \"Script Editor\"  (osascript banners post under that name)")
    print("  ->  turn ON \"Allow Notifications\"; set the style to Banners or Alerts.")
    print("  Then re-run:  python main.py --test-notify")
    print("\n(If you use a notification 'Focus'/Do-Not-Disturb mode, allow it there too.)")
    return 0


def run_calibrate(cfg) -> int:
    from core.calibration import run_calibration

    imu = None
    if cfg["imu"]["enabled"]:
        try:
            from core.imu_bridge import IMUBridge
            imu = IMUBridge(cfg).start()
            time.sleep(0.5)  # let a few samples arrive
        except Exception as exc:
            logging.warning("IMU unavailable for calibration: %s", exc)
    try:
        run_calibration(cfg, imu_bridge=imu)
    finally:
        if imu is not None:
            imu.stop()
    return 0


def run_background(cfg, baseline, settings=None) -> int:
    """Standalone tray icon + notifications (Section 7). No GUI buffers.

    This is the lightweight, GUI-free path (pystray). The default full mode
    already includes its own system tray, so use this only when you never want
    the camera window — e.g. launch-at-login background monitoring.
    """
    from core.pipeline import Processor
    from notifications.notifier import Notifier
    from ui.tray import TrayApp

    notifier = Notifier(cfg)
    # Status banners (e.g. post-wake "recording new baseline") go straight out.
    proc = Processor(cfg, baseline,
                     on_status=lambda title, msg: notifier.notify(title, msg))
    if settings is not None and settings.get("paused", False):
        proc.state.set_paused(True)  # restore persisted pause (Issue 4)
    # Start the camera up front so a missing device is reported immediately.
    try:
        proc.start_camera()
    except RuntimeError as exc:
        print(f"\nCamera error: {exc}", file=sys.stderr)
        notifier.notify("Posture Tracker — no camera", str(exc))
        return 1
    _spawn_hardware_check(cfg, proc, notifier, settings)

    def open_full():
        print("[tray] 'Open full view' — relaunch with: python main.py --full")

    tray = TrayApp(
        proc, notifier, cfg,
        on_open_full=open_full,
        on_recalibrate=lambda: proc.trigger_recalibration("manual"),
        on_quit=lambda: proc.stop(),
        settings=settings,
    )
    try:
        tray.run()  # blocks
    except KeyboardInterrupt:
        pass
    finally:
        tray.stop()
        proc.stop()
    return 0


def run_full(cfg, baseline, settings=None, screenshot: bool = False) -> int:
    """Full-screen PyQt6 correction GUI with built-in system tray (Sections 6-7).

    A single QApplication / Processor / camera drives both the visible window
    and the tray-backed background mode, so 'minimise to tray' and 'open full
    view' just hide/show one window — no second camera handle.
    """
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from core.pipeline import Processor
    from notifications.notifier import Notifier
    from ui.full_view import FullView

    notifier = Notifier(cfg)
    proc = Processor(cfg, baseline)
    if settings is not None and settings.get("paused", False):
        proc.state.set_paused(True)  # restore persisted pause (Issue 4)
    try:
        proc.start_camera()
    except RuntimeError as exc:
        print(f"\nCamera error: {exc}", file=sys.stderr)
        notifier.notify("Posture Tracker — no camera", str(exc))
        return 1

    app = QApplication(sys.argv)
    # Keep running when the window closes to the tray (don't quit on last window).
    app.setQuitOnLastWindowClosed(False)
    window = FullView(
        proc, cfg, notifier=notifier,
        on_recalibrate=lambda: run_calibrate(cfg),
        on_quit=lambda: proc.stop(),
        screenshot=screenshot,
        settings=settings,
    )
    window.show()

    # Make Ctrl-C in the launching terminal quit cleanly. Qt's C++ event loop
    # normally swallows SIGINT, so we route it to a clean quit and pump a no-op
    # timer periodically to give Python a chance to run the handler.
    import signal
    signal.signal(signal.SIGINT, lambda *_: window.quit_app())
    wake = QTimer()
    wake.start(250)
    wake.timeout.connect(lambda: None)

    try:
        code = app.exec()
    finally:
        proc.stop()
    return code


# ----------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Posture & Eye-Strain Tracker")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true",
                      help="record ideal-posture baseline (run this first)")
    mode.add_argument("--console", action="store_true",
                      help="headless: stream metrics to the terminal")
    mode.add_argument("--background", "--tray", dest="background",
                      action="store_true", help="tray icon + notifications only")
    mode.add_argument("--full", action="store_true",
                      help="full-screen GUI (default)")
    mode.add_argument("--diagnostics", action="store_true",
                      help="check camera + CPU performance and print a report")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="path to config.yaml")
    parser.add_argument("--screenshot", action="store_true",
                        help="save the rendered frame to screenshots/ ~1/s "
                             "(rolling 10, wiped each run) for remote inspection")
    parser.add_argument("--test-notify", action="store_true",
                        help="send test notifications and print macOS permission help")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    cfg = load_config(args.config)

    # Restore persisted UI toggles (reminders, etc.) over the loaded config so
    # they survive between sessions (Issue 4).
    from core.settings import Settings
    settings = Settings()
    settings.apply_to_config(cfg)

    baseline = load_baseline()

    if args.test_notify:
        return run_test_notify(cfg)

    if args.diagnostics:
        return run_diagnostics(cfg)

    if args.calibrate:
        return run_calibrate(cfg)

    if args.console:
        if not _require_baseline(baseline):
            print("(continuing without baseline — scores will be neutral)\n")
        return run_console(cfg, baseline, screenshot=args.screenshot, settings=settings)

    if args.background:
        baseline = _ensure_baseline(cfg, baseline)
        if baseline is None:
            return 1
        return run_background(cfg, baseline, settings)

    # default: full GUI
    baseline = _ensure_baseline(cfg, baseline)
    if baseline is None:
        return 1
    return run_full(cfg, baseline, settings, screenshot=args.screenshot)


def _ensure_baseline(cfg, baseline):
    """Return a usable, current baseline, calibrating if missing or outdated."""
    from core.config import baseline_is_current

    if baseline is not None and not baseline_is_current(baseline):
        print("Your saved baseline is from an older version — recalibrating "
              "with the improved posture tracking.\n")
        baseline = None
    if baseline is not None:
        return baseline
    print("No current posture baseline — running calibration first.\n")
    run_calibrate(cfg)
    baseline = load_baseline()
    if baseline is None:
        print("Calibration did not complete. Re-run: python main.py --calibrate",
              file=sys.stderr)
    return baseline


if __name__ == "__main__":
    raise SystemExit(main())
