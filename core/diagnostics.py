"""Hardware / performance diagnostics.

Runs a lightweight check of the host (OS, CPU, camera, achieved throughput) so
the app works across machines and warns the user *once, on first use* when the
hardware looks insufficient or is likely hurting performance.

Two entry points:
  - ``assess_running(cfg, processor)`` — reuses an already-running Processor's
    camera + measured throughput (used for the first-use GUI/tray check).
  - ``run_standalone(cfg)``           — opens its own pipeline for a few seconds
    (used by ``python main.py --diagnostics``).
"""
from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("diagnostics")


@dataclass
class DiagnosticsReport:
    system: dict = field(default_factory=dict)
    camera: dict = field(default_factory=dict)
    perf: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    critical: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings and not self.critical

    @property
    def has_issues(self) -> bool:
        return bool(self.warnings or self.critical)

    def short(self) -> str:
        """One-line message suitable for an OS notification."""
        issues = self.critical + self.warnings
        if not issues:
            return "Hardware looks fine — you're good to go."
        head = "Performance note: " if not self.critical else "Heads up: "
        return head + " ".join(issues[:2])

    def full(self) -> str:
        """Multi-line detail for the terminal or a dialog box."""
        s = self.system
        lines = [
            f"OS:      {s.get('os','?')} {s.get('os_release','')} ({s.get('machine','?')})",
            f"CPU:     {s.get('cpu_count','?')} logical cores",
            f"Python:  {s.get('python','?')}   OpenCV: {s.get('opencv','?')}",
        ]
        c = self.camera
        if c.get("ok"):
            lines.append(
                f"Camera:  index {c.get('index')} @ "
                f"{c.get('native_width','?')}x{c.get('native_height','?')} native")
        else:
            lines.append(f"Camera:  NOT AVAILABLE ({c.get('error','no device')})")
        p = self.perf
        if p.get("frames"):
            lines.append(
                f"Speed:   ~{p.get('fps',0):.0f} fps, "
                f"{p.get('proc_ms',0):.0f} ms/frame inference")
        if self.critical:
            lines.append("")
            lines += [f"  ✖ {w}" for w in self.critical]
        if self.warnings:
            lines.append("")
            lines += [f"  ⚠ {w}" for w in self.warnings]
        if self.ok:
            lines.append("\nNo issues detected.")
        return "\n".join(lines)


# ----------------------------------------------------------------------
def collect_system_info() -> dict:
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
    }
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        info["opencv"] = "?"
    try:  # optional: only if psutil happens to be installed
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    return info


def _camera_info(camera) -> dict:
    ok = (camera.active_index is not None) or (camera.native_width > 0)
    return {
        "ok": bool(ok),
        "index": camera.active_index,
        "backend": camera.active_backend,
        "native_width": camera.native_width,
        "native_height": camera.native_height,
    }


def assess(cfg: dict, system: dict, camera: dict, perf: dict) -> DiagnosticsReport:
    """Pure decision logic: turn measurements into warnings/criticals."""
    d = cfg.get("diagnostics", {})
    warnings: List[str] = []
    critical: List[str] = []

    if not camera.get("ok", False):
        critical.append(
            "No camera detected — posture tracking can't run without a webcam.")
    else:
        nw, nh = camera.get("native_width", 0), camera.get("native_height", 0)
        if nw and nh and (nw < d.get("min_camera_width", 320)
                          or nh < d.get("min_camera_height", 240)):
            warnings.append(
                f"Low-resolution camera ({nw}x{nh}); posture detection may be "
                "less accurate.")

    cores = system.get("cpu_count", 0)
    if cores and cores < d.get("min_cpu_cores", 2):
        warnings.append(
            f"Only {cores} CPU core(s) detected; tracking may run slowly.")

    if perf.get("frames", 0) >= d.get("min_frames_for_perf", 15):
        fps = perf.get("fps", 0.0)
        if fps and fps < d.get("min_fps", 10):
            warnings.append(
                f"Your computer is processing only ~{fps:.0f} fps, so posture "
                "updates may lag. Lower camera.width/height (e.g. 320x240) in "
                "config.yaml to speed things up.")

    return DiagnosticsReport(system, camera, perf, warnings, critical)


# ----------------------------------------------------------------------
def assess_running(cfg: dict, processor) -> DiagnosticsReport:
    """Assess using a Processor that is already capturing/processing frames."""
    return assess(cfg, collect_system_info(),
                  _camera_info(processor.camera), processor.get_perf())


def run_standalone(cfg: dict) -> DiagnosticsReport:
    """Spin up the pipeline briefly and produce a full report (CLI use)."""
    from .pipeline import Processor

    warmup = float(cfg.get("diagnostics", {}).get("warmup_sec", 5))
    proc = Processor(cfg, None)
    try:
        proc.start_camera()
    except Exception as exc:
        log.warning("camera unavailable: %s", exc)
        return assess(cfg, collect_system_info(),
                      {"ok": False, "error": str(exc)}, {})
    try:
        deadline = time.monotonic() + warmup
        while time.monotonic() < deadline:
            proc.process_once()
            time.sleep(0.005)
        report = assess_running(cfg, proc)
    finally:
        proc.stop()
    return report
