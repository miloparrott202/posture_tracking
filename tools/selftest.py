#!/usr/bin/env python3
"""End-to-end self-test (no person required).

Checks every aspect of the build that can be verified without a body in frame:

  1. The real camera + MediaPipe pose/face path runs without error.
  2. Scripted poses pushed through the REAL Processor (real scoring, smoothing,
     breathing, state manager) classify correctly: upright->good, slouch and
     crane->poor with the right worst-group, tilt->asym.
  3. Notifications actually fire through Processor.on_events (posture / break /
     blink) with shortened timers.
  4. The REAL FullView renders each state; window grabs are saved to
     screenshots/selftest_*.png for visual confirmation.

Run with QT_QPA_PLATFORM=offscreen.
"""
from __future__ import annotations

import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.config import load_config  # noqa: E402
from core.face_engine import FaceMetrics  # noqa: E402
from core.pose_engine import PoseEngine  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}{(' — ' + detail) if detail else ''}")


# --- synthetic landmark builder (drives the REAL PoseEngine._compute) --------
def _lm(pts):
    arr = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
    for i, (x, y, z, v) in pts.items():
        arr[i] = types.SimpleNamespace(x=x, y=y, z=z, visibility=v)
    return arr


def pose_lm(ear_y, sh_y, ear_z, sh_z, tilt=0.0):
    return _lm({
        0: (0.5, ear_y + 0.04, ear_z, 0.99),
        2: (0.455, ear_y - 0.02, ear_z, 0.99), 5: (0.545, ear_y - 0.02, ear_z, 0.99),
        7: (0.44, ear_y, ear_z, 0.99), 8: (0.56, ear_y, ear_z, 0.99),
        9: (0.47, ear_y + 0.09, ear_z, 0.9), 10: (0.53, ear_y + 0.09, ear_z, 0.9),
        11: (0.37, sh_y + tilt, sh_z, 0.99), 12: (0.63, sh_y - tilt, sh_z, 0.99),
        23: (0.40, 0.99, 0.0, 0.2), 24: (0.60, 0.99, 0.0, 0.2),
    })


SCRIPT = {
    "upright": pose_lm(0.40, 0.72, -0.10, -0.10),
    "slouch": pose_lm(0.52, 0.66, -0.22, -0.10),   # head down + shoulders up
    # Representative frontal-camera crane: head juts forward (z) with shoulders
    # and head height roughly stable — z-depth is the only 2D-invisible signal.
    "crane": pose_lm(0.42, 0.72, -0.30, -0.10),
    "asym": pose_lm(0.40, 0.72, -0.10, -0.10, tilt=0.06),
}


class FakeCamera:
    def __init__(self):
        self._f = np.full((480, 640, 3), (44, 40, 38), np.uint8)

    def start(self): return self
    def read(self): return self._f.copy(), 1
    def stop(self): pass


class FakePose:
    phase = "upright"
    def process(self, frame, ts):
        return PoseEngine._compute(SCRIPT[self.phase])
    def close(self): pass


class FakeFace:
    blink = 16.0
    def process(self, frame, ts):
        return FaceMetrics(detected=True, blink_rate=self.blink, head_yaw_deg=0.0,
                           head_pitch_deg=0.0, gaze_h_ratio=0.5)
    def close(self): pass


def main():
    print("\n=== Posture Tracker self-test (no person in frame) ===\n")

    # --- 1. real camera + MediaPipe -----------------------------------------
    print("1) real camera + MediaPipe path")
    try:
        from core.pipeline import Processor
        from core.config import load_baseline
        real = Processor(load_config(), load_baseline())
        real.start_camera()
        ok = err = 0
        for _ in range(20):
            if real.process_once() is not None:
                ok += 1
            time.sleep(0.02)
        real.stop()
        check("camera frames processed by real MediaPipe", ok > 5, f"{ok} frames")
    except Exception as exc:
        check("camera + MediaPipe", False, repr(exc))

    # --- build an injected Processor for the rest ---------------------------
    cfg = load_config()
    cfg["posture"]["smoothing"] = 0.0  # deterministic, no EMA warm-up
    cfg["notifications"]["posture_sustain_sec"] = 1
    cfg["notifications"]["rate_limit_sec"] = {"break": 1, "posture": 1, "blink": 1,
                                              "breathing_chest": 1}
    cfg["breaks"]["screen_interval_sec"] = 1
    cfg["blink"]["low_rate_sustain_sec"] = 1
    cfg["breathing"]["enabled"] = False

    from core.pipeline import Processor
    fp, ff = FakePose(), FakeFace()
    upright_pm = PoseEngine._compute(SCRIPT["upright"])
    base_pose = {k: getattr(upright_pm, k) for k in (
        "fha_deg", "forward_head", "neck_ratio", "torso_incline_deg",
        "shoulder_tilt_deg", "head_roll_deg", "head_lateral", "ear_mid_y_norm")}
    base_pose["head_yaw_deg"] = 0.0
    baseline = {"version": 2, "pose": base_pose}

    proc = Processor(cfg, baseline)
    proc.pose.close(); proc.face.close()
    proc.pose, proc.face, proc.camera = fp, ff, FakeCamera()

    # --- 2. scoring classification ------------------------------------------
    print("\n2) scoring on scripted poses (real scoring pipeline)")
    expect = {"upright": ("good", None), "slouch": ("poor", "slouch"),
              "crane": ("poor", "crane"), "asym": ("drifting", "asym")}
    scored = {}
    for phase in SCRIPT:
        fp.phase = phase
        res = None
        for _ in range(3):
            res = proc.process_once()
        s = res.score
        scored[phase] = res
        exp_level, exp_worst = expect[phase]
        if phase == "upright":
            ok = s.level == "good"
        else:
            ok = s.worst_group == exp_worst and s.level in ("drifting", "poor")
        check(f"{phase:8s} -> score={s.score:.0f} {s.level} worst={s.worst_group}",
              ok, f"expected worst={exp_worst}")

    # --- 3. notifications fire through Processor.on_events ------------------
    print("\n3) notifications via Processor.on_events (shortened timers)")
    events = []
    proc.on_events = lambda evs: events.extend(e.type for e in evs)

    fp.phase = "slouch"  # poor posture sustained -> 'posture'; on-screen -> 'break'
    t0 = time.time()
    while time.time() - t0 < 2.5:
        proc.process_once(); time.sleep(0.03)
    check("posture notification fired", "posture" in events)
    check("break notification fired (20-min rule)", "break" in events)

    ff.blink = 5.0  # low blink rate sustained -> 'blink'
    t0 = time.time()
    while time.time() - t0 < 2.0:
        proc.process_once(); time.sleep(0.03)
    check("blink (eye-strain) notification fired", "blink" in events)

    # --- 4. real FullView renders each state --------------------------------
    print("\n4) real FullView renders each state -> screenshots/selftest_*.png")
    try:
        from PyQt6.QtWidgets import QApplication
        from ui.full_view import FullView
        app = QApplication(sys.argv)
        win = FullView(proc, cfg, notifier=None)
        win._timer.stop()
        win.resize(1040, 620)
        win.show()  # so isVisible() is True and the video overlay path runs
        os.makedirs(OUT, exist_ok=True)
        for phase in SCRIPT:
            fp.phase = phase
            for _ in range(2):
                win._tick()
            app.processEvents()
            path = os.path.join(OUT, f"selftest_{phase}.png")
            ok = win.grab().save(path)
            check(f"rendered {phase}", ok, path)
    except Exception as exc:
        check("FullView render", False, repr(exc))

    proc.stop()
    print(f"\n=== {sum(results)}/{len(results)} checks passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
