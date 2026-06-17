#!/usr/bin/env python3
"""Diagnostic harness: stream rich posture/breathing metrics for a fixed time.

Run this in YOUR OWN terminal (so macOS grants camera access) and paste the
output back for analysis:

    source .venv/bin/activate
    python tools/watch.py            # 30 seconds
    python tools/watch.py 45         # 45 seconds

Sit upright for the first half, then deliberately slump forward and drop one
shoulder for the second half so the deviations move. Ctrl-C stops early.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_baseline, load_config  # noqa: E402
from core.pipeline import Processor  # noqa: E402


class _Tee:
    """Write to the terminal and a log file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "watch_last.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"(logging to {log_path})")
    cfg = load_config()
    baseline = load_baseline()
    if baseline is None:
        print("(no baseline.json — scores will be neutral; run --calibrate first)")

    proc = Processor(cfg, baseline)
    try:
        proc.start_camera()
    except Exception as exc:
        print(f"CAMERA ERROR: {exc}")
        return 1

    print(f"\nwatching ~{duration:.0f}s — upright first half, then slump + tilt a "
          f"shoulder...\n", flush=True)

    t0 = time.time()
    n = det = face_det = 0
    scores, tilts, fhas, blinks, necks = [], [], [], [], []
    gaze_flips = 0
    prev_gaze = None
    last_print = -1
    try:
        while time.time() - t0 < duration:
            res = proc.process_once()
            if res is None:
                time.sleep(0.05)
                continue
            n += 1
            pm, fm, s, snap = res.pose, res.face, res.score, res.snapshot
            if pm and pm.detected:
                det += 1
                tilts.append(pm.shoulder_tilt_deg)
                fhas.append(pm.fha_deg)
                necks.append(pm.neck_ratio)
            if fm and fm.detected:
                face_det += 1
            if s:
                scores.append(s.score)
            if snap and snap.blink_rate:
                blinks.append(snap.blink_rate)
            if prev_gaze is not None and res.gaze_on_screen != prev_gaze:
                gaze_flips += 1
            prev_gaze = res.gaze_on_screen

            now = time.time() - t0
            if int(now) != last_print:
                last_print = int(now)
                gp = (s.group_penalties if s else {}) or {}
                rel = "rel" if (pm and getattr(pm, "reliable", False)) else "UNREL"
                frm = "frm" if (pm and getattr(pm, "well_framed", False)) else "NOFRM"
                print(
                    f"t={now:4.1f} pose={int(bool(pm and pm.detected))}/{rel:5s}/{frm:5s} "
                    f"score={(s.score if s else 0):5.1f} {s.level if s else '--':8s} "
                    f"worst={str(s.worst_group if s else '-'):6s} "
                    f"[crane={gp.get('crane',0):4.0f} slouch={gp.get('slouch',0):4.0f} "
                    f"asym={gp.get('asym',0):4.0f}] "
                    f"neck={pm.neck_ratio if pm else 0:4.2f} "
                    f"fwd={pm.forward_head if pm else 0:+5.2f} "
                    f"pitch={fm.head_pitch_deg if fm else 0:+5.1f} "
                    f"breath={snap.breathing_pattern if snap else '-':7s} "
                    f"blink={snap.blink_rate if snap else 0:4.0f} "
                    f"gaze={'ON ' if res.gaze_on_screen else 'AWAY'} "
                    f"cue='{s.correction_text if s else ''}'",
                    flush=True)
    except KeyboardInterrupt:
        print("\n(stopped early)")
    finally:
        proc.stop()

    def rng(xs):
        return f"{min(xs):.1f}..{max(xs):.1f} (mean {sum(xs)/len(xs):.1f})" if xs else "n/a"

    elapsed = time.time() - t0
    print("\n===== SUMMARY =====")
    print(f"frames={n}  fps={n/max(elapsed,1e-3):.1f}")
    print(f"pose detection: {det/max(n,1)*100:.0f}%   face detection: {face_det/max(n,1)*100:.0f}%")
    print(f"score: {rng(scores)}")
    print(f"shoulder_tilt deg: {rng(tilts)}")
    print(f"fha deg: {rng(fhas)}")
    print(f"neck_ratio (slouch; lower=hunched): {rng(necks)}")
    print(f"blink rate seen: {rng(blinks)}")
    print(f"gaze on/away flips: {gaze_flips}")
    snap = proc.state.snapshot()
    print(f"posture mix: {snap.pct_good:.0f}% good / {snap.pct_drifting:.0f}% drift "
          f"/ {snap.pct_poor:.0f}% poor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
