"""One-time baseline calibration (Section 4.3 + 8.4).

Records the user's self-assessed ideal posture for 5 seconds and writes the
averaged metrics to baseline.json. If an IMU is connected, its pitch/roll/yaw
baseline is captured in the same pass so fused deviations zero correctly.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from .capture import Camera
from .config import BASELINE_VERSION, save_baseline
from .face_engine import FaceEngine
from .pose_engine import PoseEngine

# Pose features averaged into the baseline (must match PoseMetrics attributes).
POSE_KEYS = ["fha_deg", "forward_head", "neck_ratio", "torso_incline_deg",
             "shoulder_tilt_deg", "head_roll_deg", "head_lateral",
             "ear_mid_y_norm", "shoulder_width", "face_width"]


def _avg(xs: Sequence[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default


def compute_baseline(pose_samples: Sequence, yaw: Sequence[float],
                     head_pitch: Sequence[float],
                     imu_samples: Optional[Sequence[Tuple[float, float, float]]] = None,
                     min_samples: int = 1) -> dict:
    """Build a baseline dict from collected per-frame metrics.

    ``pose_samples`` is a sequence of reliable PoseMetrics; ``yaw``/``head_pitch``
    come from FaceMetrics. Shared by the one-time CLI calibration and the
    in-pipeline post-resume recalibration so the maths stays identical.
    """
    acc = {k: [] for k in POSE_KEYS}
    for pm in pose_samples:
        for k in POSE_KEYS:
            if k == "torso_incline_deg" and not getattr(pm, "torso_valid", False):
                continue
            acc[k].append(getattr(pm, k))

    if len(acc["fha_deg"]) < max(1, min_samples):
        raise RuntimeError(
            "No reliable pose detected during calibration. Check lighting/framing "
            "and that your head and both shoulders are clearly visible."
        )

    pose_baseline = {k: _avg(acc[k], 1.0 if k.endswith("width") else 0.0)
                     for k in POSE_KEYS}
    pose_baseline["head_yaw_deg"] = _avg(yaw)
    pose_baseline["head_pitch_deg"] = _avg(head_pitch)

    baseline = {
        "version": BASELINE_VERSION,
        "pose": pose_baseline,
        "samples": len(acc["fha_deg"]),
        "created": time.time(),
    }
    if imu_samples:
        baseline["imu"] = {
            "pitch": _avg([s[0] for s in imu_samples]),
            "roll": _avg([s[1] for s in imu_samples]),
            "yaw": _avg([s[2] for s in imu_samples]),
        }
    return baseline


def run_calibration(cfg: dict, duration: float = 5.0,
                    imu_bridge=None) -> dict:
    """Capture `duration` seconds of posture and persist baseline.json."""
    cam_cfg = cfg["camera"]
    print("\n=== Posture Calibration ===")
    print("Sit in your ideal, comfortable upright posture.")
    for n in (3, 2, 1):
        print(f"  recording in {n}...", flush=True)
        time.sleep(1.0)
    print(f"Recording for {duration:.0f} seconds — hold still...\n", flush=True)

    camera = Camera(
        index=cam_cfg["index"], width=cam_cfg["width"], height=cam_cfg["height"],
        crop_top_fraction=cam_cfg["crop_top_fraction"], mirror=cam_cfg.get("mirror", True),
    ).start()
    pose = PoseEngine(cfg["models"]["pose_task"])
    face = FaceEngine(cfg, cfg["models"]["face_task"])

    pose_samples: List = []
    yaw, head_pitch = [], []
    imu_samples: List[Tuple[float, float, float]] = []

    start = time.monotonic()
    ts0 = time.time()
    try:
        while time.monotonic() - start < duration:
            frame, _ = camera.read()
            if frame is None:
                time.sleep(0.02)
                continue
            ts_ms = int((time.time() - ts0) * 1000)
            pm = pose.process(frame, ts_ms)
            fm = face.process(frame, ts_ms)
            if pm.detected and pm.reliable:
                pose_samples.append(pm)
            if fm.detected:
                yaw.append(fm.head_yaw_deg)
                head_pitch.append(fm.head_pitch_deg)
            if imu_bridge is not None:
                r = imu_bridge.read()
                if r.fresh:
                    imu_samples.append((r.pitch, r.roll, r.yaw))
            time.sleep(0.03)
    finally:
        camera.stop()
        pose.close()
        face.close()

    baseline = compute_baseline(pose_samples, yaw, head_pitch, imu_samples)
    path = save_baseline(baseline)
    p = baseline["pose"]
    print("Calibration complete. Baseline saved to", path)
    print(f"  neck_ratio={p['neck_ratio']:.2f}  fwd_head={p['forward_head']:+.3f}  "
          f"FHA={p['fha_deg']:.1f}deg  shoulder_tilt={p['shoulder_tilt_deg']:+.1f}deg  "
          f"head_roll={p['head_roll_deg']:+.1f}deg  ({baseline['samples']} samples)")
    if "imu" in baseline:
        print(f"  IMU pitch={baseline['imu']['pitch']:.1f}deg captured")
    return baseline
