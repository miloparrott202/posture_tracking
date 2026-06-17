"""Configuration + baseline loading.

Built-in defaults live in ``DEFAULT_CONFIG`` and are deep-merged with the
user's ``config.yaml`` so the file only needs to specify overrides.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency
    yaml = None


# Mirror of config.yaml; the YAML file overrides these. Keeping defaults in
# code means the app still runs if a key is missing from the user's file.
DEFAULT_CONFIG: Dict[str, Any] = {
    # Capture/display/inference resolution. Bumped from 320x240 for a crisp
    # overlay; MediaPipe still rescales to its own model input, and there is
    # ample CPU headroom. Drop back to 320x240 on very weak machines.
    "camera": {"index": 0, "width": 640, "height": 480,
               "crop_top_fraction": 1.0, "mirror": True},
    "framerate": {"background_fps": 3, "full_fps": 30},
    "models": {
        "pose_task": "pose_landmarker_lite.task",
        "face_task": "face_landmarker.task",
    },
    "posture": {
        # Penalty points per unit of deviation beyond tolerance. Slouch + crane
        # features carry the most weight; asymmetry counts but for less.
        "weights": {
            # z-depth (forward_head) is noisier, so it supports rather than
            # drives; reliable 2D signals (fha, neck_ratio, head_drop) carry it.
            "fha": 1.6, "forward_head": 120.0,           # craning + head jutting forward
            # slouch: spinal collapse (neck_ratio) carries the most weight; head
            # tilt/drop matter less so simply glancing at the bottom of the
            # monitor doesn't read as a slump.
            "neck_ratio": 230.0, "torso_incline": 2.0, "head_drop": 180.0,
            "head_pitch": 1.3,
            "shoulder_tilt": 1.3, "head_roll": 1.1, "head_lateral": 45.0,   # asymmetry
        },
        "thresholds": {"good": 85, "drifting": 60},
        # Deadband per feature: deviation within tolerance is free (no jitter).
        "tolerances": {
            "fha_deg": 5.0,
            "forward_head": 0.04,
            "neck_ratio": 0.04,
            "torso_incline_deg": 6.0,
            "head_drop": 0.035,
            "head_pitch_deg": 9.0,
            "shoulder_tilt_deg": 4.0,
            "head_roll_deg": 5.0,
            "head_lateral": 0.05,
        },
        # EMA smoothing of metrics fed to scoring: fraction of previous kept each
        # frame. 0 = off; higher = smoother but laggier. Tames landmark jitter.
        "smoothing": 0.6,
    },
    "blink": {
        "ear_threshold": 0.20,
        "consec_frames": 2,
        "window_seconds": 60,
        "low_rate_threshold": 10,
        "low_rate_sustain_sec": 300,
    },
    "gaze": {
        "yaw_threshold_deg": 25,
        "pitch_threshold_deg": 20,
        "iris_extreme_ratio": 0.30,
    },
    "breaks": {
        "screen_interval_sec": 1200,
        "look_away_reset_sec": 20,
        "away_pause_sec": 30,
    },
    "breathing": {
        "enabled": True,
        "reminder_interval_sec": 1800,   # periodic diaphragmatic-breathing nudge
        "window_sec": 30,                # rolling analysis window
        "min_fps": 5,                    # below this sample rate -> no detection
        "rate_min_bpm": 6,
        "rate_max_bpm": 30,
        "chest_amp_threshold": 0.02,     # norm. shoulder RMS above this = chest breathing
        "min_concentration": 0.25,       # spectral concentration to trust a rhythm
        "chest_sustain_sec": 25,         # chest breathing this long -> escalate reminder
    },
    "notifications": {
        "rate_limit_sec": {"break": 1200, "posture": 300, "blink": 600,
                           "breathing_chest": 600},
        "posture_sustain_sec": 15,
        "backend": "auto",
        "sound": False,           # keep alerts non-distracting: silent banners only
    },
    # Master + per-type reminder toggles (also exposed in the UI / tray).
    "reminders": {
        "enabled": True,
        "posture": True,          # "sit up / ease your head back" nudges
        "blink": True,            # low blink-rate (eye strain) nudges
        "break": True,            # 20-20-20 screen-break nudges
    },
    "health_log": {
        "enabled": True,
        "path": "posture_health_log.jsonl",
        "interval_sec": 300,
    },
    # First-use hardware/performance check (warns the user once if the
    # machine looks too weak or the camera too low-quality).
    "diagnostics": {
        "enabled": True,
        "warmup_sec": 5,             # how long to measure throughput before judging
        "min_fps": 10,               # achieved processing fps below this -> warn
        "min_cpu_cores": 2,
        "min_camera_width": 320,     # native capture below this -> quality warning
        "min_camera_height": 240,
        "min_frames_for_perf": 15,   # need this many frames before trusting fps
    },
    # Crash/freeze recovery after sleep/wake (Issue 1/2).
    "recovery": {
        # Wall-clock gap between processed frames that means the machine slept.
        "sleep_resume_gap_sec": 20,
        # Consecutive failed camera reads before the device is re-opened.
        "camera_fail_threshold": 30,
        # Re-record the baseline automatically after a camera restart/resume.
        "recalibrate_on_resume": True,
        # Length of the automatic post-resume recalibration.
        "recalibration_duration_sec": 5,
        # Consecutive inference exceptions before the engines are rebuilt.
        "inference_error_threshold": 10,
    },
    "imu": {
        "enabled": False,
        "connection": "serial",
        "port": "/dev/cu.usbmodem14201",
        "baud": 115200,
        "ble_address": "",
        "ble_char_uuid": "",
        "fusion_weight": 0.6,
        "format": "json",
    },
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline.json")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load config.yaml merged over defaults. Missing file -> pure defaults."""
    path = path or DEFAULT_CONFIG_PATH
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    user_cfg: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


BASELINE_VERSION = 3  # bump when the pose feature set changes


def load_baseline(path: str | None = None) -> Dict[str, Any] | None:
    """Load baseline.json, or None if calibration has not been run yet."""
    path = path or DEFAULT_BASELINE_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def baseline_is_current(baseline: Dict[str, Any] | None) -> bool:
    """True if the baseline was recorded with the current feature set."""
    return bool(baseline) and baseline.get("version") == BASELINE_VERSION \
        and "neck_ratio" in baseline.get("pose", {})


def save_baseline(data: Dict[str, Any], path: str | None = None) -> str:
    path = path or DEFAULT_BASELINE_PATH
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path
