# Posture & Eye-Strain Tracker

A lightweight, always-on desktop app that watches your **posture**, **blink
rate**, and **screen attention** through your webcam (optionally fused with an
IMU), gives **real-time correction guidance** in a full-screen view, and sends
**OS notifications** while running quietly in the system tray.

Built with MediaPipe Tasks (CPU-only, low idle RAM), OpenCV, and PyQt6.

---

## Quick start

One command does everything:

```bash
./run.sh
```

On the **first run** this creates a virtualenv (`.venv`), installs all
dependencies, downloads the MediaPipe models, and walks you through a 5-second
posture calibration. **Every run after that** just activates the env and opens
the app. Pick a mode by passing the flag straight through:

```bash
./run.sh               # full-screen correction GUI + tray (default)
./run.sh --console     # headless: live metrics streamed to the terminal
./run.sh --background  # standalone tray icon + notifications only
./run.sh --calibrate   # re-record your ideal-posture baseline
./run.sh --screenshot  # also dump the rendered frame to screenshots/ (~1/s)
```

The default GUI **minimises to a system-tray / menu-bar icon** instead of
quitting: closing the window keeps it monitoring in the background at the
reduced frame rate, and the tray menu ("Open full view", "Pause", "Recalibrate",
"Quit") brings it back — all in one process sharing one camera.

<details>
<summary>Manual setup (without <code>run.sh</code>)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --calibrate   # one-time baseline
python main.py               # or --console / --background
```
</details>

> Model bundles (`pose_landmarker_lite.task`, `face_landmarker.task`) are
> **downloaded automatically** into `models/` on first run.
> Optional IMU / native-macOS-menu-bar extras:
> `pip install -r requirements-optional.txt`.

---

## Run modes

| Command | Mode | What it does |
|---|---|---|
| `python main.py --calibrate` | Calibration | Records baseline FHA / shoulder tilt / ear-Y (+ IMU if connected) → `baseline.json` |
| `python main.py --console` | Console | Streams posture score, deviations, blink rate, gaze, break timer to the terminal. No GUI deps needed beyond core. Great for prototyping (Section 10, step 1). |
| `python main.py --background` | Background | Standalone (GUI-free) tray/menu-bar icon, reduced 2–4 fps, OS notifications only (Section 7). Use for launch-at-login. |
| `python main.py` / `--full` | Full GUI | Live video + skeleton overlay + correction arrows/text + sidebar (Section 6), **plus a built-in tray** — closing the window keeps it monitoring in the background and the tray reopens it (Section 7). |

Common flags: `--config path/to/config.yaml`, `-v/--verbose`.

---

## How it works

```
webcam ─► capture.py (shared frame buffer, mirrored, 640x480)
            │
            ├─► pose_engine.py   (MediaPipe Pose → slouch/crane/asymmetry features)
            ├─► face_engine.py   (Face Mesh → EAR/blink, gaze, head pose)
            ├─► imu_bridge.py    (optional: serial/BLE pitch/roll/yaw → fusion)
            ▼
        posture_score.py  (deviation vs. baseline → 0–100 score + cues)
            ▼
        state_manager.py  (screen-time/break timers, blink sustain, debounce)
            ▼
   ┌────────┴─────────┐
 full_view.py     tray.py + notifier.py
 (PyQt6 GUI)      (background notifications)
```

Everything is orchestrated by `core/pipeline.py` (`Processor`), which every run
mode shares. See the build spec for the full design.

---

## Configuration

All thresholds, weights, frame rates, and the IMU setup live in
[`config.yaml`](config.yaml). Values are deep-merged over the defaults in
`core/config.py`, so you only specify what you want to change.

Key knobs:

- `posture.weights` — scoring weights `w1/w2/w3` (FHA, shoulder tilt, yaw).
- `posture.thresholds` — good (≥85) / drifting (≥60) / poor bands.
- `framerate.background_fps` / `full_fps` — resource vs. responsiveness.
- `breaks.screen_interval_sec` — the 20-20-20 break interval.
- `notifications.rate_limit_sec` — anti-fatigue per-type cooldowns.

---

## Optional IMU (ESP32 / Arduino)

Enable in `config.yaml`:

```yaml
imu:
  enabled: true
  connection: "serial"      # or "ble"
  port: "/dev/cu.usbmodem14201"
  baud: 115200
  fusion_weight: 0.6
```

Your firmware should stream JSON (or CSV) at ~20–50 Hz:

```json
{"pitch": -12.4, "roll": 2.1, "yaw": 5.6, "ts": 1234567890}
```

IMU pitch is fused with the camera FHA estimate. If the link drops, the app
logs a warning and falls back to camera-only automatically. Re-run
`--calibrate` while the IMU is connected to capture its baseline too.

---

## Posture model

The tracker reads 11 body points (nose, both eyes, ears, mouth corners,
shoulders, and hips when visible) and derives features grouped by what they
mean, scored against your calibrated upright baseline. **Slouching and head
craning are weighted most heavily**; left/right asymmetry counts but for less:

- **Slouch / hunch** — `neck_ratio` (vertical ear→shoulder gap over shoulder
  width — collapses when you hunch), head drop, and torso incline (when hips
  are in frame).
- **Head craning** — forward-head angle plus a z-depth term (ears moving ahead
  of the shoulders, which a flat 2D angle can't see).
- **Asymmetry** — shoulder tilt, head roll (cocked sideways), and lateral head
  offset.

The sidebar collapses these into a **Head / Back / Shoulders** readout, and the
on-video skeleton turns the worst group red with an arrow pointing the way to
correct. Every feature uses a deadband so normal micro-movement scores 100.

Run `QT_QPA_PLATFORM=offscreen python tools/gui_preview.py` to render the GUI in
each state to `screenshots/` without a camera.

---

## Quality-of-life features

- **EMA smoothing** (`posture.smoothing`, default 0.6) — temporally smooths the
  metrics fed to scoring so the score and readouts don't jitter frame-to-frame.
- **Background notifications** — while minimised to the menu-bar/tray the app
  keeps watching and sends silent OS banners for slouching, low blink rate, and
  screen breaks. **Reminder toggles** (master + per-type: posture / low blink /
  screen breaks) live in the window sidebar and the tray menu, backed by
  `reminders:` in config.yaml.
- **Non-distracting alerts** — banners are silent by default; set
  `notifications.sound: true` to also play a short sound for posture warnings.
- **Sleep/wake recovery** — when you close and reopen your laptop the camera
  handle goes stale; the app detects the resume (and any long freeze), re-opens
  the camera, rebuilds the MediaPipe engines, and **auto-records a fresh
  baseline** (notified: "Recording a new baseline…"), then resumes — instead of
  freezing or crashing. Recalibration runs *in the live pipeline*, so the video
  feed never freezes while it records. Tunable under `recovery:` in config.yaml.
- **Show window while recalibrating** — an opt-in sidebar/tray toggle (default
  **off**): when on, the window un-minimises during a post-wake recalibration so
  you can see it re-recording, then re-minimises when done.
- **Toggles persist between sessions** — reminder switches, pause state, and the
  recalibration-visibility toggle are saved to `settings.json` and restored on
  next launch.
- **Cross-platform camera + first-use hardware check** — the camera layer tries
  OS-appropriate backends (DirectShow/MSMF on Windows, AVFoundation on macOS,
  V4L2 on Linux) and falls back across device indices, so it opens on varied
  machines and webcams. On the **first launch** it measures CPU throughput and
  camera resolution and, if the hardware looks insufficient (slow fps, few CPU
  cores, low-res or missing camera), pops a one-time notification (and a dialog
  in the GUI). Re-run anytime with `python main.py --diagnostics`; tune
  thresholds under `diagnostics:` in config.yaml.
- **Session posture stats** — the sidebar, health log, and `watch.py` summary
  report the share of time spent in good / drifting / poor posture this session.
- **`--screenshot`** — writes the rendered frame (video + skeleton + cues) to
  `screenshots/` about once a second, keeping the newest 10 and wiping the
  folder at the start of each run. Handy for sharing what the app is displaying
  without a live screen (e.g. pasting a frame for review).

---

## Breathing (experimental)

The camera sits above the monitor and sees your **shoulders and upper chest**,
not your belly. So the tracker can positively detect **chest / shoulder
("thoracic") breathing** — the shoulder line rising on each inhale — and read
off a breathing rate, but it can only *infer* relaxed diaphragmatic breathing
from the absence of shoulder motion. It will never claim to "see" your belly.

Two things ship here:

1. **A periodic reminder** (reliable, every mode) — every
   `breathing.reminder_interval_sec` (default 30 min) it nudges you to *"take a
   few slow breaths into your belly — relax your shoulders."*
2. **An experimental detector** (full mode only — it needs the higher frame
   rate) — buffers shoulder-Y over a 30 s window, runs an FFT to estimate
   breathing rate and shoulder-oscillation amplitude, and classifies
   `chest` / `relaxed` / `unknown`. When `chest` breathing persists for
   `chest_sustain_sec`, the reminder escalates to *"you're chest-breathing —
   drop your shoulders."* The sidebar shows `Breathing: 15/min, chest`.

Tuning lives under `breathing:` in [`config.yaml`](config.yaml).
`chest_amp_threshold` is the main knob — raise it if relaxed breathing is being
flagged as chest, lower it if real shoulder breathing is missed. Set
`breathing.enabled: false` to turn the whole feature off.

> Accuracy caveat: the signal is sub-pixel and easily disturbed by typing,
> talking, laughing, or shifting in your seat, so treat the live readout as a
> hint, not a clinical measurement. The detector stays `unknown` in background
> mode (2–4 fps is too slow to resolve breathing).

---

## Health logging

When `health_log.enabled` is true, a JSON-lines summary is appended every
`interval_sec` (default 5 min) to `posture_health_log.jsonl`:

```json
{"ts": "2026-06-15T14:30:00+00:00", "avg_posture_score": 78, "blink_rate": 14, "breathing_rate": 15, "screen_time_min": 22, "breaks_taken": 1}
```

Ready to be ingested by an external personal-health pipeline.

---

## Resource budget

Targets **<150 MB resident** and **<5% average CPU** in background mode via:
lite models, 320×240 inference, 2–4 fps when minimised, a single shared frame
buffer, released GUI buffers in background mode, and lazy IMU loading.

---

## Packaging (macOS)

```bash
pip install pyinstaller
pyinstaller --windowed --name "PostureTracker" \
  --add-data "models:models" --add-data "config.yaml:." main.py
```

For launch-at-login, add the built `.app` to *System Settings → General →
Login Items*, or use a `LaunchAgent` plist.

---

## Project layout

```
posture_tracking/
├── run.sh                  # one-command launcher (venv + deps + calibrate + run)
├── main.py                 # entry point, mode dispatch
├── config.yaml             # user settings / thresholds / IMU
├── baseline.json           # written by --calibrate
├── requirements.txt        # core deps
├── requirements-optional.txt  # IMU (pyserial/bleak) + macOS rumps
├── core/
│   ├── config.py           # defaults + YAML merge, baseline IO
│   ├── models.py           # auto-download .task bundles
│   ├── capture.py          # threaded camera, shared frame buffer
│   ├── geometry.py         # angle/euler helpers
│   ├── pose_engine.py      # MediaPipe Pose → posture metrics
│   ├── face_engine.py      # Face Mesh → EAR/blink/gaze/head pose
│   ├── gaze.py             # looking-at-screen classifier
│   ├── posture_score.py    # deviation math + scoring formula
│   ├── breathing.py        # experimental breathing-rate / chest-breathing FFT
│   ├── state_manager.py    # timers, debounce, event emission
│   ├── calibration.py      # baseline recording routine
│   ├── imu_bridge.py       # optional serial/BLE reader + fusion
│   ├── health_log.py       # periodic JSONL summaries
│   └── pipeline.py         # Processor: ties it all together
├── ui/
│   ├── overlay.py          # OpenCV skeleton/arrow drawing
│   ├── full_view.py        # PyQt6 main window
│   └── tray.py             # pystray icon + menu
└── notifications/
    └── notifier.py         # plyer / osascript wrapper
```
