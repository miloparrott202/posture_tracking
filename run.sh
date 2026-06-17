#!/usr/bin/env bash
# =============================================================================
# Posture & Eye-Strain Tracker — one-command launcher.
#
#   ./run.sh                # set up (first run) then open the full GUI
#   ./run.sh --console      # headless metrics stream
#   ./run.sh --background    # standalone tray icon + notifications only
#   ./run.sh --calibrate     # (re)record your ideal-posture baseline
#
# First run:  creates a virtualenv (.venv) and installs all dependencies.
# Every run:  activates the virtualenv and launches the tracker.
# First run also downloads the MediaPipe models and records a posture baseline.
# Any extra arguments are passed straight through to main.py.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
STAMP="$VENV/.requirements.sha"

# --- FIRST RUN: create the virtualenv and install dependencies ---------------
if [ ! -d "$VENV" ]; then
  PY_BOOT=""
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY_BOOT="$cand"; break; fi
  done
  if [ -z "$PY_BOOT" ]; then
    echo "error: no python3 interpreter found on PATH." >&2
    exit 1
  fi
  echo "[run] first run: creating virtualenv with $PY_BOOT ($("$PY_BOOT" --version 2>&1)) ..."
  "$PY_BOOT" -m venv "$VENV"
fi

# --- EVERY RUN: activate the virtualenv --------------------------------------
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- install deps on first run, or refresh them if requirements.txt changed --
req_hash="$(shasum -a 256 requirements.txt | awk '{print $1}')"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$req_hash" ]; then
  echo "[run] installing dependencies ..."
  # --no-compile: pip byte-compiles every .py it installs by default, but the
  #   mediapipe wheel bundles test files with non-ASCII chars that fail to
  #   compile on some Python builds and crash pip's error printer. We never
  #   import those test files, so skipping byte-compilation is safe.
  # PYTHONIOENCODING: guarantees pip's output stream has an encoding even when
  #   stdout isn't a TTY (the other half of that same pip crash).
  PYTHONIOENCODING=utf-8 python -m pip install --upgrade pip >/dev/null
  PYTHONIOENCODING=utf-8 python -m pip install --no-compile -r requirements.txt
  echo "$req_hash" > "$STAMP"
fi

# --- pre-fetch the MediaPipe model bundles so first launch isn't blocked ------
python - <<'PYEOF'
from core.models import ensure_model
for m in ("pose_landmarker_lite.task", "face_landmarker.task"):
    ensure_model(m)
PYEOF

# --- first-run calibration: no baseline yet and not already a calibrate run ---
want_calibrate=1
for a in "$@"; do
  case "$a" in
    --calibrate|--help|-h) want_calibrate=0 ;;
  esac
done
if [ ! -f "baseline.json" ] && [ "$want_calibrate" -eq 1 ]; then
  echo "[run] no baseline found — running one-time calibration first."
  python main.py --calibrate
fi

# --- launch (default: full GUI). Extra args pass straight through. ------------
echo "[run] launching posture tracker ..."
exec python main.py "$@"
