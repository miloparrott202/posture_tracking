"""Ensure MediaPipe Tasks model bundles are present locally.

The Tasks API needs ``.task`` bundles on disk. On first run we download the
lite/standard variants into ``models/`` so the app is self-bootstrapping.
"""
from __future__ import annotations

import os
import sys
import urllib.request

from .config import PROJECT_ROOT

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Official MediaPipe model URLs. Lite pose variant keeps idle RAM low.
MODEL_URLS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}


def ensure_model(filename: str) -> str:
    """Return an absolute path to the model, downloading it if necessary."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    url = MODEL_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(
            f"Model '{filename}' not found in {MODELS_DIR} and no download URL "
            f"is known. Place the .task bundle there manually."
        )

    print(f"[models] Downloading {filename} ...", file=sys.stderr)
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"[models] Saved -> {dest}", file=sys.stderr)
    return dest
