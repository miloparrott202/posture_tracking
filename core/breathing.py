"""Best-effort breathing estimation from shoulder motion (experimental).

The webcam sits above the monitor and sees the head, shoulders and upper chest
— **not** the abdomen. So this module can positively observe *thoracic / chest*
breathing (the shoulder line rising on each inhale) and read off a breathing
rate, but it cannot directly confirm diaphragmatic ("belly") breathing; it only
infers it from the *absence* of shoulder motion. Treat ``pattern`` accordingly:

    "chest"   -> clear shoulder/clavicular breathing detected
    "relaxed" -> a breathing rhythm is present but shoulders are quiet
    "unknown" -> not enough clean signal to decide (too few frames, low fps,
                 or no dominant periodic component)

Method: buffer the shoulder-midpoint Y over a rolling window, resample to a
uniform grid, linearly detrend (removes slow posture drift), Hann-window, and
take an FFT. The dominant peak inside the plausible breathing band gives the
rate; the RMS oscillation amplitude (normalised to face width, so it is
distance/resolution independent) gives the chest-vs-relaxed call. A spectral
concentration check rejects noise that isn't actually periodic.

Reliable only at the higher full-mode frame rate; at 2-4 fps background rate it
stays ``unknown`` by design.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class BreathingMetrics:
    valid: bool = False
    rate_bpm: float = 0.0          # breaths per minute (only when valid)
    amplitude_norm: float = 0.0    # RMS shoulder oscillation / face width
    pattern: str = "unknown"       # chest | relaxed | unknown


class BreathingEstimator:
    """Estimates breathing rate + chest-breathing amplitude from pose frames."""

    def __init__(self, cfg: dict):
        bc = cfg["breathing"]
        self.window_sec = float(bc["window_sec"])
        self.min_fps = float(bc["min_fps"])
        self.fmin = float(bc["rate_min_bpm"]) / 60.0
        self.fmax = float(bc["rate_max_bpm"]) / 60.0
        self.chest_threshold = float(bc["chest_amp_threshold"])
        self.min_concentration = float(bc.get("min_concentration", 0.25))
        # (timestamp, shoulder_mid_y, face_width) samples within the window.
        self._samples: Deque[Tuple[float, float, float]] = deque()

    def update(self, pose_metrics, now: float) -> BreathingMetrics:
        """Feed one frame's pose and return the current breathing estimate."""
        if pose_metrics is not None and pose_metrics.detected:
            sh = pose_metrics.landmarks.get("shoulder_mid")
            if sh is not None:
                self._samples.append((now, float(sh[1]), float(pose_metrics.face_width)))
        cutoff = now - self.window_sec
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return self._analyze()

    # ------------------------------------------------------------------
    def _analyze(self) -> BreathingMetrics:
        n = len(self._samples)
        if n < 16:
            return BreathingMetrics()

        t = np.fromiter((s[0] for s in self._samples), dtype=float, count=n)
        y = np.fromiter((s[1] for s in self._samples), dtype=float, count=n)
        fw = np.fromiter((s[2] for s in self._samples), dtype=float, count=n)

        span = float(t[-1] - t[0])
        if span < self.window_sec * 0.5:          # need a decent chunk of history
            return BreathingMetrics()
        if (n - 1) / span < self.min_fps:         # too slow to resolve breathing
            return BreathingMetrics()

        # Uniform resample so the FFT bins are meaningful.
        fs = max(8.0, self.fmax * 8.0)
        m = int(span * fs)
        if m < 16:
            return BreathingMetrics()
        tu = np.linspace(t[0], t[-1], m)
        yu = np.interp(tu, t, y)

        # Linear detrend removes DC + slow posture drift over the window.
        x = tu - tu[0]
        yu = yu - np.polyval(np.polyfit(x, yu, 1), x)

        yw = yu * np.hanning(m)
        spec = np.abs(np.fft.rfft(yw)) ** 2
        freqs = np.fft.rfftfreq(m, d=1.0 / fs)

        band = (freqs >= self.fmin) & (freqs <= self.fmax)
        total = float(spec[1:].sum())             # exclude DC bin
        if not band.any() or total <= 0.0:
            return BreathingMetrics()

        band_idx = np.where(band)[0]
        peak = int(band_idx[int(np.argmax(spec[band_idx]))])
        rate_bpm = float(freqs[peak] * 60.0)
        concentration = float(spec[band].sum() / total)

        face_w = float(np.median(fw)) or 1.0
        amp_norm = float(np.sqrt(np.mean(yu ** 2)) / max(face_w, 1e-3))

        if concentration < self.min_concentration or amp_norm <= 1e-4:
            return BreathingMetrics(valid=False, amplitude_norm=amp_norm,
                                    pattern="unknown")

        pattern = "chest" if amp_norm >= self.chest_threshold else "relaxed"
        return BreathingMetrics(valid=True, rate_bpm=round(rate_bpm, 1),
                                amplitude_norm=amp_norm, pattern=pattern)

    def reset(self) -> None:
        self._samples.clear()
