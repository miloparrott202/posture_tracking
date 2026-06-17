"""Optional IMU bridge: serial/BLE reader + camera fusion (Section 8).

Lazy-loaded only when ``imu.enabled`` is true. Runs in its own thread with
non-blocking reads; if the link drops it falls back to camera-only and logs a
warning rather than crashing.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("imu")


@dataclass
class IMUReading:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    ts: float = 0.0
    fresh: bool = False   # whether we've received data recently


class IMUBridge:
    """Reads pitch/roll/yaw from a microcontroller over serial or BLE."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["imu"]
        self.fusion_weight = float(self.cfg.get("fusion_weight", 0.6))
        self._latest = IMUReading()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stale_after = 1.0  # seconds without data => not fresh

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "IMUBridge":
        if self._thread is not None:
            return self
        self._stop.clear()
        target = self._serial_loop if self.cfg.get("connection") == "serial" else self._ble_loop
        self._thread = threading.Thread(target=self._guarded(target), name="imu", daemon=True)
        self._thread.start()
        return self

    def _guarded(self, target):
        def runner():
            try:
                target()
            except Exception as exc:  # never crash the app over the IMU
                log.warning("IMU bridge stopped: %s (falling back to camera-only)", exc)
        return runner

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- reads -------------------------------------------------------------
    def read(self) -> IMUReading:
        with self._lock:
            r = self._latest
            fresh = r.fresh and (time.monotonic() - r.ts) < self._stale_after
            return IMUReading(r.pitch, r.roll, r.yaw, r.ts, fresh)

    def _store(self, pitch: float, roll: float, yaw: float) -> None:
        with self._lock:
            self._latest = IMUReading(pitch, roll, yaw, time.monotonic(), True)

    def _parse(self, line: str) -> Optional[tuple]:
        line = line.strip()
        if not line:
            return None
        try:
            if self.cfg.get("format", "json") == "json":
                d = json.loads(line)
                return float(d["pitch"]), float(d["roll"]), float(d["yaw"])
            parts = line.split(",")
            return float(parts[0]), float(parts[1]), float(parts[2])
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            return None

    # -- serial transport --------------------------------------------------
    def _serial_loop(self) -> None:
        import serial  # lazy: pyserial only needed when IMU enabled

        ser = serial.Serial(
            self.cfg["port"], self.cfg.get("baud", 115200), timeout=0.1
        )
        log.info("IMU serial connected on %s", self.cfg["port"])
        try:
            while not self._stop.is_set():
                line = ser.readline().decode("utf-8", errors="ignore")
                parsed = self._parse(line)
                if parsed:
                    self._store(*parsed)
        finally:
            ser.close()

    # -- BLE transport -----------------------------------------------------
    def _ble_loop(self) -> None:
        import asyncio

        from bleak import BleakClient  # lazy: bleak only when BLE configured

        address = self.cfg["ble_address"]
        char = self.cfg["ble_char_uuid"]

        async def run():
            def handler(_, data: bytearray):
                parsed = self._parse(data.decode("utf-8", errors="ignore"))
                if parsed:
                    self._store(*parsed)

            async with BleakClient(address) as client:
                log.info("IMU BLE connected to %s", address)
                await client.start_notify(char, handler)
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)

        asyncio.run(run())

    # -- fusion ------------------------------------------------------------
    def fused_fha_dev(self, camera_fha_dev: float, baseline: Optional[dict]) -> Optional[float]:
        """Fuse IMU pitch deviation with the camera FHA deviation (Section 8.3).

        Returns None when no fresh IMU data is available so the caller keeps
        using the camera-only deviation.
        """
        r = self.read()
        if not r.fresh or baseline is None:
            return None
        imu_base = baseline.get("imu") or {}
        imu_pitch_dev = r.pitch - imu_base.get("pitch", 0.0)
        w = self.fusion_weight
        return w * imu_pitch_dev + (1.0 - w) * camera_fha_dev
