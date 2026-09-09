"""
serial_reader.py
-----------------
Responsible for ONE thing: talking to the ESP32-S3 over /dev/ttyACM0 and
turning its raw CSI serial output into clean numpy arrays.

This module does NOT do any filtering, feature extraction, or decision
making. Keeping it dumb-and-simple means the ESP32 firmware and the wire
format can stay exactly as-is (Espressif's official csi_recv_router example),
and everything downstream (signal_processing, human_detector) only ever
deals with numpy arrays, never raw strings.

Espressif's ESP-CSI example prints CSV lines like:

    CSI_DATA,<mac>,<rssi>,<rate>,...,<len>,[raw csi int8 array]

The raw CSI array is a flat list of interleaved (Imaginary, Real) int8 pairs
per subcarrier: [I0, Q0, I1, Q1, I2, Q2, ...]

We parse that into a complex-valued numpy array of shape (num_subcarriers,).
"""

from __future__ import annotations

import re
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np
import serial

logger = logging.getLogger(__name__)

# The bracketed raw CSI array in the CSV line, e.g. "[1,2,-3,4,...]"
_CSI_ARRAY_RE = re.compile(r"\[(.*?)\]")


@dataclass
class CSIFrame:
    """One parsed CSI frame."""
    rssi: int
    rate: int
    raw_iq: np.ndarray          # shape (2*num_subcarriers,), dtype int16
    complex_csi: np.ndarray     # shape (num_subcarriers,), dtype complex64
    timestamp: float            # host-side receive time (perf_counter)


class CSISerialReader:
    """
    Wraps a pyserial connection to the ESP32 and exposes:
      - read_frame(): blocking read of exactly one parsed CSIFrame
      - read_window(n): blocking read of exactly n CSIFrame objects
      - stream_windows(window_size, stride): generator yielding sliding
        windows of frames, e.g. windows of 50 frames advancing by 10
        frames each time (so consecutive windows overlap -> smoother,
        more responsive detection than non-overlapping windows).
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 921600,
        timeout: float = 2.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None

    # ------------------------------------------------------------------ #
    # connection lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        logger.info("Opened serial port %s @ %d baud", self.port, self.baudrate)

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            logger.info("Closed serial port %s", self.port)

    def __enter__(self) -> "CSISerialReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_line(line: str) -> Optional[CSIFrame]:
        """
        Parse one CSI_DATA CSV line from the ESP-CSI firmware into a
        CSIFrame. Returns None if the line isn't a valid CSI_DATA line
        (garbage, partial writes, boot logs, etc. are common on serial
        and must be silently skipped rather than crashing the pipeline).
        """
        if not line.startswith("CSI_DATA"):
            return None

        fields = line.split(",")
        # Expected columns per Espressif's csi_recv_router example:
        # CSI_DATA, mac, rssi, rate, sig_mode, mcs, bandwidth, ...,
        # len, "[raw csi array]"
        if len(fields) < 8:
            return None

        try:
            rssi = int(fields[3])
            rate = int(fields[4])
        except (ValueError, IndexError):
            return None

        match = _CSI_ARRAY_RE.search(line)
        if not match:
            return None

        raw_str = match.group(1).strip()
        if not raw_str:
            return None

        try:
            raw_iq = np.array(
                [int(x) for x in raw_str.split(",") if x.strip() != ""],
                dtype=np.int16,
            )
        except ValueError:
            return None

        if raw_iq.size < 2 or raw_iq.size % 2 != 0:
            # malformed / truncated packet
            return None

        # Interleaved [I0,Q0,I1,Q1,...] -> complex array
        imag = raw_iq[0::2].astype(np.float32)
        real = raw_iq[1::2].astype(np.float32)
        complex_csi = real + 1j * imag

        import time
        return CSIFrame(
            rssi=rssi,
            rate=rate,
            raw_iq=raw_iq,
            complex_csi=complex_csi,
            timestamp=time.perf_counter(),
        )

    # ------------------------------------------------------------------ #
    # public read API
    # ------------------------------------------------------------------ #
    def read_frame(self, max_attempts: int = 2000) -> CSIFrame:
        """
        Block until exactly one valid CSIFrame is parsed.
        Skips non-CSI lines (logs, corrupted lines) automatically.
        Raises TimeoutError if nothing valid arrives within max_attempts
        lines (protects against a silently dead serial connection).
        """
        if self._ser is None:
            raise RuntimeError("Serial port not open. Call open() first.")

        for _ in range(max_attempts):
            raw_line = self._ser.readline()
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8", errors="ignore").strip()
            except UnicodeDecodeError:
                continue

            frame = self._parse_line(line)
            if frame is not None:
                return frame

        raise TimeoutError(
            f"No valid CSI frame received after {max_attempts} lines "
            f"on {self.port}. Check ESP32 is running csi_recv_router "
            f"and connected to the same 2.4GHz network."
        )

    def read_window(self, n: int) -> List[CSIFrame]:
        """Block until n valid CSIFrame objects are collected."""
        return [self.read_frame() for _ in range(n)]

    def stream_windows(self, window_size: int = 50, stride: int = 10):
        """
        Generator that yields overlapping sliding windows of CSIFrame
        lists, each of length `window_size`, advancing by `stride`
        frames each time.

        Overlapping windows (stride < window_size) matter for a live
        monitor: a non-overlapping window (stride == window_size) only
        produces a new decision every window_size frames, which feels
        laggy. Overlap gives you a fresh decision every `stride` frames
        while each decision still sees `window_size` frames of context.

        Example: window_size=50, stride=10 -> a new decision roughly
        every 10 frames, each based on the most recent 50 frames.
        """
        if stride > window_size:
            raise ValueError("stride cannot be larger than window_size")

        buffer: Deque[CSIFrame] = deque(maxlen=window_size)

        # Fill the initial window
        while len(buffer) < window_size:
            buffer.append(self.read_frame())
        yield list(buffer)

        while True:
            # Advance by `stride` new frames, buffer auto-evicts old ones
            for _ in range(stride):
                buffer.append(self.read_frame())
            yield list(buffer)


def frames_to_amplitude_matrix(frames: List[CSIFrame]) -> np.ndarray:
    """
    Convert a list of CSIFrame into a 2D amplitude matrix of shape
    (num_frames, num_subcarriers). This is the common hand-off point
    into signal_processing.py.

    Frames with a differing subcarrier count (can happen transiently if
    bandwidth mode changes) are truncated to the minimum common length
    so the matrix stays rectangular.
    """
    if not frames:
        return np.empty((0, 0))

    min_len = min(f.complex_csi.size for f in frames)
    amp_matrix = np.array(
        [np.abs(f.complex_csi[:min_len]) for f in frames],
        dtype=np.float32,
    )
    return amp_matrix