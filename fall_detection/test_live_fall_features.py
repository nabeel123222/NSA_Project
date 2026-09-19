import sys
import time
import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from serial_reader import CSISerialReader
from fall_features import extract_two_receiver_fall_features


FRIEND_PORT = "/dev/ttyACM0"
YOUR_PORT = "/dev/ttyACM1"
BAUDRATE = 921600

CAPTURE_SECONDS = 4.0
EXPECTED_SUBCARRIERS = 64


def capture_stream(reader, result, stop_event):
    frames = []
    timestamps = []

    while not stop_event.is_set():
        frame = reader.read_frame()

        if frame is None:
            continue

        csi = np.asarray(
            frame.complex_csi,
            dtype=np.complex128
        )

        timestamp = float(frame.timestamp)

        if csi.shape != (EXPECTED_SUBCARRIERS,):
            continue

        if not np.isfinite(csi).all():
            continue

        if timestamps and timestamp <= timestamps[-1]:
            continue

        frames.append(csi)
        timestamps.append(timestamp)

    result["csi"] = np.asarray(
        frames,
        dtype=np.complex128
    )

    result["timestamps"] = np.asarray(
        timestamps,
        dtype=float
    )


def calculate_rate(timestamps):
    if len(timestamps) < 2:
        return 0.0

    duration = timestamps[-1] - timestamps[0]

    if duration <= 0:
        return 0.0

    return (len(timestamps) - 1) / duration


def main():

    print("=" * 60)
    print("LIVE FALL FEATURE TEST")
    print("=" * 60)

    friend_reader = CSISerialReader(
        port=FRIEND_PORT,
        baudrate=BAUDRATE
    )

    you_reader = CSISerialReader(
        port=YOUR_PORT,
        baudrate=BAUDRATE
    )

    print("\nOpening ESP32 receivers...")

    friend_reader.open()
    you_reader.open()

    time.sleep(2)

    friend_result = {}
    you_result = {}

    stop_event = threading.Event()

    friend_thread = threading.Thread(
        target=capture_stream,
        args=(friend_reader, friend_result, stop_event),
        daemon=True
    )

    you_thread = threading.Thread(
        target=capture_stream,
        args=(you_reader, you_result, stop_event),
        daemon=True
    )

    try:
        print("\nGet into the sensing zone.")
        print("Remain normally still / standing.")
        input("Press ENTER to capture 4 seconds...")

        friend_thread.start()
        you_thread.start()

        start = time.monotonic()

        time.sleep(CAPTURE_SECONDS)

        stop_event.set()

        friend_thread.join(timeout=2)
        you_thread.join(timeout=2)

        elapsed = time.monotonic() - start

        friend_csi = friend_result.get(
            "csi",
            np.empty((0, 64), dtype=np.complex128)
        )

        friend_t = friend_result.get(
            "timestamps",
            np.empty(0, dtype=float)
        )

        you_csi = you_result.get(
            "csi",
            np.empty((0, 64), dtype=np.complex128)
        )

        you_t = you_result.get(
            "timestamps",
            np.empty(0, dtype=float)
        )

        friend_rate = calculate_rate(friend_t)
        you_rate = calculate_rate(you_t)

        print("\nCapture complete.")
        print(f"Wall time: {elapsed:.3f} sec")

        print(
            f"Friend: {friend_csi.shape}, "
            f"rate={friend_rate:.2f} Hz"
        )

        print(
            f"You:    {you_csi.shape}, "
            f"rate={you_rate:.2f} Hz"
        )

        if len(friend_csi) < 30 or len(you_csi) < 30:
            print("\nERROR: Not enough CSI frames.")
            return

        # Convert complex CSI to amplitude.
        friend_amp = np.abs(friend_csi)
        you_amp = np.abs(you_csi)

        # Use the same number of frames from both receivers.
        n = min(
            len(friend_amp),
            len(you_amp)
        )

        friend_amp = friend_amp[:n]
        you_amp = you_amp[:n]

        print(
            f"\nUsing {n} synchronized-by-length frames "
            "for feature test."
        )

        features = extract_two_receiver_fall_features(
            friend_amp,
            you_amp,
            friend_rate,
            you_rate
        )

        names = [
            "friend_peak_motion_energy",
            "friend_motion_jerk",
            "friend_post_event_stillness",
            "you_peak_motion_energy",
            "you_motion_jerk",
            "you_post_event_stillness",
        ]

        print("\nFall-specific features:")
        print("-" * 50)

        for name, value in zip(names, features):
            print(f"{name:35s}: {value:.6f}")

        print("-" * 50)
        print("Feature count:", len(features))
        print("Finite:", np.isfinite(features).all())

    finally:
        stop_event.set()

        friend_reader.close()
        you_reader.close()

        print("\nESP32 receivers closed.")


if __name__ == "__main__":
    main()
