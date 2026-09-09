import threading
import time
import numpy as np

from serial_reader import CSISerialReader


PORTS = {
    "FRIEND": "/dev/ttyACM0",
    "YOU": "/dev/ttyACM1",
}

BAUDRATE = 921600
WINDOW_SIZE = 50
EXPECTED_SUBCARRIERS = 64

results = {
    "FRIEND": [],
    "YOU": [],
}

errors = {
    "FRIEND": None,
    "YOU": None,
}


def capture_window(name, port):
    try:
        print(f"[{name}] Opening {port}...")

        reader = CSISerialReader(
            port=port,
            baudrate=BAUDRATE,
            timeout=2.0,
        )

        reader.open()

        print(f"[{name}] Connected.")

        frames = []

        while len(frames) < WINDOW_SIZE:
            frame = reader.read_frame()

            if frame is None:
                continue

            subcarriers = frame.complex_csi.size

            if subcarriers != EXPECTED_SUBCARRIERS:
                continue

            frames.append(frame)

        reader.close()

        results[name] = frames

        print(
            f"[{name}] Captured "
            f"{len(frames)} valid frames."
        )

    except Exception as e:
        errors[name] = str(e)
        print(f"[{name}] ERROR: {e}")


print("=" * 60)
print("TWO ESP32 CSI WINDOW TEST")
print("=" * 60)

threads = []

start_time = time.perf_counter()

for name, port in PORTS.items():
    thread = threading.Thread(
        target=capture_window,
        args=(name, port),
    )

    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

elapsed = time.perf_counter() - start_time

print()
print("=" * 60)
print("CAPTURE RESULT")
print("=" * 60)

for name in ["FRIEND", "YOU"]:
    frames = results[name]

    if frames:
        matrix = np.array(
            [
                np.abs(frame.complex_csi)
                for frame in frames
            ],
            dtype=np.float32,
        )

        print(
            f"{name}: "
            f"{len(frames)} frames, "
            f"matrix shape = {matrix.shape}"
        )

        print(
            f"       Time range: "
            f"{frames[0].timestamp:.6f} → "
            f"{frames[-1].timestamp:.6f}"
        )

        print(
            f"       Duration: "
            f"{frames[-1].timestamp - frames[0].timestamp:.3f} sec"
        )

    else:
        print(f"{name}: NO DATA")

    if errors[name]:
        print(f"       ERROR: {errors[name]}")

print()
print(f"Total capture time: {elapsed:.3f} sec")

friend_ok = len(results["FRIEND"]) == WINDOW_SIZE
you_ok = len(results["YOU"]) == WINDOW_SIZE

print()

if friend_ok and you_ok:
    print("SUCCESS: Both ESP32s captured 50 valid CSI frames.")
else:
    print("FAILED: One or both ESP32s did not capture 50 valid frames.")

print("=" * 60)
