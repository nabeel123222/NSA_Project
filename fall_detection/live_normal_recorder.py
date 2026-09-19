import sys
import time
import threading
from pathlib import Path

import numpy as np

# Allow importing human_detection from the project root
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from human_detection.serial_reader import CSISerialReader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRIEND_PORT = (
    "/dev/serial/by-id/"
    "usb-Espressif_USB_JTAG_serial_debug_unit_94:A9:90:D2:EF:BC-if00"
)

YOUR_PORT = (
    "/dev/serial/by-id/"
    "usb-Espressif_USB_JTAG_serial_debug_unit_E0:72:A1:D6:F5:D0-if00"
)

BAUDRATE = 921600

EXPECTED_SUBCARRIERS = 64

WINDOW_SECONDS = 4.0

OUTPUT_DIR = BASE_DIR / "dataset" / "live_normal"

START_DELAY = 10.0

SAMPLE_GAP = 6.0


# ============================================================
# CAPTURE THREAD
# ============================================================

def capture_stream(reader, result, stop_event):

    frames = []
    timestamps = []

    while not stop_event.is_set():

        try:
            frame = reader.read_frame()
        except Exception:
            continue

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

        # Keep approximately the most recent 6 seconds.
        if len(timestamps) > 1000:
            frames = frames[-1000:]
            timestamps = timestamps[-1000:]

        result["frames"] = frames
        result["timestamps"] = timestamps


# ============================================================
# GET RECENT WINDOW
# ============================================================

def get_recent_window(result):

    frames = result.get("frames", [])
    timestamps = result.get("timestamps", [])

    # Make sure both arrays have identical length.
    n = min(len(frames), len(timestamps))

    if n < 30:
        return None, None

    frames = frames[-n:]
    timestamps = timestamps[-n:]

    timestamps_array = np.asarray(
        timestamps,
        dtype=float
    )

    latest_time = timestamps_array[-1]

    start_time = latest_time - WINDOW_SECONDS

    mask = timestamps_array >= start_time

    selected_frames = np.asarray(
        frames,
        dtype=np.complex128
    )[mask]

    selected_timestamps = timestamps_array[mask]

    if len(selected_frames) < 30:
        return None, None

    return selected_frames, selected_timestamps


# ============================================================
# SAVE SAMPLE
# ============================================================

def save_sample(
    sample_number,
    friend_csi,
    friend_ts,
    you_csi,
    you_ts,
    friend_rate,
    you_rate,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        OUTPUT_DIR
        / f"live_normal_{sample_number:03d}.npz"
    )

    np.savez_compressed(

        filename,

        label="LIVE_NORMAL",

        friend_csi=friend_csi,

        friend_timestamps=friend_ts,

        friend_rate=friend_rate,

        you_csi=you_csi,

        you_timestamps=you_ts,

        you_rate=you_rate,

        capture_seconds=WINDOW_SECONDS,

        subcarriers=EXPECTED_SUBCARRIERS,

        sample_number=sample_number
    )

    return filename


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NSA - LIVE NORMAL CSI RECORDER")
    print("=" * 70)

    print()
    print("Purpose:")
    print("Record one CSI window for each normal activity.")
    print()

    print("SAFETY:")
    print("Remain standing normally.")
    print("Do not perform a fall during this recording.")
    print()

    print(f"Window : {WINDOW_SECONDS} seconds")
    print(f"Output : {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Open ESP32 receivers
    # --------------------------------------------------------

    friend_reader = CSISerialReader(
        port=FRIEND_PORT,
        baudrate=BAUDRATE
    )

    you_reader = CSISerialReader(
        port=YOUR_PORT,
        baudrate=BAUDRATE
    )

    print("Opening ESP32 receivers...")

    friend_reader.open()
    you_reader.open()

    print("Both ESP32 receivers opened.")

    # --------------------------------------------------------
    # Shared buffers
    # --------------------------------------------------------

    friend_result = {
        "frames": [],
        "timestamps": []
    }

    you_result = {
        "frames": [],
        "timestamps": []
    }

    stop_event = threading.Event()

    friend_thread = threading.Thread(
        target=capture_stream,
        args=(
            friend_reader,
            friend_result,
            stop_event
        ),
        daemon=True
    )

    you_thread = threading.Thread(
        target=capture_stream,
        args=(
            you_reader,
            you_result,
            stop_event
        ),
        daemon=True
    )

    friend_thread.start()
    you_thread.start()

    print()
    print("Live CSI acquisition started.")
    print()

    print("Stand in the calibrated sensing zone.")
    print()

    print("NORMAL ACTIVITY PLAN:")
    print("001 - Stand still")
    print("002 - Small weight shift")
    print("003 - Small hand movement")
    print("004 - Small head/body movement")
    print("005 - Natural standing adjustment")
    print("006 - Slight arm movement")
    print("007 - Occasional weight shift")
    print("008 - Normal standing movement")
    print()

    print("For each sample:")
    print("10 seconds normal standing")
    print("4 seconds activity")
    print("Then remain standing.")
    print()

    sample_number = 9

    try:

        # Give the serial streams time to stabilize.
        print("Stabilizing CSI acquisition...")
        time.sleep(3)

        while sample_number <= 16:

            print()
            print("=" * 70)
            print(f"PREPARING NORMAL SAMPLE {sample_number}")
            print("=" * 70)

            print()
            print("Stand normally in the sensing zone.")
            print("Recording starts in 10 seconds.")

            for remaining in range(10, 0, -1):
                print(
                    f"Starting in {remaining}...",
                    flush=True
                )
                time.sleep(1)

            print()
            print(
                f"CAPTURING NORMAL SAMPLE {sample_number}"
            )
            print(
                "Perform the planned small NORMAL activity."
            )
            print(
                f"Window: {WINDOW_SECONDS} seconds"
            )

            # Wait for the exact rolling window.
            time.sleep(WINDOW_SECONDS)

            friend_csi, friend_ts = (
                get_recent_window(friend_result)
            )

            you_csi, you_ts = (
                get_recent_window(you_result)
            )

            if friend_csi is None or you_csi is None:

                print(
                    "ERROR: Not enough CSI data."
                )

                print(
                    "This sample was NOT saved."
                )

                continue

            friend_duration = (
                friend_ts[-1]
                - friend_ts[0]
            )

            you_duration = (
                you_ts[-1]
                - you_ts[0]
            )

            if (
                friend_duration <= 0
                or you_duration <= 0
            ):

                print(
                    "ERROR: Invalid timestamps."
                )

                continue

            friend_rate = (
                (len(friend_ts) - 1)
                / friend_duration
            )

            you_rate = (
                (len(you_ts) - 1)
                / you_duration
            )

            filename = save_sample(

                sample_number,

                friend_csi,
                friend_ts,

                you_csi,
                you_ts,

                friend_rate,
                you_rate
            )

            print()
            print(
                f"Friend CSI : {friend_csi.shape}"
            )

            print(
                f"You CSI    : {you_csi.shape}"
            )

            print(
                f"Friend rate: {friend_rate:.2f} Hz"
            )

            print(
                f"You rate   : {you_rate:.2f} Hz"
            )

            print()
            print(
                f"SAVED: {filename}"
            )

            sample_number += 1

            if sample_number <= 8:

                print()
                print(
                    f"Rest for {SAMPLE_GAP:.0f} seconds."
                )

                time.sleep(SAMPLE_GAP)

    except KeyboardInterrupt:

        print()
        print("Stopping recorder...")

    finally:

        stop_event.set()

        friend_thread.join(
            timeout=2
        )

        you_thread.join(
            timeout=2
        )

        friend_reader.close()
        you_reader.close()

        print()
        print(
            "Both ESP32 receivers closed."
        )

        print(
            "Recorder stopped."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
