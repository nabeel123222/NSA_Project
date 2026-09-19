import sys
import time
import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from serial_reader import CSISerialReader
from fall_features import extract_two_receiver_fall_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FRIEND_PORT = "/dev/ttyACM0"
YOUR_PORT = "/dev/ttyACM1"

BAUDRATE = 921600

EXPECTED_SUBCARRIERS = 64

CAPTURE_SECONDS = 4.0

NUM_SAMPLES = 30

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "dataset"
    / "fall"
)


# --------------------------------------------------
# Capture one ESP32 stream
# --------------------------------------------------

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


# --------------------------------------------------
# Calculate sampling rate
# --------------------------------------------------

def calculate_rate(timestamps):

    if len(timestamps) < 2:
        return 0.0

    duration = timestamps[-1] - timestamps[0]

    if duration <= 0:
        return 0.0

    return (len(timestamps) - 1) / duration


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("NSA - FALL DETECTION")
    print("SIMULATED FALL DATASET COLLECTOR")
    print("=" * 65)

    print(f"\nOutput folder:")
    print(OUTPUT_DIR)

    print("\nDataset:")
    print(f"  FALL samples : {NUM_SAMPLES}")
    print(f"  Duration     : {CAPTURE_SECONDS} seconds")
    print("  Receivers    : 2 ESP32-S3")
    print("  Subcarriers  : 64")

    print("\nIMPORTANT SAFETY:")
    print("- Do NOT perform an uncontrolled real fall.")
    print("- Use a very low-height controlled movement.")
    print("- Use a thick mattress / padded surface.")
    print("- Ideally have another person nearby.")
    print("- Stop immediately if anything feels unsafe.")

    print("\nOpen both ESP32 receivers...")

    friend_reader = CSISerialReader(
        port=FRIEND_PORT,
        baudrate=BAUDRATE
    )

    you_reader = CSISerialReader(
        port=YOUR_PORT,
        baudrate=BAUDRATE
    )

    friend_reader.open()
    you_reader.open()

    time.sleep(2)

    print("\nBoth ESP32 receivers opened.")

    input(
        "\nPress ENTER when you are ready "
        "to begin FALL collection..."
    )

    try:

        sample_number = 1

        while sample_number <= NUM_SAMPLES:

            print("\n" + "=" * 65)
            print(
                f"SIMULATED FALL SAMPLE "
                f"{sample_number}/{NUM_SAMPLES}"
            )
            print("=" * 65)

            print(
                "\nPrepare your SAFE, LOW-HEIGHT "
                "fall-like movement."
            )

            print(
                "Press ENTER, then perform the "
                "controlled movement."
            )

            input()

            friend_result = {}
            you_result = {}

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

            # Start both receivers
            friend_thread.start()
            you_thread.start()

            start_time = time.monotonic()

            time.sleep(CAPTURE_SECONDS)

            stop_event.set()

            friend_thread.join(timeout=2)
            you_thread.join(timeout=2)

            elapsed = time.monotonic() - start_time

            # --------------------------------------------------
            # Retrieve CSI
            # --------------------------------------------------

            friend_csi = friend_result.get(
                "csi",
                np.empty(
                    (0, EXPECTED_SUBCARRIERS),
                    dtype=np.complex128
                )
            )

            friend_timestamps = friend_result.get(
                "timestamps",
                np.empty(0, dtype=float)
            )

            you_csi = you_result.get(
                "csi",
                np.empty(
                    (0, EXPECTED_SUBCARRIERS),
                    dtype=np.complex128
                )
            )

            you_timestamps = you_result.get(
                "timestamps",
                np.empty(0, dtype=float)
            )

            friend_rate = calculate_rate(
                friend_timestamps
            )

            you_rate = calculate_rate(
                you_timestamps
            )

            print("\nCapture complete.")

            print(
                f"Friend: {friend_csi.shape} | "
                f"{friend_rate:.2f} Hz"
            )

            print(
                f"You:    {you_csi.shape} | "
                f"{you_rate:.2f} Hz"
            )

            print(
                f"Wall time: {elapsed:.2f} sec"
            )

            # --------------------------------------------------
            # Validate
            # --------------------------------------------------

            valid = True

            if len(friend_csi) < 100:

                print(
                    "ERROR: Friend has fewer than 100 frames."
                )

                valid = False

            if len(you_csi) < 100:

                print(
                    "ERROR: You has fewer than 100 frames."
                )

                valid = False

            if not np.isfinite(friend_csi).all():

                print(
                    "ERROR: Friend CSI contains invalid values."
                )

                valid = False

            if not np.isfinite(you_csi).all():

                print(
                    "ERROR: You CSI contains invalid values."
                )

                valid = False

            if valid:

                # --------------------------------------------------
                # Convert CSI to amplitude
                # --------------------------------------------------

                friend_amp = np.abs(friend_csi)
                you_amp = np.abs(you_csi)

                # Use common number of frames
                n = min(
                    len(friend_amp),
                    len(you_amp)
                )

                friend_amp = friend_amp[:n]
                you_amp = you_amp[:n]

                # --------------------------------------------------
                # Extract six fall-specific features
                # --------------------------------------------------

                fall_features = (
                    extract_two_receiver_fall_features(
                        friend_amp,
                        you_amp,
                        friend_rate,
                        you_rate
                    )
                )

                if len(fall_features) != 6:

                    print(
                        "ERROR: Expected 6 fall features."
                    )

                    valid = False

                elif not np.isfinite(
                    fall_features
                ).all():

                    print(
                        "ERROR: Fall features contain "
                        "invalid values."
                    )

                    valid = False

            # --------------------------------------------------
            # Save
            # --------------------------------------------------

            if valid:

                filename = (
                    OUTPUT_DIR
                    / f"fall_{sample_number:03d}.npz"
                )

                np.savez_compressed(

                    filename,

                    label="FALL",

                    friend_csi=friend_csi,

                    friend_timestamps=friend_timestamps,

                    friend_rate=friend_rate,

                    you_csi=you_csi,

                    you_timestamps=you_timestamps,

                    you_rate=you_rate,

                    fall_features=fall_features,

                    capture_seconds=CAPTURE_SECONDS,

                    subcarriers=EXPECTED_SUBCARRIERS,

                    sample_number=sample_number
                )

                print(
                    f"\nSAVED: {filename.name}"
                )

                print(
                    "\nSix fall features:"
                )

                names = [
                    "Friend peak motion",
                    "Friend jerk",
                    "Friend stillness",
                    "You peak motion",
                    "You jerk",
                    "You stillness"
                ]

                for name, value in zip(
                    names,
                    fall_features
                ):

                    print(
                        f"  {name:25s}: "
                        f"{value:.6f}"
                    )

                sample_number += 1

            else:

                print(
                    "\nSample rejected."
                )

                print(
                    "This sample was NOT saved."
                )

                print(
                    "Repeat the sample."
                )

            if sample_number <= NUM_SAMPLES:

                print(
                    "\nRest briefly before the next sample."
                )

                time.sleep(2)

    except KeyboardInterrupt:

        print(
            "\n\nCollection stopped by user."
        )

    finally:

        friend_reader.close()
        you_reader.close()

        saved_files = sorted(
            OUTPUT_DIR.glob("fall_*.npz")
        )

        print(
            "\nESP32 receivers closed."
        )

        print(
            f"\nFALL samples currently saved: "
            f"{len(saved_files)}"
        )

        print("\nCollector finished.")


if __name__ == "__main__":
    main()