import sys
import time
import threading
from pathlib import Path


import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from serial_reader import CSISerialReader
from fall_features import extract_two_receiver_fall_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

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

CAPTURE_SECONDS = 4.0

NUM_SAMPLES = 10

OUTPUT_DIR = BASE_DIR / "dataset" / "test" / "normal"


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

    print("=" * 70)
    print("NSA - FALL DETECTION")
    print("NORMAL SET 2 DATASET COLLECTOR")
    print("=" * 70)

    print("\nOutput folder:")
    print(OUTPUT_DIR)

    print("\nDataset:")
    print(f"  NORMAL samples : {NUM_SAMPLES}")
    print(f"  Duration       : {CAPTURE_SECONDS} seconds")
    print("  Receivers      : 2 ESP32-S3")
    print("  Subcarriers    : 64")

    print("\nIMPORTANT:")
    print("- Keep the router in exactly the same position.")
    print("- Keep both ESP32s in exactly the same position.")
    print("- Stand inside the same sensing zone used for FALL data.")
    print("- Fan OFF.")
    print("- Do NOT perform a fall or sudden large movement.")

    print("\nNORMAL movement plan:")
    print("  001-005 : completely still")
    print("  006-010 : very small natural movement")
    print("  011-015 : slight hand/arm movement")
    print("  016-020 : slight head/body movement")
    print("  021-025 : occasionally shift body weight")
    print("  026-030 : natural small standing movements")

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

    print("\nBoth ESP32 receivers opened successfully.")

    input(
        "\nPress ENTER when you are ready "
        "to begin NORMAL Set 2 collection..."
    )

    try:

        sample_number = 1

        while sample_number <= NUM_SAMPLES:

            print("\n" + "=" * 70)
            print(
                f"NORMAL SET 2 SAMPLE "
                f"{sample_number}/{NUM_SAMPLES}"
            )
            print("=" * 70)

            # ------------------------------------------
            # Instructions for this sample
            # ------------------------------------------

            if sample_number <= 5:

                instruction = (
                    "Stand completely still."
                )

            elif sample_number <= 10:

                instruction = (
                    "Stand normally with very small "
                    "natural movement."
                )

            elif sample_number <= 15:

                instruction = (
                    "Stand normally and make slight "
                    "hand/arm movements."
                )

            elif sample_number <= 20:

                instruction = (
                    "Stand normally with slight "
                    "head/body movement."
                )

            elif sample_number <= 25:

                instruction = (
                    "Stand normally and occasionally "
                    "shift your body weight."
                )

            else:

                instruction = (
                    "Stand normally with small natural "
                    "movements."
                )

            print("\nActivity:")
            print("  " + instruction)

            print(
                "\nPress ENTER and immediately perform "
                "the NORMAL activity."
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

            # ------------------------------------------
            # Retrieve CSI
            # ------------------------------------------

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

            # ------------------------------------------
            # Validate
            # ------------------------------------------

            friend_rate = calculate_rate(
                friend_timestamps
            )

            you_rate = calculate_rate(
                you_timestamps
            )

            print("\nCapture result:")

            print(
                f"  Friend frames : {len(friend_csi)}"
            )

            print(
                f"  You frames    : {len(you_csi)}"
            )

            print(
                f"  Friend rate   : {friend_rate:.2f} Hz"
            )

            print(
                f"  You rate      : {you_rate:.2f} Hz"
            )

            print(
                f"  Elapsed       : {elapsed:.2f} sec"
            )

            if len(friend_csi) < 100:

                print(
                    "\nERROR: Friend ESP32 produced "
                    "too few frames."
                )

                print(
                    "This sample will NOT be saved."
                )

                continue

            if len(you_csi) < 100:

                print(
                    "\nERROR: You ESP32 produced "
                    "too few frames."
                )

                print(
                    "This sample will NOT be saved."
                )

                continue

            # ------------------------------------------
            # Use common number of frames
            # ------------------------------------------

            common_frames = min(
                len(friend_csi),
                len(you_csi)
            )

            friend_csi = friend_csi[
                :common_frames
            ]

            friend_timestamps = friend_timestamps[
                :common_frames
            ]

            you_csi = you_csi[
                :common_frames
            ]

            you_timestamps = you_timestamps[
                :common_frames
            ]

            # ------------------------------------------
            # Amplitude
            # ------------------------------------------

            friend_amp = np.abs(
                friend_csi
            )

            you_amp = np.abs(
                you_csi
            )

            # ------------------------------------------
            # Fall-specific feature vector
            # ------------------------------------------

            fall_features = (
                extract_two_receiver_fall_features(
                    friend_amp,
                    you_amp,
                    friend_rate,
                    you_rate
                )
            )

            if not np.isfinite(
                fall_features
            ).all():

                print(
                    "\nERROR: Non-finite features."
                )

                print(
                    "This sample will NOT be saved."
                )

                continue

            # ------------------------------------------
            # Save
            # ------------------------------------------

            output_file = (
                OUTPUT_DIR
                / f"normal_{sample_number:03d}.npz"
            )

            np.savez_compressed(

                output_file,

                label="NORMAL",

                friend_csi=friend_csi,

                friend_timestamps=friend_timestamps,

                friend_rate=friend_rate,

                you_csi=you_csi,

                you_timestamps=you_timestamps,

                you_rate=you_rate,

                fall_features=fall_features,

                capture_seconds=elapsed,

                subcarriers=EXPECTED_SUBCARRIERS,

                sample_number=sample_number
            )

            print(
                f"\nSAVED: {output_file.name}"
            )

            print(
                "  Fall features:",
                np.round(
                    fall_features,
                    4
                )
            )

            sample_number += 1

            # Small pause before next sample
            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "\n\nCollection stopped by user."
        )

    finally:

        try:
            friend_reader.close()
        except Exception:
            pass

        try:
            you_reader.close()
        except Exception:
            pass

        print(
            "\nBoth ESP32 receivers closed."
        )

    print("\n" + "=" * 70)
    print("NORMAL SET 2 COLLECTION COMPLETE")
    print("=" * 70)

    saved_files = sorted(
        OUTPUT_DIR.glob("normal_*.npz")
    )

    print(
        f"\nSaved NORMAL Set 2 samples: "
        f"{len(saved_files)}"
    )

    print(
        f"Location: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
