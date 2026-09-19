import sys
import time
import threading
from pathlib import Path

import numpy as np
import joblib


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT / "human_detection"))

from serial_reader import CSISerialReader
from .fall_features import extract_two_receiver_fall_features

def calculate_motion(amp_matrix):
    if len(amp_matrix) < 2:
        return 0.0

    difference = np.diff(amp_matrix, axis=0)
    motion = np.mean(np.abs(difference), axis=1)

    if len(motion) == 0:
        return 0.0

    return float(np.max(motion))


# ============================================================
# CONFIGURATION
# ============================================================

FRIEND_PORT = (
    "/dev/serial/by-id/"
    "usb-Espressif_USB_JTAG_serial_debug_unit_94:A9:90:D2:EF:BC-if00"
)

YOUR_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5B8E073970-if00"
)

BAUDRATE = 921600

EXPECTED_SUBCARRIERS = 64

WINDOW_SECONDS = 4.0

DECISION_INTERVAL = 1.0

FALL_THRESHOLD = 0.70

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "fall_detector_90samples.joblib"
)

# ============================================================
# CSI CAPTURE THREAD
# ============================================================

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

        # Keep approximately the most recent 6 seconds.
        # This prevents unlimited memory growth.
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

    if len(frames) < 30:
        return None, None

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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NSA - FALL DETECTION")
    print("REAL-TIME TERMINAL DETECTOR")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Load saved model
    # --------------------------------------------------------

    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]

    print("Model loaded successfully.")

    print(f"\nFall threshold: {FALL_THRESHOLD:.2f}")
    print(f"Window: {WINDOW_SECONDS:.1f} seconds")

    # --------------------------------------------------------
    # Open ESP32 receivers
    # --------------------------------------------------------

    print("\nOpening ESP32 receivers...")

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

    print("Both ESP32 receivers opened successfully.")

    # --------------------------------------------------------
    # Start capture threads
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

    print("\nLive CSI acquisition started.")

    print("\nStand inside the calibrated sensing zone.")

    print("\nPress Ctrl+C to stop.\n")

    # --------------------------------------------------------
    # Wait for enough CSI
    # --------------------------------------------------------

    try:

        while True:

            time.sleep(DECISION_INTERVAL)

            print(
                f"DEBUG frames -> Friend: {len(friend_result['frames'])} | "
                f"You: {len(you_result['frames'])}",
                flush=True
            )

            friend_csi, friend_ts = get_recent_window(
                friend_result
            )

            you_csi, you_ts = get_recent_window(
                you_result
            )

            if friend_csi is None or you_csi is None:

                print(
                    "Waiting for enough CSI data...",
                    flush=True
                )

                continue

            # ------------------------------------------------
            # Convert complex CSI to amplitude
            # ------------------------------------------------

            friend_amp = np.abs(friend_csi)
            you_amp = np.abs(you_csi)

            # ------------------------------------------------
            # Calculate sampling rates
            # ------------------------------------------------

            friend_duration = (
                friend_ts[-1] - friend_ts[0]
            )

            you_duration = (
                you_ts[-1] - you_ts[0]
            )

            if friend_duration <= 0:
                continue

            if you_duration <= 0:
                continue

            friend_rate = (
                (len(friend_ts) - 1)
                / friend_duration
            )

            you_rate = (
                (len(you_ts) - 1)
                / you_duration
            )

            # ------------------------------------------------
            # Extract six fall features
            # ------------------------------------------------

            features = extract_two_receiver_fall_features(
                friend_amp,
                you_amp,
                friend_rate,
                you_rate
            )

            print(
                "FEATURES | "
                f"FPeak={features[0]:.4f} | "
                f"FJerk={features[1]:.4f} | "
                f"FStill={features[2]:.4f} | "
                f"YPeak={features[3]:.4f} | "
                f"YJerk={features[4]:.4f} | "
                f"YStill={features[5]:.4f}",
                flush=True
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            probabilities = model.predict_proba(
                features.reshape(1, -1)
            )[0]

            classes = model.classes_

            fall_index = list(classes).index("FALL")

            fall_probability = float(
                probabilities[fall_index]
            )

            normal_probability = float(
                probabilities[
                    list(classes).index("NORMAL")
                ]
            )

                        # ------------------------------------------------
            # Safety feature fallback
            # ------------------------------------------------

            # strong_fall_pattern =(
            #     (
            #         features[0] >= 50.0
            #         and features[1] >= 3.0
            #         and features[2] <= 0.25
            #     )
            #     or
            #     (
            #         features[3] >= 45.0
            #         and features[4] >= 3.0
            #         and features[5] <= 0.25
            #     )
            # )

            # if (
            #     fall_probability < FALL_THRESHOLD
            #     and strong_fall_pattern
            # ):
            #     fall_probability = 1.0
            #     normal_probability = 0.0

            #     print(
            #         "Strong fall-motion pattern detected",
            #         flush=True
            #     )

            # Supplementary fall pattern for strong sudden-motion events
            strong_fall_pattern = (
                (
                    features[0] >= 100.0
                    and features[1] >= 8.0
                )
                or
                (
                    features[3] >= 150.0
                    and features[4] >= 10.0
                )
            )
            # ------------------------------------------------
            # Decision
            # ------------------------------------------------

            if fall_probability >= FALL_THRESHOLD or strong_fall_pattern:
                status = "FALL DETECTED"
            else:
                status = "NORMAL"

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            print(
                f"\r"
                f"Status: {status:14s} | "
                f"Fall: {fall_probability * 100:5.1f}% | "
                f"Normal: {normal_probability * 100:5.1f}% | "
                f"Friend: {friend_rate:5.1f} Hz | "
                f"You: {you_rate:5.1f} Hz",
                end="",
                flush=True
            )

    except KeyboardInterrupt:

        print("\n\nStopping detector...")

    finally:

        stop_event.set()

        time.sleep(0.5)

        friend_reader.close()
        you_reader.close()

        print("Both ESP32 receivers closed.")

        print("\nDetector stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()