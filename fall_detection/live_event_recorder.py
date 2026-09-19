import sys
import time
import threading
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import joblib
import numpy as np

from human_detection.serial_reader import CSISerialReader
from fall_features import extract_two_receiver_fall_features


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
    "usb-1a86_USB_Single_Serial_5B8E073970-if00"
)

BAUDRATE = 921600

EXPECTED_SUBCARRIERS = 64

WINDOW_SECONDS = 4.0

OUTPUT_DIR = BASE_DIR / "dataset" / "live_falls"


MODEL_PATH = (
    BASE_DIR
    / "model"
    / "fall_detector_90samples.joblib"
)

# Motion level used to identify a strong event.
MOTION_TRIGGER = 10.0

# After one event, ignore triggers for this long.
COOLDOWN_SECONDS = 6.0


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

    # Make sure frames and timestamps always have the same length
    n = min(len(frames), len(timestamps))

    if n < 30:
        return None, None

    frames = frames[-n:]
    timestamps = timestamps[-n:]

    timestamps_array = np.asarray(timestamps, dtype=float)

    latest_time = timestamps_array[-1]
    start_time = latest_time - WINDOW_SECONDS

    mask = timestamps_array >= start_time

    selected_frames = np.asarray(frames, dtype=np.complex128)[mask]
    selected_timestamps = timestamps_array[mask]

    if len(selected_frames) < 30:
        return None, None

    return selected_frames, selected_timestamps


# ============================================================
# MOTION
# ============================================================

def calculate_motion(amp_matrix):

    if len(amp_matrix) < 2:
        return 0.0

    difference = np.diff(
        amp_matrix,
        axis=0
    )

    motion = np.mean(
        np.abs(difference),
        axis=1
    )

    if len(motion) == 0:
        return 0.0

    return float(
        np.max(motion)
    )


# ============================================================
# SAVE EVENT
# ============================================================

def save_event(
    sample_number,
    friend_csi,
    friend_ts,
    you_csi,
    you_ts,
    friend_rate,
    you_rate,
    features,
    fall_probability,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        OUTPUT_DIR
        / f"live_fall_{sample_number:03d}.npz"
    )

    np.savez_compressed(

        filename,

        label="LIVE_FALL",

        friend_csi=friend_csi,

        friend_timestamps=friend_ts,

        friend_rate=friend_rate,

        you_csi=you_csi,

        you_timestamps=you_ts,

        you_rate=you_rate,

        fall_features=features,

        fall_probability=fall_probability,

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
    print("NSA - LIVE FALL EVENT RECORDER")
    print("=" * 70)

    print()
    print("Purpose:")
    print("Record ONE CSI window for each physical fall event.")
    print()
    print("The existing model will NOT be modified.")
    print()
    print("SAFETY:")
    print("Use only a safe, controlled, low-height movement")
    print("on a padded/safe surface.")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]

    print("Model loaded successfully.")

    print()
    print(f"Motion trigger : {MOTION_TRIGGER}")
    print(f"Window         : {WINDOW_SECONDS} seconds")
    print(f"Cooldown       : {COOLDOWN_SECONDS} seconds")
    print(f"Output         : {OUTPUT_DIR}")

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

    print()
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
    print("Stand normally in the calibrated sensing zone.")
    print()
    print("Perform ONE safe simulated fall.")
    print()
    print("The same fall will NOT be saved multiple times.")
    print()
    print("Press Ctrl+C to stop.")
    print()

    sample_number = 9

    # Prevent repeated triggers from the same physical event.
    cooldown_until = 0.0

    # Track whether the system has returned to low motion.
    event_active = False

    try:

        while True:

            time.sleep(0.25)

            friend_csi, friend_ts = (
                get_recent_window(friend_result)
            )

            you_csi, you_ts = (
                get_recent_window(you_result)
            )

            if friend_csi is None or you_csi is None:
                continue

            # ------------------------------------------------
            # Convert CSI to amplitude
            # ------------------------------------------------

            friend_amp = np.abs(
                friend_csi
            )

            you_amp = np.abs(
                you_csi
            )

            # ------------------------------------------------
            # Calculate current motion
            # ------------------------------------------------

            friend_motion = calculate_motion(
                friend_amp
            )

            you_motion = calculate_motion(
                you_amp
            )

            combined_motion = max(
                friend_motion,
                you_motion
            )

            print(
                f"MOTION DEBUG | Friend={friend_motion:.4f} | "
                f"You={you_motion:.4f} | "
                f"Combined={combined_motion:.4f}",
                flush=True
            )

            now = time.monotonic()

            # ------------------------------------------------
            # Reset event_active after motion becomes low.
            # ------------------------------------------------

            if combined_motion < MOTION_TRIGGER * 0.50:

                event_active = False

            # ------------------------------------------------
            # Ignore events during cooldown.
            # ------------------------------------------------

            if now < cooldown_until:
                continue

            # ------------------------------------------------
            # Already recorded this event.
            # ------------------------------------------------

            if event_active:
                continue

            # ------------------------------------------------
            # Strong new motion event
            # ------------------------------------------------

            if combined_motion < MOTION_TRIGGER:
                continue

            event_active = True

            cooldown_until = (
                now + COOLDOWN_SECONDS
            )

            # ------------------------------------------------
            # Calculate rates
            # ------------------------------------------------

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
                or
                you_duration <= 0
            ):
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
            # Extract features
            # ------------------------------------------------

            features = (
                extract_two_receiver_fall_features(
                    friend_amp,
                    you_amp,
                    friend_rate,
                    you_rate
                )
            )

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            probabilities = (
                model.predict_proba(
                    features.reshape(1, -1)
                )[0]
            )

            classes = model.classes_

            fall_index = list(
                classes
            ).index("FALL")

            fall_probability = float(
                probabilities[fall_index]
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            print()
            print("=" * 70)
            print(
                f"LIVE FALL EVENT {sample_number}"
            )
            print("=" * 70)

            print(
                f"Combined motion : "
                f"{combined_motion:.4f}"
            )

            print(
                f"Friend motion   : "
                f"{friend_motion:.4f}"
            )

            print(
                f"You motion      : "
                f"{you_motion:.4f}"
            )

            print()

            print(
                f"Friend rate     : "
                f"{friend_rate:.2f} Hz"
            )

            print(
                f"You rate        : "
                f"{you_rate:.2f} Hz"
            )

            print()

            print(
                f"FPeak           : "
                f"{features[0]:.4f}"
            )

            print(
                f"FJerk           : "
                f"{features[1]:.4f}"
            )

            print(
                f"FStill          : "
                f"{features[2]:.4f}"
            )

            print(
                f"YPeak           : "
                f"{features[3]:.4f}"
            )

            print(
                f"YJerk           : "
                f"{features[4]:.4f}"
            )

            print(
                f"YStill          : "
                f"{features[5]:.4f}"
            )

            print()

            print(
                f"Existing model FALL probability: "
                f"{fall_probability * 100:.1f}%"
            )

            # ------------------------------------------------
            # Save exactly ONE sample
            # ------------------------------------------------

            filename = save_event(

                sample_number,

                friend_csi,
                friend_ts,

                you_csi,
                you_ts,

                friend_rate,
                you_rate,

                features,
                fall_probability
            )

            print()
            print(
                f"SAVED: {filename}"
            )

            print(
                f"Cooldown: "
                f"{COOLDOWN_SECONDS:.0f} seconds"
            )

            print("=" * 70)

            sample_number += 1

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

        print(
            "Both ESP32 receivers closed."
        )

        print(
            "Recorder stopped."
        )


if __name__ == "__main__":
    main()