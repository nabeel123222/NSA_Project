import os
import time
from collections import deque

import joblib
import numpy as np

from serial_reader import CSISerialReader, frames_to_amplitude_matrix
from features import extract_feature_vector


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/human_detector.joblib"

PORT = "/dev/ttyACM0"
BAUDRATE = 921600

WINDOW_SIZE = 50
EXPECTED_SUBCARRIERS = 64

# Number of recent predictions used for smoothing
SMOOTHING_WINDOWS = 5

# Minimum number of PRESENT predictions required
# inside the recent prediction history.
PRESENT_REQUIRED = 3

# Minimum confidence required for a PRESENT prediction
PRESENT_CONFIDENCE = 0.60


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    data = joblib.load(MODEL_PATH)

    model = data["model"]
    feature_count = data["feature_count"]
    classes = data["classes"]

    print("Model loaded successfully.")
    print(f"Expected features: {feature_count}")
    print(f"Classes: {classes}")

    return model


# ============================================================
# CALCULATE SAMPLE RATE
# ============================================================

def calculate_sample_rate(frames):
    if len(frames) < 2:
        return 0.0

    timestamps = np.array(
        [frame.timestamp for frame in frames],
        dtype=np.float64
    )

    differences = np.diff(timestamps)

    differences = differences[
        np.isfinite(differences) & (differences > 0)
    ]

    if len(differences) == 0:
        return 0.0

    return float(1.0 / np.median(differences))


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

def predict_window(model, frames):
    if len(frames) != WINDOW_SIZE:
        return None

    amp_matrix = frames_to_amplitude_matrix(frames)

    # Safety check
    if amp_matrix.shape != (
        WINDOW_SIZE,
        EXPECTED_SUBCARRIERS
    ):
        print(
            f"[WARNING] Unexpected CSI matrix shape: "
            f"{amp_matrix.shape}"
        )
        return None

    if not np.isfinite(amp_matrix).all():
        print("[WARNING] CSI window contains invalid values.")
        return None

    if np.all(amp_matrix == 0):
        print("[WARNING] CSI window is all zeros.")
        return None

    sample_rate = calculate_sample_rate(frames)

    # Use a safe fallback only if timestamp calculation fails.
    if sample_rate <= 0:
        sample_rate = 180.0

    try:
        feature_vector = extract_feature_vector(
            amp_matrix,
            sample_rate_hz=sample_rate
        )
    except Exception as exc:
        print(f"[WARNING] Feature extraction failed: {exc}")
        return None

    if len(feature_vector) != 258:
        print(
            f"[WARNING] Unexpected feature count: "
            f"{len(feature_vector)}"
        )
        return None

    feature_vector = feature_vector.reshape(1, -1)

    prediction = model.predict(feature_vector)[0]

    probabilities = model.predict_proba(feature_vector)[0]

    # The Random Forest was trained with:
    # empty = 0
    # present = 1

    empty_probability = probabilities[0]

    present_probability = probabilities[1]

    return {
        "prediction": prediction,
        "empty_probability": float(empty_probability),
        "present_probability": float(present_probability),
        "sample_rate": sample_rate,
    }


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

def update_smoothed_prediction(
    history,
    raw_prediction,
    present_probability,
    current_state
):
    """
    Stable temporal state machine.

    EMPTY -> PRESENT:
        Requires 3 confident PRESENT windows
        within the recent history.

    PRESENT -> EMPTY:
        Requires 3 confident EMPTY windows
        within the recent history.

    This prevents the final state from flickering
    when one or two prediction windows are weak.
    """

    # --------------------------------------------------------
    # Convert the current prediction into a stable label
    # --------------------------------------------------------

    if (
        raw_prediction == "present"
        and present_probability >= PRESENT_CONFIDENCE
    ):
        history.append("present")
    else:
        history.append("empty")

    present_count = history.count("present")
    empty_count = history.count("empty")

    # --------------------------------------------------------
    # If currently EMPTY
    # Only change to PRESENT after enough evidence.
    # --------------------------------------------------------

    if current_state == "EMPTY":

        if present_count >= PRESENT_REQUIRED:
            return "PRESENT"

        return "EMPTY"

    # --------------------------------------------------------
    # If currently PRESENT
    # Do NOT immediately drop to EMPTY because of
    # one weak prediction.
    # --------------------------------------------------------

    if current_state == "PRESENT":

        if empty_count >= PRESENT_REQUIRED:
            return "EMPTY"

        return "PRESENT"

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    if present_count >= PRESENT_REQUIRED:
        return "PRESENT"

    return "EMPTY"

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIVE HUMAN DETECTION")
    print("=" * 60)

    model = load_model()

    print(f"Window size: {WINDOW_SIZE}")
    print(f"Serial port: {PORT}")
    print(f"Baud rate: {BAUDRATE}")
    print()
    print(
        f"Temporal smoothing: "
        f"{SMOOTHING_WINDOWS} windows"
    )
    print(
        f"PRESENT required: "
        f"{PRESENT_REQUIRED}/{SMOOTHING_WINDOWS}"
    )
    print(
        f"PRESENT confidence threshold: "
        f"{PRESENT_CONFIDENCE * 100:.0f}%"
    )
    print()
    print("Starting ESP32 CSI reader...")
    print("Make sure the ESP32 is connected.")
    print("Press Ctrl+C to stop.")
    print()

    reader = CSISerialReader(
        port=PORT,
        baudrate=BAUDRATE,
        timeout=2.0
    )

    frames = []

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOWS
    )

    last_smoothed_state = None

    try:
        with reader:

            while True:

                frame = reader.read_frame()

                if frame is None:
                    continue

                # ====================================================
                # ACCEPT ONLY 64-SUBCARRIER FRAMES
                # ====================================================

                if frame.complex_csi.size != EXPECTED_SUBCARRIERS:

                    print(
                        f"[WARNING] Ignoring CSI frame with "
                        f"{frame.complex_csi.size} subcarriers"
                    )

                    continue

                frames.append(frame)

                # Wait until 50 valid frames are collected
                if len(frames) < WINDOW_SIZE:
                    continue

                # ====================================================
                # PREDICTION
                # ====================================================

                result = predict_window(
                    model,
                    frames
                )

                # Clear window after processing
                frames.clear()

                if result is None:
                    continue

                raw_prediction = result["prediction"]
                if raw_prediction == 1:
                    raw_prediction = "present"
                else:
                    raw_prediction = "empty"

                empty_probability = result[
                    "empty_probability"
                ]

                present_probability = result[
                    "present_probability"
                ]

                sample_rate = result[
                    "sample_rate"
                ]

                # ====================================================
                # TEMPORAL SMOOTHING
                # ====================================================

                smoothed_state = update_smoothed_prediction(
                    prediction_history,
                    raw_prediction,
                    present_probability,
                    last_smoothed_state or "EMPTY"
                )
                # ====================================================
                # PRINT RESULT
                # ====================================================

                timestamp = time.strftime("%H:%M:%S")

                print(
                    f"[{timestamp}] "
                    f"Raw: {raw_prediction.upper():7s} | "
                    f"Final: {smoothed_state:7s} | "
                    f"Present: {present_probability * 100:6.2f}% | "
                    f"Empty: {empty_probability * 100:6.2f}% | "
                    f"Rate: {sample_rate:6.1f} Hz"
                )

                # Print when final state changes
                if (
                    last_smoothed_state is not None
                    and smoothed_state != last_smoothed_state
                ):
                    print(
                        f"    >>> HUMAN DETECTION STATE CHANGED: "
                        f"{last_smoothed_state} -> "
                        f"{smoothed_state}"
                    )

                last_smoothed_state = smoothed_state

    except KeyboardInterrupt:

        print()
        print("=" * 60)
        print("LIVE DETECTION STOPPED")
        print("=" * 60)


if __name__ == "__main__":
    main()