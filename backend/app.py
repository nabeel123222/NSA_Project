from flask import Flask, jsonify, render_template
import threading
import time
import os
import sys
from collections import deque

import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

HUMAN_DETECTION_DIR = os.path.join(
    PROJECT_ROOT,
    "human_detection"
)

if HUMAN_DETECTION_DIR not in sys.path:
    sys.path.insert(0, HUMAN_DETECTION_DIR)


# ============================================================
# HUMAN DETECTION IMPORTS
# ============================================================

from serial_reader import (
    CSISerialReader,
    frames_to_amplitude_matrix
)

from features import extract_feature_vector


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        PROJECT_ROOT,
        "frontend",
        "templates"
    ),
    static_folder=os.path.join(
        PROJECT_ROOT,
        "frontend",
        "static"
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    HUMAN_DETECTION_DIR,
    "model",
    "calibrated_model.joblib"
)

PORT = "/dev/ttyACM0"
BAUDRATE = 921600

WINDOW_SIZE = 50
EXPECTED_SUBCARRIERS = 64

SMOOTHING_WINDOWS = 5
PRESENT_REQUIRED = 3
PRESENT_CONFIDENCE = 0.60


# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]

print("=" * 60)
print("NSA FLASK BACKEND")
print("=" * 60)
print("Human detection model loaded.")
print(f"Model features: {model_data['feature_count']}")
print(f"Model classes: {model_data['classes']}")


# ============================================================
# GLOBAL STATE
# ============================================================

detector_thread = None
detector_running = False

latest_status = {
    "state": "STOPPED",
    "raw_prediction": "UNKNOWN",
    "confidence": 0.0,
    "present_probability": 0.0,
    "empty_probability": 0.0,
    "sample_rate": 0.0,
    "timestamp": None
}

status_lock = threading.Lock()


# ============================================================
# SAMPLE RATE
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
        np.isfinite(differences)
        & (differences > 0)
    ]

    if len(differences) == 0:
        return 0.0

    return float(
        1.0 / np.median(differences)
    )


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

def predict_window(frames):

    if len(frames) != WINDOW_SIZE:
        return None

    # --------------------------------------------------------
    # Convert frames to amplitude matrix
    # --------------------------------------------------------

    amp_matrix = frames_to_amplitude_matrix(
        frames
    )

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if amp_matrix.shape != (
        WINDOW_SIZE,
        EXPECTED_SUBCARRIERS
    ):

        print(
            f"[WARNING] Unexpected CSI matrix shape: "
            f"{amp_matrix.shape}"
        )

        return None

    # --------------------------------------------------------
    # Check values
    # --------------------------------------------------------

    if not np.isfinite(amp_matrix).all():

        print(
            "[WARNING] CSI window contains "
            "invalid values."
        )

        return None

    if np.all(amp_matrix == 0):

        print(
            "[WARNING] CSI window is all zeros."
        )

        return None

    # --------------------------------------------------------
    # Calculate sample rate BEFORE frames are cleared
    # --------------------------------------------------------

    sample_rate = calculate_sample_rate(
        frames
    )

    if sample_rate <= 0:
        sample_rate = 180.0

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    try:

        feature_vector = extract_feature_vector(
            amp_matrix,
            sample_rate_hz=sample_rate
        )

    except Exception as exc:

        print(
            f"[WARNING] Feature extraction failed: "
            f"{exc}"
        )

        return None

    # --------------------------------------------------------
    # Feature count check
    # --------------------------------------------------------

    if len(feature_vector) != 258:

        print(
            f"[WARNING] Unexpected feature count: "
            f"{len(feature_vector)}"
        )

        return None

    feature_vector = feature_vector.reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Random Forest prediction
    # --------------------------------------------------------

    prediction = model.predict(
        feature_vector
    )[0]

    probabilities = model.predict_proba(
        feature_vector
    )[0]

    # Model:
    # 0 = empty
    # 1 = present

    empty_probability = float(
        probabilities[0]
    )

    present_probability = float(
        probabilities[1]
    )

    if prediction == 1:

        raw_prediction = "present"
        confidence = present_probability

    else:

        raw_prediction = "empty"
        confidence = empty_probability

    return {
        "prediction": raw_prediction,
        "confidence": confidence,
        "empty_probability": empty_probability,
        "present_probability": present_probability,
        "sample_rate": sample_rate
    }


# ============================================================
# UPDATE API STATUS
# ============================================================

def update_status(result, final_state):

    global latest_status

    with status_lock:

        latest_status = {
            "state": final_state,
            "raw_prediction": result["prediction"].upper(),
            "confidence": round(
                result["confidence"],
                4
            ),
            "present_probability": round(
                result["present_probability"],
                4
            ),
            "empty_probability": round(
                result["empty_probability"],
                4
            ),
            "sample_rate": round(
                result["sample_rate"],
                2
            ),
            "timestamp": time.time()
        }


# ============================================================
# DETECTOR LOOP
# ============================================================

def detector_loop():

    global detector_running

    print()
    print("Starting human detection thread...")
    print("Serial port:", PORT)
    print("Baud rate:", BAUDRATE)
    print()

    frames = []

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOWS
    )

    try:

        reader = CSISerialReader(
            port=PORT,
            baudrate=BAUDRATE,
            timeout=2.0
        )

        with reader:

            while detector_running:

                frame = reader.read_frame()

                if frame is None:
                    continue

                # ------------------------------------------------
                # Accept ONLY 64-subcarrier frames
                # ------------------------------------------------

                if (
                    frame.complex_csi.size
                    != EXPECTED_SUBCARRIERS
                ):

                    print(
                        f"[WARNING] Ignoring CSI frame "
                        f"with "
                        f"{frame.complex_csi.size} "
                        f"subcarriers"
                    )

                    continue

                frames.append(frame)

                # ------------------------------------------------
                # Wait for 50 valid frames
                # ------------------------------------------------

                if len(frames) < WINDOW_SIZE:
                    continue

                # ------------------------------------------------
                # EXACT SAME PREDICTION PIPELINE
                # ------------------------------------------------

                result = predict_window(
                    frames
                )

                # Clear AFTER prediction
                frames.clear()

                if result is None:
                    continue

                # ------------------------------------------------
                # Raw prediction
                # ------------------------------------------------

                raw_prediction = result[
                    "prediction"
                ]

                present_probability = result[
                    "present_probability"
                ]

                # ------------------------------------------------
                # Temporal smoothing
                # ------------------------------------------------

                if (
                    raw_prediction == "present"
                    and present_probability >= PRESENT_CONFIDENCE
                ):

                    prediction_history.append("present")

                else:

                    prediction_history.append("empty")


                present_count = prediction_history.count("present")
                empty_count = prediction_history.count("empty")


                # ------------------------------------------------
                # Hysteresis state machine
                # ------------------------------------------------

                previous_state = latest_status["state"]

                if previous_state == "PRESENT":

                    # Stay PRESENT until 3 confident EMPTY
                    # windows are observed.
                    if empty_count >= PRESENT_REQUIRED:
                        final_state = "EMPTY"
                    else:
                        final_state = "PRESENT"

                else:

                    # Change EMPTY -> PRESENT only after
                    # 3 confident PRESENT windows.
                    if present_count >= PRESENT_REQUIRED:
                        final_state = "PRESENT"
                    else:
                        final_state = "EMPTY"

                # ------------------------------------------------
                # Update API
                # ------------------------------------------------

                update_status(
                    result,
                    final_state
                )

                # ------------------------------------------------
                # Terminal output
                # ------------------------------------------------

                print(
                    f"[DETECTOR] "
                    f"Raw={raw_prediction.upper()} | "
                    f"Final={final_state} | "
                    f"Present="
                    f"{present_probability * 100:.1f}% | "
                    f"Rate="
                    f"{result['sample_rate']:.1f} Hz"
                )

    except Exception as exc:

        print(
            f"[ERROR] Detector stopped: {exc}"
        )

    finally:

        detector_running = False

        print(
            "[DETECTOR] Detection thread stopped."
        )


# ============================================================
# START DETECTOR
# ============================================================

def start_detector():

    global detector_thread
    global detector_running

    if detector_running:
        return False

    detector_running = True

    with status_lock:

        latest_status["state"] = "STARTING"
        latest_status["raw_prediction"] = "UNKNOWN"
        latest_status["timestamp"] = time.time()

    detector_thread = threading.Thread(
        target=detector_loop,
        daemon=True
    )

    detector_thread.start()

    return True


# ============================================================
# STOP DETECTOR
# ============================================================

def stop_detector():

    global detector_running

    detector_running = False

    with status_lock:

        latest_status["state"] = "STOPPED"
        latest_status["raw_prediction"] = "UNKNOWN"
        latest_status["timestamp"] = time.time()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# START API
# ============================================================

@app.route(
    "/api/detection/start"
)
def api_start():

    started = start_detector()

    if started:

        return jsonify({
            "status": "success",
            "message": "Human detection started"
        })

    return jsonify({
        "status": "already_running",
        "message": "Human detection is already running"
    })


# ============================================================
# STOP API
# ============================================================

@app.route(
    "/api/detection/stop"
)
def api_stop():

    stop_detector()

    return jsonify({
        "status": "success",
        "message": "Human detection stopped"
    })


# ============================================================
# STATUS API
# ============================================================

@app.route(
    "/api/detection/status"
)
def api_status():

    with status_lock:

        return jsonify(
            latest_status
        )


# ============================================================
# OLD ROUTES
# ============================================================

@app.route("/start")
def start():

    return api_start()


@app.route("/stop")
def stop():

    return api_stop()


@app.route("/status")
def status():

    return api_status()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )