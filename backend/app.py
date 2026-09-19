import sys
import os



from flask import Flask, jsonify, render_template, request
import threading
import time
from collections import deque
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
from fall_detection.fall_features import (
    extract_two_receiver_fall_features
)
from collection_service import (
    start_collection,
    stop_collection,
    get_collection_status,
    archive_current_dataset,
    get_dataset_counts
)
from training_service import (
    start_training,
    get_training_status
)

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

FALL_DETECTION_DIR = os.path.join(
    PROJECT_ROOT,
    "fall_detection"
)

FALL_MODEL_PATH = os.path.join(
    FALL_DETECTION_DIR,
    "model",
    "fall_detector_90samples.joblib"
)

# FRIEND_PORT = (
#     "/dev/serial/by-id/"
#     "usb-Espressif_USB_JTAG_serial_debug_unit_94:A9:90:D2:EF:BC-if00"
# )

# YOUR_PORT = (
#     "/dev/serial/by-id/"
#     "usb-1a86_USB_Single_Serial_5B8E073970-if00"
# )

# FALL_FRIEND_PORT = (
#     "/dev/serial/by-id/"
#     "usb-Espressif_USB_JTAG_serial_debug_unit_94:A9:90:D2:EF:BC-if00"
# )

# FALL_YOUR_PORT = (
#     "/dev/serial/by-id/"
#     "usb-1a86_USB_Single_Serial_5B8E073970-if00"
# )

# ============================================================
# FALL DETECTION SERIAL PORTS
# ============================================================

FALL_FRIEND_PORT = (
    "/dev/serial/by-id/"
    "usb-Espressif_USB_JTAG_serial_debug_unit_94:A9:90:D2:EF:BC-if00"
)

FALL_YOUR_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5B8E073970-if00"
)


# ============================================================
# HUMAN DETECTION SERIAL PORTS
# ============================================================

FRIEND_PORT = "/dev/ttyACM0"
YOUR_PORT = "/dev/ttyACM1"

FALL_BAUDRATE = 921600
FALL_WINDOW_SECONDS = 4.0
FALL_THRESHOLD = 0.70

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
    "ui_two_esp_human_detector.joblib"
)

FRIEND_PORT = "/dev/ttyACM0"
YOUR_PORT = "/dev/ttyACM1"

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

# ============================================================
# FALL DETECTION MODEL
# ============================================================

fall_model_data = joblib.load(FALL_MODEL_PATH)

fall_model = fall_model_data["model"]

# ============================================================
# FALL DETECTION RUNTIME STATE
# ============================================================

fall_status = {
    "status": "OFFLINE",
    "fall_probability": 0.0,
    "normal_probability": 1.0,
    "friend_rate": 0.0,
    "you_rate": 0.0,
    "frames_friend": 0,
    "frames_you": 0,
    "last_update": None
}

fall_lock = threading.Lock()

# ============================================================
# TEMPORARY ALERT HISTORY
# ============================================================

alert_history = []
alert_lock = threading.Lock()
previous_fall_state = "NORMAL"

print("=" * 60)
print("NSA FALL DETECTION MODEL")
print("=" * 60)
print("Fall model loaded:")
print(FALL_MODEL_PATH)
print("Fall model object:", type(fall_model).__name__)
print("Fall model classes:", fall_model.classes_)
print("=" * 60)


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

    "fall_state": "NORMAL",
    "fall_probability": 0.0,
    "normal_probability": 1.0,
    "friend_rate": 0.0,
    "you_rate": 0.0,

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
# PREDICT ONE TWO-ESP32 WINDOW
# ============================================================

def predict_window(friend_frames, your_frames):

    if (
        len(friend_frames) != WINDOW_SIZE
        or
        len(your_frames) != WINDOW_SIZE
    ):
        return None

    # --------------------------------------------------------
    # Convert both ESP32 streams to amplitude matrices
    # --------------------------------------------------------

    friend_amp = frames_to_amplitude_matrix(
        friend_frames
    )

    your_amp = frames_to_amplitude_matrix(
        your_frames
    )

    # --------------------------------------------------------
    # Check shapes
    # --------------------------------------------------------

    expected_shape = (
        WINDOW_SIZE,
        EXPECTED_SUBCARRIERS
    )

    if friend_amp.shape != expected_shape:
        print(
            "[WARNING] Friend CSI shape:",
            friend_amp.shape
        )
        return None

    if your_amp.shape != expected_shape:
        print(
            "[WARNING] Your CSI shape:",
            your_amp.shape
        )
        return None

    # --------------------------------------------------------
    # Check values
    # --------------------------------------------------------

    if not np.isfinite(friend_amp).all():
        print(
            "[WARNING] Friend CSI contains invalid values."
        )
        return None

    if not np.isfinite(your_amp).all():
        print(
            "[WARNING] Your CSI contains invalid values."
        )
        return None

    if np.all(friend_amp == 0):
        print(
            "[WARNING] Friend CSI window is all zeros."
        )
        return None

    if np.all(your_amp == 0):
        print(
            "[WARNING] Your CSI window is all zeros."
        )
        return None

    # --------------------------------------------------------
    # Calculate sample rates
    # --------------------------------------------------------

    friend_rate = calculate_sample_rate(
        friend_frames
    )

    your_rate = calculate_sample_rate(
        your_frames
    )

    if friend_rate <= 0:
        friend_rate = 180.0

    if your_rate <= 0:
        your_rate = 180.0

    # --------------------------------------------------------
    # Extract 258 features from each ESP32
    # --------------------------------------------------------

    try:

        friend_features = extract_feature_vector(
            friend_amp,
            sample_rate_hz=friend_rate
        )

        your_features = extract_feature_vector(
            your_amp,
            sample_rate_hz=your_rate
        )

    except Exception as exc:

        print(
            f"[WARNING] Feature extraction failed: {exc}"
        )

        return None

    # --------------------------------------------------------
    # Verify feature counts
    # --------------------------------------------------------

    if len(friend_features) != 258:
        print(
            "[WARNING] Friend feature count:",
            len(friend_features)
        )
        return None

    if len(your_features) != 258:
        print(
            "[WARNING] Your feature count:",
            len(your_features)
        )
        return None

    # --------------------------------------------------------
    # Combine:
    #
    # Friend = 258
    # You    = 258
    # ----------------
    # Total  = 516
    # --------------------------------------------------------

    feature_vector = np.concatenate(
        [
            friend_features,
            your_features
        ]
    )

    if len(feature_vector) != 516:
        print(
            "[WARNING] Combined feature count:",
            len(feature_vector)
        )
        return None

    if not np.isfinite(feature_vector).all():
        print(
            "[WARNING] Combined features contain invalid values."
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

    # --------------------------------------------------------
    # Model:
    # 0 = empty
    # 1 = present
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": raw_prediction,
        "confidence": confidence,
        "empty_probability": empty_probability,
        "present_probability": present_probability,
        "sample_rate": (
            friend_rate + your_rate
        ) / 2.0
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
# TWO-ESP32 DETECTOR LOOP
# ============================================================

def detector_loop():

    global detector_running

    print()
    print("Starting two-ESP32 human detection thread...")
    print("Friend serial port:", FRIEND_PORT)
    print("Your serial port   :", YOUR_PORT)
    print("Baud rate          :", BAUDRATE)
    print()

    friend_frames = []
    your_frames = []

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOWS
    )

    try:

        # ----------------------------------------------------
        # Open BOTH readers once
        # ----------------------------------------------------

        friend_reader = CSISerialReader(
            port=FRIEND_PORT,
            baudrate=BAUDRATE,
            timeout=2.0
        )

        your_reader = CSISerialReader(
            port=YOUR_PORT,
            baudrate=BAUDRATE,
            timeout=2.0
        )

        friend_reader.open()
        your_reader.open()

        try:

            while detector_running:

                # ------------------------------------------------
                # Read one frame from each ESP32
                # ------------------------------------------------

                friend_frame = friend_reader.read_frame()

                if (
                    friend_frame is not None
                    and
                    friend_frame.complex_csi.size
                    == EXPECTED_SUBCARRIERS
                ):

                    friend_frames.append(
                        friend_frame
                    )

                your_frame = your_reader.read_frame()

                if (
                    your_frame is not None
                    and
                    your_frame.complex_csi.size
                    == EXPECTED_SUBCARRIERS
                ):

                    your_frames.append(
                        your_frame
                    )

                # ------------------------------------------------
                # Wait for 50 valid frames from BOTH
                # ------------------------------------------------

                if (
                    len(friend_frames) < WINDOW_SIZE
                    or
                    len(your_frames) < WINDOW_SIZE
                ):
                    continue

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                result = predict_window(
                    friend_frames,
                    your_frames
                )

                # ------------------------------------------------
                # Clear AFTER prediction
                # ------------------------------------------------

                friend_frames.clear()
                your_frames.clear()

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
                    and
                    present_probability
                    >= PRESENT_CONFIDENCE
                ):

                    prediction_history.append(
                        "present"
                    )

                else:

                    prediction_history.append(
                        "empty"
                    )

                present_count = (
                    prediction_history.count(
                        "present"
                    )
                )

                empty_count = (
                    prediction_history.count(
                        "empty"
                    )
                )

                # ------------------------------------------------
                # Hysteresis state machine
                # ------------------------------------------------

                previous_state = (
                    latest_status["state"]
                )

                if previous_state == "PRESENT":

                    # Stay PRESENT until 3 confident
                    # EMPTY windows are observed.

                    if (
                        empty_count
                        >= PRESENT_REQUIRED
                    ):

                        final_state = "EMPTY"

                    else:

                        final_state = "PRESENT"

                else:

                    # Change EMPTY -> PRESENT only after
                    # 3 confident PRESENT windows.

                    if (
                        present_count
                        >= PRESENT_REQUIRED
                    ):

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

        finally:

            # ----------------------------------------------------
            # Close both serial readers
            # ----------------------------------------------------

            try:
                friend_reader.close()
            except Exception:
                pass

            try:
                your_reader.close()
            except Exception:
                pass

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
# FALL DETECTION
# ============================================================

fall_detector_thread = None
fall_detector_running = False


def calculate_rate(frames):
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

    return float(1.0 / np.median(differences))


def fall_detection_loop():

    global fall_detector_running

    print()
    print("=" * 60)
    print("NSA FALL DETECTION")
    print("=" * 60)
    print("Friend:", FALL_FRIEND_PORT)
    print("You   :", FALL_YOUR_PORT)
    print("=" * 60)

    friend_reader = None
    you_reader = None

    friend_frames = deque(maxlen=1000)
    you_frames = deque(maxlen=1000)

    try:

        friend_reader = CSISerialReader(
            port=FALL_FRIEND_PORT,
            baudrate=FALL_BAUDRATE,
            timeout=2.0
        )

        you_reader = CSISerialReader(
             port=FALL_FRIEND_PORT,
            baudrate=FALL_BAUDRATE,
            timeout=2.0
        )

        friend_reader.open()
        you_reader.open()

        while fall_detector_running:

            friend_frame = friend_reader.read_frame()
            you_frame = you_reader.read_frame()

            if friend_frame is not None:
                if friend_frame.complex_csi.size == 64:
                    friend_frames.append(friend_frame)

            if you_frame is not None:
                if you_frame.complex_csi.size == 64:
                    you_frames.append(you_frame)

            if len(friend_frames) < 30 or len(you_frames) < 30:
                continue

            # ------------------------------------------------
            # 4-second rolling window
            # ------------------------------------------------

            friend_latest_time = friend_frames[-1].timestamp
            you_latest_time = you_frames[-1].timestamp

            friend_window = [
                f for f in friend_frames
                if friend_latest_time - f.timestamp
                <= FALL_WINDOW_SECONDS
            ]

            you_window = [
                f for f in you_frames
                if you_latest_time - f.timestamp
                <= FALL_WINDOW_SECONDS
            ]

            if len(friend_window) < 30:
                continue

            if len(you_window) < 30:
                continue

            friend_csi = np.abs(
                np.array(
                    [f.complex_csi for f in friend_window]
                )
            )

            you_csi = np.abs(
                np.array(
                    [f.complex_csi for f in you_window]
                )
            )

            friend_rate = calculate_rate(friend_window)
            you_rate = calculate_rate(you_window)

            try:

                features = extract_two_receiver_fall_features(
                    friend_csi,
                    you_csi,
                    friend_rate,
                    you_rate
                )

                features_2d = features.reshape(1, -1)

                probabilities = fall_model.predict_proba(
                    features_2d
                )[0]

                classes = fall_model.classes_

                fall_probability = 0.0
                normal_probability = 0.0

                for cls, probability in zip(
                    classes,
                    probabilities
                ):

                    if str(cls).upper() == "FALL":
                        fall_probability = float(probability)

                    elif str(cls).upper() == "NORMAL":
                        normal_probability = float(probability)

                # ------------------------------------------------
                # Final fall decision
                # ------------------------------------------------

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

                if (
                    fall_probability >= FALL_THRESHOLD
                    or strong_fall_pattern
                ):
                    fall_state = "FALL DETECTED"
                else:
                    fall_state = "NORMAL"

                 # ------------------------------------------------
                # Create alert on new fall event
                # ------------------------------------------------

                global previous_fall_state

                if (
                    fall_state == "FALL DETECTED"
                    and previous_fall_state != "FALL DETECTED"
                ):

                    alert_record = {
                        "id": len(alert_history) + 1,
                        "date": time.strftime("%d-%m-%Y"),
                        "time": time.strftime("%H:%M:%S"),
                        "fall_probability": round(
                            fall_probability * 100,
                            1
                        ),
                        "status": "Acknowledgement Pending"
                    }

                    with alert_lock:
                        alert_history.append(alert_record)

                    print(
                        "[ALERT] New fall event recorded:",
                        alert_record,
                        flush=True
                    )

                previous_fall_state = fall_state
                # ------------------------------------------------
                # Update Flask state
                # ------------------------------------------------

                with status_lock:

                    latest_status[
                        "fall_state"
                    ] = fall_state

                    latest_status[
                        "fall_probability"
                    ] = round(
                        fall_probability,
                        4
                    )

                    latest_status[
                        "normal_probability"
                    ] = round(
                        normal_probability,
                        4
                    )

                    latest_status[
                        "friend_rate"
                    ] = round(
                        friend_rate,
                        2
                    )

                    latest_status[
                        "you_rate"
                    ] = round(
                        you_rate,
                        2
                    )

                    latest_status[
                        "timestamp"
                    ] = time.time()

                print(
                    f"[FALL] "
                    f"{fall_state} | "
                    f"Fall={fall_probability * 100:.1f}% | "
                    f"Friend={friend_rate:.1f} Hz | "
                    f"You={you_rate:.1f} Hz",
                    flush=True
                )

            except Exception as exc:

                print(
                    f"[FALL WARNING] {exc}",
                    flush=True
                )

    except Exception as exc:

        print(
            f"[FALL ERROR] {exc}",
            flush=True
        )

    finally:

        fall_detector_running = False

        try:
            if friend_reader is not None:
                friend_reader.close()
        except Exception:
            pass

        try:
            if you_reader is not None:
                you_reader.close()
        except Exception:
            pass

        print("[FALL] Detection thread stopped.")


def start_fall_detector():

    global fall_detector_thread
    global fall_detector_running

    if fall_detector_running:
        return False

    fall_detector_running = True

    fall_detector_thread = threading.Thread(
        target=fall_detection_loop,
        daemon=True
    )

    fall_detector_thread.start()

    return True


def stop_fall_detector():

    global fall_detector_running

    fall_detector_running = False


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

    # Stop Fall Detection before using Human Detection
    stop_fall_detector()

    return render_template(
        "index.html"
    )

# ============================================================
# FALL DETECTION API
# ============================================================

@app.route("/fall-detection")
def fall_detection_page():

    # Stop Human Detection before starting Fall Detection
    stop_detector()

    # Start Fall Detection
    start_fall_detector()

    return render_template("fall_detection.html")

@app.route("/alerts")
def alerts_page():
    stop_fall_detector()

    today = time.strftime("%d-%m-%Y")

    with alert_lock:
        alerts = list(alert_history)

    today_alerts = sum(
        1
        for alert in alerts
        if alert["date"] == today
    )

    return render_template(
        "alerts.html",
        alerts=alerts,
        today_alerts=today_alerts
    )

@app.route("/api/alerts/acknowledge", methods=["POST"])
def acknowledge_alert():

    with alert_lock:

        if not alert_history:
            return jsonify({
                "success": False,
                "message": "No alerts available"
            }), 404

        # Find the most recent pending alert
        for alert in reversed(alert_history):

            if alert["status"] == "Acknowledgement Pending":

                alert["status"] = "Acknowledged"

                return jsonify({
                    "success": True,
                    "alert_id": alert["id"],
                    "status": "Acknowledged"
                })

        return jsonify({
            "success": False,
            "message": "No pending alert"
        }), 404


@app.route("/api/fall/status")
def fall_detection_status():
    with fall_lock:
        return jsonify(latest_status)

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
# TRAINING DATA COLLECTION API
# ============================================================

@app.route("/api/collection/start", methods=["POST"])
def api_collection_start():

    data = request.get_json(
        silent=True
    ) or {}

    label = data.get(
        "label",
        "EMPTY"
    )

    samples = data.get(
        "samples",
        30
    )

    started, message = start_collection(
        label,
        samples
    )

    if started:
        return jsonify({
            "success": True,
            "message": message
        })

    return jsonify({
        "success": False,
        "message": message
    }), 400


@app.route("/api/collection/stop", methods=["POST"])
def api_collection_stop():

    stop_collection()

    return jsonify({
        "status": "success",
        "message": "Collection stopping."
    })


@app.route("/api/collection/status")
def api_collection_status():

    return jsonify(
        get_collection_status()
    )


# ============================================================
# AI TRAINING API
# ============================================================

@app.route("/api/training/start", methods=["POST"])
def api_training_start():

    started, message = start_training()

    if started:
        return jsonify({
            "success": True,
            "message": message
        })

    return jsonify({
        "success": False,
        "message": message
    }), 400


@app.route("/api/training/status")
def api_training_status():

    return jsonify(
        get_training_status()
    )


# ============================================================
# NEW ROOM / RECALIBRATION API
# ============================================================

@app.route("/api/dataset/new-room", methods=["POST"])
def api_new_room():

    success, message = archive_current_dataset()

    if success:
        return jsonify({
            "success": True,
            "message": message,
            "counts": get_dataset_counts()
        })

    return jsonify({
        "success": False,
        "message": message
    }), 400


@app.route("/api/dataset/counts")
def api_dataset_counts():

    return jsonify(
        get_dataset_counts()
    )
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