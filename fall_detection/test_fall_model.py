import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from fall_features import extract_two_receiver_fall_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "fall_detector_90samples.joblib"

NORMAL_DIR = BASE_DIR / "dataset" / "test" / "normal"
FALL_DIR = BASE_DIR / "dataset" / "test" / "fall"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("NSA - FALL DETECTION")
print("UNSEEN TEST DATA EVALUATION")
print("=" * 70)

print("\nLoading model:")
print(MODEL_PATH)

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]

print("Model loaded successfully.")


# ============================================================
# LOAD ONE SAMPLE
# ============================================================

def load_sample(file_path):
    data = np.load(file_path)

    friend_csi = data["friend_csi"]
    you_csi = data["you_csi"]

    friend_rate = float(data["friend_rate"])
    you_rate = float(data["you_rate"])

    features = extract_two_receiver_fall_features(
        friend_csi,
        you_csi,
        friend_rate,
        you_rate
    )

    return features


# ============================================================
# TEST DATA
# ============================================================

X = []
y = []
file_names = []

print("\n" + "=" * 70)
print("TESTING UNSEEN SAMPLES")
print("=" * 70)


# NORMAL samples
normal_files = sorted(NORMAL_DIR.glob("normal_*.npz"))

print(f"\nNORMAL test samples: {len(normal_files)}")

for file_path in normal_files:
    features = load_sample(file_path)

    X.append(features)
    y.append("NORMAL")
    file_names.append(file_path.name)


# FALL samples
fall_files = sorted(FALL_DIR.glob("fall_*.npz"))

print(f"FALL test samples: {len(fall_files)}")

for file_path in fall_files:
    features = load_sample(file_path)

    X.append(features)
    y.append("FALL")
    file_names.append(file_path.name)


X = np.asarray(X)
y = np.asarray(y)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

predictions = model.predict(X)

for name, actual, predicted in zip(file_names, y, predictions):

    status = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"{name:15s} | "
        f"Actual: {actual:6s} | "
        f"Predicted: {predicted:6s} | "
        f"{status}"
    )


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(y, predictions)

cm = confusion_matrix(
    y,
    predictions,
    labels=["NORMAL", "FALL"]
)

print("\n" + "=" * 70)
print("UNSEEN TEST RESULTS")
print("=" * 70)

print(f"\nTotal test samples : {len(y)}")
print(f"NORMAL samples     : {len(normal_files)}")
print(f"FALL samples       : {len(fall_files)}")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nConfusion matrix:")
print("                 Predicted")
print("                 NORMAL  FALL")
print(
    f"Actual NORMAL   {cm[0,0]:6d} {cm[0,1]:5d}"
)
print(
    f"Actual FALL     {cm[1,0]:6d} {cm[1,1]:5d}"
)

print("\nClassification report:")

print(
    classification_report(
        y,
        predictions,
        labels=["NORMAL", "FALL"],
        zero_division=0
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

normal_correct = cm[0, 0]
normal_wrong = cm[0, 1]

fall_correct = cm[1, 1]
fall_wrong = cm[1, 0]

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nNORMAL correctly detected : "
    f"{normal_correct}/{len(normal_files)}"
)

print(
    f"FALL correctly detected   : "
    f"{fall_correct}/{len(fall_files)}"
)

print(
    f"False FALL detections     : "
    f"{normal_wrong}"
)

print(
    f"Missed FALL detections    : "
    f"{fall_wrong}"
)

print("\nThis test uses samples that were NOT used during training.")
print("=" * 70)
