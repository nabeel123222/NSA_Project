from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from features import extract_feature_vector


DATASET_DIR = Path("dataset")

TRAIN_SETUPS = ["setup_A", "setup_B"]
CALIBRATION_SETUP = "setup_C"

CALIBRATION_EMPTY = 10

LABELS = {
    "empty": 0,
    "present": 1,
}


def load_data(setups, limit_per_class=None):
    X = []
    y = []

    for setup in setups:

        print("\n" + "-" * 60)
        print(f"Loading {setup.upper()}")
        print("-" * 60)

        for label_name, label in LABELS.items():

            folder = DATASET_DIR / setup / label_name
            files = sorted(folder.glob("*.npy"))

            if limit_per_class is not None:
                files = files[:limit_per_class]

            loaded = 0

            for file_path in files:

                try:
                    amp_matrix = np.load(file_path)

                    if amp_matrix.shape != (50, 64):
                        continue

                    if not np.isfinite(amp_matrix).all():
                        continue

                    if np.count_nonzero(amp_matrix) == 0:
                        continue

                    features = extract_feature_vector(
                        amp_matrix,
                        sample_rate_hz=180.0,
                    )

                    X.append(features)
                    y.append(label)

                    loaded += 1

                except Exception as e:
                    print(
                        f"WARNING: {file_path.name}: {e}"
                    )

            print(
                f"{setup} - {label_name.upper()}: "
                f"{loaded} samples"
            )

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
    )


print("=" * 70)
print("CALIBRATION EXPERIMENT")
print("=" * 70)

# ---------------------------------------------------------
# Train general model using A + B
# ---------------------------------------------------------

print("\nTRAINING GENERAL MODEL USING A + B")

X_train, y_train = load_data(TRAIN_SETUPS)

print("\nTraining samples:", len(X_train))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("General model trained.")

# ---------------------------------------------------------
# Load Setup C
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING SETUP C")
print("=" * 70)

empty_folder = DATASET_DIR / CALIBRATION_SETUP / "empty"
present_folder = DATASET_DIR / CALIBRATION_SETUP / "present"

empty_files = sorted(empty_folder.glob("*.npy"))
present_files = sorted(present_folder.glob("*.npy"))

# First 10 EMPTY samples = calibration
calibration_files = empty_files[:CALIBRATION_EMPTY]

# Remaining EMPTY samples = test
test_empty_files = empty_files[CALIBRATION_EMPTY:]

# All PRESENT samples = test
test_present_files = present_files

print("\nCalibration EMPTY samples:", len(calibration_files))
print("Test EMPTY samples:", len(test_empty_files))
print("Test PRESENT samples:", len(test_present_files))

# ---------------------------------------------------------
# Build calibration baseline
# ---------------------------------------------------------

calibration_features = []

for file_path in calibration_files:

    amp_matrix = np.load(file_path)

    features = extract_feature_vector(
        amp_matrix,
        sample_rate_hz=180.0,
    )

    calibration_features.append(features)

calibration_features = np.asarray(
    calibration_features,
    dtype=np.float64,
)

baseline = np.mean(
    calibration_features,
    axis=0,
)

print("\nCalibration baseline created.")
print("Baseline feature count:", len(baseline))

# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

X_test = []
y_test = []

for file_path in test_empty_files:

    amp_matrix = np.load(file_path)

    features = extract_feature_vector(
        amp_matrix,
        sample_rate_hz=180.0,
    )

    X_test.append(features)
    y_test.append(0)


for file_path in test_present_files:

    amp_matrix = np.load(file_path)

    features = extract_feature_vector(
        amp_matrix,
        sample_rate_hz=180.0,
    )

    X_test.append(features)
    y_test.append(1)


X_test = np.asarray(X_test, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.int64)

# ---------------------------------------------------------
# Apply calibration correction
# ---------------------------------------------------------

X_test_calibrated = X_test - baseline

# Training data must receive the same transformation.
# Since the general model was trained on absolute features,
# we train a second experimental model using differences
# from the A+B baseline.

general_baseline = np.mean(
    X_train[y_train == 0],
    axis=0,
)

X_train_calibrated = X_train - general_baseline

calibrated_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

calibrated_model.fit(
    X_train_calibrated,
    y_train,
)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CALIBRATED EVALUATION")
print("=" * 70)

y_pred = calibrated_model.predict(X_test_calibrated)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["EMPTY", "PRESENT"],
        digits=4,
    )
)

print("\n" + "=" * 70)
print("CALIBRATION EXPERIMENT COMPLETED")
print("=" * 70)

print("\nIMPORTANT:")
print("This experiment did not modify the dataset.")
print("This experiment did not modify human_detector.joblib.")
