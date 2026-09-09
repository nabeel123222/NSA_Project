"""
train_calibration.py
--------------------
Train a Random Forest specifically for the
current room/setup using local calibration data.

Dataset:
    calibration/current_room/empty/
    calibration/current_room/present/

Classes:
    0 = EMPTY
    1 = PRESENT
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from features import extract_feature_vector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CALIBRATION_DIR = Path("calibration")

SETUPS = [
    "set_A",
    "set_B",
    "set_C",
]

MODEL_DIR = Path("model")

MODEL_PATH = MODEL_DIR / "calibrated_model.joblib"

LABELS = {
    "empty": 0,
    "present": 1,
}

WINDOW_SIZE = 50
EXPECTED_SUBCARRIERS = 64


# ---------------------------------------------------------
# Load calibration dataset
# ---------------------------------------------------------

X = []
y = []

print("=" * 60)
print("LOCAL CALIBRATION - RANDOM FOREST TRAINING")
print("=" * 60)

for setup_name in SETUPS:

    print("\n" + "-" * 60)
    print(f"Loading {setup_name.upper()}")
    print("-" * 60)

    for folder_name, label in LABELS.items():

        folder = (
            CALIBRATION_DIR
            / setup_name
            / folder_name
        )

        if not folder.exists():
            raise FileNotFoundError(
                f"Calibration folder not found: {folder}"
            )

        files = sorted(folder.glob("*.npy"))

        print(f"\n{setup_name} - {folder_name.upper()}")
        print("Files found:", len(files))

        loaded = 0

        for file_path in files:

            try:

                amp_matrix = np.load(file_path)

                if amp_matrix.shape != (
                    WINDOW_SIZE,
                    EXPECTED_SUBCARRIERS,
                ):
                    print(
                        f"WARNING: Skipping {file_path.name} "
                        f"because shape is {amp_matrix.shape}"
                    )
                    continue

                if not np.isfinite(amp_matrix).all():
                    print(
                        f"WARNING: Skipping {file_path.name} "
                        f"because it contains NaN/Inf"
                    )
                    continue

                if np.count_nonzero(amp_matrix) == 0:
                    print(
                        f"WARNING: Skipping {file_path.name} "
                        f"because it is all zero"
                    )
                    continue

                feature_vector = extract_feature_vector(
                    amp_matrix
                )

                X.append(feature_vector)
                y.append(label)

                loaded += 1

            except Exception as e:

                print(
                    f"WARNING: Failed to process "
                    f"{file_path.name}: {e}"
                )

        print("Loaded:", loaded)

    

# ---------------------------------------------------------
# Convert to NumPy arrays
# ---------------------------------------------------------

X = np.asarray(X, dtype=np.float64)

y = np.asarray(y, dtype=np.int64)

print("\n" + "=" * 60)
print("CALIBRATION FEATURE DATASET")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("EMPTY samples:", np.sum(y == 0))
print("PRESENT samples:", np.sum(y == 1))

if len(X) == 0:
    raise RuntimeError(
        "No calibration samples were loaded."
    )

if np.isnan(X).any() or np.isinf(X).any():
    raise RuntimeError(
        "Feature matrix contains NaN or Inf values."
    )

if np.sum(y == 0) < 2 or np.sum(y == 1) < 2:
    raise RuntimeError(
        "Need at least two samples from each class."
    )


# ---------------------------------------------------------
# Train / test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

print("\nTraining local Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)

print("Training completed.")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print("\n" + "=" * 60)
print("LOCAL CALIBRATION MODEL EVALUATION")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "EMPTY",
            "PRESENT",
        ],
        digits=4,
    )
)


# ---------------------------------------------------------
# Save calibrated model
# ---------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "feature_count": X.shape[1],
        "classes": LABELS,
        "model_type": "local_calibration",
    },
    MODEL_PATH,
)

print("\n" + "=" * 60)
print("CALIBRATED MODEL SAVED")
print("=" * 60)

print("Model:", MODEL_PATH)
print("Features:", X.shape[1])
print("Trees:", model.n_estimators)

print("\nLocal calibration training finished successfully.")
