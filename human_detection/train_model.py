"""
train_model.py
--------------
Train a Random Forest classifier for CSI-based human detection.

Classes:
    0 = EMPTY
    1 = PRESENT

Dataset:
    dataset/empty/
    dataset/present/

Each .npy file contains:
    (50, 64) CSI amplitude matrix

The actual CSI sampling rate is read from metadata.csv.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_feature_vector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "human_detector.joblib"

LABELS = {
    "empty": 0,
    "present": 1,
}

SETUPS = [
    "setup_A",
    "setup_B",
    "setup_C",
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

X = []
y = []

print("=" * 60)
print("RANDOM FOREST HUMAN DETECTION - TRAINING")
print("=" * 60)

for setup_name in SETUPS:

    print("\n" + "-" * 60)
    print(f"Loading {setup_name.upper()}...")
    print("-" * 60)

    for folder_name, label in LABELS.items():

        folder = DATASET_DIR / setup_name / folder_name
        metadata_path = folder / "metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        metadata = pd.read_csv(metadata_path)

        print(f"\n{setup_name} - {folder_name.upper()}")
        print(f"Metadata rows: {len(metadata)}")

        loaded = 0

        for _, row in metadata.iterrows():

            filename = row["filename"]
            sample_rate = float(row["rate_hz"])

            file_path = folder / filename

            if not file_path.exists():
                print(f"WARNING: Missing file: {file_path}")
                continue

            try:
                amp_matrix = np.load(file_path)

                if amp_matrix.shape != (50, 64):
                    print(
                        f"WARNING: Skipping {filename} "
                        f"because shape is {amp_matrix.shape}"
                    )
                    continue

                if not np.isfinite(amp_matrix).all():
                    print(
                        f"WARNING: Skipping {filename} "
                        f"because it contains NaN/Inf"
                    )
                    continue

                if np.count_nonzero(amp_matrix) == 0:
                    print(
                        f"WARNING: Skipping {filename} "
                        f"because it is all zero"
                    )
                    continue

                feature_vector = extract_feature_vector(
                    amp_matrix,
                    sample_rate_hz=sample_rate,
                )

                X.append(feature_vector)
                y.append(label)

                loaded += 1

            except Exception as e:
                print(
                    f"WARNING: Failed to process "
                    f"{filename}: {e}"
                )

        print(f"Loaded: {loaded}")

# ---------------------------------------------------------
# Convert to NumPy arrays
# ---------------------------------------------------------

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.int64)

print("\n" + "=" * 60)
print("FEATURE DATASET")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("EMPTY samples:", np.sum(y == 0))
print("PRESENT samples:", np.sum(y == 1))

if len(X) == 0:
    raise RuntimeError("No training samples were loaded.")

if np.isnan(X).any() or np.isinf(X).any():
    raise RuntimeError("Feature matrix contains NaN or Inf values.")


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

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training completed.")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

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


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_count": X.shape[1],
        "classes": LABELS,
    },
    MODEL_PATH,
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Model:", MODEL_PATH)
print("Features:", X.shape[1])
print("Trees:", model.n_estimators)

print("\nRandom Forest training finished successfully.")
