from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from features import extract_feature_vector


DATASET_DIR = Path("dataset")

TRAIN_SETUPS = ["setup_A", "setup_B"]
TEST_SETUP = "setup_C"

LABELS = {
    "empty": 0,
    "present": 1,
}


def load_setups(setups):
    X = []
    y = []

    for setup in setups:

        print("\n" + "-" * 60)
        print(f"Loading {setup.upper()}")
        print("-" * 60)

        for folder_name, label in LABELS.items():

            folder = DATASET_DIR / setup / folder_name
            metadata_path = folder / "metadata.csv"

            metadata = pd.read_csv(metadata_path)

            print(f"\n{setup} - {folder_name.upper()}")
            print(f"Metadata rows: {len(metadata)}")

            loaded = 0

            for _, row in metadata.iterrows():

                filename = row["filename"]
                sample_rate = float(row["rate_hz"])

                file_path = folder / filename

                if not file_path.exists():
                    print("WARNING: Missing:", file_path)
                    continue

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
                        sample_rate_hz=sample_rate,
                    )

                    X.append(features)
                    y.append(label)

                    loaded += 1

                except Exception as e:
                    print(
                        f"WARNING: Failed {filename}: {e}"
                    )

            print(f"Loaded: {loaded}")

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
    )


print("=" * 60)
print("TRUE GENERALIZATION TEST")
print("=" * 60)

print("\nTraining setups:", ", ".join(TRAIN_SETUPS))
print("Testing setup :", TEST_SETUP)

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

X_train, y_train = load_setups(TRAIN_SETUPS)

print("\nTraining dataset:")
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)
print("EMPTY:", np.sum(y_train == 0))
print("PRESENT:", np.sum(y_train == 1))

# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test, y_test = load_setups([TEST_SETUP])

print("\nTest dataset:")
print("X shape:", X_test.shape)
print("y shape:", y_test.shape)
print("EMPTY:", np.sum(y_test == 0))
print("PRESENT:", np.sum(y_test == 1))

# ---------------------------------------------------------
# Train separate Random Forest
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING A+B MODEL")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training completed.")

# ---------------------------------------------------------
# Test ONLY on Setup C
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TESTING ONLY ON SETUP C")
print("=" * 60)

y_pred = model.predict(X_test)

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

print("\n" + "=" * 60)
print("TRUE GENERALIZATION TEST COMPLETED")
print("=" * 60)

print("\nIMPORTANT:")
print("This test model was trained ONLY on Setup A + Setup B.")
print("Setup C was used ONLY for testing.")
print("The saved human_detector.joblib was NOT changed.")
