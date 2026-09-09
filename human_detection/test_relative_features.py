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
TEST_SETUP = "setup_C"

LABELS = {
    "empty": 0,
    "present": 1,
}


def normalize_window(amp_matrix):
    """
    Convert absolute CSI amplitudes into relative CSI.

    Each subcarrier is normalized by its mean amplitude
    inside the current window.
    """

    baseline = np.mean(amp_matrix, axis=0)

    baseline = np.maximum(baseline, 1e-6)

    relative = amp_matrix / baseline

    return relative


def load_features(setups):
    X = []
    y = []

    for setup in setups:

        print("\n" + "-" * 60)
        print(f"Loading {setup.upper()}")
        print("-" * 60)

        for label_name, label in LABELS.items():

            folder = DATASET_DIR / setup / label_name
            files = sorted(folder.glob("*.npy"))

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

                    relative_matrix = normalize_window(
                        amp_matrix
                    )

                    vector = extract_feature_vector(
                        relative_matrix,
                        sample_rate_hz=180.0,
                    )

                    X.append(vector)
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
print("RELATIVE CSI FEATURE EXPERIMENT")
print("=" * 70)

print("\nTraining setups:", TRAIN_SETUPS)
print("Testing setup :", TEST_SETUP)

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING DATA")
print("=" * 70)

X_train, y_train = load_features(TRAIN_SETUPS)

print("\nTraining shape:", X_train.shape)
print("EMPTY:", np.sum(y_train == 0))
print("PRESENT:", np.sum(y_train == 1))

# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEST DATA")
print("=" * 70)

X_test, y_test = load_features([TEST_SETUP])

print("\nTest shape:", X_test.shape)
print("EMPTY:", np.sum(y_test == 0))
print("PRESENT:", np.sum(y_test == 1))

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING RELATIVE-FEATURE RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training completed.")

# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TESTING ONLY ON SETUP C")
print("=" * 70)

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

print("\n" + "=" * 70)
print("RELATIVE FEATURE EXPERIMENT COMPLETED")
print("=" * 70)

print("\nIMPORTANT:")
print("This is an experimental model only.")
print("The existing human_detector.joblib was NOT changed.")
