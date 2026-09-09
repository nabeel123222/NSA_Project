from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from features import extract_feature_vector


DATASET_DIR = Path("dataset")
SETUP = "setup_C"

LABELS = {
    "empty": 0,
    "present": 1,
}


X = []
y = []

print("=" * 60)
print("SETUP C INTERNAL TEST")
print("=" * 60)

# ---------------------------------------------------------
# Load Setup C
# ---------------------------------------------------------

for label_name, label in LABELS.items():

    folder = DATASET_DIR / SETUP / label_name
    files = sorted(folder.glob("*.npy"))

    print(f"\nLoading {label_name.upper()}...")
    print("Files:", len(files))

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

            feature_vector = extract_feature_vector(
                amp_matrix,
                sample_rate_hz=180.0,
            )

            X.append(feature_vector)
            y.append(label)

            loaded += 1

        except Exception as e:
            print(
                f"WARNING: {file_path.name}: {e}"
            )

    print("Loaded:", loaded)


X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.int64)

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("EMPTY:", np.sum(y == 0))
print("PRESENT:", np.sum(y == 1))


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

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING SETUP C MODEL")
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
# Evaluation
# ---------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("SETUP C INTERNAL EVALUATION")
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

print("\n" + "=" * 60)
print("SETUP C INTERNAL TEST COMPLETED")
print("=" * 60)

print("\nIMPORTANT:")
print("This model was trained and tested only using Setup C.")
print("The existing human_detector.joblib was NOT changed.")
