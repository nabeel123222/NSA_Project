from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import extract_feature_vector


MODEL_PATH = Path("model/human_detector.joblib")
DATASET_DIR = Path("dataset")

TEST_SETUP = "setup_C"

LABELS = {
    "empty": 0,
    "present": 1,
}


print("=" * 60)
print("UNSEEN SETUP TEST")
print("=" * 60)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

print("Model loaded:", MODEL_PATH)
print("Testing setup:", TEST_SETUP)

# ---------------------------------------------------------
# Load ONLY Setup C
# ---------------------------------------------------------

X_test = []
y_test = []

for folder_name, label in LABELS.items():

    folder = DATASET_DIR / TEST_SETUP / folder_name
    metadata_path = folder / "metadata.csv"

    metadata = pd.read_csv(metadata_path)

    print(f"\nLoading {folder_name.upper()}...")
    print("Metadata rows:", len(metadata))

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
                print(
                    f"WARNING: Skipping {filename} "
                    f"shape={amp_matrix.shape}"
                )
                continue

            if not np.isfinite(amp_matrix).all():
                print(
                    f"WARNING: Skipping {filename} "
                    "because of NaN/Inf"
                )
                continue

            if np.count_nonzero(amp_matrix) == 0:
                print(
                    f"WARNING: Skipping {filename} "
                    "because it is all zero"
                )
                continue

            features = extract_feature_vector(
                amp_matrix,
                sample_rate_hz=sample_rate,
            )

            X_test.append(features)
            y_test.append(label)

            loaded += 1

        except Exception as e:
            print(
                f"WARNING: Failed to process "
                f"{filename}: {e}"
            )

    print("Loaded:", loaded)


# ---------------------------------------------------------
# Convert to arrays
# ---------------------------------------------------------

X_test = np.asarray(X_test, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.int64)

print("\n" + "=" * 60)
print("UNSEEN SETUP DATASET")
print("=" * 60)

print("X shape:", X_test.shape)
print("y shape:", y_test.shape)

print("EMPTY samples:", np.sum(y_test == 0))
print("PRESENT samples:", np.sum(y_test == 1))


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("UNSEEN SETUP EVALUATION")
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
print("UNSEEN SETUP TEST COMPLETED")
print("=" * 60)
