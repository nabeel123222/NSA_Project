import joblib
import numpy as np
from pathlib import Path

from features import extract_feature_vector


MODEL_PATH = Path("model/calibrated_model.joblib")

TEST_SAMPLES = [
    ("EMPTY", Path("dataset/empty/empty_00003.npy")),
    ("PRESENT", Path("dataset/present/present_00003.npy")),
]


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

print("=" * 60)
print("RANDOM FOREST MODEL TEST")
print("=" * 60)

data = joblib.load(MODEL_PATH)

model = data["model"]
feature_count = data["feature_count"]
classes = data["classes"]

print("Model loaded successfully.")
print("Expected features:", feature_count)
print("Classes:", classes)


# ---------------------------------------------------------
# Test samples
# ---------------------------------------------------------

for actual_label, file_path in TEST_SAMPLES:

    print("\n" + "-" * 60)
    print("File  :", file_path)
    print("Actual:", actual_label)

    if not file_path.exists():
        print("ERROR: File not found.")
        continue

    amp_matrix = np.load(file_path)

    print("Shape :", amp_matrix.shape)

    # Use the sample's recorded rate if available.
    # For this first test, use the actual approximate collection rate
    # stored in metadata.csv.

    if actual_label == "EMPTY":
        metadata_path = Path("dataset/empty/metadata.csv")
    else:
        metadata_path = Path("dataset/present/metadata.csv")

    import pandas as pd

    metadata = pd.read_csv(metadata_path)

    row = metadata[metadata["filename"] == file_path.name]

    if len(row) == 0:
        print("ERROR: Metadata not found.")
        continue

    sample_rate = float(row.iloc[0]["rate_hz"])

    print("Rate  :", sample_rate, "Hz")

    # Feature extraction
    feature_vector = extract_feature_vector(
        amp_matrix,
        sample_rate_hz=sample_rate,
    )

    print("Features:", len(feature_vector))

    # Prediction
    prediction = model.predict(feature_vector.reshape(1, -1))[0]

    probabilities = model.predict_proba(
        feature_vector.reshape(1, -1)
    )[0]

    predicted_label = "EMPTY" if prediction == 0 else "PRESENT"

    print("Predicted:", predicted_label)

    print(
        f"Confidence - EMPTY: {probabilities[0] * 100:.2f}% | "
        f"PRESENT: {probabilities[1] * 100:.2f}%"
    )

    if predicted_label == actual_label:
        print("Result: CORRECT ✅")
    else:
        print("Result: WRONG ❌")


print("\n" + "=" * 60)
print("MODEL TEST COMPLETED")
print("=" * 60)
