from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

from fall_features import (
    extract_two_receiver_fall_features,
    FALL_FEATURE_NAMES
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

NORMAL_DIR = DATASET_DIR / "normal_set2"
FALL_DIR_1 = DATASET_DIR / "fall"
FALL_DIR_2 = DATASET_DIR / "fall_set2"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fall_detector_90samples.joblib"


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

def load_dataset(directory, pattern, label):
    samples = []

    for i in range(1, 31):

        file_path = directory / f"{pattern}_{i:03d}.npz"

        if not file_path.exists():
            print(
                f"WARNING: Missing sample {i:03d} "
                f"in {directory}"
            )
            continue

        try:
            features = load_sample(file_path)

            if not np.all(np.isfinite(features)):
                print(
                    f"WARNING: Non-finite features "
                    f"in {file_path}"
                )
                continue

            samples.append((features, label))

        except Exception as e:
            print(
                f"WARNING: Failed {file_path}: {e}"
            )

    return samples


print("=" * 75)
print("NSA - FALL DETECTION")
print("RANDOM FOREST TRAINING - 90 SAMPLE DATASET")
print("=" * 75)

print("\nTraining datasets:")
print(f"  NORMAL   : {NORMAL_DIR}")
print(f"  FALL 1   : {FALL_DIR_1}")
print(f"  FALL 2   : {FALL_DIR_2}")


normal_samples = load_dataset(
    NORMAL_DIR,
    "normal",
    "NORMAL"
)

fall_samples_1 = load_dataset(
    FALL_DIR_1,
    "fall",
    "FALL"
)

fall_samples_2 = load_dataset(
    FALL_DIR_2,
    "fall",
    "FALL"
)


all_samples = (
    normal_samples
    + fall_samples_1
    + fall_samples_2
)


X = np.array(
    [features for features, label in all_samples]
)

y = np.array(
    [label for features, label in all_samples]
)


print("\nDataset:")
print(f"  NORMAL samples : {len(normal_samples)}")
print(f"  FALL Set 1     : {len(fall_samples_1)}")
print(f"  FALL Set 2     : {len(fall_samples_2)}")
print(f"  TOTAL samples  : {len(all_samples)}")


print("\nFeature matrix:")
print(f"  Shape: {X.shape}")
print(f"  Features: {X.shape[1]}")
print(
    f"  Feature names: "
    f"{FALL_FEATURE_NAMES}"
)


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])


print("\n" + "=" * 75)
print("5-FOLD CROSS-VALIDATION")
print("=" * 75)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("\nFold accuracies:")

for i, score in enumerate(scores, start=1):
    print(
        f"  Fold {i}: "
        f"{score * 100:.2f}%"
    )


print(
    f"\nMean accuracy : "
    f"{scores.mean() * 100:.2f}%"
)

print(
    f"Std deviation : "
    f"{scores.std() * 100:.2f}%"
)


print("\n" + "=" * 75)
print("TRAINING FINAL MODEL")
print("=" * 75)


model.fit(X, y)

train_predictions = model.predict(X)

train_accuracy = np.mean(
    train_predictions == y
)


print(
    f"\nTraining accuracy: "
    f"{train_accuracy * 100:.2f}%"
)


cm = confusion_matrix(
    y,
    train_predictions,
    labels=["NORMAL", "FALL"]
)


print("\nConfusion matrix:")
print("                 Predicted")
print("                 NORMAL  FALL")
print(
    f"Actual NORMAL    "
    f"{cm[0, 0]:8d} "
    f"{cm[0, 1]:5d}"
)
print(
    f"Actual FALL      "
    f"{cm[1, 0]:8d} "
    f"{cm[1, 1]:5d}"
)


print("\nClassification report:")

print(
    classification_report(
        y,
        train_predictions,
        labels=["NORMAL", "FALL"],
        zero_division=0
    )
)


print("=" * 75)
print("FEATURE IMPORTANCE")
print("=" * 75)


rf = model.named_steps["classifier"]


for name, importance in sorted(
    zip(
        FALL_FEATURE_NAMES,
        rf.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"{name:28s} "
        f"{importance:.4f}"
    )


metadata = {
    "model": model,
    "feature_names": list(FALL_FEATURE_NAMES),
    "classes": ["NORMAL", "FALL"],
    "training_samples": len(X),
    "normal_samples": len(normal_samples),
    "fall_set1_samples": len(fall_samples_1),
    "fall_set2_samples": len(fall_samples_2),
    "cv_mean": float(scores.mean()),
    "cv_std": float(scores.std()),
}


joblib.dump(
    metadata,
    MODEL_PATH
)


print("\n" + "=" * 75)
print("MODEL SAVED")
print("=" * 75)

print(f"\n{MODEL_PATH}")