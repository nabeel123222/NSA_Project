import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from fall_features import extract_two_receiver_fall_features


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

TRAIN_NORMAL = DATASET_DIR / "normal_set2"
TRAIN_FALL = DATASET_DIR / "fall_set2"

TEST_NORMAL = DATASET_DIR / "test" / "normal"
TEST_FALL = DATASET_DIR / "test" / "fall"


def load_sample(path):
    data = np.load(path)

    return extract_two_receiver_fall_features(
        data["friend_csi"],
        data["you_csi"],
        float(data["friend_rate"]),
        float(data["you_rate"]),
    )


def load_dir(directory, prefix, label):
    X = []
    y = []

    for path in sorted(directory.glob(f"{prefix}_*.npz")):
        X.append(load_sample(path))
        y.append(label)

    return np.asarray(X), np.asarray(y)


# ------------------------------------------------------------
# Training data
# ------------------------------------------------------------

normal_X, normal_y = load_dir(
    TRAIN_NORMAL, "normal", "NORMAL"
)

fall_X, fall_y = load_dir(
    TRAIN_FALL, "fall", "FALL"
)

X_train = np.vstack([normal_X, fall_X])
y_train = np.concatenate([normal_y, fall_y])


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)


# ------------------------------------------------------------
# Unseen test data
# ------------------------------------------------------------

normal_X, normal_y = load_dir(
    TEST_NORMAL, "normal", "NORMAL"
)

fall_X, fall_y = load_dir(
    TEST_FALL, "fall", "FALL"
)

X_test = np.vstack([normal_X, fall_X])
y_test = np.concatenate([normal_y, fall_y])


# ------------------------------------------------------------
# FALL probabilities
# ------------------------------------------------------------

probabilities = model.predict_proba(X_test)

classes = model.classes_
fall_index = list(classes).index("FALL")

fall_prob = probabilities[:, fall_index]


print("=" * 70)
print("NSA - FALL DETECTION")
print("FALL PROBABILITY THRESHOLD ANALYSIS")
print("=" * 70)

print("\nFALL probability for each unseen sample:")
print("-" * 70)

for i, probability in enumerate(fall_prob):
    print(
        f"{i + 1:02d} | "
        f"Actual: {y_test[i]:6s} | "
        f"FALL probability: {probability:.3f}"
    )


# ------------------------------------------------------------
# Threshold testing
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:

    predictions = np.where(
        fall_prob >= threshold,
        "FALL",
        "NORMAL"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["NORMAL", "FALL"]
    )

    normal_correct = cm[0, 0]
    false_fall = cm[0, 1]
    missed_fall = cm[1, 0]
    fall_correct = cm[1, 1]

    accuracy = np.mean(predictions == y_test)

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        f"  Accuracy       : {accuracy * 100:.2f}%"
    )

    print(
        f"  NORMAL correct : {normal_correct}/10"
    )

    print(
        f"  FALL correct   : {fall_correct}/10"
    )

    print(
        f"  False FALL     : {false_fall}"
    )

    print(
        f"  Missed FALL    : {missed_fall}"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
