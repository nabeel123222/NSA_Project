import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from fall_features import extract_two_receiver_fall_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

NORMAL_TRAIN_DIR = DATASET_DIR / "normal_set2"
FALL_TRAIN_DIR = DATASET_DIR / "fall_set2"

NORMAL_TEST_DIR = DATASET_DIR / "test" / "normal"
FALL_TEST_DIR = DATASET_DIR / "test" / "fall"


FEATURE_NAMES = [
    "friend_peak_motion_energy",
    "friend_motion_jerk",
    "friend_post_event_stillness",
    "you_peak_motion_energy",
    "you_motion_jerk",
    "you_post_event_stillness",
]


# ============================================================
# LOAD ONE SAMPLE
# ============================================================

def load_sample(file_path):

    data = np.load(file_path)

    return extract_two_receiver_fall_features(
        data["friend_csi"],
        data["you_csi"],
        float(data["friend_rate"]),
        float(data["you_rate"]),
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_directory(directory, prefix, label):

    X = []
    y = []

    files = sorted(directory.glob(f"{prefix}_*.npz"))

    for file_path in files:

        try:
            features = load_sample(file_path)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"ERROR: {file_path.name}: {e}")

    return np.asarray(X), np.asarray(y)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NSA - FALL DETECTION")
print("SET 2 ONLY MODEL EXPERIMENT")
print("=" * 70)

print("\nTraining:")
print(f"  NORMAL Set 2 : {NORMAL_TRAIN_DIR}")
print(f"  FALL Set 2   : {FALL_TRAIN_DIR}")

print("\nUnseen test:")
print(f"  NORMAL       : {NORMAL_TEST_DIR}")
print(f"  FALL         : {FALL_TEST_DIR}")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

normal_X, normal_y = load_directory(
    NORMAL_TRAIN_DIR,
    "normal",
    "NORMAL"
)

fall_X, fall_y = load_directory(
    FALL_TRAIN_DIR,
    "fall",
    "FALL"
)

X_train = np.vstack([normal_X, fall_X])
y_train = np.concatenate([normal_y, fall_y])


print("\n" + "=" * 70)
print("TRAINING DATASET")
print("=" * 70)

print(f"\nNORMAL samples : {len(normal_X)}")
print(f"FALL samples   : {len(fall_X)}")
print(f"TOTAL          : {len(X_train)}")
print(f"Feature shape  : {X_train.shape}")


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"\nMean accuracy : {scores.mean() * 100:.2f}%")
print(f"Std deviation : {scores.std() * 100:.2f}%")


# ============================================================
# TRAIN FINAL TEMPORARY MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING TEMPORARY SET 2 MODEL")
print("=" * 70)

model.fit(X_train, y_train)

print("Temporary model trained.")


# ============================================================
# LOAD UNSEEN TEST DATA
# ============================================================

normal_test_X, normal_test_y = load_directory(
    NORMAL_TEST_DIR,
    "normal",
    "NORMAL"
)

fall_test_X, fall_test_y = load_directory(
    FALL_TEST_DIR,
    "fall",
    "FALL"
)

X_test = np.vstack([
    normal_test_X,
    fall_test_X
])

y_test = np.concatenate([
    normal_test_y,
    fall_test_y
])


print("\n" + "=" * 70)
print("UNSEEN TEST DATA")
print("=" * 70)

print(f"\nNORMAL test samples : {len(normal_test_X)}")
print(f"FALL test samples   : {len(fall_test_X)}")
print(f"TOTAL test samples  : {len(X_test)}")


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["NORMAL", "FALL"]
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN TEST RESULTS - SET 2 ONLY MODEL")
print("=" * 70)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nConfusion matrix:")
print("                 Predicted")
print("                 NORMAL  FALL")
print(
    f"Actual NORMAL   {cm[0,0]:6d} {cm[0,1]:5d}"
)
print(
    f"Actual FALL     {cm[1,0]:6d} {cm[1,1]:5d}"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=["NORMAL", "FALL"],
        zero_division=0
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nNORMAL correctly detected : "
    f"{cm[0,0]}/{len(normal_test_X)}"
)

print(
    f"FALL correctly detected   : "
    f"{cm[1,1]}/{len(fall_test_X)}"
)

print(
    f"False FALL detections     : "
    f"{cm[0,1]}"
)

print(
    f"Missed FALL detections    : "
    f"{cm[1,0]}"
)

print("\nExisting 90-sample model was NOT modified.")
print("=" * 70)
