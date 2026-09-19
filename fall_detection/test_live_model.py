import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# LOAD LIVE DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NORMAL_DIR = BASE_DIR / "dataset" / "live_normal"
FALL_DIR = BASE_DIR / "dataset" / "live_falls"


X = []
y = []


# ------------------------------------------------------------
# LIVE NORMAL
# ------------------------------------------------------------

for file in sorted(NORMAL_DIR.glob("*.npz")):

    data = np.load(file)

    friend_csi = np.abs(data["friend_csi"])
    you_csi = np.abs(data["you_csi"])

    from .fall_features import extract_two_receiver_fall_features

    features = extract_two_receiver_fall_features(
        friend_csi,
        you_csi,
        float(data["friend_rate"]),
        float(data["you_rate"])
    )

    X.append(features)
    y.append("NORMAL")


# ------------------------------------------------------------
# LIVE FALL
# ------------------------------------------------------------

for file in sorted(FALL_DIR.glob("*.npz")):

    data = np.load(file)

    features = data["fall_features"]

    X.append(features)
    y.append("FALL")


X = np.asarray(X)
y = np.asarray(y)


print("=" * 70)
print("LIVE DEPLOYMENT MODEL TEST")
print("=" * 70)

print()
print(f"Samples : {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"NORMAL  : {np.sum(y == 'NORMAL')}")
print(f"FALL    : {np.sum(y == 'FALL')}")


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
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# LEAVE-ONE-OUT VALIDATION
# ============================================================

print()
print("=" * 70)
print("LEAVE-ONE-OUT VALIDATION")
print("=" * 70)

loo = LeaveOneOut()

scores = cross_val_score(
    model,
    X,
    y,
    cv=loo,
    scoring="accuracy"
)

print()
print(
    f"Accuracy: {scores.mean() * 100:.2f}%"
)

print(
    f"Correct : {scores.sum():.0f}/{len(scores)}"
)


# ============================================================
# MANUAL PREDICTIONS FOR CONFUSION MATRIX
# ============================================================

predictions = []

for train_index, test_index in loo.split(X):

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )[0]

    predictions.append(prediction)


predictions = np.asarray(predictions)
# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

print()
print("=" * 70)
print("MISCLASSIFIED SAMPLES")
print("=" * 70)

sample_files = []

for file in sorted(NORMAL_DIR.glob("*.npz")):
    sample_files.append(("NORMAL", file.name))

for file in sorted(FALL_DIR.glob("*.npz")):
    sample_files.append(("FALL", file.name))

for i, (actual, predicted) in enumerate(zip(y, predictions)):

    if actual != predicted:

        print(
            f"Actual: {actual:6s} | "
            f"Predicted: {predicted:6s} | "
            f"File: {sample_files[i][1]}"
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = ["NORMAL", "FALL"]

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)

print()
print("             Predicted")
print("             NORMAL  FALL")
print(
    f"Actual NORMAL   {cm[0,0]:2d}     {cm[0,1]:2d}"
)
print(
    f"       FALL     {cm[1,0]:2d}     {cm[1,1]:2d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y,
        predictions,
        labels=labels,
        zero_division=0
    )
)


# ============================================================
# TRAIN FINAL TEMPORARY MODEL
# ============================================================

model.fit(
    X,
    y
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf = model.named_steps["classifier"]

feature_names = [
    "Friend Peak",
    "Friend Jerk",
    "Friend Stillness",
    "You Peak",
    "You Jerk",
    "You Stillness",
]

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for name, importance in sorted(
    zip(feature_names, rf.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{name:20s}: {importance:.4f}"
    )

print()
print("This is an experiment only.")
print("The existing 90-sample model was NOT modified.")
