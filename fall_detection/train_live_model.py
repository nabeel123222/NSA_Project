import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score

from .fall_features import extract_two_receiver_fall_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NORMAL_DIR = BASE_DIR / "dataset" / "live_normal"
FALL_DIR = BASE_DIR / "dataset" / "live_falls"

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = (
    MODEL_DIR
    / "fall_detector_live_24samples.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

X = []
y = []


print("=" * 70)
print("NSA - LIVE DEPLOYMENT FALL MODEL TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# LIVE NORMAL
# ------------------------------------------------------------

normal_files = sorted(
    NORMAL_DIR.glob("*.npz")
)

print()
print(f"Loading LIVE NORMAL: {len(normal_files)} samples")


for file in normal_files:

    data = np.load(file)

    friend_csi = np.abs(
        data["friend_csi"]
    )

    you_csi = np.abs(
        data["you_csi"]
    )

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

fall_files = sorted(
    FALL_DIR.glob("*.npz")
)

print(
    f"Loading LIVE FALL: {len(fall_files)} samples"
)


for file in fall_files:

    data = np.load(file)

    friend_csi = np.abs(
        data["friend_csi"]
    )

    you_csi = np.abs(
        data["you_csi"]
    )

    features = extract_two_receiver_fall_features(
        friend_csi,
        you_csi,
        float(data["friend_rate"]),
        float(data["you_rate"])
    )

    X.append(features)
    y.append("FALL")


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

X = np.asarray(X, dtype=float)
y = np.asarray(y)


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print()
print(f"Total samples : {len(X)}")
print(f"Features      : {X.shape[1]}")
print(f"NORMAL        : {np.sum(y == 'NORMAL')}")
print(f"FALL          : {np.sum(y == 'FALL')}")


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
# VALIDATION
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
    f"Accuracy : {scores.mean() * 100:.2f}%"
)

print(
    f"Correct  : {int(scores.sum())}/{len(scores)}"
)


# ============================================================
# TRAIN FINAL MODEL ON ALL 24 SAMPLES
# ============================================================

print()
print("=" * 70)
print("TRAINING FINAL LIVE MODEL")
print("=" * 70)

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
    "You Stillness"
]

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for name, importance in sorted(
    zip(
        feature_names,
        rf.feature_importances_
    ),
    key=lambda item: item[1],
    reverse=True
):

    print(
        f"{name:20s}: {importance:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_data = {
    "model": model,
    "feature_names": feature_names,
    "classes": list(model.classes_),
    "training_samples": len(X),
    "description": (
        "Deployment-specific fall detection model "
        "trained using 16 LIVE NORMAL and 8 LIVE FALL samples."
    )
}

joblib.dump(
    model_data,
    MODEL_PATH
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print()
print(MODEL_PATH)

print()
print("Existing 90-sample model was NOT modified.")
print("This is a separate deployment-specific model.")
